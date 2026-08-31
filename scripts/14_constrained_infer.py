# -*- coding: utf-8 -*-
"""
【流水线 14/16】生产加固 · 推理期约束管线（校验→带答案重试→规则兜底）
运行位置：魔搭实例 GPU
输入：data/v2/seeds/validation/ 或 final_test_questions.jsonl
输出：output/constrained_validation.jsonl
前置步骤：09　｜　后续步骤：15 约束效果验收

推理期约束管线（生产用）：在生成后对回答做业务规则校验，违规则带修正指令重生成，
多次重试仍失败则规则化兜底修正。目标：在不重训的情况下，消除 final_test 暴露的
三类高频问题——幻觉工具名、tool_required=false 乱发 tool_call、cat3/cat6 安全话术缺失。

校验规则（与 eval_sft.py 评分口径一致，保证可比）：
  1. tool_required=true  → 必须发起 <tool_call>，工具名必须在 tool_schemas.json，
     参数值必须逐字来自用户话语（或日期格式）；
  2. tool_required=false → 不得发起 <tool_call>；
  3. category 3 → 回答必须含安全劝阻话术（请勿继续行驶/立即靠边停车等）；
  4. category 6 → 回答必须含救援确认话术（已发起救援/已为您安排等）；
  5. 半截话（<80 字且无实质内容）→ 重试。

用法（实例 /mnt/workspace）:
    python scripts/constrained_infer.py --selftest                 # 离线自检校验+兜底逻辑
    python scripts/constrained_infer.py \
        --input data/v2/seeds/validation --input-is-seed-dir \
        --output output/constrained_validation.jsonl               # 验证约束效果（推荐先用 validation）
    # 输入也可为含 rules 字段的 jsonl（如 final_test_questions.jsonl）
    # 可选 --retries N（默认3）--adapter finetuned/dpo_model_r2（默认，生产最佳模型）
"""
import argparse
import glob
import json
import os
import re

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import (RESCUE_CONFIRM_WORDS, SAFETY_STOP_WORDS, SYSTEM_PROMPT, _locate,
                      build_user_message)
from eval_sft import _TOOL_CALL_RE

BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
SFT_MERGED_DIR = os.environ.get("SFT_MERGED_DIR", "finetuned/sft_model_merged")
DPO_ADAPTER_DIR = os.environ.get("DPO_ADAPTER_DIR", "finetuned/dpo_model_r2")
TOOL_SCHEMA_PATH = "data/v2/tool_schemas.json"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
HALF_ANSWER_MIN = 80

_RETRY_GUIDE = (
    "检测到您的回答存在以下问题，请按指引修正后重新回答（只修正问题点，不要改变其余内容）：\n{issues}"
)


def build_retry_guide(seed: dict, violations: list) -> str:
    """把违规转成带正确答案的修正指引（答案来自种子标注/业务规则，非模型猜测）。

    工具选择在生产中由业务规则决定（种子 tool_name 即规则答案），重试提示带答案
    才能让模型改正——validation 首跑 28/64 fallback 的根因就是提示只有"不对"没有"该用哪个"。
    """
    lines = []
    tn = seed.get("tool_name") or ""
    for v in violations:
        if "不在可用工具列表" in v:
            lines.append(f"- 工具名错误：本场景应使用工具 '{tn}'，请按该名称发起 <tool_call>" if tn else v)
        elif "不是客户提供的信息" in v:
            lines.append(f"- 工具参数必须留空或只使用客户已提供的信息：{v}（缺失参数直接询问客户）")
        elif "回答中没有 <tool_call>" in v:
            lines.append(f"- 本场景需要发起工具查询：使用工具 '{tn}' 发起 <tool_call>（参数留空，先询问客户）")
        elif "不需要工具查询" in v:
            lines.append("- 本场景不需要任何工具查询，请直接给出安全建议和处理步骤，不要发起 <tool_call>")
        elif "安全劝阻" in v:
            lines.append("- 故障场景必须包含安全劝阻话术（如：请勿继续行驶，立即靠边停车）")
        elif "救援场景" in v:
            lines.append("- 救援场景必须包含确认话术（如：已为您发起救援，请保持电话畅通）")
        elif "半截话" in v:
            lines.append("- 回答过短：请补充处理步骤、后续安排和收尾")
        else:
            lines.append(f"- {v}")
    return _RETRY_GUIDE.format(issues="\n".join(lines))


def load_tool_names() -> set:
    """读取 tool_schemas.json 里的工具名白名单（文件缺失则返回空集，跳过工具名校验）。"""
    p = _locate(TOOL_SCHEMA_PATH)
    if not os.path.isfile(p):
        print(f"[警告] 找不到工具 schema: {p}，工具名校验跳过")
        return set()
    with open(p, encoding="utf-8") as f:
        return {t["name"] for t in json.load(f)["tools"]}


def check_violations(answer: str, seed: dict, tool_names: set) -> list:
    """按业务规则校验回答，返回违规描述列表（空=通过）。"""
    a = answer or ""
    v = []
    tool_calls = _TOOL_CALL_RE.findall(a)
    if seed.get("tool_required"):
        if not tool_calls:
            v.append("该场景需要发起工具查询，但回答中没有 <tool_call>")
        else:
            for tc_raw in tool_calls:
                try:
                    tc = json.loads(tc_raw)
                except json.JSONDecodeError:
                    v.append(f"<tool_call> 内容不是合法 JSON: {tc_raw[:40]}")
                    continue
                name = tc.get("name", "")
                if tool_names and name not in tool_names:
                    v.append(f"工具名 '{name}' 不在可用工具列表中（可用工具见 tool_schemas.json）")
                user_text = seed.get("scenario", "") + seed.get("user_goal", "")
                for k, val in (tc.get("arguments") or tc.get("parameters") or {}).items():
                    vs = str(val)
                    if vs and vs not in user_text and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", vs):
                        v.append(f"工具参数 {k}='{vs}' 不是客户提供的信息，应改为先询问客户")
    elif tool_calls:
        v.append("该场景不需要工具查询，回答不应包含 <tool_call>")
    if seed.get("category_id") == 3 and not any(w in a for w in SAFETY_STOP_WORDS):
        v.append("故障场景必须先给出安全劝阻（如：请勿继续行驶，立即靠边停车）")
    if seed.get("category_id") == 6 and not any(w in a for w in RESCUE_CONFIRM_WORDS):
        v.append("救援场景必须给出确认话术（如：已发起救援/已为您安排救援）")
    if len(a) < HALF_ANSWER_MIN and "①" not in a and "<tool_call>" not in a:
        v.append("回答过短（半截话），应补充安全提示、查询/处理步骤与后续安排")
    return v


def fix_answer(answer: str, seed: dict, tool_names: set) -> str:
    """规则化兜底修正（重试耗尽后使用）：删非法 tool_call、补安全/救援话术。

    删除用 _TOOL_CALL_RE.sub 而非 str.replace——模型输出的标签内常带换行，
    拼字符串匹配不上会静默失效。
    """
    a = answer or ""
    BLANK = "我先登记您的需求，待核实相关信息后第一时间答复您。"

    def drop_illegal(m):
        """re.sub 回调：白名单内的工具名原样保留，否则整段 tool_call 换成兜底话术。"""
        try:
            name = json.loads(m.group(1)).get("name", "")
        except json.JSONDecodeError:
            return BLANK
        return m.group(0) if (not tool_names or name in tool_names) else BLANK

    if seed.get("tool_required"):
        a = _TOOL_CALL_RE.sub(drop_illegal, a)
        if not _TOOL_CALL_RE.search(a):  # 补一个正确工具名的查询（参数留空 + 追问）
            tn = seed.get("tool_name")
            if tn:
                tc = json.dumps({"name": tn, "arguments": {}}, ensure_ascii=False)
                a = a.rstrip("。") + "。我先帮您在系统里查询一下，<tool_call>" + tc + "</tool_call>"
                if "请您确认" not in a and "还请您" not in a:
                    a += "为了更精准地处理，请您提供车辆识别码或相关凭证。"
                a += "查到结果后我会第一时间告诉您。"
    else:
        a = _TOOL_CALL_RE.sub("", a)
    if seed.get("category_id") == 3 and not any(w in a for w in SAFETY_STOP_WORDS):
        a = "请注意安全：请勿继续行驶，立即靠边停车，确保人员安全。\n" + a
    if seed.get("category_id") == 6 and not any(w in a for w in RESCUE_CONFIRM_WORDS):
        a = a + "\n我已为您发起救援安排，请保持电话畅通，等待救援人员联系。"
    return a


def generate_with_constraints(model, tokenizer, seed: dict, tool_names: set,
                              retries: int = 3) -> dict:
    """生成 → 校验 → 带修正指令重试 → 兜底修正。返回 {answer, status, attempts, violations}。"""
    import torch
    user_text = build_user_message(seed)
    guide_tail = ""
    for attempt in range(retries + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + ("\n" + guide_tail if guide_tail else "")},
            {"role": "user", "content": user_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # 首次贪心（稳定基线）；重试必须采样——贪心下同一 prompt 输出确定，
            # 修正指令没有机会跳出原违规回答（validation 首跑 4/4 fallback 即此根因）
            if attempt == 0:
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            else:
                torch.manual_seed(20260831 + attempt * 7919)
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.0,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        violations = check_violations(ans, seed, tool_names)
        if not violations:
            return {"answer": ans, "status": "pass" if attempt == 0 else f"retry{attempt}",
                    "attempts": attempt + 1, "violations": []}
        guide_tail = build_retry_guide(seed, violations)
    ans = fix_answer(ans, seed, tool_names)
    return {"answer": ans, "status": "fallback", "attempts": retries + 1,
            "violations": violations}


def load_rules_inputs(path: str, is_seed_dir: bool) -> list:
    """加载带规则字段的输入：validation/final_test 种子目录，或含 rules 的 jsonl。"""
    if is_seed_dir or os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        seeds = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                seeds += [json.loads(l) for l in f if l.strip()]
        return seeds
    seeds = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                seeds.append({
                    "seed_id": r.get("seed_id", "q"),
                    "category_id": r.get("category_id", 0),
                    "scenario": r.get("scenario", ""),
                    "user_goal": r.get("user_goal", ""),
                    "tool_required": r.get("tool_required", False),
                    "tool_name": r.get("tool_name", ""),
                    "required_facts": r.get("required_facts", r.get("facts", [])),
                    "required_questions": r.get("required_questions", r.get("questions", [])),
                    "required_actions": r.get("required_actions", r.get("actions", [])),
                    "prohibited_actions": r.get("prohibited_actions", r.get("prohibited", [])),
                })
    return seeds


def selftest() -> None:
    """离线自检校验与兜底修正逻辑（不加载模型）。"""
    names = {"vehicle_feature_query", "rescue_dispatch"}
    cat3 = {"seed_id": "T3", "category_id": 3, "scenario": "行驶中仪表故障灯亮",
            "user_goal": "安全停车并检查", "tool_required": False, "tool_name": None}
    cat6 = {"seed_id": "T6", "category_id": 6, "scenario": "高速抛锚",
            "user_goal": "尽快获得救援", "tool_required": True, "tool_name": "rescue_dispatch"}

    ok = "您好，请先确认车辆位置安全。" \
         "请注意安全：请勿继续行驶，立即靠边停车。我为您安排救援：<tool_call>{\"name\": \"rescue_dispatch\", \"arguments\": {}}</tool_call>已为您发起救援，请保持电话畅通。"
    bad_tool = "我帮您查一下：<tool_call>{\"name\": \"battery_health_query\", \"arguments\": {\"vehicle_id\": \"VIN999\"}}</tool_call>"
    bad_false_call = "请确认：<tool_call>{\"name\": \"rescue_dispatch\", \"arguments\": {}}</tool_call>"
    half = "请问您现在在哪？"

    assert check_violations(ok, cat6, names) == [], check_violations(ok, cat6, names)
    v = check_violations(bad_tool, cat6, names)
    assert any("不在可用工具列表" in x for x in v), v
    assert any("不是客户提供的信息" in x for x in v), v
    v = check_violations(bad_false_call, cat3, names)
    assert any("不需要工具查询" in x for x in v), v
    v = check_violations(half, cat3, names)
    assert any("半截话" in x for x in v), v
    v = check_violations("您好，请先确认位置。", cat6, names)
    assert any("救援场景" in x for x in v), v

    f = fix_answer("我帮您查一下：<tool_call>{\"name\": \"battery_health_query\", \"arguments\": {}}</tool_call>", cat6, names)
    assert "battery_health_query" not in f and "登记您的需求" in f, f
    f = fix_answer("请确认：<tool_call>{\"name\": \"rescue_dispatch\", \"arguments\": {}}</tool_call>", cat3, names)
    assert "<tool_call>" not in f, f
    f = fix_answer("您好，请确认位置。", cat6, names)
    assert "已为您发起救援" in f, f
    f = fix_answer("您好，请确认位置。", cat6, names)
    assert "rescue_dispatch" in f and "<tool_call>" in f, f  # 缺 tool_call 时兜底补正确工具
    f = fix_answer("您好，请确认位置。", cat3, names)
    assert "请勿继续行驶" in f, f

    # 修正指引：必须带正确答案（工具名/必含话术），而不是只说"不对"
    g = build_retry_guide(cat6, ["工具名 'battery_health_query' 不在可用工具列表中（可用工具见 tool_schemas.json）",
                                 "工具参数 vehicle_id='V123' 不是客户提供的信息，应改为先询问客户"])
    assert "rescue_dispatch" in g, g
    g = build_retry_guide(cat3, ["故障场景必须先给出安全劝阻（如：请勿继续行驶，立即靠边停车）",
                                 "该场景不需要工具查询，回答不应包含 <tool_call>"])
    assert "不需要任何工具查询" in g and "请勿继续行驶" in g, g
    g = build_retry_guide(cat6, ["救援场景必须给出确认话术（如：已发起救援/已为您安排救援）"])
    assert "已为您发起救援" in g, g
    print("[自检通过] 校验 5 类规则 + 兜底修正 4 类场景 + 修正指引带正确答案 全部符合预期")


def main() -> None:
    """约束推理主流程：加载工具白名单与输入 → 挂 DPO Adapter 逐条约束生成 → 增量落盘并统计
    直接通过/重试通过/兜底修正三类占比（支持断点续跑）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--input", default="data/v2/seeds/validation",
                    help="种子目录或含 rules 的 jsonl（默认 validation 种子目录）")
    ap.add_argument("--input-is-seed-dir", action="store_true", help="输入是种子目录")
    ap.add_argument("--output", default="output/constrained_validation.jsonl")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--adapter", default=DPO_ADAPTER_DIR, help="推理加载的 LoRA adapter（默认 dpo_model_r2）")
    ap.add_argument("--base-only", action="store_true", help="只用 SFT 合并权重（不挂 adapter）")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tool_names = load_tool_names()
    seeds = load_rules_inputs(args.input, args.input_is_seed_dir)
    print(f"[输入] {len(seeds)} 条，工具白名单 {len(tool_names)} 个")
    print(f"[加载] {SFT_MERGED_DIR}" + ("" if args.base_only else f" + {args.adapter}"))
    tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if not args.base_only:
        if not os.path.isdir(args.adapter):
            raise SystemExit(f"[错误] 找不到 Adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stats = {"pass": 0, "retry": 0, "fallback": 0}
    done = set()
    if os.path.exists(args.output):  # 断点续跑：已有 seed_id 跳过
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line).get("seed_id"))
        if done:
            print(f"[续跑] 已有 {len(done)} 条，跳过继续")
    with open(args.output, "a", encoding="utf-8") as f:
        for i, seed in enumerate(seeds, 1):
            if seed.get("seed_id") in done:
                continue
            r = generate_with_constraints(model, tokenizer, seed, tool_names, retries=args.retries)
            stats["pass" if r["status"] == "pass" else ("retry" if r["status"].startswith("retry") else "fallback")] += 1
            f.write(json.dumps({
                "seed_id": seed.get("seed_id"), "category_id": seed.get("category_id"),
                "question": build_user_message(seed), "answer": r["answer"],
                "status": r["status"], "attempts": r["attempts"],
                "violations": r["violations"],
            }, ensure_ascii=False) + "\n")
            f.flush()
            tail = f"  违规: {'; '.join(r['violations'][:2])}" if r["violations"] else ""
            print(f"[{i}/{len(seeds)}] {seed.get('seed_id')} {r['status']} "
                  f"({r['attempts']}次尝试){tail}", flush=True)
    print(f"\n[完成] 约束结果统计: 直接通过 {stats['pass']} / 重试后通过 {stats['retry']} / "
          f"兜底修正 {stats['fallback']}（共 {len(seeds)} 条）")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
