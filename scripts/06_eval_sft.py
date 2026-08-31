# -*- coding: utf-8 -*-
"""
【流水线 06/16】里程碑三 · SFT 模型系统评估（7 维打分 + Bad Case）
运行位置：实例 GPU 生成 ＋ Judge API 打分
输入：data/v2/seeds/validation/（64 条）、finetuned/sft_model_r2/
输出：output/sft_eval_results.json、sft_eval_report.md、sft_bad_cases.jsonl
前置步骤：04　｜　后续步骤：07 构造 DPO 数据

SFT 模型系统评估：对 data/v2/seeds/validation/ 的 64 条种子逐条生成回答，
按种子规则字段做规则化打分（7 维，每维 0/1/2），统计 DPO 靶向缺陷命中率，
产出 output/sft_eval_results.json / sft_eval_report.md / sft_bad_cases.jsonl。

用法（实例项目根目录 /mnt/workspace）:
    python scripts/eval_sft.py                    # 完整评估（需 GPU + 模型）
    python scripts/eval_sft.py --selftest         # 本地自检打分逻辑（不加载模型）

注意：final_test/ 绝不参与本评估。
"""
import argparse
import glob
import json
import os
import re
import sys

# ---------------- 固定常量 ----------------
BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "finetuned/sft_model")
SEED_DIR = "data/v2/seeds/validation"
TOOL_SCHEMA_PATH = "data/v2/tool_schemas.json"
OUTPUT_DIR = "output"
MAX_NEW_TOKENS = 512
DO_SAMPLE = False  # 评估用确定性贪心解码，保证结果可复现

SYSTEM_PROMPT = (
    "你是汽车售后服务智能客服助手，请以专业、亲切、安全的口吻回答客户问题。"
    "涉及具体政策或数据时，如需查询请说明已发起查询或请客户提供信息，不得编造。"
)

# ---------------- 缺陷检测关键词 ----------------
SAFETY_STOP_WORDS = ["请勿继续行驶", "不要继续行驶", "停止使用", "立即靠边", "靠边停车",
                     "尽快停车", "不要启动", "切勿继续", "不要再开", "停车检查", "停止驾驶"]
RESCUE_CONFIRM_WORDS = ["已发起救援", "已为您安排", "已安排救援", "已为您发起", "救援已发起",
                        "已为您联系", "已发起", "已为您预约救援"]
HALF_ANSWER_MAX_LEN = 80          # 少于此长度且无①②条目，视为模板式"半截话"
FACT_COVER_RATIO = 0.45           # 长句关键词 bigram 覆盖率阈值（中文换词容错）
PROHIBIT_HIT_RATIO = 0.5
NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|年|个月|月|天|日|次|公里|km|万|元|小时|km/h)")
_PUNCT = re.compile(r"[，。！？、；：“”‘’（）,.!?;:()\s]")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

# ---------------- LLM Judge（优先，规则分作降级）----------------
# 优先用 ModelScope 免费推理 API（实例自带 MODELSCOPE_API_KEY），也可换 DashScope/OpenAI 兼容端点
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api-inference.modelscope.cn/v1")
JUDGE_API_KEY = (os.environ.get("JUDGE_API_KEY") or os.environ.get("MODELSCOPE_API_KEY")
                 or os.environ.get("DASHSCOPE_API_KEY", ""))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
JUDGE_DIMS = ["理解正确性", "required_facts覆盖", "required_questions追问", "required_actions正确",
              "prohibited_actions违规", "完整性自然度", "是否编造"]

JUDGE_PROMPT = """你是汽车售后客服回答的质检专家。给定客户问题、业务规则和客服回答，请严格逐项打分（每项0/1/2分）：
1. 理解正确性：是否正确理解客户问题与诉求（2=完全正确，1=部分理解或答偏，0=理解错误/空答）
2. required_facts覆盖：回答是否覆盖全部必需事实要点（2=全部，1=过半，0=不足半）
3. required_questions追问：是否包含所需的追问/确认问题（2=全部，1=过半，0=不足半）
4. required_actions正确：是否完成要求的动作（如安全分流、发起查询、给出路径）（2=全部，1=过半，0=不足半）
5. prohibited_actions违规：是否违反禁止项（2=无违规，0=有违规）
6. 完整性自然度：回答是否完整自然、非套话模板（2=完整自然，1=一般，0=半截话/纯模板/一句反问）
7. 是否编造：是否编造政策数字、工单号、查询结果、用户未提供的参数值（2=无编造，1=疑似，0=明确编造）
注意：tool_required=true 时正确做法是发起 <tool_call> 且参数只能取自用户话语；直接给查询结果属于编造。
只输出JSON：{{"理解正确性":x,"required_facts覆盖":x,"required_questions追问":x,"required_actions正确":x,"prohibited_actions违规":x,"完整性自然度":x,"是否编造":x,"原因":"一句话"}}

[客户问题] {question}
[业务规则] 必需事实：{facts}；必需追问：{questions}；必需动作：{actions}；禁止项：{prohibited}；需要工具：{tool_required}（工具名 {tool_name}）
[客服回答] {answer}"""


def judge_with_llm(seed: dict, answer: str) -> dict | None:
    """调 LLM Judge 打分；失败返回 None（降级用规则分）。"""
    import requests
    prompt = JUDGE_PROMPT.format(
        question=build_user_message(seed),
        facts="；".join(seed["required_facts"]) or "无",
        questions="；".join(seed["required_questions"]) or "无",
        actions="；".join(seed["required_actions"]) or "无",
        prohibited="；".join(seed["prohibited_actions"]) or "无",
        tool_required=seed["tool_required"], tool_name=seed["tool_name"] or "-",
        answer=answer or "（空）")
    try:
        import time
        resp = None
        for attempt in range(6):  # 429 限速退避：5/10/20/40/80 秒
            payload = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 300,
                       "messages": [{"role": "user", "content": prompt}]}
            if "deepseek" in JUDGE_MODEL:  # 压短思考，避免 content 为空
                payload["reasoning_effort"] = "low"
            resp = requests.post(f"{JUDGE_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"}, json=payload, timeout=120)
            if resp.status_code == 429 and attempt < 5:
                wait = 5 * (2 ** attempt)
                print(f"  [429限速] 第{attempt + 1}次重试，等待{wait}s")
                time.sleep(wait)
                continue
            break
        resp.raise_for_status()
        time.sleep(2)  # 请求间隔，降低限速概率
        content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        m = re.search(r"\{.*\}", content, re.S)
        if m is None:
            print(f"  [Judge输出无JSON] 内容片段: {content[:100]!r}")
            return None
        out = json.loads(m.group(0))
        if not all(d in out for d in JUDGE_DIMS):
            raise ValueError(f"维度缺失: {out}")
        return {d: int(out[d]) for d in JUDGE_DIMS} | {"judge_reason": str(out.get("原因", ""))}
    except Exception as e:  # noqa: BLE001 - 任何失败都降级到规则分
        print(f"  [Judge失败→规则分] {type(e).__name__}: {e}")
        return None


def _bigrams(text: str) -> set:
    t = _PUNCT.sub("", text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t} if t else set()


def _covered(sentence: str, answer: str, ratio: float = FACT_COVER_RATIO) -> bool:
    """长句无法整句命中时，用 bigram 覆盖率近似判断答案是否覆盖了该句要点；
    兜底：句中任一 8 字连续片段原样出现即算命中（换词容错 + 精确片段强信号）。"""
    if not sentence:
        return True
    if sentence in answer:
        return True
    t = _PUNCT.sub("", sentence)
    if len(t) >= 8 and any(t[i:i + 8] in answer for i in range(len(t) - 7)):
        return True
    grams = _bigrams(sentence)
    if not grams:
        return True
    hit = sum(1 for g in grams if g in answer)
    return hit / len(grams) >= ratio


def _cover_level(items, answer: str) -> tuple:
    """返回 (covered_count, total, level 0/1/2)。"""
    total = len(items)
    if total == 0:
        return 0, 0, 2
    ok = sum(1 for it in items if _covered(it, answer))
    level = 2 if ok == total else (1 if ok >= max(1, (total + 1) // 2) else 0)
    return ok, total, level


def score_answer(seed: dict, answer: str, tool_schemas: dict) -> dict:
    """对单条回答做规则化打分，返回 7 维得分 + 缺陷标记。"""
    a = answer.strip()
    d = {}
    flags = {}

    # 1 理解正确性：回答是否围绕场景话题展开
    grams = _bigrams(seed["scenario"])
    topic_hit = sum(1 for g in grams if g in a) / max(len(grams), 1)
    d["理解正确性"] = 2 if (a and topic_hit >= 0.12) else (1 if a else 0)
    flags["话题相关度"] = round(topic_hit, 2)

    # 2 required_facts 覆盖
    ok_f, tot_f, d["required_facts覆盖"] = _cover_level(seed["required_facts"], a)
    # 3 required_questions 追问
    ok_q, tot_q, d["required_questions追问"] = _cover_level(seed["required_questions"], a)
    # 4 required_actions 正确
    ok_act, tot_act, d["required_actions正确"] = _cover_level(seed["required_actions"], a)
    flags["facts覆盖数"] = f"{ok_f}/{tot_f}"
    flags["questions覆盖数"] = f"{ok_q}/{tot_q}"
    flags["actions覆盖数"] = f"{ok_act}/{tot_act}"

    # 5 prohibited_actions 违规（违规得 0）
    prohibited = [p for p in seed.get("prohibited_actions") or [] if p]
    violation = any(_covered(p, a, PROHIBIT_HIT_RATIO) for p in prohibited)
    d["prohibited_actions违规"] = 0 if violation else 2
    flags["违规条款"] = next((p for p in prohibited if _covered(p, a, PROHIBIT_HIT_RATIO)), "")

    # 6 完整性自然度 + 半截话检测（模板式短句：极短、无①②条目、无工具调用、无实质内容）
    sentences = [s for s in re.split(r"[。！？!?\n]", a) if s.strip()]
    is_half = len(a) < HALF_ANSWER_MAX_LEN and "①" not in a and "<tool_call>" not in a
    if is_half:
        d["完整性自然度"] = 0
    elif len(a) >= 100 and len(sentences) >= 2:
        d["完整性自然度"] = 2
    else:
        d["完整性自然度"] = 1
    flags["半截话"] = is_half

    # 7 是否编造政策数字 / 工单 / 查询结果
    fab_level = 2
    fab_reasons = []
    nums = NUM_PATTERN.findall(a)
    seed_text = "".join(seed["required_facts"] + seed["required_actions"] + seed["required_questions"])
    suspicious = [n for n in nums if n.strip() not in seed_text]
    if suspicious:
        fab_level = 1
        fab_reasons.append(f"疑似编造具体数字: {suspicious}")

    tool_calls = _TOOL_CALL_RE.findall(a)
    if seed.get("tool_required"):
        if not tool_calls:
            fab_reasons.append("tool_required=true 但未发起 <tool_call>")
            fab_level = min(fab_level, 1)
        else:
            try:
                tc = json.loads(tool_calls[0])
                name = tc.get("name", "")
                if tool_schemas and name not in tool_schemas:
                    fab_reasons.append(f"工具名不匹配 schema: {name}")
                    fab_level = 0
                # 参数编造检测：参数值必须能在用户话语/场景描述中找到来源
                user_text = seed["scenario"] + seed["user_goal"]
                for k, v in (tc.get("arguments") or tc.get("parameters") or {}).items():
                    vs = str(v)
                    if vs and vs not in user_text and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", vs):
                        fab_reasons.append(f"参数值疑似编造: {k}={vs}（用户未提供）")
                        fab_level = 0
            except json.JSONDecodeError:
                fab_reasons.append("<tool_call> 内容非法 JSON")
                fab_level = 0
    elif tool_calls:
        fab_reasons.append("tool_required=false 却发起了 <tool_call>")
        fab_level = 0
    # 直接伪造查询结果（未发起工具调用却给出"查询到/查到…结果"）
    if re.search(r"查询(到|结果[为是])", a) and not tool_calls:
        fab_reasons.append("未发起工具调用却给出查询结果")
        fab_level = 0
    d["是否编造"] = fab_level
    flags["编造疑点"] = fab_reasons

    # ---- DPO 靶向缺陷命中 ----
    if seed["category_id"] == 3:
        flags["安全劝阻缺失"] = not any(w in a for w in SAFETY_STOP_WORDS)
    if seed["category_id"] == 6:
        flags["救援确认缺失"] = not any(w in a for w in RESCUE_CONFIRM_WORDS)
    if seed.get("tool_required"):
        flags["tool_call缺失"] = not tool_calls

    d["总分"] = sum(d[k] for k in ["理解正确性", "required_facts覆盖", "required_questions追问",
                                   "required_actions正确", "prohibited_actions违规",
                                   "完整性自然度", "是否编造"])
    flags["缺陷原因"] = [r for r in fab_reasons] + (
        [flags["违规条款"] + "（违反禁止项）"] if violation else [])
    return {"scores": d, "flags": flags}


def _locate(rel: str, marker: str = "") -> str:
    """在 cwd 及其子目录/上级目录中定位数据路径，兼容实例上项目不在 cwd 的情况。"""
    if os.path.exists(os.path.join(rel, marker) if marker else rel):
        return rel
    root = os.getcwd()
    script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根（scripts/ 的上一级）
    for base in (root, os.path.dirname(root), script_root, os.path.dirname(script_root)):
        for dirpath, dirnames, _ in os.walk(base):
            # 跳过模型权重等大目录
            dirnames[:] = [d for d in dirnames if d not in
                           ("models", "finetuned", "output", "__pycache__", ".git")]
            cand = os.path.join(dirpath, rel)
            if os.path.exists(os.path.join(cand, marker) if marker else cand):
                return cand
    return rel  # 返回原路径，让后续报错信息直接可读


def load_seeds() -> list:
    seed_dir = _locate(SEED_DIR, marker="category1_validation.jsonl")
    if seed_dir != SEED_DIR:
        print(f"[提示] 数据目录定位为: {seed_dir}")
    seeds = []
    for fp in sorted(glob.glob(os.path.join(seed_dir, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            seeds += [json.loads(l) for l in f if l.strip()]
    assert len(seeds) == 64, \
        f"预期 64 条 validation 种子，实际 {len(seeds)}（查找目录: {seed_dir}，当前 cwd: {os.getcwd()}）"
    return seeds


def load_tool_schemas() -> dict:
    p = _locate(TOOL_SCHEMA_PATH)
    if not os.path.isfile(p):
        print(f"[警告] 找不到工具 schema: {p}，将跳过工具名校验")
        return {}
    with open(p, encoding="utf-8") as f:
        return {t["name"] for t in json.load(f)["tools"]}


def build_user_message(seed: dict) -> str:
    return f"您好，{seed['scenario']}。我的诉求是：{seed['user_goal']}。请帮我处理。"


def generate_answers(seeds: list) -> list:
    """加载 base + LoRA，逐条贪心生成（同 smoke_infer.py 的加载方式）。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit(f"[错误] 找不到 Adapter 目录: {ADAPTER_DIR}")
    print("[加载] 基座模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    print("[加载] LoRA Adapter ...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR).eval()

    answers = []
    for i, seed in enumerate(seeds, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(seed)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
                temperature=None, top_p=None, top_k=None,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answers.append(ans)
        print(f"[{i}/64] {seed['seed_id']} 生成完毕（{len(ans)} 字）")
    return answers


def selftest() -> None:
    """用 3 个构造样例自检打分逻辑，不加载模型。"""
    seed = {
        "seed_id": "T1", "category_id": 3, "category": "故障预诊断与安全分流",
        "scenario": "夜间行驶时一侧近光灯突然不亮", "user_goal": "保证照明安全并安排检查",
        "required_facts": ["远程预诊断不能代替现场检测，需先确认影响安全的伴随现象"],
        "required_questions": ["请确认车辆是否已在安全位置以及是否有警告灯、异响、烟雾或操控异常"],
        "required_actions": ["先完成安全分流，再给出观察、停车、救援或检查建议"],
        "prohibited_actions": ["不得远程确定具体故障零件，也不得在存在安全异常时建议继续驾驶"],
        "tool_required": False, "tool_name": None,
    }
    schemas = {"battery_warranty_query", "bodyshop_booking"}
    cases = [
        # 应为高分
        ("好的，夜间灯光问题涉及行车安全，请先确认车辆是否已在安全位置，是否有警告灯、异响、烟雾或操控异常。"
         "远程预诊断不能代替现场检测，建议先完成安全分流：请勿继续行驶，立即靠边停车检查。"
         "如伴随异常我会为您安排救援，否则再预约到店检修。", 14),
        # 半截话 + 无劝阻，应为低分
        ("您的车灯不亮了，请问您在哪？", 5),
    ]
    # 编造参数（tool_required=true 场景），应为低分
    tc_seed = dict(seed, tool_required=True, tool_name="battery_warranty_query")
    tc_text = ('<tool_call>{"name": "battery_warranty_query", "arguments": {"vehicle_id": "VIN123", '
               '"trim": "2024款长续航版"}}</tool_call>')
    r1 = score_answer(seed, cases[0][0], schemas)
    r2 = score_answer(seed, cases[1][0], schemas)
    r3 = score_answer(tc_seed, tc_text, schemas)
    for label, s, text in [("高分样例", seed, cases[0][0]),
                           ("半截话样例", seed, cases[1][0]),
                           ("编造参数样例", tc_seed, tc_text)]:
        r = score_answer(s, text, schemas)
        print(f"[{label}] 输入: {text[:50]}...\n得分: {r['scores']}\n标记: {r['flags']}\n{'-' * 50}")
    assert r1["scores"]["总分"] >= 12 and r1["scores"]["prohibited_actions违规"] == 2
    assert r2["scores"]["完整性自然度"] == 0 and r2["flags"]["半截话"]
    assert r3["scores"]["是否编造"] == 0 and any("编造" in x for x in r3["flags"]["编造疑点"])
    print("[自检通过] 打分逻辑符合预期")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="复用 output/sft_eval_results.json 里已生成的回答重新打分，不重新生成")
    ap.add_argument("--judge", action="store_true",
                    help="用 LLM Judge（ModelScope API）对7维重新打分，失败降级规则分")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    seeds = load_seeds()
    schemas = load_tool_schemas()
    use_judge = args.judge and bool(JUDGE_API_KEY)
    if args.judge and not JUDGE_API_KEY:
        print("[警告] 未设置 MODELSCOPE_API_KEY/DASHSCOPE_API_KEY，仅用规则打分")

    def maybe_judge(seed, answer, rule_result):
        if not use_judge:
            return rule_result
        out = judge_with_llm(seed, answer)
        if out is None:
            return rule_result
        reason = out.pop("judge_reason", "")
        rule_result = dict(rule_result)
        rule_result["scores"] = dict(out, 总分=sum(out.values()))
        rule_result["flags"] = dict(rule_result["flags"])
        rule_result["flags"]["缺陷原因"] = ([reason] if reason else []) + rule_result["flags"]["缺陷原因"]
        rule_result["flags"]["judge"] = True
        return rule_result

    if args.rescore:
        with open(os.path.join(OUTPUT_DIR, "sft_eval_results.json"), encoding="utf-8") as f:
            prev = json.load(f)
        by_id = {s["seed_id"]: s for s in seeds}
        results = []
        for i, p in enumerate(prev, 1):
            seed = by_id[p["seed_id"]]
            r = maybe_judge(seed, p["answer"], score_answer(seed, p["answer"], schemas))
            results.append({k: p[k] for k in ("seed_id", "category_id", "category",
                                              "subcategory", "question", "answer")} | r)
            print(f"[rescore {i}/{len(prev)}] {p['seed_id']} 总分 {r['scores']['总分']}"
                  + ("（LLM Judge）" if r["flags"].get("judge") else "（规则）"))
    else:
        answers = generate_answers(seeds)
        results = []
        for seed, ans in zip(seeds, answers):
            r = maybe_judge(seed, ans, score_answer(seed, ans, schemas))
            results.append({
                "seed_id": seed["seed_id"], "category_id": seed["category_id"],
                "category": seed["category"], "subcategory": seed["subcategory"],
                "question": build_user_message(seed), "answer": ans,
                **r,
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "sft_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---- Bad Cases：总分<=10 或任一维度 0 分 ----
    bad = [r for r in results if r["scores"]["总分"] <= 10 or
           any(r["scores"][k] == 0 for k in ["required_facts覆盖", "required_questions追问",
                                             "required_actions正确", "prohibited_actions违规",
                                             "完整性自然度", "是否编造"])]
    with open(os.path.join(OUTPUT_DIR, "sft_bad_cases.jsonl"), "w", encoding="utf-8") as f:
        for r in bad:
            seed = next(s for s in seeds if s["seed_id"] == r["seed_id"])
            f.write(json.dumps({
                "seed_id": r["seed_id"], "category_id": r["category_id"],
                "question": r["question"], "answer": r["answer"],
                "scores": r["scores"], "seed_rules": {
                    "required_facts": seed["required_facts"],
                    "required_questions": seed["required_questions"],
                    "required_actions": seed["required_actions"],
                    "prohibited_actions": seed["prohibited_actions"],
                    "tool_required": seed["tool_required"], "tool_name": seed["tool_name"],
                },
                "judge_failure_reasons": r["flags"]["缺陷原因"] or
                    [f"{k} 得 0 分" for k, v in r["scores"].items() if v == 0 and k != "总分"],
            }, ensure_ascii=False) + "\n")

    write_report(results, bad, seeds)
    print(f"\n[完成] 共 64 条，Bad Case {len(bad)} 条。")
    print("输出: output/sft_eval_results.json, output/sft_eval_report.md, output/sft_bad_cases.jsonl")


DIM_KEYS = ["理解正确性", "required_facts覆盖", "required_questions追问", "required_actions正确",
            "prohibited_actions违规", "完整性自然度", "是否编造"]


def _rate(results: list, key: str) -> float:
    return round(sum(r["scores"][key] for r in results) / (2 * len(results)) * 100, 1)


def write_report(results: list, bad: list, seeds: list) -> None:
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r["category_id"], []).append(r)

    lines = ["# SFT 模型评估报告（validation 64 条，规则化打分）", "",
             f"- 模型：{BASE_MODEL_DIR} + LoRA（{ADAPTER_DIR}），bf16 / sdpa，贪心解码（do_sample=False），max_new_tokens={MAX_NEW_TOKENS}",
             f"- 数据：data/v2/seeds/validation/ 64 条（8 类 × 8 条），未使用任何训练样本，final_test 未触碰",
             f"- 总均分：{sum(r['scores']['总分'] for r in results) / len(results):.2f} / 14　"
             f"Bad Case：{len(bad)} 条（{len(bad) / len(results) * 100:.0f}%）", "",
             "## 一、8 类场景分组统计", "",
             "| 类别 | 条数 | 均分/14 | 理解 | facts | questions | actions | 无违规 | 完整 | 无编造 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cid in sorted(by_cat):
        rs = by_cat[cid]
        row = [rs[0]["category"], str(len(rs)),
               f"{sum(r['scores']['总分'] for r in rs) / len(rs):.2f}"]
        row += [f"{_rate(rs, k)}%" for k in DIM_KEYS]
        lines.append("| " + " | ".join(row) + " |")

    # 缺陷命中率
    def hit(name): return sum(1 for r in results if r["flags"].get(name))

    lines += ["", "## 二、DPO 靶向缺陷命中率", "",
              f"- 安全分流不彻底（cat3 缺少明确劝阻话术）：{hit('安全劝阻缺失')}/8 "
              f"（判定词：{'、'.join(SAFETY_STOP_WORDS[:5])} 等）",
              f"- 救援确认话术缺失（cat6 无\u201c已发起救援/已为您安排\u201d类确认）：{hit('救援确认缺失')}/8",
              f"- tool_call 缺失（tool_required=true 却未发起调用）：{hit('tool_call缺失')}"
              f"/{sum(1 for r in results if 'tool_call缺失' in r['flags'] or True) and sum(1 for s in seeds if s['tool_required'])}",
              f"- 半截话（仅一句反问、无实质内容）：{hit('半截话')} 条",
              f"- prohibited 违规：{sum(1 for r in results if r['scores']['prohibited_actions违规'] == 0)} 条",
              f"- 编造问题（数字/工具名/参数/查询结果）：{sum(1 for r in results if r['scores']['是否编造'] < 2)} 条", ""]

    # 各类缺陷样本
    lines.append("## 三、缺陷样本清单（附回答原文）")
    for cid in sorted(by_cat):
        cat_bad = [r for r in by_cat[cid] if r in bad]
        if not cat_bad:
            continue
        lines += ["", f"### 类别{cid} {by_cat[cid][0]['category']}（{len(cat_bad)} 条）"]
        for r in cat_bad:
            lines += [f"- **{r['seed_id']}**（{r['subcategory']}）得分 {r['scores']['总分']}/14，"
                      f"原因：{('；'.join(r['flags']['缺陷原因'])) or '、'.join(k for k in DIM_KEYS if r['scores'][k] == 0)}",
                      f"  - 问题：{r['question']}",
                      f"  - 回答原文：{r['answer'][:600]}{'…' if len(r['answer']) > 600 else ''}"]

    # 最主要问题
    defect_counts = {
        "安全分流不彻底": sum(1 for r in results if r["flags"].get("安全劝阻缺失")),
        "救援确认话术缺失": sum(1 for r in results if r["flags"].get("救援确认缺失")),
        "tool_call 缺失/参数编造": sum(1 for r in results if r["flags"].get("tool_call缺失")) +
                                   sum(1 for r in results if r["scores"]["是否编造"] == 0),
        "半截话": hit("半截话"),
        "prohibited 违规": sum(1 for r in results if r["scores"]["prohibited_actions违规"] == 0),
        "facts 覆盖不足": sum(1 for r in results if r["scores"]["required_facts覆盖"] < 2),
    }
    top3 = sorted(defect_counts.items(), key=lambda x: -x[1])[:3]
    lines += ["", "## 四、SFT 模型当前 3 个最主要的问题", ""]
    for i, (name, n) in enumerate(top3, 1):
        lines.append(f"{i}. **{name}**（命中 {n}/64）")
    lines += ["", "## 五、DPO 偏好数据建议（把高频缺陷写成 rejected 特征）", "",
              "对上述每类缺陷，从 validation 缺陷样本出发构造偏好对（chosen 参照种子 required_* 字段人工/规则改写）："]
    dpo_map = [
        ("安全劝阻缺失", "rejected：故障场景只谈排查步骤、无任何停车/勿继续行驶的安全劝阻；"
         "chosen：先明确\u201c请勿继续行驶、立即靠边停车\u201d类安全分流，再给排查与救援安排。"),
        ("救援确认缺失", "rejected：救援场景仅口头建议或让客户\u201c稍等\u201d；"
         "chosen：包含\u201c已发起救援/已为您安排\u201d确认话术并告知等待要点。"),
        ("tool_call缺失", "rejected：tool_required=true 时只说不查、或凭空口播查询结果；"
         "chosen：发起 <tool_call>，工具名与 schema 一致，参数仅取自用户话语，缺参时先追问。"),
        ("半截话", "rejected：整条回答只有一句反问、无实质内容；"
         "chosen：追问同时给出安全提示、可执行步骤与后续安排。"),
        ("prohibited违规", "rejected：远程断言具体故障零件、或在安全异常时建议继续驾驶；"
         "chosen：说明远程诊断边界并优先安全分流。"),
        ("编造", "rejected：编造政策数字（\u201c3年10万公里\u201d类）、用户未提供的参数值（如\u201c2024款长续航版\u201d）、"
         "或无工具调用却给出查询结果；chosen：数字/结果一律说明需查询，参数留空追问客户。"),
    ]
    for name, desc in dpo_map:
        n = defect_counts.get(name, sum(1 for r in results if r["flags"].get(name)))
        if n:
            lines.append(f"- **{name}**（{n} 条）：{desc}")
    lines.append("\n> 注：rejected 特征均来自本评估实测命中率最高的缺陷；"
                 "构造偏好对时 chosen 必须逐条满足对应种子的 required_facts/questions/actions。")

    with open(os.path.join(OUTPUT_DIR, "sft_eval_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
