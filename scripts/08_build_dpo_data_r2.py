# -*- coding: utf-8 -*-
"""
【流水线 08/16】里程碑四 · DPO 偏好数据二轮扩充（450 对，正式采用）
运行位置：本机（无需 GPU）
输入：data/v2/seeds/train/ 种子（复用 07 的加载与参数表）
输出：data/v2/dpo_r2/dpo_train.jsonl、dpo_train_trl.jsonl、报告
前置步骤：07　｜　后续步骤：09 DPO 训练

二轮 DPO 数据扩充脚本（基于种子合成，无 GPU 依赖）。

核心问题诊断：
  round 1 的 180 对数据 chosen 模板高度统一（全是"这里先跟您说明X点关键信息..."），
  模型过拟合到"模板区分"而非"规则理解"。本轮目标：

  1. 量：从 180 → 400+ 对（target 500）
  2. 质：chosen 多样化（3 种以上结构变体，不再统一模板）
  3. 点：重点补半截话（42/64→目标<20）、cat3 安全劝阻（0/8→5+）、cat6 救援确认（0/8→5+）
  4. 去：去掉 round 1 的 chosen 模板，用多样化手写风格替代

数据来源（全部种子合成，不依赖真实模型输出）：
  - train 种子 640 条 × 多种 chosen 变体 + 多种 rejected 变体
  - 从 validation 种子中额外补充（但 validation 不参与 DPO 训练，仅在 chosen 生成时用种子规则校验）

输出：data/v2/dpo_r2/ 目录下 5 个文件，格式与 round 1 一致。

用法：
    python scripts/build_dpo_data_r2.py --selftest
    python scripts/build_dpo_data_r2.py
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import SYSTEM_PROMPT, score_answer, _locate  # noqa: E402
from build_dpo_data import (load_seed_dir, load_schemas, load_jsonl,
                             PARAM_LABELS, FREE_TEXT_PARAMS, FAB_POOL,
                             CIRCLED, cn)

TRAIN_DIR = "data/v2/seeds/train"
OUT_DIR = "data/v2/dpo_r2"

# ---------- 生成参数 ----------
TARGET_PAIRS = 500
MIN_SCORE = 12          # chosen 最低分
MIN_MARGIN = 4          # 分差
LEN_RATIO = 1.5         # 长度惩罚阈值
SEED = 42

# ---------- ① 多样化 chosen 模板（3 种风格，打破 round 1 的单一模板） ----------

def _chosen_style_a(seed: dict, tool_call_seg: str, missing_seg: str, question_seg: str,
                    action_seg: str) -> str:
    """风格 A：先共情 + 说明关键信息 + 工具 + 追问 + 处理建议。"""
    parts = []
    cat = seed["category_id"]
    sc = seed["scenario"]

    # 开场多样化（非 cat3/cat6 也给共情变体）
    openers = [
        f"您好，您的情况「{sc}」我已经详细了解了，",
        f"您好，感谢您的反馈。「{sc}」这个问题我来帮您处理。",
        f"您好，关于「{sc}」，理解您的着急心情。",
    ]
    parts.append(random.choice(openers))

    if cat == 3:
        parts.append("首先请您注意行车安全，如正在行驶请尽快安全停车，我们优先处理安全问题。")
    elif cat == 6:
        parts.append("请您首先确保自身和现场人员安全，我立即为您协调处理。")

    facts = seed.get("required_facts") or []
    if facts:
        parts.append("这里先说明几点关键信息：" + "；".join(
            f"{CIRCLED[i]}{f}" for i, f in enumerate(facts)) + "。")

    if tool_call_seg:
        parts.append(tool_call_seg)
    if missing_seg:
        parts.append(missing_seg)
    if question_seg:
        parts.append(question_seg)
    if action_seg:
        parts.append(action_seg)

    parts.append("后续有任何进展或疑问，欢迎随时联系我们。")
    return "".join(parts)


def _chosen_style_b(seed: dict, tool_call_seg: str, missing_seg: str, question_seg: str,
                    action_seg: str) -> str:
    """风格 B：问题定位型——直接分析问题原因 + 给出排查路径。"""
    parts = []
    cat = seed["category_id"]
    sc = seed["scenario"]
    facts = seed.get("required_facts") or []

    if cat == 3:
        parts.append(f"您好，关于「{sc}」，安全提醒：请勿继续行驶，立即靠边停车确保安全。")
    elif cat == 6:
        parts.append(f"您好，「{sc}」情况紧急，请先确保人员安全，我马上为您安排救援。")
    else:
        parts.append(f"您好，您反映的「{sc}」问题，我来帮您逐步排查。")

    if facts:
        parts.append("需要先确认" + cn(len(facts)) + "个关键信息：" + "；".join(
            f"{CIRCLED[i]}{f}" for i, f in enumerate(facts)) + "。")

    if tool_call_seg:
        parts.append(tool_call_seg)
    if missing_seg:
        parts.append(missing_seg)
    if question_seg:
        parts.append(question_seg)
    if action_seg:
        parts.append(action_seg)

    tail = "祝您行车平安。" if cat in (3, 6) else "祝您用车愉快。"
    parts.append(tail)
    return "".join(parts)


def _chosen_style_c(seed: dict, tool_call_seg: str, missing_seg: str, question_seg: str,
                    action_seg: str) -> str:
    """风格 C：步骤导向型——明确告诉客户下一步做什么。"""
    parts = []
    cat = seed["category_id"]
    sc = seed["scenario"]

    parts.append(f"您好，您反映的「{sc}」我已记录，接下来帮您处理。")

    if cat == 3:
        parts.append("安全第一——请勿继续行驶，立即靠边停车，打开双闪并在车后放置警告标志。")
    elif cat == 6:
        parts.append("请您先撤离到安全区域，开启双闪并在车后 150 米放置警告标志，"
                     "我已为您联系救援单位，救援正在路上。")

    facts = seed.get("required_facts") or []
    if facts:
        parts.append("处理前需要确认以下信息：" + "；".join(
            f"{CIRCLED[i]}{f}" for i, f in enumerate(facts)) + "。")

    if tool_call_seg:
        parts.append(tool_call_seg)
    if missing_seg:
        parts.append(missing_seg)
    if question_seg:
        parts.append(question_seg)
    if action_seg:
        parts.append(action_seg)

    parts.append("请按以上步骤操作，我们会持续跟进。")
    return "".join(parts)


CHOSEN_STYLES = [_chosen_style_a, _chosen_style_b, _chosen_style_c]

# ---------- ② 多样化 rejected 模板（打破 round 1 的单一"确认一个问题就停"） ----------

def _rej_half_answer(seed: dict) -> str:
    """半截话：只有开场 + 一个反问，无任何实质内容。"""
    qs = seed.get("required_questions") or []
    q0 = qs[0].split("以及")[0].split("和")[0].split("、")[0].split("，")[0][:14] if qs else "车型年款"
    return (f"您好，感谢您联系我们，您的情况我已经记录下来了。"
            f"为了给您准确的处理方案，我先跟您确认一个信息：{q0}。")


def _rej_wrong_tool(seed: dict, schemas: dict) -> str:
    """工具规范类：不调用正确工具或调用错误工具。"""
    base = _rej_half_answer(seed)
    return base + "这边先帮您登记一下，稍后有专人回复您，请您保持电话畅通。"


def _rej_fabricate(seed: dict, schemas: dict) -> str:
    """参数编造类：编造工具名/参数值/查询结果。"""
    base = _rej_half_answer(seed)
    tool = seed.get("tool_name")
    if not tool or tool not in schemas:
        return base + "我这边帮您查过了，您的车辆是2024款长续航版，相关功能都支持的。"
    props = schemas.get(tool, {}).get("parameters", {}).get("properties", {})
    args = {p: FAB_POOL.get(p, f"VID{random.randint(10000,99999)}") for p in props}
    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    call = '<tool_call>\n' + '{"name": "' + tool + '", "arguments": ' + args_json + '}\n</tool_call>'
    return base + "我先帮您查一下，" + call + "查到结果后我直接告诉您。"


def _rej_no_safety(seed: dict) -> str:
    """安全话术类（cat3/cat6）：缺安全劝阻/救援确认。"""
    base = _rej_half_answer(seed)
    if seed["category_id"] == 3:
        return base + "这个问题比较常见，影响不大，您先继续正常行驶，有空到店检查就行。"
    return base + "救援稍后安排，请保持电话畅通，先不用着急。"


REJECTED_FNS = [_rej_half_answer, _rej_wrong_tool, _rej_fabricate, _rej_no_safety]

# ---------- 工具调用段落（多样化表达） ----------

def _make_tool_call_seg(seed: dict, schemas: dict, variant: int) -> tuple:
    """返回 (tool_call_segment, missing_segment)。"""
    tr = seed.get("tool_required", False)
    tool = seed.get("tool_name")
    if not tr or not tool or tool not in schemas:
        return "", ""

    props = schemas.get(tool, {}).get("parameters", {}).get("properties", {})
    user_text = seed["scenario"] + seed["user_goal"]
    args, missing = {}, []
    for p in props:
        if p in FREE_TEXT_PARAMS and seed["user_goal"] and seed["user_goal"] in user_text:
            args[p] = seed["user_goal"]
        else:
            missing.append(p)

    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    call = '<tool_call>\n' + '{"name": "' + tool + '", "arguments": ' + args_json + '}\n</tool_call>'

    leads = [
        "我现在就帮您在系统里发起查询，",
        "我已在系统中发起查询，",
        "我马上帮您查一下，",
    ]
    lead = "我已为您安排救援，" if seed["category_id"] == 6 else leads[variant % len(leads)]

    seg = lead + call + "查询结果出来后我会第一时间同步给您，在此之前不给您任何猜测性结论。"
    missing_seg = ""
    if missing:
        missing_seg = "为了确保查准，还请您补充：" + "；".join(
            f"{CIRCLED[i]}{PARAM_LABELS.get(p, p)}"
            for i, p in enumerate(missing)) + "。"
    return seg, missing_seg


def _make_question_seg(seed: dict, variant: int) -> str:
    qs = seed.get("required_questions") or []
    if not qs:
        return ""
    leads = [
        "在处理之前请您配合确认",
        "为准确判断，请您确认",
        "还需要您确认以下信息",
    ]
    return leads[variant % len(leads)] + cn(len(qs)) + "点：" + "；".join(
        f"{CIRCLED[i]}{q}" for i, q in enumerate(qs)) + "。"


def _make_action_seg(seed: dict, variant: int) -> str:
    acts = seed.get("required_actions") or []
    if not acts:
        return ""
    leads = [
        "接下来您可以这样操作",
        "建议您按以下步骤处理",
        "接下来建议您",
    ]
    return leads[variant % len(leads)] + "：" + "；".join(
        f"{CIRCLED[i]}{a}" for i, a in enumerate(acts)) + "。"

# ---------- 主流程 ----------

def build_one_pair(seed: dict, schemas: dict, chosen_style: int, rejected_fn,
                   margin: int, len_ratio: float) -> dict | None:
    """为一条 seed 构建一个 DPO 对。"""
    cat = seed["category_id"]
    scenario = seed["scenario"]
    user_goal = seed.get("user_goal", "")
    question = f"您好，{scenario}。我的诉求是：{user_goal}。请帮我处理。"

    # --- 构造 chosen ---
    tc_seg, miss_seg = _make_tool_call_seg(seed, schemas, chosen_style)
    q_seg = _make_question_seg(seed, chosen_style)
    a_seg = _make_action_seg(seed, chosen_style)
    style_fn = CHOSEN_STYLES[chosen_style % len(CHOSEN_STYLES)]
    chosen_text = style_fn(seed, tc_seg, miss_seg, q_seg, a_seg)

    # --- 评分 ---
    r_chosen = score_answer(seed, chosen_text, schemas)
    sc = r_chosen["scores"]["总分"]
    if sc < MIN_SCORE:
        return None

    # --- 构造 rejected ---
    if rejected_fn in (_rej_wrong_tool, _rej_fabricate):
        rej_text = rejected_fn(seed, schemas)
    else:
        rej_text = rejected_fn(seed)
    r_rejected = score_answer(seed, rej_text, schemas)
    sr = r_rejected["scores"]["总分"]

    # --- 过滤 ---
    m = sc - sr
    if m < MIN_MARGIN:
        return None
    lc, lr = len(chosen_text), len(rej_text)
    if lr > 0 and lc >= len_ratio * lr and m < 6:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": {"role": "assistant", "content": chosen_text},
        "rejected": {"role": "assistant", "content": rej_text},
        "seed_id": seed["seed_id"],
        "category_id": cat,
        "defect_type": {
            _rej_half_answer: "完整性类", _rej_wrong_tool: "工具规范类",
            _rej_fabricate: "参数编造类", _rej_no_safety: "安全话术类",
        }.get(rejected_fn, "完整性类"),
        "source": "synthetic_r2",
        "score_chosen": sc, "score_rejected": sr, "margin": m,
        "len_chosen": lc, "len_rejected": lr,
    }


def to_trl(p: dict) -> dict:
    return {"prompt": p["messages"], "chosen": [p["chosen"]], "rejected": [p["rejected"]]}


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    seeds = load_seed_dir(TRAIN_DIR, marker="category*_train.jsonl")
    schemas = load_schemas()
    print(f"[数据] 加载 {len(seeds)} 条 train 种子，{len(schemas)} 个工具 schema")

    # 每条 seed × 多种 chosen 风格 × 多种 rejected 类型 → 生成候选对
    all_pairs = []
    seen = set()  # (seed_id, chosen_style_idx, rej_fn_name) 去重

    rej_fns = [_rej_half_answer, _rej_wrong_tool, _rej_fabricate, _rej_no_safety]

    for seed in seeds.values():
        for cs in range(len(CHOSEN_STYLES)):
            for rf in rej_fns:
                key = (seed["seed_id"], cs, rf.__name__)
                if key in seen:
                    continue
                seen.add(key)
                pair = build_one_pair(seed, schemas, cs, rf, MIN_MARGIN, LEN_RATIO)
                if pair:
                    all_pairs.append(pair)

    print(f"[生成] 候选对: {len(all_pairs)}")

    # 配额控制：每类缺陷至少 100 对（重点补安全性），总量控制在 TARGET 附近
    by_defect = {}
    for p in all_pairs:
        by_defect.setdefault(p["defect_type"], []).append(p)

    selected = []
    for defect, pairs in by_defect.items():
        random.shuffle(pairs)
        cap = 120 if defect == "安全话术类" else 110  # 安全类多给配额
        selected.extend(pairs[:cap])
        print(f"  {defect}: 选取 {min(len(pairs), cap)}/{len(pairs)}")

    random.shuffle(selected)
    if len(selected) > TARGET_PAIRS:
        selected = selected[:TARGET_PAIRS]
    print(f"[最终] 选取 {len(selected)} 对")

    # 输出 ms-swift 格式
    with open(os.path.join(OUT_DIR, "dpo_train.jsonl"), "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 输出 TRL 格式
    with open(os.path.join(OUT_DIR, "dpo_train_trl.jsonl"), "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps(to_trl(p), ensure_ascii=False) + "\n")

    # 统计报告
    report = []
    report.append(f"# 二轮 DPO 数据报告（{len(selected)} 对）\n")
    report.append("## 配比统计\n")
    report.append("| 缺陷类型 | 条数 | 占比 |")
    report.append("|---|---|---|")
    for defect, pairs in sorted(by_defect.items()):
        n = min(len(pairs), 120 if defect == "安全话术类" else 110)
        report.append(f"| {defect} | {n} | {n/len(selected)*100:.1f}% |")
    report.append(f"| **合计** | **{len(selected)}** | |")
    report.append("")

    # chosen 长度分布
    lens = [p["len_chosen"] for p in selected]
    margins = [p["margin"] for p in selected]
    report.append("## chosen 长度与 margin\n")
    report.append(f"- chosen 均长: {sum(lens)/len(lens):.0f} 字")
    report.append(f"- chosen 最短: {min(lens)} 字，最长: {max(lens)} 字")
    report.append(f"- margin 均值: {sum(margins)/len(margins):.1f}")
    report.append(f"- margin 最小: {min(margins)}，最大: {max(margins)}")
    report.append("")

    # cat3/cat6 统计
    cat3 = [p for p in selected if p["category_id"] == 3]
    cat6 = [p for p in selected if p["category_id"] == 6]
    report.append("## 靶向缺陷覆盖\n")
    report.append(f"- cat3（故障分流）: {len(cat3)} 对，其中安全话术类: "
                  f"{sum(1 for p in cat3 if p['defect_type']=='安全话术类')}")
    report.append(f"- cat6（道路救援）: {len(cat6)} 对，其中安全话术类: "
                  f"{sum(1 for p in cat6 if p['defect_type']=='安全话术类')}")

    with open(os.path.join(OUT_DIR, "dpo_r2_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n[完成] 报告已写入 {OUT_DIR}/dpo_r2_report.md")

    # 抽样 10 对
    samples = random.sample(selected, min(10, len(selected)))
    with open(os.path.join(OUT_DIR, "sample_10_r2.txt"), "w", encoding="utf-8") as f:
        for i, p in enumerate(samples, 1):
            q = p["messages"][-1]["content"][:80]
            c = p["chosen"]["content"][:120]
            r = p["rejected"]["content"][:120]
            f.write(f"--- 样本 {i} [{p['defect_type']}] ---\n")
            f.write(f"问题: {q}...\n")
            f.write(f"chosen ({p['len_chosen']}字, {p['score_chosen']}分): {c}...\n")
            f.write(f"rejected ({p['len_rejected']}字, {p['score_rejected']}分): {r}...\n\n")
    print(f"[抽样] 10 对已写入 {OUT_DIR}/sample_10_r2.txt")


def selftest():
    """自检：生成 20 对 + 验证格式 + 打印样例。"""
    random.seed(SEED)
    seeds = load_seed_dir(TRAIN_DIR, marker="category*_train.jsonl")
    schemas = load_schemas()
    seeds_list = list(seeds.values())[:20]

    ok = 0
    for seed in seeds_list:
        for cs in range(len(CHOSEN_STYLES)):
            for rf in [REJECTED_FNS[0]]:
                pair = build_one_pair(seed, schemas, cs, rf, MIN_MARGIN, LEN_RATIO)
                if pair:
                    ok += 1
                    trl = to_trl(pair)
                    assert "prompt" in trl and "chosen" in trl and "rejected" in trl
                    assert trl["prompt"][0]["role"] == "system"
                    assert trl["chosen"][0]["role"] == "assistant"
                    assert trl["rejected"][0]["role"] == "assistant"
    print(f"[自检] 20 条种子 × 3 风格 → {ok} 对通过格式校验")
    if ok < 5:
        sys.exit("[自检] 太少对通过，请检查生成逻辑")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        main()
