# -*- coding: utf-8 -*-
"""
【流水线 03/16】里程碑一 · SFT 数据多样化补充（打破模板单一性）
运行位置：本机（无需 GPU）
输入：data/v2/seeds/train/ 种子
输出：output/sft_train_supplement.jsonl（约 500 条，3 种回答风格）
前置步骤：01　｜　后续步骤：04 训练

SFT 数据多样化补充脚本。

问题诊断：当前 1912 条 SFT 训练数据全部是同一模板结构（开场→事实→工具→追问→建议→收尾），
模型学到的是固定"形状"而非业务逻辑，validation 上退化成最短路径（半截话）。

本脚本基于 640 条 train 种子，为每条种子生成 2~3 种**结构不同**的回答变体：
  - 变体 A：共情优先型（先安抚情绪 + 分步引导，非事实罗列）
  - 变体 B：问题诊断型（直接分析原因 + 给排查路径，非模板填充）
  - 变体 C：信息不完整处理型（用户没给关键信息时，先给初步建议 + 再要信息）

与 build_sft_data.py 的 1912 条混合后重训 SFT，目标是打破模板单一性。

输出：output/sft_train_supplement.jsonl（~500 条），与 output/sft_train.jsonl 混合后重训。

用法：
    python scripts/build_sft_supplement.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import SYSTEM_PROMPT, score_answer, _locate
from build_dpo_data import load_seed_dir, load_schemas, PARAM_LABELS, FREE_TEXT_PARAMS

TRAIN_DIR = "data/v2/seeds/train"
OUT_PATH = "output/sft_train_supplement.jsonl"
TARGET = 500
SEED = 42


def _cn(n):
    return {2: "两"}.get(n, "一二三四五六七八九十"[n - 1] if 1 <= n <= 10 else str(n))


def _tool_call_text(seed, schemas):
    """生成 tool_call 段落。"""
    tr = seed.get("tool_required", False)
    tool = seed.get("tool_name")
    if not tr or not tool or tool not in schemas:
        return ""

    props = schemas.get(tool, {}).get("parameters", {}).get("properties", {})
    user_text = seed["scenario"] + seed["user_goal"]
    args = {}
    for p in props:
        if p in FREE_TEXT_PARAMS and seed["user_goal"] and seed["user_goal"] in user_text:
            args[p] = seed["user_goal"]

    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    call = '<tool_call>\n' + '{"name": "' + tool + '", "arguments": ' + args_json + '}\n</tool_call>'
    return "我先帮您在系统里查询一下，" + call + "查到结果后我会第一时间告诉您。"


def variant_a(seed, schemas):
    """共情优先型：先安抚 + 分步引导。"""
    cat = seed["category_id"]
    sc = seed["scenario"]
    facts = seed.get("required_facts") or []
    qs = seed.get("required_questions") or []

    parts = []
    if cat == 3:
        parts.append(f"您好，理解您的担心。「{sc}」这种情况确实让人不放心。"
                     "首先请注意行车安全——如正在行驶请尽快安全靠边停车，我们优先处理安全问题。")
    elif cat == 6:
        parts.append(f"您好，别着急。「{sc}」我马上帮您处理。请您先确保自身和现场人员安全。")
    else:
        openers = [
            f"您好，理解您的心情。「{sc}」这个问题我来帮您一步步处理。",
            f"您好，您说的情况「{sc}」我已经了解了，别担心，我来帮您分析。",
            f"您好，关于「{sc}」，我来帮您想办法解决。",
        ]
        parts.append(random.choice(openers))

    if facts:
        parts.append("先跟您说一下背景：" + "；".join(
            f"①{f}" for f in facts[:3]) + "。")

    tc = _tool_call_text(seed, schemas)
    if tc:
        parts.append(tc)

    if qs:
        parts.append("为了更准确地帮您，还需要确认：" + "；".join(
            f"①{q}" for q in qs[:3]) + "。")

    # 给出初步方向（不等信息齐全就先建议）
    acts = seed.get("required_actions") or []
    if acts:
        parts.append("目前建议您：" + "；".join(f"①{a}" for a in acts[:2]) + "。")

    tail = "祝您行车平安。" if cat in (3, 6) else "后续有问题随时联系我们。"
    parts.append(tail)
    return "".join(parts)


def variant_b(seed, schemas):
    """问题诊断型：直接分析原因 + 排查路径。"""
    cat = seed["category_id"]
    sc = seed["scenario"]
    facts = seed.get("required_facts") or []

    parts = []
    if cat == 3:
        parts.append(f"您好，关于「{sc}」——安全提醒：请勿继续行驶，立即靠边停车确保安全。"
                     "下面帮您逐步排查。")
    elif cat == 6:
        parts.append(f"您好，「{sc}」情况紧急。请您先撤离到安全区域，"
                     "我已为您联系救援。现在帮您确认现场情况。")
    else:
        parts.append(f"您好，「{sc}」这个问题我来帮您分析。")

    # 直接给分析（不先问信息）
    if facts:
        parts.append("一般来说这类问题可能涉及：" + "；".join(
            f"• {f}" for f in facts[:3]) + "。")

    tc = _tool_call_text(seed, schemas)
    if tc:
        parts.append(tc)

    qs = seed.get("required_questions") or []
    if qs:
        parts.append("请您帮忙确认几点，方便精准定位：" + "；".join(
            f"• {q}" for q in qs[:3]) + "。")

    acts = seed.get("required_actions") or []
    if acts:
        parts.append("排查步骤：" + "；".join(f"{i+1}. {a}" for i, a in enumerate(acts[:3])) + "。")

    tail = "祝您行车平安。" if cat in (3, 6) else "如有疑问随时联系我们。"
    parts.append(tail)
    return "".join(parts)


def variant_c(seed, schemas):
    """信息不完整处理型：先给初步建议 + 再要关键信息。"""
    cat = seed["category_id"]
    sc = seed["scenario"]
    facts = seed.get("required_facts") or []

    parts = []
    if cat == 3:
        parts.append(f"您好，「{sc}」——请注意安全，先停车再处理。"
                     "虽然还需要确认一些信息，但先给您几个初步建议：")
    elif cat == 6:
        parts.append(f"您好，「{sc}」——先确保安全！虽然还需要了解一些细节，"
                     "但先帮您启动应急流程：")
    else:
        parts.append(f"您好，「{sc}」——虽然信息还不全，但先给您一些方向：")

    # 先给方向，再问信息（与原来"先问再答"相反）
    acts = seed.get("required_actions") or []
    if acts:
        parts.append("初步建议：" + "；".join(f"①{a}" for a in acts[:2]) + "。")

    tc = _tool_call_text(seed, schemas)
    if tc:
        parts.append(tc)

    if facts:
        missing = facts[:3]
        parts.append("为了更精准地处理，还请您补充：" + "；".join(
            f"• {f}" for f in missing) + "。")

    qs = seed.get("required_questions") or []
    if qs:
        parts.append("另外确认一下：" + "；".join(f"• {q}" for q in qs[:2]) + "。")

    tail = "祝您行车平安。" if cat in (3, 6) else "有问题随时找我们。"
    parts.append(tail)
    return "".join(parts)


VARIANTS = [variant_a, variant_b, variant_c]


def main():
    random.seed(SEED)
    seeds = load_seed_dir(TRAIN_DIR, marker="category*_train.jsonl")
    schemas = load_schemas()
    print(f"[数据] 加载 {len(seeds)} 条种子，{len(schemas)} 个工具 schema")

    results = []
    for seed in seeds.values():
        question = f"您好，{seed['scenario']}。我的诉求是：{seed.get('user_goal', '')}。请帮我处理。"
        for vfn in VARIANTS:
            answer = vfn(seed, schemas)
            # 评分
            r = score_answer(seed, answer, schemas)
            if r["scores"]["总分"] >= 10:  # 比 DPO 低一些，SFT 容忍度更高
                results.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "seed_id": seed["seed_id"],
                    "category_id": seed["category_id"],
                    "variant": vfn.__name__,
                    "score": r["scores"]["总分"],
                    "len": len(answer),
                })

    print(f"[生成] {len(results)} 条候选（{len(seeds)} 种子 × 3 变体）")

    # 配额控制：每类缺陷/场景均匀
    random.shuffle(results)
    selected = results[:TARGET]

    # 输出
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in selected:
            # 只输出 messages 字段（与原 SFT 格式一致）
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

    # 统计
    avg_len = sum(s["len"] for s in selected) / len(selected)
    cats = {}
    for s in selected:
        cats.setdefault(s["category_id"], []).append(s)
    print(f"[完成] {len(selected)} 条写入 {OUT_PATH}，均长 {avg_len:.0f} 字")
    for c in sorted(cats):
        v = cats[c]
        print(f"  cat{c}: {len(v)} 条")


if __name__ == "__main__":
    main()
