# -*- coding: utf-8 -*-
"""
【流水线 15/16】生产加固 · 约束版 vs 裸模型 64 条双盲竞技场验收
运行位置：本机 ＋ Judge API（无需 GPU）
输入：output/constrained_validation.jsonl、sft_eval_results_before_constraint.json
输出：output/validation_arena_report.md、validation_arena_results.json
前置步骤：14　｜　后续步骤：—

Validation Arena：约束版（DPO R2 + 约束层）vs 基线（DPO R2 裸）64 条双盲竞技场。
复用 run_final_eval.arena_pair 的评审逻辑（随机站位 + 双向对调，方向冲突判平）。

输入：
    A = output/constrained_validation.jsonl             （约束版回答，含 answer）
    B = output/sft_eval_results_before_constraint.json  （基线回答，含 answer）
输出：
    output/validation_arena_report.md / validation_arena_results.json

用法（实例 /mnt/workspace）:
    export JUDGE_API_KEY=... JUDGE_BASE_URL=... JUDGE_MODEL=...   # 与打分同一 Judge 保持一致
    python scripts/run_arena_validation.py --selftest
    python scripts/run_arena_validation.py
"""
import argparse
import json
import os
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import _locate, build_user_message
from run_final_eval import arena_pair, elo_diff, _fmt_sec, _progress

A_FILE = "output/constrained_validation.jsonl"
B_FILE = "output/sft_eval_results_before_constraint.json"
OUT_DIR = "output"
REPORT_FILE = os.path.join(OUT_DIR, "validation_arena_report.md")
RESULTS_FILE = os.path.join(OUT_DIR, "validation_arena_results.json")


def load_answers(path: str) -> dict:
    """读取回答文件（.jsonl 逐行 / 其余按 json 数组），返回 seed_id -> answer 映射。"""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()] if path.endswith(".jsonl") \
            else json.load(f)
    return {r["seed_id"]: r["answer"].strip() for r in rows}


def load_seeds() -> list:
    """加载 validation 的 64 条种子，按 seed_id 排序以固定评审顺序。"""
    import glob
    d = _locate("data/v2/seeds/validation", marker="category1_validation.jsonl")
    seeds = []
    for fp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            seeds += [json.loads(l) for l in f if l.strip()]
    seeds.sort(key=lambda s: s["seed_id"])
    assert len(seeds) == 64
    return seeds


def mock_judge(prompt: str) -> dict:
    """离线假评审：从 prompt 里抽出 A/B 回答，判更长的一方胜，仅供 selftest 跑通管线。"""
    m = __import__("re").search(r"\[回答A\] (.*?)\n\[回答B\] (.*)", prompt, __import__("re").S)
    a, b = m.group(1), m.group(2)
    return {"better": "A" if len(a) >= len(b) else "B", "reason": "mock: 更长更完整"}


def selftest() -> None:
    """离线：mock 评审跑通统计与报告生成（不打 API，跳过请求间隔）。"""
    import run_final_eval
    run_final_eval.JUDGE_INTERVAL = 0  # arena_pair 内部的 sleep 走模块全局，mock 时清零
    seeds = load_seeds()
    a = {s["seed_id"]: "您好，这是约束版回答，内容完整覆盖必需事实、追问与动作，并包含安全提示。" for s in seeds}
    b = {s["seed_id"]: "您好。" for s in seeds}
    items = []
    for s in seeds:
        arena = arena_pair(s, a[s["seed_id"]], b[s["seed_id"]], judge_fn=mock_judge)
        items.append({"seed_id": s["seed_id"], "category_id": s["category_id"],
                      "category": s["category"], "subcategory": s["subcategory"],
                      "question": build_user_message(s), "arena": arena})
    w = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    l = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    t = len(items) - w - l
    assert w == 64 and l == 0, (w, l, t)
    print(f"[自检通过] 64 场 mock 评审：约束版 64 胜，统计与报告管线正常")


def main() -> None:
    """约束版 vs 基线 64 场盲测主流程：对齐两侧回答 → 逐场评审 → 统计战绩并写报告。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pair-single", action="store_true", help="单向评审（省一半调用）")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    seeds = load_seeds()
    a = load_answers(A_FILE)
    b = load_answers(B_FILE)
    missing = [s["seed_id"] for s in seeds if s["seed_id"] not in a or s["seed_id"] not in b]
    assert not missing, f"缺 {len(missing)} 条: {missing[:5]}"

    t0 = time.time()
    print(f"[评审] {len(seeds)} 场盲测，模型 A=约束版（DPO R2+约束） vs B=基线（DPO R2 裸），"
          f"{'单向' if args.pair_single else '双向对调'}，Judge={os.environ.get('JUDGE_MODEL', '默认')}")
    items = []
    for i, s in enumerate(seeds, 1):
        arena = arena_pair(s, a[s["seed_id"]], b[s["seed_id"]], pair_single=args.pair_single)
        items.append({"seed_id": s["seed_id"], "category_id": s["category_id"],
                      "category": s["category"], "subcategory": s["subcategory"],
                      "question": build_user_message(s), "arena": arena})
        print(f"[{_progress(i, len(seeds), t0)}] {s['seed_id']}  判定: "
              f"{arena['verdict']}（{arena['method']}）", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    w = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    l = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    t = len(items) - w - l
    elo = elo_diff(w, l, t)
    wr = w / len(items) * 100
    wr_nt = w / (w + l) * 100 if w + l else 0.0

    by_cat: dict = {}
    for r in items:
        by_cat.setdefault(r["category_id"], []).append(r)
    L = ["# Validation Arena 报告：约束版 vs 基线（64 条双盲）", "",
         f"> 评审：随机站位 + 双向对调（方向一致才判胜负，冲突判平）| 模型 A=约束版（DPO R2 + 约束层）"
         f" vs B=基线（DPO R2 裸）| Judge={os.environ.get('JUDGE_MODEL', '默认')}", "",
         f"## 总战绩", "",
         f"- **约束版胜 {w} / 基线胜 {l} / 平局 {t}**",
         f"- Win Rate（约束版胜率）= **{wr:.1f}%**；非平局胜率 = {wr_nt:.1f}%",
         f"- Elo 分差（约束版 vs 基线）= {'+' if elo >= 0 else ''}{elo}", "",
         "## 分类别战绩", "",
         "| 类别 | 约束版胜 | 基线胜 | 平局 |", "|---|---|---|---|"]
    for cid in sorted(by_cat):
        rs = by_cat[cid]
        cw = sum(1 for r in rs if r["arena"]["verdict"] == "dpo")
        cl = sum(1 for r in rs if r["arena"]["verdict"] == "sft")
        L.append(f"| {cid} {rs[0]['category']} | {cw} | {cl} | {len(rs) - cw - cl} |")
    L += ["", "## 逐场判定（含判词）", ""]
    for r in sorted(items, key=lambda x: (x["category_id"], x["seed_id"])):
        v = {"dpo": "约束版胜", "sft": "基线胜", "tie": "平局"}[r["arena"]["verdict"]]
        reason = "；".join(x for x in r["arena"]["reasons"] if x) or "—"
        L.append(f"- **{r['seed_id']}**（cat{r['category_id']} {r['subcategory']}）→ **{v}**"
                 f"（{r['arena']['method']}）\n  - 判词：{reason}")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[完成] 总用时 {_fmt_sec(time.time() - t0)} | 约束版 {w} / 基线 {l} / 平 {t} | "
          f"Win Rate {wr:.1f}% | Elo {'+' if elo >= 0 else ''}{elo}")
    print(f"报告: {REPORT_FILE}")


if __name__ == "__main__":
    main()
