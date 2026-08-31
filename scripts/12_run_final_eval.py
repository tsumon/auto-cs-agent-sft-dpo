# -*- coding: utf-8 -*-
"""
【流水线 12/16】里程碑六 · final_test 评分阶段（7 维打分 + Arena 盲测）
运行位置：本机或实例 ＋ Judge API（无需 GPU）
输入：output/final_sft_answers.jsonl、final_dpo_answers.jsonl
输出：output/final_eval_results.json、final_bad_cases.jsonl、final_eval_report.md、看板 HTML
前置步骤：11　｜　后续步骤：13 看板（管线末尾自动触发）

final_test 最终评估——评分阶段：SFT 基线 vs DPO 模型终局对比。

输入（由 scripts/run_final_gen.py 生成）：
    output/final_sft_answers.jsonl   SFT 基线回答（160 条）
    output/final_dpo_answers.jsonl   DPO 模型回答（160 条）

评分口径与里程碑三完全一致：
    1) 7 维打分（理解/facts/questions/actions/prohibited违规/完整性/编造，各 0/1/2，满分 14）：
       规则打分器 + LLM Judge（ModelScope API）覆盖，--judge 开启；
    2) Arena 竞技场盲测两两对比：两个回答匿名化为"回答A/回答B"，随机站位 + 双向对调复审
       （两次方向一致才判定胜负，不一致判平局，消除位置偏置），统计 胜/平/负、Win Rate、Elo 分差。

产物：
    output/final_eval_results.json   逐条明细（两个模型得分 + 竞技场判词）
    output/final_bad_cases.jsonl     DPO 最终 Bad Case（含业务规则与缺陷归因）
    output/final_eval_report.md     最终评估报告（分组对比、Win Rate、Good/Bad Case、核心结论）

用法（实例项目根目录 /mnt/workspace，或本机——只需回答文件 + MODELSCOPE_API_KEY）：
    python scripts/run_final_eval.py --selftest          # 离线自检全流程（不打 API、不需要 GPU）
    python scripts/run_final_eval.py --judge             # Judge 打分 + Arena 盲测（推荐）
    python scripts/run_final_eval.py --judge --pair-single  # API 配额紧张时 Arena 单向评审（减一半调用）
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import time

# 评分逻辑直接复用里程碑三的 eval_sft.py，保证"与 validation 评估同一把尺子"
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import (DIM_KEYS, _locate, build_user_message, judge_with_llm,
                      load_tool_schemas, score_answer)

SEED_DIR = "data/v2/seeds/final_test"
OUTPUT_DIR = "output"
SFT_ANSWERS_FILE = os.path.join(OUTPUT_DIR, "final_sft_answers.jsonl")
DPO_ANSWERS_FILE = os.path.join(OUTPUT_DIR, "final_dpo_answers.jsonl")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "final_eval_results.json")
BAD_FILE = os.path.join(OUTPUT_DIR, "final_bad_cases.jsonl")
REPORT_FILE = os.path.join(OUTPUT_DIR, "final_eval_report.md")

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api-inference.modelscope.cn/v1")
JUDGE_API_KEY = (os.environ.get("JUDGE_API_KEY") or os.environ.get("MODELSCOPE_API_KEY")
                 or os.environ.get("DASHSCOPE_API_KEY", ""))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
ARENA_SEED = 20260831          # 竞技场随机站位种子（固定，保证可复现）
ELO_BASE = 1200.0              # SFT 基线锚定分，DPO = 基线 + Elo 分差
JUDGE_INTERVAL = float(os.environ.get("JUDGE_INTERVAL", "2"))  # 请求间隔秒数，429 频繁时可调大
JUDGE_CACHE = os.environ.get("JUDGE_CACHE", os.path.join(OUTPUT_DIR, ".judge_cache.jsonl"))

# ---------------- Judge 结果缓存（断点续跑）----------------
# 429 限速下评分随时可能中断；成功过的 Judge 结果逐条落盘，重跑同一命令只补失败缺口，
# 已缓存条目不再消耗 API 配额。缓存按 (Judge模型, 内容哈希) 键控，换 Judge 模型不串结果。
_cache: dict = {}


def _cache_load() -> None:
    """把落盘的 Judge 缓存逐行读回内存 _cache；缓存文件不存在就什么都不做。"""
    if os.path.exists(JUDGE_CACHE):
        with open(JUDGE_CACHE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    _cache[r["k"]] = r["v"]


def _cache_get(key: str):
    """读一条缓存；内存为空时先惰性加载缓存文件。未命中返回 None。"""
    if not _cache:
        _cache_load()
    return _cache.get(key)


def _cache_put(key: str, value: dict) -> None:
    """写一条缓存：存内存并追加落盘。

    已存在的键直接返回，避免同一条结果被重复追加进 jsonl。
    """
    if key in _cache:
        return
    _cache[key] = value
    os.makedirs(os.path.dirname(JUDGE_CACHE) or ".", exist_ok=True)
    with open(JUDGE_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


def _hash(text: str) -> str:
    """取文本 sha1 的前 16 位，作为缓存键里的内容指纹。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

BAD_THRESH = 10                # Bad Case：总分<=10 或任一维度 0 分（与里程碑三一致）
CAT_IMPROVE, CAT_FLAT = 0.25, 0.25   # 类别均分 Δ 判定阈值：>=+0.25 改善 / <=-0.25 下降
DIM_IMPROVE, DIM_FLAT = 0.15, 0.15   # 维度均分 Δ 判定阈值


# ---------------- Arena 竞技场 ----------------
ARENA_PROMPT = """你是汽车售后客服回答质量评审员。这是一场盲测竞技场：两个匿名客服助手对同一客户问题各给出一个回答（回答A、回答B），请综合判断哪个更好。
评判标准（与质检维度一致）：
1) 对必需事实、必要追问、必需动作的覆盖程度；
2) 是否违反禁止项、是否编造政策数字/工单号/查询结果/用户未提供的参数值（需要工具时应发起 <tool_call> 且参数只能取自用户话语，直接口播查询结果属于编造）；
3) 完整性与自然度：半截话、纯套话、答非所问者更差；安全场景是否有明确的安全分流话术。
两个回答差距明显才判 A 或 B，差距很小判 tie。只输出JSON：{{"better":"A","reason":"一句话"}}（better 取 A/B/tie）

[客户问题] {question}
[业务规则] 必需事实：{facts}；必需追问：{questions}；必需动作：{actions}；禁止项：{prohibited}；需要工具：{tool_required}（工具名 {tool_name}）
[回答A] {answer_a}
[回答B] {answer_b}"""


def call_judge(prompt: str, max_tokens: int = 1200) -> dict | None:
    """调 Judge 模型返回解析后的 JSON（带 429 退避），失败返回 None。

    max_tokens 默认 1200：部分端点/模型的 thinking 与 content 共用输出预算且思考偶发
    超长（实测 600 仍会截断），给足预算显著降低空返回概率；按实际输出计费，成本影响小。
    """
    import requests
    payload = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    if "deepseek" in JUDGE_MODEL:  # 推理模型：压短思考，避免 thinking 吃光 max_tokens 导致 content 为空
        payload["reasoning_effort"] = "low"
    waits = [5, 10, 20, 40, 80, 80]  # 6 次重试，最多约 4 分钟
    for attempt, wait in enumerate([0] + waits):
        if wait:
            print(f"  [429限速] 第{attempt}次重试，等待{wait}s")
            time.sleep(wait)
        try:
            resp = requests.post(
                f"{JUDGE_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
                json=payload, timeout=120)
            if resp.status_code == 429 and attempt < len(waits):
                continue
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            m = re.search(r"\{.*\}", content, re.S)
            if m is None:  # content 为空时兜底解析 reasoning_content
                m = re.search(r"\{.*\}", msg.get("reasoning_content") or "", re.S)
            if m is None:
                print(f"  [Judge输出无JSON] 内容片段: {content[:100]!r}")
                return None
            return json.loads(m.group(0))
        except Exception as e:  # noqa: BLE001
            print(f"  [Judge调用失败] {type(e).__name__}: {e}")
            return None
    return None


def call_judge_cached(prompt: str, max_tokens: int = 1200) -> dict | None:
    """带缓存的竞技场评审调用；未命中缓存才真正打 API。

    缓存键为 arena|Judge模型名|prompt哈希：带模型名是为了换 Judge 时不串用旧结果，
    带 prompt 哈希是因为站位对调后 prompt 不同，两个方向各存一份。
    失败返回的 None 不写缓存，留给重跑时补缺口。
    """
    key = "arena|" + JUDGE_MODEL + "|" + _hash(prompt)
    hit = _cache_get(key)
    if hit is not None:
        return hit
    out = call_judge(prompt, max_tokens)
    if out is not None:
        _cache_put(key, out)
    return out


def arena_pair(seed: dict, dpo_ans: str, sft_ans: str, pair_single: bool = False,
               judge_fn=call_judge_cached) -> dict:
    """一场盲测竞技场评审。

    随机站位（DPO 是否放 A 由固定种子决定）+ 双向对调复审：只有两个方向判定一致才
    判胜负，否则平局——消除 LLM Judge 的位置偏置。返回 {verdict, method, dpo_is_a, reasons}。
    verdict 取 "dpo" / "sft" / "tie"。
    """
    rng = random.Random(f"{ARENA_SEED}_{seed['seed_id']}")
    dpo_is_a = rng.random() < 0.5
    rule = None  # 兜底用：懒计算规则分差

    def one_call(first_is_dpo: bool) -> tuple:
        """按指定站位调一次评审，把 A/B 判词还原成 dpo/sft（非法值归 tie）；失败返回 (None, "")。"""
        a, b = (dpo_ans, sft_ans) if first_is_dpo else (sft_ans, dpo_ans)
        out = judge_fn(ARENA_PROMPT.format(
            question=build_user_message(seed),
            facts="；".join(seed["required_facts"]) or "无",
            questions="；".join(seed["required_questions"]) or "无",
            actions="；".join(seed["required_actions"]) or "无",
            prohibited="；".join(seed["prohibited_actions"]) or "无",
            tool_required=seed["tool_required"], tool_name=seed["tool_name"] or "-",
            answer_a=a or "（空）", answer_b=b or "（空）"))
        time.sleep(JUDGE_INTERVAL)  # 请求间隔，降低限速概率
        if out is None:
            return None, ""
        better = str(out.get("better", "tie")).strip().upper()
        if better not in ("A", "B", "TIE"):
            better = "TIE"
        winner = {"A": ("dpo" if first_is_dpo else "sft"),
                  "B": ("sft" if first_is_dpo else "dpo"),
                  "TIE": "tie"}[better]
        return winner, str(out.get("reason", ""))

    w1, r1 = one_call(dpo_is_a)
    if pair_single:
        if w1 is None:
            return {"verdict": _rule_fallback(seed, dpo_ans, sft_ans), "method": "规则分兜底（评审失败）",
                    "dpo_is_a": dpo_is_a, "reasons": ["竞技场评审调用失败"]}
        return {"verdict": w1, "method": "单向盲测", "dpo_is_a": dpo_is_a, "reasons": [r1]}
    w2, r2 = one_call(not dpo_is_a)
    if w1 is None and w2 is None:
        return {"verdict": _rule_fallback(seed, dpo_ans, sft_ans), "method": "规则分兜底（评审失败）",
                "dpo_is_a": dpo_is_a, "reasons": ["双向评审均调用失败"]}
    if w1 is None:
        return {"verdict": w2, "method": "单向盲测（对向调用失败）", "dpo_is_a": dpo_is_a, "reasons": [r2]}
    if w2 is None:
        return {"verdict": w1, "method": "单向盲测（对向调用失败）", "dpo_is_a": dpo_is_a, "reasons": [r1]}
    if w1 == w2:
        return {"verdict": w1, "method": "双向一致", "dpo_is_a": dpo_is_a, "reasons": [r1, r2]}
    return {"verdict": "tie", "method": "双向冲突判平", "dpo_is_a": dpo_is_a, "reasons": [r1, r2]}


def _rule_fallback(seed: dict, dpo_ans: str, sft_ans: str, schemas: dict | None = None) -> str:
    """竞技场评审不可用时，用规则分差兜底（分差>=2 判胜负，否则平局）。"""
    global _SCHEMAS
    schemas = schemas or _SCHEMAS or {}
    rd = score_answer(seed, dpo_ans, schemas)["scores"]["总分"]
    rs = score_answer(seed, sft_ans, schemas)["scores"]["总分"]
    if rd - rs >= 2:
        return "dpo"
    if rs - rd >= 2:
        return "sft"
    return "tie"


_SCHEMAS: dict = {}


# ---------------- 7 维打分（Judge 覆盖，规则兜底）----------------
def judge_cached(seed: dict, answer: str, tag: str):
    """带缓存的 7 维 Judge 打分；返回 (结果dict|None, 是否命中缓存)。"""
    key = "7dim|" + JUDGE_MODEL + "|" + seed["seed_id"] + "|" + tag + "|" + _hash(answer)
    hit = _cache_get(key)
    if hit is not None:
        return dict(hit), True
    out = judge_with_llm(seed, answer)
    if out is not None:
        _cache_put(key, out)
    return out, False


def score_one(seed: dict, answer: str, schemas: dict, use_judge: bool, tag: str = "sft") -> dict:
    """给一条回答打 7 维分：先跑规则打分器，use_judge 时用 Judge 结果覆盖分数。

    Judge 成功则把判词并入 flags["缺陷原因"]，并置 flags 的 judge / cached 标记；
    Judge 失败（返回 None）保留规则分，即降级。tag 取 'sft'/'dpo'，只用于缓存键区分。
    """
    result = score_answer(seed, answer, schemas)
    if use_judge:
        out, from_cache = judge_cached(seed, answer, tag)
        if out is not None:
            reason = out.pop("judge_reason", "")
            result = dict(result)
            result["scores"] = dict(out, 总分=sum(out.values()))
            result["flags"] = dict(result["flags"])
            result["flags"]["缺陷原因"] = ([reason] if reason else []) + result["flags"]["缺陷原因"]
            result["flags"]["judge"] = True
            result["flags"]["cached"] = from_cache
    return result


# ---------------- 输入加载 ----------------
def load_answers(path: str) -> dict:
    """读回答 jsonl，返回 {seed_id: 记录} 字典；文件不存在直接断言失败并提示先跑生成脚本。"""
    assert os.path.isfile(path), f"找不到回答文件: {path}（请先运行 scripts/run_final_gen.py）"
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                by_id[r["seed_id"]] = r
    return by_id


def load_final_seeds() -> list:
    """加载 final_test 全部种子（按 seed_id 排序）；条数不足 160 直接断言失败。"""
    seed_dir = _locate(SEED_DIR, marker="category1_final_test.jsonl")
    seeds = []
    for fp in sorted(glob.glob(os.path.join(seed_dir, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            seeds += [json.loads(l) for l in f if l.strip()]
    assert len(seeds) == 160, f"预期 160 条 final_test 种子，实际 {len(seeds)}"
    seeds.sort(key=lambda s: s["seed_id"])
    return seeds


# ---------------- 聚合统计 ----------------
def _mean(xs):
    """算术均值；空序列返回 0.0（避免除零）。"""
    return sum(xs) / len(xs) if xs else 0.0


def _fmt_sec(sec: float) -> str:
    """把秒数格式化成 h:mm:ss（满 1 小时）或 m:ss，用于耗时/ETA 打印。"""
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 \
        else f"{sec // 60}:{sec % 60:02d}"


def _progress(i: int, n: int, t0: float) -> str:
    """进度条片段：i/n 百分比 | 已用 | 预计还需（按平均单条耗时线性估计，限速时波动大，仅供参考）"""
    el = time.time() - t0
    eta = el / i * (n - i) if i else 0
    return f"{i}/{n} {i * 100 // n:3d}% | 已用 {_fmt_sec(el)} | 预计还需 ~{_fmt_sec(eta)}"


def agg_scores(items: list, model: str) -> dict:
    """model: 'sft' / 'dpo'。返回总分与各维度均值、Bad Case 数。"""
    dim_means = {k: round(_mean([r[model]["scores"][k] for r in items]), 3) for k in DIM_KEYS}
    totals = [r[model]["scores"]["总分"] for r in items]
    bad = [r for r in items if r[model]["scores"]["总分"] <= BAD_THRESH or
           any(r[model]["scores"][k] == 0 for k in DIM_KEYS if k != "理解正确性")]
    return {"mean_total": round(_mean(totals), 3), "dim_means": dim_means,
            "bad_count": len(bad), "n": len(items)}


def flag_count(items: list, model: str, name: str) -> int:
    """统计某模型命中指定缺陷标记（如"半截话"）的条数。"""
    return sum(1 for r in items if r[model]["flags"].get(name))


def dim_zero_count(items: list, model: str, dim: str) -> int:
    """统计某维度失分条数：一般维度按 ==0 计，"是否编造"从严按 <2 计（扣分即算编造）。"""
    return sum(1 for r in items if r[model]["scores"][dim] < 2) if dim == "是否编造" else \
        sum(1 for r in items if r[model]["scores"][dim] == 0)


def elo_diff(w_dpo: int, w_sft: int, t: int) -> float:
    """双模型 MLE Elo 分差 = 400*log10(DPO得分期望/SFT得分期望)，平局各计 0.5。"""
    import math
    a, b = w_dpo + 0.5 * t, w_sft + 0.5 * t
    if a <= 0:
        return -800.0
    if b <= 0:
        return 800.0
    return round(400 * math.log10(a / b), 1)


# ---------------- 主管线 ----------------
def run_pipeline(seeds: list, schemas: dict, sft_by_id: dict, dpo_by_id: dict,
                 out_dir: str, use_judge: bool, pair_single: bool = False,
                 pre_items: list | None = None) -> dict:
    """终局评估主流程，返回 write_report 的汇总 dict。

    步骤：1) 逐条对两个模型打 7 维分（传了 pre_items 就跳过，供 --arena-only 复用旧分数），
    打分前断言两模型的问题完全一致；2) 每题跑一场 Arena 盲测，未开 --judge 则用规则分兜底；
    3) 打印 Judge 覆盖率与补缺口提示；4) 落盘明细 json 与 DPO Bad Case；
    5) 生成报告；6) 渲染 Arena 看板（失败只告警，不影响评估产物）。
    """
    global _SCHEMAS
    _SCHEMAS = schemas
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) 逐条打分（--arena-only 时复用已有结果，跳过）----
    if pre_items is not None:
        items = pre_items
        print(f"[跳过打分] 复用已有 7 维得分（--arena-only，竞技场评审模型: {JUDGE_MODEL}）")
        t0 = time.time()
    else:
        t0 = time.time()
        n_calls = len(seeds) * (2 + (1 if pair_single else 2))
        print(f"\n{'=' * 66}\n[阶段 1/2] 7 维打分（{len(seeds)} 条 × 2 模型）{time.strftime('%H:%M:%S')} 开始")
        print(f"[预估] 共 {n_calls} 次 Judge 调用；不限速约 "
              f"{max(1, round(n_calls / 640 * 40))} 分钟，高峰限速会显著拉长；"
              f"随时可中断，重跑同命令走缓存只补缺口\n{'=' * 66}", flush=True)
        items = []
        for i, seed in enumerate(seeds, 1):
            sid = seed["seed_id"]
            sa, da = sft_by_id[sid]["answer"].strip(), dpo_by_id[sid]["answer"].strip()
            # 审计：确认两个模型用的是同一问题
            assert sft_by_id[sid]["question"] == build_user_message(seed) == dpo_by_id[sid]["question"], \
                f"{sid} 问题不一致，统一测试集被破坏"
            sr = score_one(seed, sa, schemas, use_judge, "sft")
            dr = score_one(seed, da, schemas, use_judge, "dpo")
            if sr["flags"].get("cached") and dr["flags"].get("cached"):
                src = "Judge·缓存"
            elif sr["flags"].get("judge") and dr["flags"].get("judge"):
                src = "Judge"
            else:
                src = "部分规则分"
            items.append({
                "seed_id": sid, "category_id": seed["category_id"], "category": seed["category"],
                "subcategory": seed["subcategory"], "question": build_user_message(seed),
                "sft": {"answer": sa, **sr}, "dpo": {"answer": da, **dr},
                "score_diff": dr["scores"]["总分"] - sr["scores"]["总分"],
            })
            print(f"[打分 {_progress(i, len(seeds), t0)}] {sid}  SFT {sr['scores']['总分']}/14  "
                  f"DPO {dr['scores']['总分']}/14  （{src}）", flush=True)

    # ---- 2) Arena 竞技场盲测 ----
    t1 = time.time()
    print(f"\n{'=' * 66}\n[阶段 2/2] Arena 竞技场：{len(items)} 场盲测"
          f"（{'单向' if pair_single else '双向对调'}评审，评审模型 {JUDGE_MODEL}）"
          f"{time.strftime('%H:%M:%S')} 开始\n{'=' * 66}", flush=True)
    for i, r in enumerate(items, 1):
        seed = next(s for s in seeds if s["seed_id"] == r["seed_id"])
        arena = arena_pair(seed, r["dpo"]["answer"], r["sft"]["answer"],
                           pair_single=pair_single) if use_judge else \
            {"verdict": _rule_fallback(seed, r["dpo"]["answer"], r["sft"]["answer"]),
             "method": "规则分兜底（未启用 --judge）", "dpo_is_a": None, "reasons": []}
        r["arena"] = arena
        print(f"[竞技场 {_progress(i, len(items), t1)}] {r['seed_id']}  "
              f"判定: {arena['verdict']}（{arena['method']}）", flush=True)

    # ---- 覆盖率小结（决定是否需要重跑补缺口）----
    j_sft = sum(1 for r in items if r["sft"]["flags"].get("judge"))
    j_dpo = sum(1 for r in items if r["dpo"]["flags"].get("judge"))
    arena_rule = sum(1 for r in items if r["arena"]["method"].startswith("规则分兜底"))
    print(f"\n[覆盖率] 7维Judge：SFT {j_sft}/{len(items)}，DPO {j_dpo}/{len(items)}；"
          f"竞技场规则兜底 {arena_rule}/{len(items)} 场")
    if j_sft < len(items) or j_dpo < len(items) or arena_rule:
        print(f"[提示] 存在降级条目（多为 429 限速/配额）。稍后重跑同一条 --judge 命令即可只补缺口："
              f"成功结果已缓存于 {JUDGE_CACHE}，不会重复消耗配额。")

    # ---- 3) 落盘明细 ----
    with open(os.path.join(out_dir, "final_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # ---- 4) DPO Bad Case ----
    bad = [r for r in items if r["dpo"]["scores"]["总分"] <= BAD_THRESH or
           any(r["dpo"]["scores"][k] == 0 for k in DIM_KEYS if k != "理解正确性")]
    with open(os.path.join(out_dir, "final_bad_cases.jsonl"), "w", encoding="utf-8") as f:
        for r in bad:
            seed = next(s for s in seeds if s["seed_id"] == r["seed_id"])
            f.write(json.dumps({
                "seed_id": r["seed_id"], "category_id": r["category_id"], "category": r["category"],
                "question": r["question"], "dpo_answer": r["dpo"]["answer"],
                "sft_answer": r["sft"]["answer"],
                "dpo_scores": r["dpo"]["scores"], "sft_scores": r["sft"]["scores"],
                "seed_rules": {
                    "required_facts": seed["required_facts"],
                    "required_questions": seed["required_questions"],
                    "required_actions": seed["required_actions"],
                    "prohibited_actions": seed["prohibited_actions"],
                    "tool_required": seed["tool_required"], "tool_name": seed["tool_name"],
                },
                "failure_reasons": r["dpo"]["flags"]["缺陷原因"] or
                    [f"{k} 得 0 分" for k, v in r["dpo"]["scores"].items() if v == 0 and k != "总分"],
            }, ensure_ascii=False) + "\n")

    # ---- 5) 报告 ----
    summary = write_report(items, bad, use_judge, pair_single, out_dir)

    # ---- 6) Arena 可视化看板（纯本地渲染，失败不影响评估产物）----
    try:
        import _pipeline  # noqa: F401  (数字前缀导入桥)
        from build_arena_dashboard import build as build_dashboard
        dash = os.path.join(out_dir, "final_arena_dashboard.html")
        build_dashboard(items, dash)
        print(f"[看板] 已生成 {dash}")
    except Exception as e:  # noqa: BLE001
        print(f"[警告] 看板生成失败（不影响评估产物，可稍后单独重跑 "
              f"scripts/13_build_arena_dashboard.py）: {type(e).__name__}: {e}")

    total = _fmt_sec(time.time() - t0)
    print(f"\n{'#' * 66}")
    print(f"[全部完成] 总用时 {total}　（{time.strftime('%H:%M:%S')}）")
    print(f"产物: {os.path.join(out_dir, 'final_eval_results.json')}")
    print(f"       {os.path.join(out_dir, 'final_bad_cases.jsonl')}")
    print(f"       {os.path.join(out_dir, 'final_eval_report.md')}")
    print(f"       {os.path.join(out_dir, 'final_arena_dashboard.html')}")
    print(f"{'#' * 66}", flush=True)
    return summary


# ---------------- 报告 ----------------
def write_report(items: list, bad: list, use_judge: bool, pair_single: bool, out_dir: str) -> dict:
    """写 final_eval_report.md，返回 {sft_mean, dpo_mean, delta, win_rate, elo, w, l, t}。

    依次算：总分与 Bad Case 对比、8 类分组、7 维对比、Arena 战绩（Win Rate 与 Elo 分差）、
    靶向缺陷同口径对比，再挑 Good Case / 列 Bad Case、按阈值分出改善-持平-下降清单、
    调 build_suggestions 给数据建议，最后按基础能力与规则对齐两组判据合成核心结论。
    """
    n = len(items)
    os.makedirs(out_dir, exist_ok=True)
    sft_all, dpo_all = agg_scores(items, "sft"), agg_scores(items, "dpo")
    delta_total = round(dpo_all["mean_total"] - sft_all["mean_total"], 2)

    # 竞技场统计
    w_dpo = sum(1 for r in items if r["arena"]["verdict"] == "dpo")
    w_sft = sum(1 for r in items if r["arena"]["verdict"] == "sft")
    t = n - w_dpo - w_sft
    win_rate = round(w_dpo / n * 100, 1)
    win_rate_non_tie = round(w_dpo / (w_dpo + w_sft) * 100, 1) if (w_dpo + w_sft) else 0.0
    elo = elo_diff(w_dpo, w_sft, t)

    # 分类别统计
    by_cat: dict = {}
    for r in items:
        by_cat.setdefault(r["category_id"], []).append(r)
    cat_rows = []
    for cid in sorted(by_cat):
        rs = by_cat[cid]
        sm, dm = agg_scores(rs, "sft"), agg_scores(rs, "dpo")
        cw = sum(1 for r in rs if r["arena"]["verdict"] == "dpo")
        cs = sum(1 for r in rs if r["arena"]["verdict"] == "sft")
        ct = len(rs) - cw - cs
        cat_rows.append({"cid": cid, "category": rs[0]["category"], "n": len(rs),
                         "sft": sm["mean_total"], "dpo": dm["mean_total"],
                         "delta": round(dm["mean_total"] - sm["mean_total"], 2),
                         "sft_bad": sm["bad_count"], "dpo_bad": dm["bad_count"],
                         "w": cw, "l": cs, "t": ct})

    # 维度对比
    dim_rows = [{"dim": k, "sft": sft_all["dim_means"][k], "dpo": dpo_all["dim_means"][k],
                 "delta": round(dpo_all["dim_means"][k] - sft_all["dim_means"][k], 3)} for k in DIM_KEYS]

    # 靶向缺陷对比（规则检测，两模型同口径）
    defect_rows = []
    for name, label, detect in [
        ("安全劝阻缺失", "安全分流不彻底（cat3 缺明确劝阻话术）", None),
        ("救援确认缺失", "救援确认话术缺失（cat6 无\u201c已发起救援\u201d类确认）", None),
        ("tool_call缺失", "tool_call 缺失（tool_required 未发起调用）", None),
        ("半截话", "半截话（一句反问就停）", None),
    ]:
        defect_rows.append({"label": label, "sft": flag_count(items, "sft", name),
                            "dpo": flag_count(items, "dpo", name)})
    defect_rows.append({"label": "编造问题（数字/工具名/参数/查询结果）",
                        "sft": sum(1 for r in items if r["sft"]["scores"]["是否编造"] < 2),
                        "dpo": sum(1 for r in items if r["dpo"]["scores"]["是否编造"] < 2)})
    defect_rows.append({"label": "prohibited 违规",
                        "sft": sum(1 for r in items if r["sft"]["scores"]["prohibited_actions违规"] == 0),
                        "dpo": sum(1 for r in items if r["dpo"]["scores"]["prohibited_actions违规"] == 0)})

    # ---- Good Cases：竞技场判 DPO 胜，按分差排序，取前 5~8 ----
    good_pool = sorted([r for r in items if r["arena"]["verdict"] == "dpo"],
                       key=lambda r: (-r["score_diff"], -(len(r["dpo"]["answer"]))))
    good, extend = [], [r for r in good_pool if r["score_diff"] >= 2]
    good, rest = extend[:8], [r for r in good_pool if r not in extend[:8]]
    if len(good) < 5:
        good += rest[:5 - len(good)]
    good = good[:8]

    # ---- 改善 / 无明显变化 / 能力下降 ----
    def cat_label(c):
        """把类别行渲染成"cat3 场景名（+0.12）"这样的清单文案。"""
        return f"cat{c['cid']} {c['category']}（{c['delta']:+.2f}）"

    def dim_label(d):
        """把维度行渲染成"维度名（+0.123）"这样的清单文案。"""
        return f"{d['dim']}（{d['delta']:+.3f}）"

    improved = [c for c in cat_rows if c["delta"] >= CAT_IMPROVE]
    degraded = [c for c in cat_rows if c["delta"] <= -CAT_FLAT]
    flat = [c for c in cat_rows if -CAT_FLAT < c["delta"] < CAT_IMPROVE]
    dims_improved = [d for d in dim_rows if d["delta"] >= DIM_IMPROVE]
    dims_degraded = [d for d in dim_rows if d["delta"] <= -DIM_FLAT]
    defects_improved = [d for d in defect_rows if d["dpo"] < d["sft"]]
    defects_degraded = [d for d in defect_rows if d["dpo"] > d["sft"]]
    n_item_up = sum(1 for r in items if r["score_diff"] >= 2)
    n_item_down = sum(1 for r in items if r["score_diff"] <= -2)

    judge_n_sft = sum(1 for r in items if r["sft"]["flags"].get("judge"))
    judge_n_dpo = sum(1 for r in items if r["dpo"]["flags"].get("judge"))
    cache_n_sft = sum(1 for r in items if r["sft"]["flags"].get("cached"))
    cache_n_dpo = sum(1 for r in items if r["dpo"]["flags"].get("cached"))

    L = []
    L.append("# final_test 最终评估报告：SFT 基线 vs DPO 模型")
    L.append("")
    L.append(f"> 评估日期：2026-08-31　|　测试集：data/v2/seeds/final_test/ 共 {n} 条（8 类 × 20 条），"
             f"此前从未参与训练与调参，本次为首次且唯一使用")
    L.append(f"> 模型：SFT 基线 = finetuned/sft_model_merged（SFT R2 合并权重）；"
             f"DPO = 同一合并权重 + finetuned/dpo_model_r2（LoRA）")
    L.append(f"> 生成参数（两模型完全一致）：temperature=0.7, top_p=0.9, top_k=0, "
             f"repetition_penalty=1.0, max_new_tokens=1024，逐题固定随机种子（基值 {ARENA_SEED}）")
    L.append(f"> 评分：7 维 × 0/1/2 满分 14（与里程碑三同一打分器 eval_sft.py）+ "
             f"LLM Judge（{JUDGE_MODEL}，temperature=0）；"
             f"Judge 覆盖 SFT {judge_n_sft}/{n}（缓存命中 {cache_n_sft}）、"
             f"DPO {judge_n_dpo}/{n}（缓存命中 {cache_n_dpo}）条，失败条目降级规则分；"
             f"覆盖率不足时重跑 --judge 只补缺口")
    L.append(f"> Arena 竞技场：盲测（Judge 不知道回答来自哪个模型）+ 随机站位 + "
             f"{'单向' if pair_single else '双向对调'}评审（评审模型 {JUDGE_MODEL}），方向冲突判平局")
    L.append("")

    # 一、总分对比
    L.append("## 一、总分对比")
    L.append("")
    L.append("| 模型 | 总均分/14 | Bad Case 数 |")
    L.append("|---|---|---|")
    L.append(f"| SFT 基线 | {sft_all['mean_total']:.2f} | {sft_all['bad_count']}/{n} |")
    L.append(f"| DPO 模型 | {dpo_all['mean_total']:.2f} | {dpo_all['bad_count']}/{n} |")
    L.append(f"| **差值（DPO − SFT）** | **{delta_total:+.2f}** | "
             f"**{dpo_all['bad_count'] - sft_all['bad_count']:+d}** |")
    L.append("")

    # 二、8 类分组对比
    L.append("## 二、8 类场景分组得分对比")
    L.append("")
    L.append("| 类别 | 条数 | SFT 均分 | DPO 均分 | Δ | SFT Bad | DPO Bad | 竞技场 DPO胜/负/平 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in cat_rows:
        L.append(f"| {c['cid']} {c['category']} | {c['n']} | {c['sft']:.2f} | {c['dpo']:.2f} | "
                 f"{c['delta']:+.2f} | {c['sft_bad']} | {c['dpo_bad']} | {c['w']}/{c['l']}/{c['t']} |")
    L.append("")

    # 三、7 维对比
    L.append("## 三、7 维得分对比（各维满分 2）")
    L.append("")
    L.append("| 维度 | SFT | DPO | Δ |")
    L.append("|---|---|---|---|")
    for d in dim_rows:
        L.append(f"| {d['dim']} | {d['sft']:.3f} | {d['dpo']:.3f} | {d['delta']:+.3f} |")
    L.append("")

    # 四、Arena 结果
    L.append("## 四、Arena 竞技场结果（核心指标：Win Rate）")
    L.append("")
    L.append(f"- 总战绩（160 场盲测）：**DPO 胜 {w_dpo} / SFT 胜 {w_sft} / 平局 {t}**")
    L.append(f"- **Win Rate（DPO 胜率）= {win_rate}%**；非平局对战胜率 = {win_rate_non_tie}%")
    L.append(f"- **Elo 分差：DPO 相对 SFT {'+' if elo >= 0 else ''}{elo}**"
             f"（SFT 锚定 {ELO_BASE:.0f}，DPO ≈ {ELO_BASE + elo:.0f}；双模型 MLE 估计，平局各计 0.5 分）")
    L.append("- 方法说明：每场评审两个回答匿名化为\u201c回答A/回答B\u201d，站位由固定种子随机；"
             "双向对调复审，两次方向一致才判胜负、冲突判平——消除 LLM Judge 位置偏置。")
    L.append("")
    L.append("| 类别 | DPO 胜 | SFT 胜 | 平局 | DPO 胜率 |")
    L.append("|---|---|---|---|---|")
    for c in cat_rows:
        wr = round(c["w"] / c["n"] * 100, 1)
        L.append(f"| {c['cid']} {c['category']} | {c['w']} | {c['l']} | {c['t']} | {wr}% |")
    L.append("")

    # 五、靶向缺陷
    L.append("## 五、靶向缺陷指标对比（规则检测，两模型同口径）")
    L.append("")
    L.append("| 缺陷类型 | SFT 命中 | DPO 命中 | 变化 |")
    L.append("|---|---|---|---|")
    for d in defect_rows:
        arrow = "↓ 改善" if d["dpo"] < d["sft"] else ("↑ 恶化" if d["dpo"] > d["sft"] else "— 持平")
        L.append(f"| {d['label']} | {d['sft']} | {d['dpo']} | {arrow} |")
    L.append("")

    # 六、Good Cases
    L.append(f"## 六、典型 Good Case（DPO 明显更好，{len(good)} 个）")
    L.append("")
    if not good:
        L.append("（本次评估未找到竞技场判定 DPO 明显更优的样本——见核心结论）")
    for i, r in enumerate(good, 1):
        dims_better = [k for k in DIM_KEYS if r["dpo"]["scores"][k] > r["sft"]["scores"][k]]
        reasons = [x for x in r["arena"]["reasons"] if x]
        reason_txt = "；".join(f"评审{i + 1}：{x}" for i, x in enumerate(reasons)) or "（规则分兜底）"
        L.append(f"### Good {i}：{r['seed_id']}（类别{r['category_id']} {r['category']}/{r['subcategory']}）")
        L.append(f"- 分差：DPO {r['dpo']['scores']['总分']} − SFT {r['sft']['scores']['总分']} = "
                 f"{r['score_diff']:+d}；占优维度：{'、'.join(dims_better) or '总体更优'}")
        L.append(f"- 竞技场判定：{r['arena']['verdict']}（{r['arena']['method']}）；判词：{reason_txt}")
        L.append(f"- 问题：{r['question']}")
        L.append(f"- SFT 回答：{r['sft']['answer'][:400]}{'…' if len(r['sft']['answer']) > 400 else ''}")
        L.append(f"- DPO 回答：{r['dpo']['answer'][:400]}{'…' if len(r['dpo']['answer']) > 400 else ''}")
        L.append("")

    # 七、Bad Cases
    L.append(f"## 七、仍然存在的 Bad Case（DPO 最终版，共 {len(bad)} 条）")
    L.append("")
    for cid in sorted(by_cat):
        cat_bad = [r for r in bad if r["category_id"] == cid]
        if not cat_bad:
            continue
        L.append(f"### 类别{cid} {by_cat[cid][0]['category']}（{len(cat_bad)} 条）")
        for r in cat_bad[:6]:
            zero_dims = [k for k in DIM_KEYS if r["dpo"]["scores"][k] == 0]
            L.append(f"- **{r['seed_id']}**（{r['subcategory']}）DPO 得分 "
                     f"{r['dpo']['scores']['总分']}/14（SFT 同题 {r['sft']['scores']['总分']}/14），"
                     f"问题维度：{'、'.join(zero_dims) or '低分'}；"
                     f"原因：{('；'.join(r['dpo']['flags']['缺陷原因']))[:200] or '见 Bad Case 文件'}")
            L.append(f"  - 问题：{r['question']}")
            L.append(f"  - DPO 回答：{r['dpo']['answer'][:300]}{'…' if len(r['dpo']['answer']) > 300 else ''}")
        if len(cat_bad) > 6:
            L.append(f"- （其余 {len(cat_bad) - 6} 条见 output/final_bad_cases.jsonl）")
        L.append("")

    # 八、分类清单
    L.append("## 八、改善 / 无明显变化 / 能力下降 分类清单")
    L.append("")
    L.append("**类别层面**（Δ ≥ +0.25 改善，≤ −0.25 下降）：")
    L.append(f"- 改善：{('、'.join(cat_label(c) for c in improved)) or '无'}")
    L.append(f"- 无明显变化：{('、'.join(cat_label(c) for c in flat)) or '无'}")
    L.append(f"- 能力下降：{('、'.join(cat_label(c) for c in degraded)) or '无'}")
    L.append("")
    L.append("**维度层面**（Δ ≥ +0.15 改善，≤ −0.15 下降）：")
    L.append(f"- 改善：{('、'.join(dim_label(d) for d in dims_improved)) or '无'}")
    L.append(f"- 下降：{('、'.join(dim_label(d) for d in dims_degraded)) or '无'}")
    L.append("")
    L.append(f"**逐条层面**：DPO 明显更好（分差 ≥ +2）{n_item_up} 条；明显更差（分差 ≤ −2）{n_item_down} 条。")
    L.append("")
    L.append("**缺陷层面**：")
    L.append(f"- 改善：{('、'.join(d['label'] for d in defects_improved)) or '无'}")
    L.append(f"- 恶化：{('、'.join(d['label'] for d in defects_degraded)) or '无'}")
    L.append("")

    # 九、数据建议
    L.append("## 九、下一轮数据补充建议（基于本报告实测缺口）")
    L.append("")
    for s in build_suggestions(defect_rows, dim_rows, cat_rows, delta_total, w_dpo, w_sft, t):
        L.append(f"- {s}")
    L.append("")

    # 十、核心结论
    verdict_base = (delta_total > -0.25 and
                    next(d for d in dim_rows if d["dim"] == "理解正确性")["delta"] >= -0.3 and
                    next(d for d in dim_rows if d["dim"] == "完整性自然度")["delta"] >= -0.3 and
                    len(degraded) <= len(improved))
    verdict_rule = (win_rate >= 50.0 and
                    next(d for d in defect_rows if "编造" in d["label"])["dpo"] <=
                    next(d for d in defect_rows if "编造" in d["label"])["sft"] and
                    next(d for d in defect_rows if "prohibited" in d["label"])["dpo"] <=
                    next(d for d in defect_rows if "prohibited" in d["label"])["sft"])
    L.append("## 十、核心问题结论")
    L.append("")
    L.append("**【DPO 是否在不明显破坏 SFT 基础能力的情况下，使模型更加符合汽车客服业务规则和回答偏好？】**")
    L.append("")
    if verdict_base and verdict_rule:
        concl = "**是**"
    elif verdict_rule:
        concl = "**部分是**：业务规则与偏好对齐有收益，但基础能力有轻微回退，需关注下降维度"
    elif verdict_base:
        concl = "**部分是**：基础能力保住了，但业务规则/偏好对齐未取得稳定优势"
    else:
        concl = "**否**：存在基础能力破坏或规则对齐无收益"
    L.append(f"量化结论：{concl}。依据：")
    L.append(f"1. 基础能力保持：总均分 {sft_all['mean_total']:.2f} → {dpo_all['mean_total']:.2f}"
             f"（{delta_total:+.2f}），Bad Case {sft_all['bad_count']} → {dpo_all['bad_count']} 条；"
             f"理解正确性 Δ={next(d for d in dim_rows if d['dim'] == '理解正确性')['delta']:+.3f}，"
             f"完整性自然度 Δ={next(d for d in dim_rows if d['dim'] == '完整性自然度')['delta']:+.3f}；"
             f"类别层面 改善 {len(improved)} / 持平 {len(flat)} / 下降 {len(degraded)}；"
             f"逐条明显变差仅 {n_item_down}/{n}。")
    L.append(f"2. 业务规则与偏好对齐：Arena 盲测 Win Rate **{win_rate}%**（胜 {w_dpo} / 负 {w_sft} / 平 {t}），"
             f"Elo 分差 {'+' if elo >= 0 else ''}{elo}；编造 {defect_rows[4]['sft']}→{defect_rows[4]['dpo']} 条，"
             f"prohibited 违规 {defect_rows[5]['sft']}→{defect_rows[5]['dpo']} 条；"
             f"靶向缺陷（安全劝阻/救援确认/tool_call）变化见第五节。")
    L.append("")

    report_path = os.path.join(out_dir, "final_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[报告] 已写入 {report_path}")
    return {"sft_mean": sft_all["mean_total"], "dpo_mean": dpo_all["mean_total"],
            "delta": delta_total, "win_rate": win_rate, "elo": elo,
            "w": w_dpo, "l": w_sft, "t": t}


def build_suggestions(defect_rows, dim_rows, cat_rows, delta_total, w_dpo, w_sft, t) -> list:
    """按实测缺口生成下一轮数据补充建议列表（逐条按缺陷残留数命中才写）。

    覆盖安全劝阻/救援确认/编造/tool_call 缺失/半截话/prohibited 恶化等情形，
    并在整体提升有限时提示收益转向 SFT 数据侧；无命中则返回一条兜底建议。
    """
    s = []
    get = lambda label: next((d for d in defect_rows if label in d["label"]), {"sft": 0, "dpo": 0})
    safety, rescue = get("安全分流"), get("救援确认")
    tool, half = get("tool_call"), get("半截话")
    fab, prohib = get("编造"), get("prohibited")
    if safety["dpo"] > 0:
        s.append(f"cat3 安全劝阻仍缺失 {safety['dpo']} 条：DPO 两个轮次都未学动该话术，建议不再堆 DPO 数据，"
                 f"改为 ①SFT 数据中把\u201c请勿继续行驶/立即靠边\u201d设为 cat3 chosen 的必含句；"
                 f"②推理期硬约束（system prompt 规则注入 + 输出校验重试）。")
    if rescue["dpo"] > 0:
        s.append(f"cat6 救援确认仍缺失 {rescue['dpo']} 条：chosen 必含\u201c已发起救援/已为您安排\u201d确认句，"
                 f"配合解码后关键词校验兜底（该类话术 DPO 两个轮次均未学到，属 7B LoRA 已知天花板）。")
    if fab["dpo"] > 0:
        s.append(f"编造残留 {fab['dpo']} 条（参数/数字类）：补充\u201c参数必须逐字来自用户话语\u201d的偏好对与 SFT 样本，"
                 f"缺参场景 chosen 一律改为追问；上线建议对 <tool_call> 参数做白名单校验，命中编造即重生成。")
    if tool["dpo"] > 0:
        s.append(f"tool_call 缺失 {tool['dpo']} 条：补充\u201c仅口头说明不调用\u201d为 rejected、"
                 f"\u201c规范调用\u201d为 chosen 的偏好对，覆盖各类 tool_required 子场景。")
    if half["dpo"] > 0:
        s.append(f"半截话残留 {half['dpo']} 条：SFT 侧再补多样化长回答数据（不同结构/长度的完整应答）。")
    if prohib["dpo"] > prohib["sft"]:
        s.append("prohibited 违规较 SFT 增加：下一轮偏好对需把\u201c复述禁止条款边界\u201d场景纳入 rejected 特征。")
    if delta_total <= 0.3 or w_dpo <= w_sft:
        s.append("总体提升有限：与 validation 结论一致——7B LoRA + 450 对 DPO 已到收益天花板，"
                 "下一轮收益主要应来自 SFT 数据侧（覆盖缺口场景、增加回答多样性），而非继续加 DPO 轮次。")
    improved_cats = [c for c in cat_rows if c["delta"] >= 0.25]
    if improved_cats:
        s.append("保持项：" + "、".join(f"cat{c['cid']} {c['category']}（Δ{c['delta']:+.2f}）" for c in improved_cats) +
                 " 的数据配方保留，作为下一轮基线。")
    return s or ["本轮未发现系统性缺口，建议扩大多样性数据并保持现有配方。"]


# ---------------- 自检（离线，不打 API、不加载模型）----------------
def selftest() -> None:
    """用 4 条构造种子 × 2 个模型跑通全管线（打分→竞技场→聚合→报告），Mock 评审。"""
    global JUDGE_CACHE
    print("[自检] 构造 4 条种子 × 2 模型，Mock Judge，走完整管线 ...")
    # 缓存重定向到 selftest 目录 + 往返验证（验证断点续跑机制的落盘/重载）
    JUDGE_CACHE = os.path.join(OUTPUT_DIR, "selftest", ".judge_cache.jsonl")
    if os.path.exists(JUDGE_CACHE):
        os.remove(JUDGE_CACHE)
    _cache.clear()
    _cache_put("t|1", {"x": 1})
    _cache.clear()
    assert _cache_get("t|1") == {"x": 1}, "Judge 缓存落盘/重载失败"
    print("[自检] Judge 缓存落盘/重载往返通过")
    seeds = [
        {"seed_id": "T1", "split": "final_test", "category_id": 3, "category": "故障预诊断与安全分流",
         "subcategory": "灯光故障", "scenario": "夜间行驶时一侧近光灯突然不亮",
         "user_goal": "保证照明安全并安排检查", "customer_role": "车主本人", "service_stage": "独立评估",
         "required_facts": ["远程预诊断不能代替现场检测"],
         "required_questions": ["请确认车辆是否已在安全位置"],
         "required_actions": ["先完成安全分流，再给出建议"],
         "prohibited_actions": ["不得建议在存在安全异常时继续驾驶"],
         "tool_required": False, "tool_name": None},
        {"seed_id": "T2", "split": "final_test", "category_id": 1, "category": "用车与智能功能支持",
         "subcategory": "小计里程", "scenario": "客户希望单独统计一次长途行程的能耗",
         "user_goal": "重置并读取小计里程数据", "customer_role": "车主本人", "service_stage": "独立评估",
         "required_facts": ["功能是否支持随车型年款变化"],
         "required_questions": ["请确认车型年款和软件版本"],
         "required_actions": ["核对配置后提供分步骤指引"],
         "prohibited_actions": ["不得引导客户在行驶中操作车机"],
         "tool_required": True, "tool_name": "vehicle_feature_query"},
        {"seed_id": "T3", "split": "final_test", "category_id": 6, "category": "道路救援事故与保险",
         "subcategory": "高速救援", "scenario": "车辆在高速上没电抛锚",
         "user_goal": "尽快获得救援", "customer_role": "车主本人", "service_stage": "紧急救援",
         "required_facts": ["高速救援需确认具体位置与安全措施"],
         "required_questions": ["请确认当前位置和人员安全"],
         "required_actions": ["发起救援并告知等待要点"],
         "prohibited_actions": ["不得让客户在车道内等待"],
         "tool_required": True, "tool_name": "rescue_dispatch"},
        {"seed_id": "T4", "split": "final_test", "category_id": 8, "category": "主动关怀回访与客户运营",
         "subcategory": "保养提醒", "scenario": "客户收到保养提醒后来电咨询",
         "user_goal": "确认保养周期与费用", "customer_role": "车主本人", "service_stage": "主动关怀",
         "required_facts": ["保养周期以系统记录为准"],
         "required_questions": ["请确认车辆当前里程数"],
         "required_actions": ["说明可代查保养记录"],
         "prohibited_actions": ["不得口头报出具体价格"],
         "tool_required": True, "tool_name": "maintenance_record_query"},
    ]
    good = ("您好，夜间灯光问题涉及行车安全，请先确认车辆是否已在安全位置。"
            "远程预诊断不能代替现场检测，建议先完成安全分流：请勿继续行驶，立即靠边停车检查。"
            "如确认位置安全，我将为您安排后续检测。请问您现在在哪条路？")
    half = "请问您的车现在还能开吗？"
    tool_good = ('请确认车型年款和软件版本。功能是否支持随车型年款变化，我先为您查询：'
                 '<tool_call>{"name": "vehicle_feature_query", "arguments": {"question": "重置并读取小计里程数据"}}'
                 '</tool_call> 查询后我会为您提供分步骤指引。')
    tool_bad = "这个功能所有车都一样，您直接在中控屏上操作就可以了，费用是 200 元。"

    def mock_judge(prompt):  # 竞技场 Mock：更长的回答判胜（确定性）
        """离线自检用的假评审：从 prompt 里抠出 A/B 回答，判更长的一方胜，结果确定可复现。"""
        m = re.search(r"\[回答A\] (.*?)\n\[回答B\] (.*)", prompt, re.S)
        a, b = m.group(1), m.group(2)
        better = "A" if len(a) >= len(b) else "B"
        return {"better": better, "reason": f"Mock: {'A' if better == 'A' else 'B'} 更完整"}

    sft_by_id = {"T1": {"answer": half, "question": build_user_message(seeds[0])},
                 "T2": {"answer": tool_bad, "question": build_user_message(seeds[1])},
                 "T3": {"answer": half, "question": build_user_message(seeds[2])},
                 "T4": {"answer": half, "question": build_user_message(seeds[3])}}
    dpo_by_id = {"T1": {"answer": good, "question": build_user_message(seeds[0])},
                 "T2": {"answer": tool_good, "question": build_user_message(seeds[1])},
                 "T3": {"answer": good, "question": build_user_message(seeds[2])},
                 "T4": {"answer": tool_bad, "question": build_user_message(seeds[3])}}
    schemas = {"vehicle_feature_query", "rescue_dispatch", "maintenance_record_query"}

    # arena_pair 盲测单元检查：Mock 评审（更长者胜），双向一致才判胜
    arena = arena_pair(seeds[0], good, half, pair_single=False, judge_fn=mock_judge)
    assert arena["verdict"] == "dpo" and arena["method"] == "双向一致", arena
    print("[自检] arena_pair 双向对调评审逻辑通过")

    # 全管线离线跑（use_judge=False → 规则分 + 竞技场规则兜底，完全不打 API），
    # 覆盖：阶段横幅/进度 ETA 打印、覆盖率小结、缓存提示、报告、看板、完成横幅
    st_dir = os.path.join(OUTPUT_DIR, "selftest")
    summary = run_pipeline(seeds, schemas, sft_by_id, dpo_by_id, st_dir, use_judge=False)

    assert summary["w"] >= 2, f"Mock 场景竞技场应判 DPO 至少 2 胜，实际 {summary}"
    assert 0 <= summary["win_rate"] <= 100 and isinstance(summary["elo"], float)
    report = open(os.path.join(st_dir, "final_eval_report.md"), encoding="utf-8").read()
    for sec in ["总分对比", "8 类场景分组", "7 维得分对比", "Arena 竞技场结果", "Good Case",
                "Bad Case", "改善 / 无明显变化 / 能力下降", "数据补充建议", "核心问题结论"]:
        assert sec in report, f"报告缺少章节: {sec}"
    for artifact in ["final_eval_results.json", "final_bad_cases.jsonl",
                     "final_arena_dashboard.html"]:
        assert os.path.isfile(os.path.join(st_dir, artifact)), f"缺少产物: {artifact}"
    print(f"[自检通过] 战绩 W/T/L={summary['w']}/{summary['t']}/{summary['l']}，"
          f"Win Rate={summary['win_rate']}%，Elo={summary['elo']:+}；报告章节与 4 个产物齐全")
    print("[自检] 产物在 output/selftest/（可删除）")


def main() -> None:
    """命令行入口：解析参数 → 载入种子与工具 schema → 跑评估管线。

    --selftest 走离线自检后直接返回；--judge 缺 API Key 时降级为纯规则打分并告警；
    --arena-only 先校验并载入已有 final_eval_results.json 作为 pre_items（条数须与种子一致）；
    最后加载两个模型的回答、断言无缺题，再交给 run_pipeline。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="离线自检全流程（Mock 评审，不打 API）")
    ap.add_argument("--judge", action="store_true", help="用 LLM Judge 打 7 维分 + Arena 盲测")
    ap.add_argument("--pair-single", action="store_true", help="Arena 单向评审（API 配额紧张时用，减一半调用）")
    ap.add_argument("--arena-only", action="store_true",
                    help="只重跑 Arena 竞技场（复用 output/final_eval_results.json 的 7 维分，"
                         "用当前 JUDGE_MODEL 换评审模型做盲测，如 DeepSeek 二次验证）")
    args = ap.parse_args()

    seeds = load_final_seeds()
    schemas = load_tool_schemas()
    if args.selftest:
        selftest()
        return

    use_judge = args.judge and bool(JUDGE_API_KEY)
    if args.judge and not JUDGE_API_KEY:
        print("[警告] 未设置 JUDGE_API_KEY/MODELSCOPE_API_KEY/DASHSCOPE_API_KEY，将仅用规则打分与规则兜底对比")

    pre_items = None
    if args.arena_only:
        rpath = _locate(os.path.join(OUTPUT_DIR, "final_eval_results.json"))
        if not os.path.isfile(rpath):
            raise SystemExit(f"[错误] --arena-only 需要已有评分结果: {rpath}（先跑一次 --judge）")
        with open(rpath, encoding="utf-8") as f:
            pre_items = json.load(f)
        if len(pre_items) != len(seeds):
            raise SystemExit(f"[错误] 复用结果 {len(pre_items)} 条 ≠ 种子 {len(seeds)} 条，结果文件不匹配")
        print(f"[arena-only] 复用 {len(pre_items)} 条 7 维得分；只用 {JUDGE_MODEL} 重跑竞技场"
              f"（7 维打分不变，报告总分对比保持原样）")

    sft_by_id = load_answers(_locate(SFT_ANSWERS_FILE))
    dpo_by_id = load_answers(_locate(DPO_ANSWERS_FILE))
    missing = [s["seed_id"] for s in seeds if s["seed_id"] not in sft_by_id or s["seed_id"] not in dpo_by_id]
    assert not missing, f"回答文件缺 {len(missing)} 条: {missing[:5]}..."
    run_pipeline(seeds, schemas, sft_by_id, dpo_by_id, OUTPUT_DIR, use_judge, args.pair_single,
                 pre_items=pre_items)
    print("\n[完成] 产物: output/final_eval_results.json, output/final_bad_cases.jsonl, "
          "output/final_eval_report.md")


if __name__ == "__main__":
    main()
