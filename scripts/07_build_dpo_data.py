# -*- coding: utf-8 -*-
"""
【流水线 07/16】里程碑四 · 构造 DPO 偏好数据（一轮，180 对）
运行位置：本机（无需 GPU）
输入：output/sft_bad_cases.jsonl + sft_eval_results.json + train 种子
输出：data/v2/dpo/dpo_train.jsonl、dpo_train_trl.jsonl、报告
前置步骤：06　｜　后续步骤：08 二轮扩充

里程碑四：构造 DPO 偏好数据（rejected=真实缺陷回答，chosen=按 seed_rules 规则模板生成）。

数据来源：
  A. output/sft_bad_cases.jsonl 27 条 Bad Case（必须包含）：rejected = SFT 模型原始回答，不改写；
  B. output/sft_eval_results.json 其余带缺陷条目（补充样本）：rejected = 模型原始回答；
  C. data/v2/seeds/train/ 种子合成补充对：chosen=规则模板，rejected=按缺陷类型模板化退化
     （复刻实测缺陷：套话半截话 / 不发起 tool_call / 幻觉工具名+编造参数 / 缺安全劝阻）。

质量红线（自检）：
  - 每条 chosen 必须通过 eval_sft.score_answer 规则打分 >= --min-chosen-score（默认 12/14）；
  - 分差 margin = score(chosen)-score(rejected) >= --margin（默认 4）才保留，宁缺毋滥；
  - 长度惩罚：chosen 长度 >= 1.5 倍 rejected 且 margin < 6 时丢弃（防止"仅因更长而胜出"）；
  - tool_required=true：chosen 输出 <tool_call>，工具名用 seed.tool_name（必须在 schema 内），
    参数值只能取自用户话语（scenario+user_goal），缺参明确追问；tool_required=false 不发起任何调用；
  - cat3 chosen 含"请勿继续行驶/立即靠边停车"类劝阻；cat6 chosen 含"已为您发起救援"类确认；
  - final_test/ 绝不使用。

输出：
  data/v2/dpo/dpo_train.jsonl        ms-swift DPO 格式（messages + chosen/rejected 消息对象）
  data/v2/dpo/dpo_train_trl.jsonl    TRL DPOTrainer 会话格式（prompt/chosen/rejected）
  data/v2/dpo/dpo_data_report.md     配比统计与构造规则说明
  data/v2/dpo/sample_10_for_review.txt  随机 10 对人工抽检

用法（项目根，本机或实例均可）:
    python scripts/build_dpo_data.py --selftest
    python scripts/build_dpo_data.py
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import SYSTEM_PROMPT, build_user_message, score_answer, _locate  # noqa: E402

VALIDATION_DIR = "data/v2/seeds/validation"
TRAIN_DIR = "data/v2/seeds/train"
BAD_CASES = "output/sft_bad_cases.jsonl"
EVAL_RESULTS = "output/sft_eval_results.json"
OUT_DIR = "data/v2/dpo"

DEFECT_TYPES = ["安全话术类", "工具规范类", "参数编造类", "完整性类"]

# chosen 中可由用户话语（user_goal）直接取值的自由文本参数；其余参数一律缺参追问，不编造
FREE_TEXT_PARAMS = {"issue_summary", "question", "description", "remark", "content",
                    "note", "reason", "issue", "summary", "demand", "request", "symptom"}
PARAM_LABELS = {"vehicle_id": "车辆识别代号/车架号", "location": "车辆当前位置",
                "work_order_id": "维修工单号", "preferred_date": "期望服务日期",
                "store_id": "服务门店", "feature_name": "功能名称", "policy_type": "政策类型",
                "part_name": "配件名称", "phone": "联系电话", "case_id": "案件编号",
                "vehicle_model": "车型年款", "topic": "咨询主题", "city": "所在城市",
                "service_type": "服务类型", "contact_channel": "联系方式/沟通渠道",
                "customer_phone": "联系电话",
                "vin": "车辆识别代号/车架号", "date": "日期",
                "address": "详细地址", "capability": "能力项", "dealer_id": "门店编号",
                "mileage": "当前行驶里程", "phone_model": "手机型号",
                "policy_topic": "政策主题", "region": "所在地区"}
# 合成 rejected 的参数编造素材池（复刻实测幻觉值：VID26…、S2463、“2024款长续航版”等）
FAB_POOL = {"vehicle_id": "VID264381759016661", "work_order_id": "RO20250812001",
            "store_id": "S2463", "location": "S2463门店", "preferred_date": "下周三上午",
            "feature_name": "2024款长续航版", "policy_type": "2024款长续航版",
            "part_name": "2024款长续航版原厂件"}
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"


def cn(n: int) -> str:
    """数量转中文量词：2 读作“两”，1-10 用中文数字，超出范围直接转字符串。"""
    return {2: "两"}.get(n, "一二三四五六七八九十"[n - 1] if 1 <= n <= 10 else str(n))


def load_jsonl(path: str) -> list:
    """逐行读 JSONL，跳过空行，返回 dict 列表。"""
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_seed_dir(path: str, marker: str) -> dict:
    """用 marker 文件定位种子目录，加载其下全部 jsonl，返回 {seed_id: 种子}。"""
    d = _locate(path, marker=marker)
    seeds = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for s in load_jsonl(fp):
            seeds[s["seed_id"]] = s
    return seeds


def load_schemas() -> dict:
    """返回 {工具名: schema对象}（score_answer 只做名字 in 判断，传 dict 兼容）。"""
    p = _locate("data/v2/tool_schemas.json")
    if not os.path.isfile(p):
        raise SystemExit(f"[错误] 找不到工具 schema: {p}")
    with open(p, encoding="utf-8") as f:
        return {t["name"]: t for t in json.load(f)["tools"]}


# ---------------- chosen：规则模板生成（基于 seed_rules 可程序化拼装） ----------------

def build_tool_args(seed: dict, tool: str, schemas: dict) -> tuple:
    """参数值只能来自用户话语（scenario+user_goal，即 question 的实义部分）；
    自由文本参数取 user_goal，其余缺参返回待追问清单。绝不编造。"""
    props = schemas.get(tool, {}).get("parameters", {}).get("properties", {})
    user_text = seed["scenario"] + seed["user_goal"]
    args, missing = {}, []
    for p in props:
        if p in FREE_TEXT_PARAMS and seed["user_goal"] and seed["user_goal"] in user_text:
            args[p] = seed["user_goal"]
        else:
            missing.append(p)
    return args, missing


def _assemble(seed: dict, schemas: dict, variant: int) -> str:
    """按 seed_rules 拼装一版 chosen：开场（cat3 加劝阻 / cat6 加救援确认）+ facts 分点
    +（tool_required 时）tool_call 与缺参追问 + questions 追问 + actions 分步 + 收尾。
    variant 只换开场措辞，用于打分不达标时重试。"""
    cat = seed["category_id"]
    scenario = seed["scenario"]
    facts = seed.get("required_facts") or []
    qs = seed.get("required_questions") or []
    acts = seed.get("required_actions") or []
    tr = seed.get("tool_required", False)
    tool = seed.get("tool_name")

    parts = []
    if variant == 0:
        if cat == 3:
            parts.append(f"您好，您反映的情况是「{scenario}」，这涉及行车安全，请勿继续行驶，"
                         f"立即靠边停车，确保人员和车辆处于安全位置后，我们再逐步处理。")
        elif cat == 6:
            parts.append(f"您好，别着急，您反映的情况是「{scenario}」，请先确保人员和现场安全，"
                         f"我马上为您协调处理。")
        else:
            parts.append(f"您好，感谢您联系我们。您反映的情况是「{scenario}」，我来帮您处理。")
    else:  # 变体开场
        if cat == 3:
            parts.append(f"您好，安全第一。关于「{scenario}」，请勿继续行驶，立即靠边停车，"
                         f"确认现场安全后我来帮您安排。")
        elif cat == 6:
            parts.append(f"您好，请您先保证自身和现场安全，关于「{scenario}」，我马上为您协调处理。")
        else:
            parts.append(f"您好，您的情况我已经记录下来了。关于「{scenario}」，我来帮您分析处理。")

    if facts:
        parts.append("这里先跟您说明" + cn(len(facts)) + "点关键信息："
                     + "；".join(f"{CIRCLED[i]}{f}" for i, f in enumerate(facts)) + "。")

    if tr and tool and tool in schemas:
        args, missing = build_tool_args(seed, tool, schemas)
        call = ('<tool_call>\n{"name": "' + tool + '", "arguments": '
                + json.dumps(args, ensure_ascii=False, separators=(",", ":")) + "}\n</tool_call>")
        lead = "我已为您发起救援调度，" if cat == 6 else "我现在就帮您在系统里发起查询，"
        seg = (lead + call + "查询提交后请您稍等，结果出来我会第一时间同步给您，"
               "在此之前我不会给您任何猜测性的结论。")
        if missing:
            seg += ("另外，为了确保一次查准，还请您提供：" + "；".join(
                f"{CIRCLED[i]}{PARAM_LABELS.get(p, '「' + p + '」对应信息')}"
                for i, p in enumerate(missing)) + "。")
        parts.append(seg)

    if qs:
        parts.append("在处理之前，为了不误判，请您配合确认" + cn(len(qs)) + "点："
                     + "；".join(f"{CIRCLED[i]}{q}" for i, q in enumerate(qs)) + "。")
    if acts:
        parts.append("接下来您可以这样处理："
                     + "；".join(f"{CIRCLED[i]}{a}" for i, a in enumerate(acts)) + "。")

    tail = "祝您行车平安。" if cat in (3, 6) else "祝您用车愉快。"
    parts.append(f"后续有任何进展或疑问，欢迎随时联系我们，{tail}")
    return "".join(parts)


def make_chosen(seed: dict, schemas: dict, min_score: int) -> tuple:
    """生成 chosen；不达标依次换模板重试，仍不达标返回 None（宁缺毋滥）。"""
    best = None
    for v in range(3):
        text = _assemble(seed, schemas, v)
        r = score_answer(seed, text, schemas)
        if best is None or r["scores"]["总分"] > best[1]["scores"]["总分"]:
            best = (text, r)
        if r["scores"]["总分"] >= min_score:
            return text, r, v
    return (*best, None) if best else (None, None, None)


# ---------------- rejected：真实缺陷回答 + 合成退化模板 ----------------

def _half_base(seed: dict) -> str:
    """复刻实测套话半截话：开场 + 一个（常常偏题的）确认问题，无实质内容。"""
    qs = seed.get("required_questions") or []
    if qs:
        import re
        q0 = re.split(r"以及|和|、|，", qs[0])[0][:14] or "车型年款"
    else:
        q0 = "车型年款"
    return (f"您好，感谢您联系我们，您的情况我已经记录下来了。"
            f"为了给您准确的处理方案，我先跟您确认一个信息：{q0}。")


def make_synthetic_rejected(seed: dict, defect: str, idx: int, schemas: dict) -> str:
    """按缺陷类型在半截话基础上退化出 rejected：完整性类只留半截话，工具规范类改成"已登记"
    不发起调用，参数编造类发起 tool_call 但参数全取幻觉池，安全话术类去掉劝阻/救援确认。"""
    base = _half_base(seed)
    if defect == "完整性类":
        return base
    if defect == "工具规范类":
        return base + "这边先帮您登记一下，稍后有专人回复您，请您保持电话畅通。"
    if defect == "参数编造类":
        tool = seed.get("tool_name")
        if not tool or tool not in schemas:  # 非工具场景：直接伪造查询结果（同样是实测缺陷）
            return (base + "我这边在系统里帮您查过了，您的车辆是2024款长续航版，"
                    "相关服务都是支持的，您放心使用。")
        props = schemas.get(tool, {}).get("parameters", {}).get("properties", {})
        args = {p: FAB_POOL.get(p, FAB_POOL["feature_name"] + str(idx % 10)) for p in props}
        call = ('<tool_call>\n{"name": "' + tool + '", "arguments": '
                + json.dumps(args, ensure_ascii=False, separators=(",", ":")) + "}\n</tool_call>")
        return (base + "我先帮您在系统里查一下，" + call + "查到结果后我直接告诉您结论，"
                "您不用再跑一趟。")
    if defect == "安全话术类":
        if seed["category_id"] == 3:
            return (base + "这个问题比较常见，影响不大，您先继续正常行驶，"
                    "有空再到店检查一下就可以了。")
        return base + "救援的事情稍后给您安排，请保持电话畅通，先不用着急。"
    return base


# ---------------- 缺陷类型判定（真实回答用） ----------------

def type_real_answer(seed: dict, answer: str, flags: dict, schemas: dict) -> str | None:
    """由打分 flags 判定真实回答属于哪类缺陷，优先级：安全话术 > 参数编造 > 工具规范 > 完整性；
    四类都不命中返回 None（无可判定缺陷，不入选）。"""
    scores = flags.pop("_scores", {})
    suspicions = flags.get("编造疑点") or []
    cat = seed["category_id"]
    if cat in (3, 6) and (flags.get("安全劝阻缺失") or flags.get("救援确认缺失")):
        return "安全话术类"
    if any(("工具名不匹配" in s or "参数值疑似编造" in s or "非法 JSON" in s) for s in suspicions):
        return "参数编造类"
    if (flags.get("tool_call缺失") or any("tool_required=false 却发起" in s for s in suspicions)
            or (seed.get("tool_required") and "<tool_call>" not in answer)):
        return "工具规范类"
    if flags.get("半截话") or scores.get("完整性自然度", 2) <= 1:
        return "完整性类"
    return None


# ---------------- 配对过滤 ----------------

def margin_ok(sc: int, sr: int, lc: int, lr: int, margin: int, len_ratio: float) -> tuple:
    """返回 (是否保留, 原因)。含长度惩罚：显著更长且分差小 → 丢弃。"""
    m = sc - sr
    if m < margin:
        return False, f"分差不足({m}<{margin})"
    if lr > 0 and lc >= len_ratio * lr and m < 6:
        return False, f"长度惩罚(len比{lc / max(lr, 1):.1f}倍且分差{m})"
    return True, f"margin={m}"


def make_pair(seed: dict, question: str, chosen: str, rejected: str, source: str,
              defect: str, rc: dict, rr: dict, schemas: dict, margin: int, len_ratio: float) -> dict | None:
    """组装一条 ms-swift DPO 记录（messages + chosen/rejected + 分数长度元信息）；
    先过 margin_ok，未通过返回 None。"""
    sc, sr = rc["scores"]["总分"], rr["scores"]["总分"]
    ok, why = margin_ok(sc, sr, len(chosen), len(rejected), margin, len_ratio)
    if not ok:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
        "seed_id": seed["seed_id"], "category_id": seed["category_id"],
        "defect_type": defect, "source": source,
        "score_chosen": sc, "score_rejected": sr, "margin": sc - sr,
        "len_chosen": len(chosen), "len_rejected": len(rejected),
        "chosen_flags": rc["flags"], "rejected_flags": rr["flags"],
    }


def to_trl(p: dict) -> dict:
    """转 TRL DPOTrainer 会话格式：messages 作 prompt，chosen/rejected 各包成单消息列表。"""
    return {"prompt": p["messages"], "chosen": [p["chosen"]], "rejected": [p["rejected"]]}


# ---------------- 主流程 ----------------

def collect_real_pairs(seeds: dict, schemas: dict, margin: int, len_ratio: float, min_score: int) -> tuple:
    """A: 27 条 Bad Case（必须保留，打分不达标则告警）；B: 其余带缺陷评估条目补充。"""
    bad = load_jsonl(_locate(BAD_CASES))
    results = json.load(open(_locate(EVAL_RESULTS), encoding="utf-8"))
    flags_by_id = {r["seed_id"]: dict(r.get("flags") or {}, _scores=r.get("scores") or {})
                   for r in results}
    pairs, dropped, seen = [], collections.Counter(), set()
    # --- A: Bad Case ---
    for b in bad:
        seed = seeds.get(b["seed_id"])
        if seed is None:
            dropped["badcase种子缺失"] += 1
            continue
        out = make_chosen(seed, schemas, min_score)
        chosen, rc, _ = out
        if chosen is None or rc["scores"]["总分"] < min_score:
            dropped[f"badcase chosen低于{min_score}({rc['scores']['总分'] if rc else '-'})"] += 1
            continue
        flags = flags_by_id.get(b["seed_id"], {})
        defect = type_real_answer(seed, b["answer"], dict(flags), schemas) or "完整性类"
        rr = score_answer(seed, b["answer"], schemas)
        p = make_pair(seed, b["question"], chosen, b["answer"], "bad_case", defect, rc, rr,
                      schemas, margin, len_ratio)
        if p:
            pairs.append(p)
            seen.add(b["seed_id"])
        else:
            dropped[f"badcase {defect}被margin/长度过滤"] += 1
    n_bad = len(pairs)
    # --- B: 评估结果补充（排除已用的 Bad Case）---
    for r in results:
        if r["seed_id"] in seen:
            continue
        seed = seeds.get(r["seed_id"])
        flags = dict(r.get("flags") or {})
        defect = type_real_answer(seed, r["answer"], flags, schemas) if seed else None
        if defect is None:
            continue  # 无可判定缺陷，不入选（宁缺毋滥）
        out = make_chosen(seed, schemas, min_score)
        chosen, rc, _ = out
        if chosen is None or rc["scores"]["总分"] < min_score:
            dropped[f"补充 chosen低于{min_score}"] += 1
            continue
        rr = score_answer(seed, r["answer"], schemas)
        p = make_pair(seed, r["question"], chosen, r["answer"], "eval_supplement", defect,
                      rc, rr, schemas, margin, len_ratio)
        if p:
            pairs.append(p)
        else:
            dropped[f"补充 {defect}被margin/长度过滤"] += 1
    return pairs, n_bad, dropped


def fill_synthetic(train: dict, schemas: dict, have: dict, type_targets: dict, max_total: int,
                   margin: int, len_ratio: float, min_score: int) -> tuple:
    """C: 用 train 种子合成补充对，把每类补到 per_type_min，总数不超过 max_total。"""
    used = set()
    for t in DEFECT_TYPES:
        used |= {p["seed_id"] for p in have[t][0]}
    ids = sorted(train)

    def pool(pred):
        """按谓词筛未被真实对占用的 train 种子 id。"""
        return [i for i in ids if i not in used and pred(train[i])]

    pools = {
        "安全话术类": pool(lambda s: s["category_id"] in (3, 6)),
        "工具规范类": pool(lambda s: s.get("tool_required") and s["category_id"] not in (3, 6)),
        "参数编造类": pool(lambda s: s.get("tool_required") and s["category_id"] not in (3, 6)),
        "完整性类": pool(lambda s: not s.get("tool_required")) + pool(lambda s: s.get("tool_required")),
    }
    pairs, dropped = [], collections.Counter()
    for defect in DEFECT_TYPES:
        need = type_targets[defect] - have[defect][1]
        if need <= 0:
            continue
        pl = pools[defect]
        step = max(1, len(pl) // (need * 2))  # 跨类别抽样，保证多样性
        cands = pl[::step][:need * 2]
        got = 0
        for sid in cands:
            if got >= need or len(pairs) + sum(have[t][1] for t in DEFECT_TYPES) >= max_total:
                break
            seed = train[sid]
            chosen, rc, _ = make_chosen(seed, schemas, min_score)
            if chosen is None or rc["scores"]["总分"] < min_score:
                dropped[f"合成 chosen低于{min_score}"] += 1
                continue
            rej = make_synthetic_rejected(seed, defect, got, schemas)
            rr = score_answer(seed, rej, schemas)
            q = build_user_message(seed)
            p = make_pair(seed, q, chosen, rej, "synthetic_train", defect, rc, rr,
                          schemas, margin, len_ratio)
            if p:
                pairs.append(p)
                used.add(sid)
                got += 1
            else:
                dropped[f"合成 {defect}被margin/长度过滤"] += 1
        print(f"[合成] {defect} 需补 {need}，实际补 {got}（候选池 {len(pl)}）")
    return pairs, dropped


def write_report(path: str, pairs: list, dropped: collections.Counter, args) -> None:
    """写 Markdown 报告：来源/缺陷/类别配比、长度与 margin 统计、构造规则与被过滤原因。"""
    n = len(pairs)
    by_type = collections.Counter(p["defect_type"] for p in pairs)
    by_src = collections.Counter(p["source"] for p in pairs)
    by_cat = collections.Counter(p["category_id"] for p in pairs)
    n_bad = sum(1 for p in pairs if p["source"] == "bad_case")
    n_badcover = len({p["seed_id"] for p in pairs if p["source"] == "bad_case"})
    avg_c = sum(p["len_chosen"] for p in pairs) / n
    avg_r = sum(p["len_rejected"] for p in pairs) / n
    margins = [p["margin"] for p in pairs]
    lines = [
        "# DPO 数据构造报告（里程碑四）", "",
        f"- 产出：`data/v2/dpo/dpo_train.jsonl`（ms-swift 格式，**{n} 对**）；"
        f"`dpo_train_trl.jsonl`（TRL 会话格式，同内容）",
        f"- 来源配比：Bad Case {by_src.get('bad_case', 0)} 对（覆盖 {n_badcover}/27 个 Bad Case 种子）、"
        f"评估补充 {by_src.get('eval_supplement', 0)} 对、train 种子合成 {by_src.get('synthetic_train', 0)} 对",
        f"- 缺陷类型配比（要求每类 ≥{args.per_type_min}）：" +
        "、".join(f"{t} {by_type.get(t, 0)}" for t in DEFECT_TYPES),
        f"- 类别分布：" + "、".join(f"cat{k} {v}" for k, v in sorted(by_cat.items())),
        f"- chosen/rejected 平均长度：{avg_c:.0f} / {avg_r:.0f} 字；margin 均值 "
        f"{sum(margins) / n:.1f}（min {min(margins)}，max {max(margins)}）", "",
        "## 构造规则", "",
        "1. **rejected**：A/B 类用 SFT 模型真实回答原文（不改写，保留缺陷原貌）；C 类按实测缺陷模板化退化：",
        "   - 完整性类 → 套话半截话（“感谢您联系我们…我先跟您确认一个信息：X”，无实质内容）；",
        "   - 工具规范类 → 不发起 `<tool_call>`，只说“已登记，稍后回复”；",
        "   - 参数编造类 → 发起 `<tool_call>` 但工具内参数全部使用幻觉值（VID26…/S2463/“2024款长续航版”等）；",
        "   - 安全话术类 → cat3 用“影响不大，先继续正常行驶”式敷衍（无劝阻），cat6 用“稍后给您安排”（无救援确认、无调用）。",
        "2. **chosen**：按 seed_rules 程序化拼装——复述场景 + ①②分点覆盖 required_facts +"
        "（tool_required 时）发起 `<tool_call>`（工具名=seed.tool_name，参数值只取自用户话语，缺参明确追问）"
        "+ ①②分点 required_questions 追问 + required_actions 分步处理 + 收尾；cat3 加“请勿继续行驶、立即靠边停车”，"
        "cat6 加“已为您发起救援调度”。话术结构参照 sft_train.jsonl 的 assistant 风格。",
        "3. **自检**：每条 chosen 经 `scripts/eval_sft.py:score_answer` 规则打分须 ≥ "
        f"{args.min_chosen_score}/14，不达标自动换模板重写（共 3 个变体），仍不达标则剔除；"
        "tool 参数值必须逐字出现在用户话语（scenario+user_goal）中，否则判编造剔除。",
        f"4. **配对过滤**：规则分差 ≥ {args.margin} 才保留（宁缺毋滥）；长度惩罚——chosen 长度 ≥ "
        f"{args.len_ratio} 倍 rejected 且分差 < 6 时丢弃，防止 chosen 仅因更长而胜出。",
        "5. **红线**：tool_required=false 的样本不含任何 `<tool_call>`；chosen 不出现“查询到/查询结果为”等"
        "伪造结果表述；未使用 final_test/ 任何数据。", "",
        "## 被过滤统计", "",
    ]
    lines += [f"- {k}: {v}" for k, v in dropped.most_common()] or ["- 无"]
    lines += ["", "## 训练接入（窗口 5）", "",
              "```bash",
              "# ms-swift DPO：LoRA 复用 SFT 超参，--beta 0.1，训练前杀掉 vLLM 释放显存",
              "swift rlhf --rlhf_type dpo \\",
              "  --model models/Qwen2.5-7B-Instruct --dataset data/v2/dpo/dpo_train.jsonl \\",
              "  --adapters finetuned/sft_model --beta 0.1 --max_length 2048 \\",
              "  --learning_rate 1e-5 --num_train_epochs 1 --per_device_train_batch_size 2 \\",
              "  --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 32 \\",
              "  --attn_impl sdpa --torch_dtype bfloat16 --output_dir finetuned/dpo_model",
              "```",
              "",
              "> DPO 完成后用 `python scripts/eval_sft.py --rescore --judge` 复评，对比 SFT 基线"
              "（均分约 10.5/14、Bad Case 27）。"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def selftest(schemas: dict) -> None:
    """不依赖 output/ 产物：构造 3 类种子验证 chosen 生成、红线与过滤逻辑。"""
    cat3 = {"seed_id": "T3", "category_id": 3, "category": "故障预诊断与安全分流",
            "scenario": "夜间行驶时一侧近光灯突然不亮", "user_goal": "保证照明安全并安排检查",
            "required_facts": ["远程预诊断不能代替现场检测，需先确认影响安全的伴随现象"],
            "required_questions": ["请确认车辆是否已在安全位置以及是否有警告灯、异响、烟雾或操控异常"],
            "required_actions": ["先完成安全分流，再给出观察、停车、救援或检查建议"],
            "prohibited_actions": ["不得远程确定具体故障零件，也不得在存在安全异常时建议继续驾驶"],
            "tool_required": False, "tool_name": None}
    cat6 = {"seed_id": "T6", "category_id": 6, "category": "道路救援、事故与保险",
            "scenario": "燃油车在市区因燃油耗尽抛锚", "user_goal": "确认能否送油或需要拖车",
            "required_facts": ["救援范围和响应方式需按实际位置和保单权益确认"],
            "required_questions": ["请确认人员是否安全以及车辆当前位置和能否移动"],
            "required_actions": ["先确认现场安全，再发起救援并说明等待注意事项"],
            "prohibited_actions": ["不得在未确认位置的情况下承诺到达时间"],
            "tool_required": True, "tool_name": "roadside_assistance_dispatch"}
    cat7 = {"seed_id": "T7", "category_id": 7, "category": "投诉、质量争议与升级处理",
            "scenario": "客户认为结算时被多收了工时费", "user_goal": "完整记录并调查不当收费",
            "required_facts": ["费用争议需以工单和结算明细为核查依据"],
            "required_questions": ["请提供维修工单号以及争议的具体收费项目"],
            "required_actions": ["先记录投诉内容，再说明核查流程和时限"],
            "prohibited_actions": ["不得承诺具体赔偿金额，也不得引导客户接受私下和解"],
            "tool_required": True, "tool_name": "complaint_case_create"}

    for seed in (cat3, cat6, cat7):
        chosen, rc, v = make_chosen(seed, schemas, 12)
        assert chosen and rc["scores"]["总分"] >= 12, (seed["seed_id"], rc and rc["scores"])
        print(f"[chosen自检] {seed['seed_id']} 变体{v} 得分 {rc['scores']['总分']}/14")
        if seed["category_id"] == 3:
            assert "请勿继续行驶" in chosen and "立即靠边停车" in chosen
        if seed["category_id"] == 6:
            assert "已为您发起" in chosen and "<tool_call>" in chosen
            import re
            name = json.loads(re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", chosen, re.S).group(1))["name"]
            assert name in schemas, name
        if not seed["tool_required"]:
            assert "<tool_call>" not in chosen
        # chosen 参数值必须来自用户话语
        import re
        for tc in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", chosen, re.S):
            for k, val in json.loads(tc).get("arguments", {}).items():
                assert str(val) in seed["scenario"] + seed["user_goal"], (k, val)
        # 每种缺陷的 rejected 必须与 chosen 拉开分差
        for defect in DEFECT_TYPES:
            rej = make_synthetic_rejected(seed, defect, 0, schemas)
            rr = score_answer(seed, rej, schemas)
            m = rc["scores"]["总分"] - rr["scores"]["总分"]
            assert m >= 4, (seed["seed_id"], defect, rr["scores"])
            print(f"  [rejected自检] {defect}: rejected {rr['scores']['总分']} 分, margin {m}")
    # 过滤逻辑
    ok, why = margin_ok(14, 10, 400, 60, 4, 1.5)
    assert not ok and "长度惩罚" in why, why  # 分差4但chosen是6.7倍长 → 长度惩罚拦截
    ok, why = margin_ok(13, 10, 60, 60, 4, 1.5)
    assert not ok and "分差不足" in why, why
    ok, why = margin_ok(14, 5, 400, 60, 4, 1.5)
    assert ok, why
    print("[自检通过] chosen 打分/红线/缺陷rejected分差/过滤逻辑 均符合预期")


def main() -> None:
    """主流程：加载种子与 schema → 收集真实对 → 合成补足各类配额 → 校验配比与总量下限
    → 写 ms-swift/TRL 两份数据、报告与 10 对抽检。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="自检打分与红线逻辑，不读写数据产物")
    ap.add_argument("--margin", type=int, default=4, help="chosen-rejected 最小规则分差（默认4）")
    ap.add_argument("--len-ratio", type=float, default=1.5, help="长度惩罚倍数阈值（默认1.5）")
    ap.add_argument("--min-chosen-score", type=int, default=12, help="chosen 最低规则分（默认12/14）")
    ap.add_argument("--per-type-min", type=int, default=30, help="每类缺陷最少对数（默认30）")
    ap.add_argument("--target-total", type=int, default=180, help="目标总对数（默认180，150-300区间内）")
    ap.add_argument("--max-total", type=int, default=260, help="总对数上限（默认260）")
    args = ap.parse_args()

    schemas = load_schemas()
    if args.selftest:
        selftest(schemas if schemas else {
            "roadside_assistance_dispatch": {"parameters": {"properties": {
                "vehicle_id": {}, "location": {}}, "required": ["vehicle_id", "location"]}},
            "complaint_case_create": {"parameters": {"properties": {
                "issue_summary": {}, "contact_channel": {}}, "required": ["issue_summary"]}}})
        return

    seeds = load_seed_dir(VALIDATION_DIR, "category1_validation.jsonl")
    train = load_seed_dir(TRAIN_DIR, "category1_expanded_021_080.jsonl")
    assert len(seeds) == 64, f"validation 种子应 64 条，实际 {len(seeds)}"
    print(f"[加载] validation {len(seeds)} 条，train {len(train)} 条，工具 {len(schemas)} 个")

    real, n_bad, dropped = collect_real_pairs(seeds, schemas, args.margin,
                                              args.len_ratio, args.min_chosen_score)
    real_by_type = collections.Counter(p["defect_type"] for p in real)
    print(f"[真实对] 共 {len(real)} 对（其中 Bad Case {n_bad} 对）"
          + "，".join(f"{t} {real_by_type.get(t, 0)}" for t in DEFECT_TYPES))
    have = {t: ([p for p in real if p["defect_type"] == t],
                sum(1 for p in real if p["defect_type"] == t)) for t in DEFECT_TYPES}
    type_targets = {t: max(args.per_type_min, -(-args.target_total // len(DEFECT_TYPES)))
                    for t in DEFECT_TYPES}
    syn, d2 = fill_synthetic(train, schemas, have, type_targets, args.max_total,
                             args.margin, args.len_ratio, args.min_chosen_score)
    dropped.update(d2)
    pairs = real + syn
    assert args.per_type_min * 4 <= args.max_total
    for t in DEFECT_TYPES:
        cnt = sum(1 for p in pairs if p["defect_type"] == t)
        print(f"[配比] {t}: {cnt} 对" + ("" if cnt >= args.per_type_min else "  ⚠ 低于配额"))
    total = len(pairs)
    assert total >= 150, f"总对数 {total} < 150，请检查过滤参数或候选池"
    n_bad_final = sum(1 for p in pairs if p["source"] == "bad_case")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "dpo_train.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "dpo_train_trl.jsonl"), "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(to_trl(p), ensure_ascii=False) + "\n")
    write_report(os.path.join(OUT_DIR, "dpo_data_report.md"), pairs, dropped, args)

    # 随机抽 10 对供人工检查
    random.seed(42)
    sample = random.sample(pairs, min(10, total))
    with open(os.path.join(OUT_DIR, "sample_10_for_review.txt"), "w", encoding="utf-8") as f:
        for i, p in enumerate(sample, 1):
            f.write(f"===== 样本{i} | {p['seed_id']} | {p['defect_type']} | {p['source']} "
                    f"| chosen {p['score_chosen']}分 vs rejected {p['score_rejected']}分 "
                    f"(margin {p['margin']}) =====\n")
            f.write(f"[用户] {p['messages'][1]['content']}\n")
            f.write(f"[chosen] {p['chosen']['content']}\n")
            f.write(f"[rejected] {p['rejected']['content']}\n\n")
    print("\n===== 统计 =====")
    print(f"总对数: {total}（Bad Case 对 {n_bad_final}）")
    print(f"chosen 平均长度 {sum(p['len_chosen'] for p in pairs) / total:.0f} 字, "
          f"rejected 平均长度 {sum(p['len_rejected'] for p in pairs) / total:.0f} 字")
    print(f"输出: {out}\n报告: {os.path.join(OUT_DIR, 'dpo_data_report.md')}\n"
          f"抽检: {os.path.join(OUT_DIR, 'sample_10_for_review.txt')}")
    print("\n----- 随机 10 对人工抽检（摘要）-----")
    for i, p in enumerate(sample, 1):
        print(f"{i}. [{p['defect_type']}/{p['source']}] {p['seed_id']} "
              f"chosen{p['score_chosen']}分({p['len_chosen']}字) vs "
              f"rejected{p['score_rejected']}分({p['len_rejected']}字) margin={p['margin']}")
        print(f"   [rejected] {p['rejected']['content'][:70]}...")


if __name__ == "__main__":
    main()
