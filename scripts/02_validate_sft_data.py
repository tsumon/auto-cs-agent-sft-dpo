#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【流水线 02/16】里程碑一 · 校验 SFT 数据
运行位置：本机（无需 GPU）
输入：output/sft_train.jsonl、output/sft_validation.jsonl
输出：校验报告打印到终端（覆盖率/违规/去重）
前置步骤：01　｜　后续步骤：03 多样化补充

SFT 数据校验脚本（汽车售后服务智能客服）

校验项：
  1. JSON 合法性与消息结构（role 交替、首条为 user、末条为 assistant、内容非空）
  2. required_facts / required_questions / required_actions 覆盖检查（归一化子串匹配）
  3. prohibited_actions 违规扫描（剥离"不得/禁止"等前缀后子串匹配，命中即违规）
  4. 工具样本：tool_call JSON 可解析、name 与 tool_name 一致、必填参数齐全、
     不伪造查询结果（黑名单短语扫描）
  5. 全局去重、长度过滤、与 final_test 种子零重叠
用法：
  python scripts/validate_sft_data.py                 # 校验两个默认文件
  python scripts/validate_sft_data.py --show 10      # 随机抽 10 条打印
退出码：存在任何 FAIL 项时为 1。
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DIR = os.path.join(ROOT, "data", "v2", "seeds")
TOOL_SCHEMA_PATH = os.path.join(ROOT, "data", "v2", "tool_schemas.json")

FABRICATION_BLACKLIST = [
    "查询到您的", "查询结果显示", "查询结果为", "查询到您已",
    "系统显示您的", "已审核通过", "已成功为您办理", "处理成功",
    "返回结果为您", "查到您的账户",
]
PROHIBITED_PREFIX = re.compile(r"^(不得|不应|不可以|不能|禁止|严禁|避免)")


def normalize(s: str) -> str:
    """归一化：仅保留中英文与数字，用于稳健子串匹配。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", s or "")


def load_seeds_by_id():
    seeds = {}
    for split in ("train", "validation", "final_test"):
        for p in glob.glob(os.path.join(SEED_DIR, split, "*.jsonl")):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        seeds[d["seed_id"]] = d
    return seeds


def load_tool_schemas():
    with open(TOOL_SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {t["name"]: t for t in data["tools"]}


def assistant_text(rec) -> str:
    return "\n".join(m["content"] for m in rec["messages"] if m["role"] == "assistant")


def validate_record(rec, seeds, schemas, seen_hashes, issues):
    sid = rec.get("seed_id")
    errs, warns = [], []

    def err(msg):
        errs.append(msg)

    seed = seeds.get(sid)
    if seed is None:
        err(f"seed_id {sid} 在种子库中不存在")
        return errs, warns
    if rec.get("split") == "train" and sid.startswith(tuple(f"cat{i}" for i in range(1, 9))):
        pass  # seed_id 无 split 信息，靠来源文件保证；此处仅防 final_test 混入：
    if seed.get("split") == "final_test":
        err("使用了 final_test 种子（禁止）")

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        err("messages 为空或长度不足")
        return errs, warns
    roles = [m.get("role") for m in msgs]
    if roles[0] != "user":
        err("首条消息不是 user")
    if roles[-1] != "assistant":
        err("末条消息不是 assistant")
    for i, r in enumerate(roles):
        if r not in ("user", "assistant"):
            err(f"第{i}条 role 非法: {r}")
        if i > 0 and r == roles[i - 1]:
            err(f"第{i}条与上一条 role 相同（未交替）")
    for i, m in enumerate(msgs):
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            err(f"第{i}条 content 为空")
        elif m["role"] == "user" and len(c) > 600:
            err(f"第{i}条 user 内容超长 ({len(c)}>600)")
        elif m["role"] == "assistant" and not (20 <= len(c) <= 2500):
            err(f"第{i}条 assistant 长度越界 ({len(c)})")

    atext_norm = normalize(assistant_text(rec))
    atext_raw = assistant_text(rec)

    # 覆盖检查
    for fact in seed.get("required_facts") or []:
        if normalize(fact) not in atext_norm:
            err(f"未覆盖 required_fact: {fact}")
    for q in seed.get("required_questions") or []:
        if normalize(q) not in atext_norm:
            err(f"未覆盖 required_question: {q}")
    for a in seed.get("required_actions") or []:
        if normalize(a) not in atext_norm:
            err(f"未覆盖 required_action: {a}")

    # prohibited 扫描
    for p in seed.get("prohibited_actions") or []:
        core = PROHIBITED_PREFIX.sub("", p)
        if normalize(core) and normalize(core) in atext_norm:
            err(f"疑似违反 prohibited_action: {p}")

    # 工具样本检查
    if rec.get("sample_type") == "tool_call":
        content = "".join(m["content"] for m in msgs)
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S)
        if not m:
            err("tool_call 样本缺少 <tool_call> 块")
        else:
            try:
                tc = json.loads(m.group(1))
            except json.JSONDecodeError as e:
                err(f"tool_call JSON 不合法: {e}")
                tc = None
            if tc:
                if tc.get("name") != seed.get("tool_name"):
                    err(f"tool name {tc.get('name')} != 期望 {seed.get('tool_name')}")
                schema = schemas.get(seed.get("tool_name") or "")
                if schema:
                    required = schema.get("parameters", {}).get("required", [])
                    for k in required:
                        if k not in tc.get("arguments", {}):
                            err(f"tool_call 缺少必填参数: {k}")
        for kw in FABRICATION_BLACKLIST:
            if kw in atext_raw:
                err(f"疑似伪造查询结果（命中黑名单短语）: {kw}")

    # 去重
    h = hashlib.md5(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if h in seen_hashes:
        err("与其他样本完全重复")
    seen_hashes.add(h)

    return errs, warns


def load_records(path):
    records, bad_lines = [], 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"  [JSON 错误] {path}:{ln} -> {e}")
    return records, bad_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(ROOT, "output", "sft_train.jsonl"))
    parser.add_argument("--val", default=os.path.join(ROOT, "output", "sft_validation.jsonl"))
    parser.add_argument("--show", type=int, default=0, help="随机打印 N 条样本供人工检查")
    args = parser.parse_args()

    seeds = load_seeds_by_id()
    schemas = load_tool_schemas()
    seen_hashes = set()
    total_pass = total_fail = 0
    all_errs = []

    for name, path in (("train", args.train), ("validation", args.val)):
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            total_fail += 1
            continue
        records, bad_lines = load_records(path)
        if bad_lines:
            all_errs.append(f"{name}: {bad_lines} 行 JSON 解析失败")
        cat_cnt = Counter(r.get("category_id") for r in records)
        type_cnt = Counter(r.get("sample_type") for r in records)
        n_fail = 0
        for i, rec in enumerate(records):
            errs, _ = validate_record(rec, seeds, schemas, seen_hashes, None)
            if errs:
                n_fail += 1
                all_errs.extend(f"{name}[{i}] {rec.get('seed_id')}: {e}" for e in errs)
        n_pass = len(records) - n_fail
        total_pass += n_pass
        total_fail += n_fail
        print(f"--- {name}: {path}")
        print(f"    样本数={len(records)}  通过={n_pass}  失败={n_fail}")
        print(f"    类目分布={dict(sorted(cat_cnt.items()))}")
        print(f"    类型分布={dict(type_cnt)}")
        if len(set(cat_cnt)) < 1 or len(records) == 0:
            all_errs.append(f"{name}: 无有效样本")

    print("========================================")
    if all_errs:
        print(f"校验结果：❌ 存在 {total_fail} 条失败样本 / {len(all_errs)} 个问题")
        for e in all_errs[:30]:
            print("  -", e)
        if len(all_errs) > 30:
            print(f"  ...（其余 {len(all_errs) - 30} 条省略）")
        sys.exit(1)
    else:
        print(f"校验结果：✅ 全部通过（train+validation 共 {total_pass} 条）")

    if args.show > 0:
        rng = random.Random(20260829)
        pool = load_records(args.train)[0]
        for rec in rng.sample(pool, min(args.show, len(pool))):
            print("\n" + "=" * 60)
            print(f"[{rec['seed_id']}] {rec['category']} / {rec.get('subcategory')} | 类型={rec['sample_type']}")
            for m in rec["messages"]:
                print(f"  [{m['role']}] {m['content']}")


if __name__ == "__main__":
    main()
