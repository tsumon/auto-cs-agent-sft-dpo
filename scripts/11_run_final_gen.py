# -*- coding: utf-8 -*-
"""
【流水线 11/16】里程碑六 · final_test 生成阶段（SFT 基线 vs DPO，各 160 条）
运行位置：魔搭实例 GPU
输入：data/v2/seeds/final_test/（160 条，仅此阶段使用）
输出：output/final_test_questions.jsonl、final_sft_answers.jsonl、final_dpo_answers.jsonl
前置步骤：09　｜　后续步骤：12 评分

final_test 最终评估——生成阶段：对全部 160 条 final_test 种子，用统一问题集、相同采样参数
（temperature=0.7, top_p=0.9, max_new_tokens=1024）一次跑完两个模型：
    SFT 基线 = finetuned/sft_model_merged（SFT R2 合并权重，无 Adapter）
    DPO 模型 = finetuned/sft_model_merged + finetuned/dpo_model_r2（LoRA）

产物：
    output/final_test_questions.jsonl   统一测试问题集（冻结存档，供审计）
    output/final_sft_answers.jsonl      SFT 基线回答
    output/final_dpo_answers.jsonl      DPO 模型回答

用法（实例项目根目录 /mnt/workspace）:
    python scripts/run_final_gen.py --selftest   # 自检测试集构造（不加载模型、不需要 GPU）
    python scripts/run_final_gen.py              # 完整生成（需 GPU；支持断点续跑）

评分阶段运行 scripts/run_final_eval.py。
注意：不要用 eval_sft.py 跑 final_test——它固定加载 validation 集，以此保证
final_test 只在本阶段通过本脚本使用，且从未参与训练与调参。
"""
import argparse
import glob
import json
import os

# 复用里程碑三的同一套 system prompt 与问题构造，保证口径可比
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _pipeline  # 注册 eval_sft / build_dpo_data / run_final_eval 别名（数字前缀无法直接 import）
from eval_sft import SYSTEM_PROMPT, build_user_message, _locate

BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
SFT_MERGED_DIR = os.environ.get("SFT_MERGED_DIR", "finetuned/sft_model_merged")
DPO_ADAPTER_DIR = os.environ.get("DPO_ADAPTER_DIR", "finetuned/dpo_model_r2")
SEED_DIR = "data/v2/seeds/final_test"
OUTPUT_DIR = "output"
QUESTIONS_FILE = os.path.join(OUTPUT_DIR, "final_test_questions.jsonl")
SFT_ANSWERS_FILE = os.path.join(OUTPUT_DIR, "final_sft_answers.jsonl")
DPO_ANSWERS_FILE = os.path.join(OUTPUT_DIR, "final_dpo_answers.jsonl")

# 任务规定的统一采样参数（两个模型完全一致）；显式关闭 top_k / repetition_penalty，
# 避免基座 generation_config.json 的默认值（top_k=20, rep=1.05）混入
GEN_SEED = 20260831
GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 1024,
    "do_sample": True,
}


def load_final_seeds() -> list:
    seed_dir = _locate(SEED_DIR, marker="category1_final_test.jsonl")
    if seed_dir != SEED_DIR:
        print(f"[提示] 数据目录定位为: {seed_dir}")
    seeds = []
    for fp in sorted(glob.glob(os.path.join(seed_dir, "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            seeds += [json.loads(l) for l in f if l.strip()]
    assert len(seeds) == 160, \
        f"预期 160 条 final_test 种子，实际 {len(seeds)}（查找目录: {seed_dir}，当前 cwd: {os.getcwd()}）"
    by_cat = {}
    for s in seeds:
        assert s["split"] == "final_test", f"{s['seed_id']} split 异常: {s['split']}"
        by_cat.setdefault(s["category_id"], []).append(s)
    assert len(by_cat) == 8 and all(len(v) == 20 for v in by_cat.values()), \
        f"8 类 × 20 条校验失败: {[{k: len(v) for k, v in by_cat.items()}]}"
    seeds.sort(key=lambda s: s["seed_id"])  # 固定顺序，保证可复现
    return seeds


def write_questions(seeds: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps({
                "seed_id": s["seed_id"], "category_id": s["category_id"],
                "category": s["category"], "subcategory": s["subcategory"],
                "question": build_user_message(s),
                "required_facts": s["required_facts"],
                "required_questions": s["required_questions"],
                "required_actions": s["required_actions"],
                "prohibited_actions": s["prohibited_actions"],
                "tool_required": s["tool_required"], "tool_name": s["tool_name"],
            }, ensure_ascii=False) + "\n")
    print(f"[问题集] 已冻结 {len(seeds)} 条统一测试问题 -> {QUESTIONS_FILE}")


def load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                done.add(json.loads(l)["seed_id"])
    return done


def generate(model, tokenizer, seeds: list, out_path: str, model_tag: str) -> None:
    """逐条采样生成并增量落盘；已有结果跳过（断点续跑）。"""
    import torch
    done = load_done(out_path)
    if done:
        print(f"[{model_tag}] 检测到已有 {len(done)} 条结果，跳过继续生成")
    with open(out_path, "a", encoding="utf-8") as f:
        for i, seed in enumerate(seeds):
            if seed["seed_id"] in done:
                continue
            torch.manual_seed(GEN_SEED + i)  # 每条固定种子，断点续跑结果一致
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(seed)},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, **GEN_PARAMS,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rec = {
                "seed_id": seed["seed_id"], "category_id": seed["category_id"],
                "category": seed["category"], "subcategory": seed["subcategory"],
                "question": build_user_message(seed), "answer": ans,
                "model": model_tag, "gen_params": GEN_PARAMS, "gen_seed": GEN_SEED + i,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{model_tag} {i + 1}/160] {seed['seed_id']} 生成完毕（{len(ans)} 字）", flush=True)


def selftest() -> None:
    seeds = load_final_seeds()
    write_questions(seeds)
    n_tool = sum(1 for s in seeds if s["tool_required"])
    print(f"[自检通过] 160 条 = 8 类 × 20 条；tool_required={n_tool} 条")
    print(f"[自检通过] 采样参数: {GEN_PARAMS}，随机种子基值 {GEN_SEED}")
    print(f"[自检通过] 示例问题: {build_user_message(seeds[0])}")
    print(f"[模型] SFT 基线: {SFT_MERGED_DIR}（无 Adapter）")
    print(f"[模型] DPO: {SFT_MERGED_DIR} + {DPO_ADAPTER_DIR}")
    for p in (SFT_MERGED_DIR, DPO_ADAPTER_DIR):
        if not os.path.isdir(p):
            print(f"[警告] 目录不存在: {p}（selftest 不加载模型，正式运行前需确认）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="只校验测试集构造与参数，不加载模型")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    seeds = load_final_seeds()
    write_questions(seeds)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for p in (SFT_MERGED_DIR, DPO_ADAPTER_DIR):
        if not os.path.isdir(p):
            raise SystemExit(f"[错误] 找不到模型目录: {p}")
    print(f"[加载] SFT 合并权重（同时作为 SFT 基线）: {SFT_MERGED_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_DIR, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    # ---- 模型一：SFT 基线 ----
    print("\n========== 第一阶段：SFT 基线生成 ==========")
    generate(model, tokenizer, seeds, SFT_ANSWERS_FILE, "SFT")

    # ---- 模型二：DPO（同一基座挂 DPO Adapter，一次跑完）----
    print(f"\n[加载] DPO LoRA Adapter: {DPO_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR).eval()
    print("========== 第二阶段：DPO 模型生成 ==========")
    generate(model, tokenizer, seeds, DPO_ANSWERS_FILE, "DPO")

    print(f"\n[完成] 两个模型 × 160 条全部生成。")
    print(f"输出: {QUESTIONS_FILE}, {SFT_ANSWERS_FILE}, {DPO_ANSWERS_FILE}")
    print("下一步: export MODELSCOPE_API_KEY=... 后运行 python scripts/run_final_eval.py --judge")


if __name__ == "__main__":
    main()
