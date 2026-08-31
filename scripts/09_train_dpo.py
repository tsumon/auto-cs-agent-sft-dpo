# -*- coding: utf-8 -*-
"""
【流水线 09/16】里程碑五 · DPO 训练（TRL DPOTrainer + IPO loss）
运行位置：魔搭实例 GPU
输入：data/v2/dpo_r2/dpo_train_trl.jsonl、finetuned/sft_model_r2/
输出：finetuned/sft_model_merged/（合并权重）、finetuned/dpo_model_r2/（DPO Adapter）
前置步骤：08　｜　后续步骤：10 冒烟

二轮 DPO 训练脚本（相对于 round 1 的调整）：

超参变更（针对 round 1 过拟合 + 半截话恶化）：
  - lr: 1e-5 → 3e-6（更保守，减少过拟合）
  - beta: 0.1 → 0.2（更强 KL 约束，拉住 SFT 行为不漂移）
  - loss_type: sigmoid → ipo（IPO 在小数据上比 DPO sigmoid 更稳定）
  - num_train_epochs: 2 → 1（防过拟合，数据量增大后 1 epoch 足够）
  - per_device_batch: 2 → 4（更大 batch 稳定梯度）
  - 新增 label_smoothing=0.1（防 margins 爆炸）
  - 数据路径 → data/v2/dpo_r2/（500 对多样化数据）
  - 输出 → finetuned/dpo_model_r2/

其余：与 train_dpo.py 共享 SFT 合并权重逻辑（MERGED_DIR 可复用，不会重新合并）。

用法：
    python scripts/train_dpo_r2.py
"""
import inspect
import json
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
SFT_ADAPTER_DIR = os.environ.get("SFT_ADAPTER_DIR", "finetuned/sft_model")
MERGED_DIR = "finetuned/sft_model_merged"  # round 1 已生成，可复用
TRAIN_FILE = "data/v2/dpo_r2/dpo_train_trl.jsonl"
OUTPUT_DIR = "finetuned/dpo_model_r2"
LOG_DIR = "output/logs"

MAX_LENGTH = 2048


def _supported_kwargs(cls, kwargs):
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _warn_dropped_kwargs(cls, kwargs):
    params = inspect.signature(cls.__init__).parameters
    for key in kwargs:
        if key not in params:
            print(f"[兼容] 当前 TRL 的 {cls.__name__} 不支持参数 {key}，已省略")


def _make_dpo_config(**kwargs):
    _warn_dropped_kwargs(DPOConfig, kwargs)
    return DPOConfig(**_supported_kwargs(DPOConfig, kwargs))


def build_merged_sft_model(tokenizer) -> str:
    if os.path.exists(os.path.join(MERGED_DIR, "config.json")):
        print(f"[合并] 已存在合并权重，跳过: {MERGED_DIR}")
        return MERGED_DIR
    if not os.path.isdir(SFT_ADAPTER_DIR):
        sys.exit(f"[错误] 找不到 SFT Adapter: {SFT_ADAPTER_DIR}，请先完成 SFT 并上传")
    print(f"[合并] 加载基座 {BASE_MODEL_DIR} + SFT Adapter {SFT_ADAPTER_DIR} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[合并] 完成: {MERGED_DIR}")
    return MERGED_DIR


def check_data() -> None:
    if not os.path.exists(TRAIN_FILE):
        sys.exit(f"[错误] 找不到训练数据: {TRAIN_FILE}，请先运行 build_dpo_data_r2.py")
    n = 0
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            assert set(s.keys()) >= {"prompt", "chosen", "rejected"}, f"字段缺失: {list(s.keys())}"
            assert s["prompt"][0]["role"] == "system"
            assert s["chosen"][0]["role"] == "assistant"
            assert s["rejected"][0]["role"] == "assistant"
            n += 1
    print(f"[数据] {TRAIN_FILE}: {n} 对")
    if n < 300:
        print(f"[警告] 预期 400+ 对，实际 {n} 对")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[环境] torch={torch.__version__}, cuda可用={torch.cuda.is_available()}, "
          f"设备={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    check_data()
    merged_dir = build_merged_sft_model(tokenizer)

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})
    dataset = dataset.remove_columns(
        [c for c in dataset["train"].column_names if c not in ("prompt", "chosen", "rejected")]
    )

    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 二轮超参：更低 lr、更强 KL、IPO loss、label_smoothing
    config_kwargs = dict(
        output_dir=OUTPUT_DIR,

        # DPO 核心超参
        beta=0.2,                          # 比 round 1(0.1) 更强 KL 约束
        max_length=MAX_LENGTH,
        loss_type="ipo",                   # IPO 比 sigmoid 更稳定（小数据防过拟合）

        # 训练轮次与 batch
        num_train_epochs=1,                # 数据量增大，1 epoch 防过拟合
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,     # 有效 batch = 4*4 = 16

        # 优化器
        learning_rate=3e-6,                # 比 round 1(1e-5) 低 3 倍
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # 精度与显存
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",

        # 防过拟合
        label_smoothing=0.1,               # 防 margins 爆炸

        # 日志与保存
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["tensorboard"],
        logging_dir=LOG_DIR,
        seed=42,
    )
    args = _make_dpo_config(**config_kwargs)

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=peft_config,
        ref_model=None,
    )
    trainer.model.print_trainable_parameters()

    print("[训练] 开始 DPO 二轮训练 ...")
    train_result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    log_history = [
        {k: v for k, v in entry.items()
         if k in ("step", "epoch", "loss", "rewards/chosen", "rewards/rejected",
                  "rewards/margins", "rewards/accuracies")}
        for entry in trainer.state.log_history if "loss" in entry
    ]
    with open(os.path.join(OUTPUT_DIR, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "log_history": log_history},
                  f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(LOG_DIR, "train_dpo_r2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2, default=str)

    print(f"[完成] DPO 二轮 Adapter 与指标已保存到 {OUTPUT_DIR}")
    print(metrics)
    print("[提示] 推理: 基座 → finetuned/sft_model_merged → finetuned/dpo_model_r2")


if __name__ == "__main__":
    main()
