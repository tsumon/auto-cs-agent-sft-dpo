# -*- coding: utf-8 -*-
"""
【流水线 04/16】里程碑二 · SFT 训练（LoRA）
运行位置：魔搭实例 GPU
输入：output/sft_train.jsonl + output/sft_train_supplement.jsonl（合计约 2412 条）
输出：finetuned/sft_model_r2/（LoRA Adapter）
前置步骤：01、03　｜　后续步骤：05 冒烟

SFT 重训脚本（基于多样化补充数据）。

数据：原 1912 条 + 补充 ~500 条 = ~2400 条
超参变更（相对于原 SFT）：
  - lr: 1e-4 → 5e-5（更多数据 + 降 lr 防过拟合）
  - epochs: 3 → 2（更多数据后 2 epoch 足够）
  - 其余不变：LoRA r=32/alpha=64 全 proj，bf16+sdpa，max_len 2048，eff batch 32

用法：
    python scripts/train_sft_r2.py
"""
import json
import os
import sys
import tempfile

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
ORIG_TRAIN = "output/sft_train.jsonl"
SUPP_TRAIN = "output/sft_train_supplement.jsonl"
EVAL_FILE = "output/sft_validation.jsonl"
OUTPUT_DIR = "finetuned/sft_model_r2"
LOG_DIR = "output/logs"
MAX_LEN = 2048


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[环境] torch={torch.__version__}, cuda={torch.cuda.is_available()}")

    # 合并两个训练文件
    merged_lines = []
    for path in [ORIG_TRAIN, SUPP_TRAIN]:
        if not os.path.exists(path):
            sys.exit(f"[错误] 找不到: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                merged_lines.append(json.dumps({"messages": s["messages"]}, ensure_ascii=False))
    print(f"[数据] 合并: {len(merged_lines)} 条")

    # 写临时文件供 datasets 加载
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for l in merged_lines:
        tmp.write(l + "\n")
    tmp.close()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    train_ds = load_dataset("json", data_files={"train": tmp.name})
    train_ds = train_ds["train"].remove_columns(
        [c for c in train_ds["train"].column_names if c != "messages"]
    )
    eval_ds = load_dataset("json", data_files={"validation": EVAL_FILE})
    eval_ds = eval_ds["validation"].remove_columns(
        [c for c in eval_ds["validation"].column_names if c != "messages"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LEN,
        packing=False,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,   # eff batch = 32
        per_device_eval_batch_size=8,
        learning_rate=5e-5,              # 从 1e-4 降到 5e-5
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=LOG_DIR,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    print("[训练] 开始 SFT 重训 ...")
    train_result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_loss": train_result.metrics,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }, f, ensure_ascii=False, indent=2, default=str)

    os.unlink(tmp.name)
    print(f"[完成] 模型已保存到 {OUTPUT_DIR}")
    print(train_result.metrics)


if __name__ == "__main__":
    main()
