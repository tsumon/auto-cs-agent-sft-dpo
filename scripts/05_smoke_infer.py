# -*- coding: utf-8 -*-
"""
【流水线 05/16】里程碑二 · SFT 冒烟测试（5 题人工看质量）
运行位置：魔搭实例 GPU
输入：finetuned/sft_model_r2/
输出：5 个典型问题的回答打印到终端
前置步骤：04　｜　后续步骤：06 系统评估

SFT 冒烟测试：加载基座 + LoRA Adapter，对 5 个汽车客服典型问题生成回答。
用法（在实例项目根目录，训练完成后）:
    python scripts/smoke_infer.py
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
ADAPTER_DIR = "finetuned/sft_model"

QUESTIONS = [
    "我的手机蓝牙搜索不到车上的蓝牙，不知道怎么回事，能帮我看看吗？",        # 查蓝牙
    "我刚提了新车，首保的时间和里程要求是多少？错过了会有什么影响？",        # 首保政策
    "车启动后仪表盘发动机故障灯常亮，动力感觉有点弱，我还能继续开吗？",      # 故障报修
    "我想预约下周六做一次保养，需要带什么材料吗？",                          # 预约保养
    "我在高速上车没电抛锚了，人没事，请问道路救援怎么申请？",                # 道路救援
]

SYSTEM_PROMPT = (
    "你是汽车售后服务智能客服助手，请以专业、亲切、安全的口吻回答客户问题。"
    "涉及具体政策或数据时，如需查询请说明已发起查询或请客户提供信息，不得编造。"
)


def main() -> None:
    """冒烟推理入口：校验 ADAPTER_DIR 存在 → 加载基座（bf16 + sdpa，转 cuda，
    路径取环境变量 BASE_MODEL_DIR）并挂上 LoRA Adapter → 对 QUESTIONS 里 5 个问题
    套 chat template 贪心生成（max_new_tokens=512）→ 逐条打印问题与回答。
    """
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit(f"[错误] 找不到 Adapter 目录: {ADAPTER_DIR}，请先完成训练")

    print("[加载] 基座模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    print("[加载] LoRA Adapter ...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    for i, q in enumerate(QUESTIONS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n{'=' * 60}\n[问题 {i}] {q}\n{'-' * 60}\n{answer.strip()}")


if __name__ == "__main__":
    main()
