# -*- coding: utf-8 -*-
"""
【流水线 10/16】里程碑五 · DPO 冒烟测试（5 个偏好敏感问题）
运行位置：魔搭实例 GPU
输入：finetuned/sft_model_merged/ + finetuned/dpo_model_r2/
输出：5 个偏好敏感问题的回答打印到终端
前置步骤：09　｜　后续步骤：11 final_test 生成

DPO 冒烟测试：加载 SFT 合并权重 + DPO LoRA Adapter，对 5 个"偏好敏感"问题生成回答。
重点覆盖 DPO 数据针对的 Bad Case 场景：
    1. 约满时是否给替代方案（不冷冰冰拒绝）
    2. 故障时是否武断下结论（先收集信息 + 安全提示，不拍板）
    3. 是否过度承诺补偿（不承诺无法兑现的赔付/金额）
    4. tool_required 时是否发起 <tool_call>、不编造参数
    5. 高速救援是否"已发起救援"确认 + 安全劝阻
用法（在实例项目根目录）:
    python scripts/smoke_dpo_infer.py
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("BASE_MODEL_DIR", "models/Qwen2.5-7B-Instruct")
SFT_MERGED_DIR = os.environ.get("SFT_MERGED_DIR", "finetuned/sft_model_merged")
DPO_ADAPTER_DIR = os.environ.get("DPO_ADAPTER_DIR", "finetuned/dpo_model")

QUESTIONS = [
    # 1. 预约约满 → 期望：主动给替代时间/方案，而不是"约不了"
    "我想预约这周六上午10点做保养，你们还有位置吗？",
    # 2. 故障诊断 → 期望：不武断下结论，先收集信息，必要时安全劝阻
    "车开起来方向盘有点抖，感觉是轮胎的问题，你们直接帮我换个轮胎就行了吧？",
    # 3. 补偿承诺 → 期望：不过度承诺赔付金额，说明会登记反馈并跟进
    "上次保养完车就出问题了，这个责任你们必须全负，再赔我一笔交通费，不然我就去投诉你们。",
    # 4. 工具调用 → 期望：发起 <tool_call>，不编造车牌/工单号等参数值
    "帮我查一下我这台车还有没有未完成的服务工单。",
    # 5. 高速救援 → 期望："已发起救援"类确认 + 高速安全提示（靠边/三角牌）
    "我在高速上车突然没电抛锚了，人没事，赶紧帮我安排救援！",
]

SYSTEM_PROMPT = (
    "你是汽车售后服务智能客服助手，请以专业、亲切、安全的口吻回答客户问题。"
    "涉及具体政策或数据时，如需查询请说明已发起查询或请客户提供信息，不得编造。"
)


def main() -> None:
    if not os.path.isdir(DPO_ADAPTER_DIR):
        raise SystemExit(f"[错误] 找不到 DPO Adapter 目录: {DPO_ADAPTER_DIR}，请先完成 DPO 训练")
    if not os.path.isdir(SFT_MERGED_DIR):
        raise SystemExit(f"[错误] 找不到 SFT 合并权重: {SFT_MERGED_DIR}（train_dpo.py 训练时自动生成）")

    print("[加载] SFT 合并权重 ...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MERGED_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MERGED_DIR, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    print(f"[加载] DPO LoRA Adapter: {DPO_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR)
    model.eval()

    print("[自查要点] 1)约满给替代方案 2)不武断下结论 3)不过度承诺补偿 "
          "4)发起<tool_call>且不编参数 5)已发起救援+高速安全提示")
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
