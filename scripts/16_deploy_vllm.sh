#!/bin/bash
# 【流水线 16/16】可选 · vLLM 服务化部署（OpenAI 兼容接口）
# 运行位置：魔搭实例 GPU
# 输入：finetuned/ 下的 Adapter
# 输出：常驻服务 http://127.0.0.1:8000
# 前置步骤：04 或 09　｜　后续步骤：—
# vLLM 部署 SFT 模型（base + LoRA Adapter，OpenAI 兼容接口）
# 用法（实例 /mnt/workspace）: bash scripts/deploy_vllm.sh
# 说明：ROCm 上 vLLM 优先尝试官方 wheel（pip show vllm 或按 vLLM ROCm 文档安装 rocm 分支）。
# 若无 wheel 或 --enable-lora 启动报错，用降级方案：
#   python scripts/eval_sft.py          # transformers 推理（函数调用式）
#   或 ms-swift: swift deploy --adapters finetuned/sft_model --model models/Qwen2.5-7B-Instruct
# 两者差别：vLLM 常驻服务、支持并发和 --enable-lora 热插拔 Adapter，OpenAI /v1/chat/completions 兼容，
#   适合批量评估与服务化；transformers 脚本单进程、逐条生成、吞吐低，但零依赖、ROCm 上最稳，
#   评估脚本 eval_sft.py 即走此路线，二者贪心解码结果一致性可接受。

set -e
BASE_MODEL=${BASE_MODEL:-models/Qwen2.5-7B-Instruct}
ADAPTER_DIR=${ADAPTER_DIR:-finetuned/sft_model}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name sft-agent \
  --enable-lora \
  --lora-modules sft-adapter="$ADAPTER_DIR" \
  --max-lora-rank 64 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port "$PORT"

# 调用示例：
# curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
#   "model": "sft-adapter",
#   "messages": [{"role":"user","content":"我在高速上车没电抛锚了，怎么申请道路救援？"}]
# }'
