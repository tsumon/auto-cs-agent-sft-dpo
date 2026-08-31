# SFT 数据构造报告（v1）

生成日期：2026-08-29
生成方式：**模板 + 规则程序化生成**（环境中无可用的 LLM API Key；话术素材由 LLM 撰写、脚本确定性组装。脚本支持 `--mode api` 切换为在线大模型逐条生成，Prompt 见 `scripts/prompts/sft_generation_prompt.md`）。

## 产出文件

| 文件 | 样本数 | 说明 |
|---|---|---|
| output/sft_train.jsonl | 1912 | 由 data/v2/seeds/train/ 640 条种子扩写（每种子 2~3 变体） |
| output/sft_validation.jsonl | 120 | 由 validation/ 64 条种子生成（每种子 single_turn + tool_call） |
| output/sample_10_for_review.txt | 10 | 随机抽样供人工检查 |
| scripts/build_sft_data.py | — | 生成脚本（可重复运行，输出哈希一致） |
| scripts/validate_sft_data.py | — | 校验脚本 |

## 训练集类目 × 类型分布

| 类目 | single_turn | tool_call | multi_turn | 合计 |
|---|---|---|---|---|
| 1 用车与智能功能支持 | 136 | 24 | 80 | 240 |
| 2 保养、质保与服务政策 | 80 | 80 | 80 | 240 |
| 3 故障预诊断与安全分流 | 160 | 0 | 80 | 240 |
| 4 预约与服务受理 | 84 | 76 | 80 | 240 |
| 5 维修过程、费用与交车 | 84 | 76 | 80 | 240 |
| 6 道路救援、事故与保险 | 92 | 68 | 80 | 240 |
| 7 投诉、质量争议与升级处理 | 80 | 80 | 80 | 240 |
| 8 主动关怀、回访与客户运营 | 80 | 80 | 72 | 232 |
| **合计** | **796** | **484** | **632** | **1912** |

注：类目 3 的种子全部 tool_required=false，故无 tool_call 样本；类目 8 有 8 条种子无 required_questions，未生成 multi_turn 变体。

## 校验结果（scripts/validate_sft_data.py）

- JSON 合法性、role 交替、首 user 末 assistant：全部通过
- required_facts / required_questions / required_actions 覆盖：**2032/2032 通过**
- prohibited_actions 违规扫描：0 命中
- 工具样本（484 条）：`<tool_call>` JSON 可解析、name 与 tool_name 一致、必填参数齐全、伪造结果黑名单 0 命中
- 全局去重、长度过滤：无剔除
- final_test 种子：0 条混入（final_test 目录未读取）

## 兼容性验证

- `datasets.load_dataset('json', ...)` 读取通过（TRL SFTTrainer 兼容的标准 messages 格式）
- 与 data/v2/smoke/sft_train_smoke.jsonl 相同的 messages schema，ms-swift 可直接以 `--dataset output/sft_train.jsonl` 使用
