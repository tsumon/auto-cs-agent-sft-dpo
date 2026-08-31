# DPO 数据构造报告（里程碑四）

- 产出：`data/v2/dpo/dpo_train.jsonl`（ms-swift 格式，**180 对**）；`dpo_train_trl.jsonl`（TRL 会话格式，同内容）
- 来源配比：Bad Case 27 对（覆盖 27/27 个 Bad Case 种子）、评估补充 35 对、train 种子合成 118 对
- 缺陷类型配比（要求每类 ≥30）：安全话术类 45、工具规范类 45、参数编造类 45、完整性类 45
- 类别分布：cat1 25、cat2 37、cat3 49、cat4 31、cat5 14、cat6 10、cat7 6、cat8 8
- chosen/rejected 平均长度：332 / 122 字；margin 均值 8.7（min 6，max 10）

## 构造规则

1. **rejected**：A/B 类用 SFT 模型真实回答原文（不改写，保留缺陷原貌）；C 类按实测缺陷模板化退化：
   - 完整性类 → 套话半截话（“感谢您联系我们…我先跟您确认一个信息：X”，无实质内容）；
   - 工具规范类 → 不发起 `<tool_call>`，只说“已登记，稍后回复”；
   - 参数编造类 → 发起 `<tool_call>` 但工具内参数全部使用幻觉值（VID26…/S2463/“2024款长续航版”等）；
   - 安全话术类 → cat3 用“影响不大，先继续正常行驶”式敷衍（无劝阻），cat6 用“稍后给您安排”（无救援确认、无调用）。
2. **chosen**：按 seed_rules 程序化拼装——复述场景 + ①②分点覆盖 required_facts +（tool_required 时）发起 `<tool_call>`（工具名=seed.tool_name，参数值只取自用户话语，缺参明确追问）+ ①②分点 required_questions 追问 + required_actions 分步处理 + 收尾；cat3 加“请勿继续行驶、立即靠边停车”，cat6 加“已为您发起救援调度”。话术结构参照 sft_train.jsonl 的 assistant 风格。
3. **自检**：每条 chosen 经 `scripts/eval_sft.py:score_answer` 规则打分须 ≥ 12/14，不达标自动换模板重写（共 3 个变体），仍不达标则剔除；tool 参数值必须逐字出现在用户话语（scenario+user_goal）中，否则判编造剔除。
4. **配对过滤**：规则分差 ≥ 4 才保留（宁缺毋滥）；长度惩罚——chosen 长度 ≥ 1.5 倍 rejected 且分差 < 6 时丢弃，防止 chosen 仅因更长而胜出。
5. **红线**：tool_required=false 的样本不含任何 `<tool_call>`；chosen 不出现“查询到/查询结果为”等伪造结果表述；未使用 final_test/ 任何数据。

## 被过滤统计

- 无

## 训练接入（窗口 5）

```bash
# ms-swift DPO：LoRA 复用 SFT 超参，--beta 0.1，训练前杀掉 vLLM 释放显存
swift rlhf --rlhf_type dpo \
  --model models/Qwen2.5-7B-Instruct --dataset data/v2/dpo/dpo_train.jsonl \
  --adapters finetuned/sft_model --beta 0.1 --max_length 2048 \
  --learning_rate 1e-5 --num_train_epochs 1 --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 --lora_rank 8 --lora_alpha 32 \
  --attn_impl sdpa --torch_dtype bfloat16 --output_dir finetuned/dpo_model
```

> DPO 完成后用 `python scripts/eval_sft.py --rescore --judge` 复评，对比 SFT 基线（均分约 10.5/14、Bad Case 27）。
