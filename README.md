# 汽车售后服务智能客服 — SFT + DPO 微调项目

把通用大模型（Qwen2.5-7B-Instruct）微调为汽车售后服务智能客服，覆盖 8 类业务场景，
两阶段训练（SFT 监督微调 → DPO 偏好对齐），并在 `final_test` 上完成 SFT vs DPO 终局对比。

- 基座模型：Qwen2.5-7B-Instruct
- 微调方式：LoRA（r=32 / alpha=64，全 proj 模块），bf16 + sdpa
- 训练环境：魔搭 ModelScope DSW 实例，单卡 AMD MI300X（192GB，ROCm）
- 框架：transformers + peft + trl（SFTTrainer / DPOTrainer）

## 一、业务场景与数据

8 类场景，种子数据严格三分且互不重叠：

| 数据集 | 条数 | 文件 | 用途 |
|---|---|---|---|
| train | 640 | `data/v2/seeds/train/` | 构造 SFT / DPO 训练数据 |
| validation | 64 | `data/v2/seeds/validation/` | 里程碑三评估、调参 |
| final_test | 160 | `data/v2/seeds/final_test/` | **仅终局评估，从未参与训练与调参** |

工具库 `data/v2/tool_schemas.json` 共 55 个工具。约定 `tool_required=true` 时回答不得伪造
查询结果，只能说明"已发起查询 / 需补充信息后查询"，工具调用格式为 `<tool_call>{json}</tool_call>`。

## 二、脚本流水线（按运行顺序 01 → 16）

每个脚本开头都有 `【流水线 NN/16】` 横幅，标注里程碑、运行位置、输入输出与前后步骤。

| 序号 | 脚本 | 里程碑 | 位置 | 说明 |
|---|---|---|---|---|
| 01 | `01_build_sft_data.py` | 一 | 本机 | 种子 → SFT 训练数据（1912 + 120 条） |
| 02 | `02_validate_sft_data.py` | 一 | 本机 | 覆盖率 / 违规 / 去重校验 |
| 03 | `03_build_sft_supplement.py` | 二 | 本机 | 补 500 条多样化回答（3 种风格，破模板单一性） |
| 04 | `04_train_sft.py` | 二 | GPU | SFT 训练（2412 条，lr 5e-5，2 epoch） |
| 05 | `05_smoke_infer.py` | 二 | GPU | 5 问冒烟 |
| 06 | `06_eval_sft.py` | 三 | GPU + API | validation 64 条 7 维打分（规则 + LLM Judge） |
| 07 | `07_build_dpo_data.py` | 四 | 本机 | DPO 偏好数据 v1（180 对） |
| 08 | `08_build_dpo_data_r2.py` | 四 | 本机 | DPO 偏好数据 v2（450 对，**生产用**） |
| 09 | `09_train_dpo.py` | 五 | GPU | DPO 训练（IPO loss，beta 0.2，lr 3e-6） |
| 10 | `10_smoke_dpo_infer.py` | 五 | GPU | 偏好敏感 5 问冒烟 |
| 11 | `11_run_final_gen.py` | 六 | GPU | final_test 160 条 × SFT/DPO 双模型生成 |
| 12 | `12_run_final_eval.py` | 六 | API | 7 维打分 + Arena 双盲竞技场 + 报告 |
| 13 | `13_build_arena_dashboard.py` | 六 | 本机 | Arena 可视化看板（自包含 HTML） |
| 14 | `14_constrained_infer.py` | 生产 | GPU | 推理期约束管线（校验 → 带答案重试 → 兜底） |
| 15 | `15_run_arena_validation.py` | 生产 | API | 约束版 vs 基线 64 条盲测 |
| 16 | `16_deploy_vllm.sh` | 部署 | GPU | vLLM + LoRA 服务化 |

`_pipeline.py` 是导入桥：脚本按运行顺序带数字前缀，而 Python 不允许 import 数字开头的模块名，
该模块把 06 / 07 / 12 / 13 注册为无前缀别名，使下游脚本共用同一套打分函数（保证"同一把尺子"）。

离线自检（不需要 GPU，不打 API）：

```bash
python scripts/06_eval_sft.py --selftest
python scripts/07_build_dpo_data.py --selftest
python scripts/12_run_final_eval.py --selftest
python scripts/14_constrained_infer.py --selftest
```

## 三、数据集产物

| 文件 | 条数 | 说明 |
|---|---|---|
| `output/sft_train.jsonl` | 1912 | SFT 训练集（messages 格式） |
| `output/sft_validation.jsonl` | 120 | SFT 验证集 |
| `output/sft_train_supplement.jsonl` | 500 | 多样化补充（与上面混合成 2412 条重训） |
| `data/v2/dpo/dpo_train.jsonl` | 180 | DPO 偏好数据 v1（含元信息） |
| `data/v2/dpo_r2/dpo_train.jsonl` | 450 | **DPO 偏好数据 v2（生产用）** |
| `data/v2/dpo_r2/dpo_train_trl.jsonl` | 450 | 同上，TRL prompt/chosen/rejected 格式 |
| `output/sft_bad_cases.jsonl` | 27 | SFT Bad Case（里程碑四输入） |
| `output/final_test_questions.jsonl` | 160 | 终局统一测试问题集（冻结存档） |
| `output/sft_eval_results.json` | 64 | SFT 评估逐条明细 |

DPO 数据构造红线：`chosen` 中 `tool_call` 参数值只能逐字取自用户话语，缺参一律追问，绝不编造；
分差 margin ≥ 4 且带长度惩罚（防"仅因更长而胜出"）。

## 四、核心结果

### 里程碑六：final_test 160 条，SFT vs DPO

| 指标 | SFT 基线 | DPO |
|---|---|---|
| 7 维总均分（满分 14） | 9.41 | 9.36（Δ −0.06） |
| Bad Case | 117 | 119 |
| Arena 战绩（DeepSeek 评审） | 胜 19 | 胜 11 / 平 130 |
| Elo 分差 | — | −17.4 |

**核心问题结论**：DPO 在不破坏 SFT 基础能力的前提下完成训练（7 维全平，±0.03 内），
但**偏好对齐无显著增量**。两个能力差距很大的评审模型（GLM-4-Flash 与 DeepSeek）都给出
高平局率（89% / 81%），说明平局反映的是"两模型输出本身高度接近"，而非 Judge 分辨力不足——
技术原因是 DPO 为 KL 受限微调（beta 0.2 / lr 3e-6 / 450 对），未触及行为、保持 SFT 原样。
结论：7B + LoRA + 450 对 DPO 的天花板。

### 迭代路径与关键发现

| 版本 | 改动 | 结果 |
|---|---|---|
| SFT R1 | 1912 条，回答结构单一 | 半截话 42/64 |
| DPO R1 | 180 对，sigmoid loss | 过拟合（accuracies 1.0），Bad Case 31 |
| DPO R2 | 450 对 + IPO + beta 0.2 | Bad Case 26，靶向缺陷未动 |
| **SFT R2** | **+500 条 3 种回答风格** | **冒烟 5/5 从半截话变完整回答** |
| DPO R3 | 450 对 on SFT R2 | 均分 10.64，半截话基本消除 |

1. **半截话的根因是训练数据回答模板单一**，不是训练不足。原 1912 条全部是"开场→事实→工具→
   追问→建议→收尾"同一结构，模型学到固定"形状"而非业务逻辑，在 validation 上退化成最短路径。
   补 500 条 3 种风格后一次解决。
2. **工具选择这类确定性映射，应交给规则层，不要指望 7B 模型记住**。曾尝试用 140 条补丁数据
   教模型工具映射（R3），结果工具名幻觉从 6 暴涨到 22——55 个工具 × 各场景的映射用每工具约 2 条
   样本学不会，模型只学到"多调工具"的表层信号。该路线已弃用。
3. **推理期约束层是本项目性价比最高的改动**（零训练成本、当天生效）：

   | 版本（validation 64 条，同一 GLM Judge） | 均分 | Bad Case |
   |---|---|---|
   | 基线（DPO R2 裸生成） | 8.66 | 49 |
   | **+ 约束层** | **10.50** | **27（−45%）** |

   约束层三个必要条件（都是踩坑得来的）：① 重试必须配采样解码（贪心下同一 prompt 输出确定，
   重试无意义）；② 修正指令必须带正确答案（只报错不给答案时，7B 模型无从改正，fallback 28/64）；
   ③ 删除非法 `tool_call` 必须用 `re.sub`（模型输出的标签内带换行，`str.replace` 拼字符串会静默失效）。

**生产方案**：基座 → `sft_model_merged` → `dpo_model_r2`，外层套 `14_constrained_infer.py`
的校验 → 带答案重试 → 兜底逻辑。最终靶标（validation 64 条，确定性规则检测）：
工具名幻觉 0、乱调用 0、cat3 缺安全劝阻 2/8、cat6 缺救援确认 0/8。

## 五、评估方法注意事项

- **跨 Judge 分数不可比**：同一批回答，Qwen3-235B 判 10.64 分 / 27 Bad Case，GLM-4-Flash 判
  8.66 分 / 49 Bad Case。任何版本对比必须用同一 Judge 重评双方，不得复用历史分数。
- **单次 Bad Case 数有 ±4 噪声带**：`06_eval_sft.py` 的 Judge 调用无缓存，且 GLM 在
  temperature=0 下仍非完全确定。版本对比的决策判据应是确定性的规则靶标（工具名幻觉 / 乱调用 /
  cat3 / cat6 计数），而非单次 Bad Case 数。
- **Arena 竞技场**规则：回答匿名为 A/B，固定种子随机站位，双向对调复审，两方向一致才判胜负，
  否则判平——消除位置偏置，代价是平局率偏高。

## 六、环境

```bash
pip install -r requirements-sft.txt
```

评分阶段需要 Judge API（三个环境变量，任选一个通道）：

```bash
export JUDGE_API_KEY=<your-key>
export JUDGE_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 或 https://api.deepseek.com/v1
export JUDGE_MODEL=glm-4-flash                              # 或 deepseek-chat
```

评分脚本请用 `python -u` 运行（脚本 print 无 flush，管道下块缓冲会看似卡死）。
