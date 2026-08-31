#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【流水线 01/16】里程碑一 · 构造 SFT 训练数据
运行位置：本机（无需 GPU）
输入：data/v2/seeds/train/ 种子 + data/v2/tool_schemas.json
输出：output/sft_train.jsonl（1912 条）、output/sft_validation.jsonl（120 条）
前置步骤：—　｜　后续步骤：02 校验

SFT 数据构造脚本（汽车售后服务智能客服）

生成方式说明：
  - 默认 mode=template：基于「LLM 撰写的话术素材池 + 规则程序化组装」。
    语言素材（问候/共情/追问/应答/收尾话术、场景改写规则、角色区分）为人工撰写并内置在
    本文件中；脚本按种子字段做确定性组装，保证 required_facts / required_questions /
    required_actions 逐条覆盖、prohibited_actions 零出现、工具调用不伪造结果。
  - 可选 mode=api：逐种子调用 OpenAI 兼容接口（如 DashScope）生成，需要环境变量
    DASHSCOPE_API_KEY（或 --api-key），Prompt 见 scripts/prompts/sft_generation_prompt.md。

输出：
  output/sft_train.jsonl
  output/sft_validation.jsonl

可重复运行：所有随机性均以 (seed_id, variant) 派生的确定性随机源实现。
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_DIR = os.path.join(ROOT, "data", "v2", "seeds")
TOOL_SCHEMA_PATH = os.path.join(ROOT, "data", "v2", "tool_schemas.json")
PROMPT_PATH = os.path.join(ROOT, "scripts", "prompts", "sft_generation_prompt.md")
OUT_DIR = os.path.join(ROOT, "output")

# ---------------------------------------------------------------------------
# 1. 话术素材池（由 LLM 撰写，脚本负责挑选与组装）
# ---------------------------------------------------------------------------

# 用户开场（信息完整 / 咨询单轮）
USER_OPENERS_SELF = [
    "您好，我想咨询一下：",
    "您好，有个问题想麻烦您帮忙看看：",
    "你好，我这边遇到点情况：",
    "您好，麻烦帮我处理一个问题：",
    "您好，想向您求助一件事：",
]
USER_OPENERS_AGENT = [
    "您好，我是车主的家人，替车主咨询一下：",
    "您好，我帮家里人来问问：",
    "您好，我是车主委托的代办人，想咨询：",
    "您好，我替我们领导（公司车）来咨询一下：",
]
USER_OPENERS_CARE = [  # 类目8：客户收到主动关怀/提醒后来电
    "您好，我收到你们发的提醒了，想具体了解一下：",
    "您好，看到你们推送的消息，我想问下：",
    "您好，我收到提醒短信了，想跟您确认下：",
]
USER_DETAIL_POOL = [
    "车子是2024款长续航版，去年6月提的。",
    "我的车是2023款四驱旗舰版，平时主要在市区通勤。",
    "车目前行驶一万四千公里左右，一直在你们店做保养。",
    "车是今年年初提的，基本每天上下班开。",
]
USER_VAGUE = [
    "您好，我想咨询个问题：{sit}。具体信息我一时说不太全，您看需要我提供哪些资料？",
    "您好，有个事想请教：{sit}。我手头的信息不太完整，麻烦您告诉我还需要补充什么。",
    "您好，想麻烦您帮个忙：{sit}。具体情况我需要再确认一下，您先帮我看下大概是什么问题？",
]
USER_REOPEN = [
    "好的，那您帮我看看接下来怎么办。",
    "明白了，麻烦您告诉我具体怎么处理。",
    "嗯嗯，那这个问题严重吗？我该怎么办？",
    "好的，那后面我需要注意些什么？",
]

# 客服共情/开场
EMPATHY = [
    "您好，很高兴为您服务，您反映的问题我来跟进。",
    "您好，感谢您的耐心等待，我来帮您处理。",
    "您好，请别着急，我这边一步一步帮您排查和解决。",
    "您好，感谢您联系我们，您的情况我已经记录下来了。",
]
EMPATHY_SAFE = [  # 涉安全（类目3/6）优先使用
    "您好，请先确保您自身处于安全环境，我再帮您处理。",
    "您好，安全第一，请先确认人和车都处在安全位置，我来帮您安排。",
    "您好，别着急，请先保证自身安全，我们马上逐步处理。",
]

# 事实/追问/行动引导语
FACT_LEADS = [
    "这里先跟您说明{num}点关键信息：",
    "有两方面的信息先跟您同步一下，一共{num}点：",
    "先给您讲清楚{num}个要点：",
    "在处理之前，有{num}点重要信息需要您了解：",
]
QUESTION_LEAD_MANY = [
    "为了给您准确的方案，麻烦您确认以下几点：",
    "需要再跟您核实{num}个信息：",
    "接下来想跟您确认{num}个问题：",
    "为了不误判，请您配合确认{num}点：",
]
QUESTION_LEAD_ONE = [
    "另外想跟您确认一下：",
    "还有一点需要您帮忙确认：",
    "麻烦您补充一个信息：",
]
ACTION_LEADS = [
    "建议您这样处理：",
    "接下来您可以按下面的步骤操作：",
    "我的处理建议如下：",
    "这边给您的处理方案是：",
]
CLOSINGS = [
    "如后续有任何问题，欢迎随时联系我们，祝您用车愉快。",
    "如果还有疑问，随时找我或拨打服务热线都可以。",
    "后续处理中有任何进展问题，欢迎随时反馈，我们会持续跟进。",
    "感谢您的理解与配合，祝您生活愉快。",
]

# 工具相关话术
TOOL_LEAD = "我先帮您在系统里{desc}，"
TOOL_ANNOUNCE = [
    "结果返回后我会第一时间同步给您，在此之前我不会给您任何猜测性的结论。",
    "查询结果出来后我会马上告知您，请您以最终反馈为准，我不会凭空估计结果。",
    "查询提交后请您稍等，结果出来我会立即同步，未出结果前我不会下结论。",
]
TOOL_MULTI_NOTE = "等您补充完这些信息后，我再为您正式发起查询。"

# 数字中文
NUM_CN = ["零", "一", "两", "三", "四", "五", "六"]

# 追问后用户的应答规则（按关键词匹配，模拟真实用户；具体规则在前，通用兜底在后）
def user_reply_for_question(q: str, rng: random.Random, last: str = None) -> str:
    """按关键词匹配生成追问后的用户应答：命中具体规则优先，否则走通用兜底。

    last 传入上一轮已用过的回复，用于避开连续重复的说法。
    """
    pairs = [
        (["限高", "车位"], ["地库限高大概两米，我的车位在B2层，坡道比较陡。"]),
        (["有人", "宠物", "被困"], ["车里没有其他人，也没有宠物，就我自己。", "没有，就我一个人在车里，宠物没带上。"]),
        (["撤离"], ["人已经撤到安全位置了，离车有一段距离。"]),
        (["水位"], ["水位还在往上涨，我已经离开车辆到高处了。"]),
        (["发热", "异味", "破损", "焦味"], ["我检查过了，没有异味和破损，摸着也不烫。"]),
        (["上电", "启动", "移动"], ["车能上电，但我没敢再启动行驶。"]),
        (["跑偏", "抖动", "踏板"], ["好像有一点轻微抖动，制动力暂时感觉还正常。"]),
        (["备用钥匙"], ["没有备用钥匙，手里就这一把，机械钥匙也不在身边。"]),
        (["重启"], ["重启过了，还是老样子。"]),
        (["过户"], ["过户手续已经办完了，合同和发票都在。"]),
        (["门店"], ["就选离我最近的那家店，时间上明天上午方便。"]),
        (["应用", "提示", "报错", "屏幕", "显示"],
         ["屏幕上提示连接失败，请稍后重试，其他功能暂时还能正常用。", "App上显示认证失败，截图我稍后发给您。"]),
        (["关系", "授权"], ["我是车主的儿子，车主本人知情，授权委托书和双方证件我都可以提供。"]),
        (["工单"], [f"工单号是{gen_work_order_id(rng)}，我这边有短信记录。"]),
        (["车型", "年款", "配置"], ["我的车是2024款长续航版。"]),
        (["里程", "公里"], ["现在仪表显示大概一万四千公里。"]),
        (["保单", "保险", "报案"], ["保险是购车时在店里投保的，报案号和保单号我稍后拍照发给您。"]),
        (["手机", "型号", "系统"], ["手机是这两年的新款，系统已经更新到最新版本。"]),
        (["城市", "哪里", "地区", "位置", "所在地", "地址"], ["我现在在杭州，具体位置我可以发定位给您。"]),
        (["时间", "多久", "什么时候", "日期", "发生"], ["大概是上周发现的，具体时间我再翻一下记录跟您确认。"]),
        (["车架号", "VIN"], ["车架号我一会儿拍行驶证发给您。"]),
        (["承诺", "之前是否"], ["之前门店没有给过我书面承诺，只是口头提过一次。"]),
    ]
    for keys, answers in pairs:
        if any(k in q for k in keys):
            candidates = [a for a in answers if a != last] or answers
            return rng.choice(candidates)
    generic = [
        "我确认了一下，情况跟您说的差不多，其他细节我可以在App里补充给您。",
        "好，我这边核对了下，基本就是您说的那样，细节我稍后补充。",
        "行，我记下来了，回头我把相关材料拍照发给您。",
    ]
    candidates = [a for a in generic if a != last] or generic
    return rng.choice(candidates)


# ---------------------------------------------------------------------------
# 2. 工具 schema 与参数填充
# ---------------------------------------------------------------------------

def load_tool_schemas():
    """读取 tool_schemas.json，返回 {工具名: schema} 的字典。"""
    with open(TOOL_SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {t["name"]: t for t in data["tools"]}


def gen_vin(rng: random.Random) -> str:
    """生成占位车架号（VIN）：固定前缀 LSVA + 13 位随机数字。"""
    return "LSVA" + "".join(str(rng.randint(0, 9)) for _ in range(13))


def gen_work_order_id(rng: random.Random) -> str:
    """生成占位工单号：固定前缀 WO26 + 8 位随机数字。"""
    return "WO26" + "".join(str(rng.randint(0, 9)) for _ in range(8))


def gen_case_id(rng: random.Random) -> str:
    """生成占位案件号：固定前缀 CC26 + 6 位随机数字。"""
    return "CC26" + "".join(str(rng.randint(0, 9)) for _ in range(6))


def fill_tool_args(tool_name: str, seed: dict, rng: random.Random, schemas: dict) -> dict:
    """按 schema required 参数填充占位值；不伪造任何查询结果。"""
    schema = schemas.get(tool_name)
    if not schema:
        return {}
    props = schema.get("parameters", {}).get("properties", {})
    required = schema.get("parameters", {}).get("required", list(props.keys()))
    sub = seed.get("subcategory") or "相关业务"
    args = {}
    for p in required:
        if p == "vehicle_id":
            args[p] = gen_vin(rng)
        elif p == "work_order_id":
            args[p] = gen_work_order_id(rng)
        elif p == "case_id":
            args[p] = gen_case_id(rng)
        elif p == "part_order_id":
            args[p] = "PTO26" + "".join(str(rng.randint(0, 9)) for _ in range(6))
        elif p == "policy_number":
            args[p] = "PIN" + "".join(str(rng.randint(0, 9)) for _ in range(10))
        elif p == "preferred_date":
            sc = seed.get("scenario", "")
            if "第二天" in sc or "明天" in sc:
                args[p] = rng.choice(["明天上午", "明天下午"])
            else:
                args[p] = rng.choice(["本周六上午", "下周三上午", "这周五下午"])
        elif p == "store_id":
            args[p] = "S" + str(rng.randint(2100, 2999))
        elif p == "city" or p == "region":
            args[p] = "杭州"
        elif p == "address":
            args[p] = "杭州市西湖区文一西路969号小区地库"
        elif p in ("policy_topic", "topic", "policy_type"):
            args[p] = sub
        elif p == "part_name":
            args[p] = sub
        elif p == "campaign_code":
            args[p] = "C26" + str(rng.randint(1000, 9999))
        elif p == "charger_model":
            args[p] = "7kW家用充电桩"
        elif p == "issue_summary":
            args[p] = clean_scenario(seed.get("scenario", ""))[:40]
        elif p == "followup_note":
            args[p] = seed.get("user_goal", "客户回访记录")[:40]
        elif p == "channel":
            args[p] = "电话"
        elif p == "consent":
            args[p] = True
        elif p == "approved":
            args[p] = True
        elif p == "vehicle_model":
            args[p] = "2024款长续航版"
        else:
            args[p] = sub
    return args


def tool_call_block(name: str, args: dict) -> str:
    """把工具名与参数包成 <tool_call>{...}</tool_call> 文本块，供模型学习该输出格式。"""
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


# ---------------------------------------------------------------------------
# 3. 种子解析与场景改写
# ---------------------------------------------------------------------------

SCENARIO_PREFIXES = [
    "客户提供的信息不完整：",
    "家属或代办人代表车主咨询：",
    "业务系统暂时无法返回结果时，",
    "客户只能通过远程渠道描述且现场情况无法复现：",
]


def clean_scenario(scenario: str) -> str:
    """剥离种子 scenario 里的固定前缀，以及开头的'客户'二字，得到干净的情境描述。"""
    s = scenario.strip()
    for p in SCENARIO_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    if s.startswith("客户"):
        s = s[2:]
    return s


def situation_for(seed) -> str:
    """面向对话的场景改写：剥离前缀、第三人称'客户'转'我/车主'、'询问X'转口语。"""
    sit = clean_scenario(seed.get("scenario", ""))
    if sit.startswith("询问"):
        sit = "想问一下，" + sit[2:]
    pron = "车主" if is_agent_role(seed.get("customer_role", "")) else "我"
    return sit.replace("客户", pron)


def is_agent_role(role: str) -> bool:
    """判断 customer_role 是否为代人咨询（家属或代办人），决定用「车主」还是「我」自称。"""
    return role in ("家属或代办人",)


def load_seeds(split: str):
    """按 split（train/validation）读入 seeds 目录下所有 jsonl 种子，返回 dict 列表。

    文件名排序后逐行读取，保证同一份种子库多次运行的顺序一致。
    """
    paths = sorted(glob.glob(os.path.join(SEED_DIR, split, "*.jsonl")))
    seeds = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))
    return seeds


def rng_for(seed_id: str, variant: int) -> random.Random:
    """用 (seed_id, variant) 派生确定性随机源，使脚本可重复运行、产出逐字节一致。"""
    return random.Random(f"{seed_id}::v{variant}")


# ---------------------------------------------------------------------------
# 4. 回答组装（保证 required_* 覆盖）
# ---------------------------------------------------------------------------

def join_numbered(items):
    """把条目串成「①xxx；②yyy。」形式，最多支持 6 条。"""
    marks = ["①", "②", "③", "④", "⑤", "⑥"]
    return "；".join(f"{marks[i]}{it}" for i, it in enumerate(items)) + "。"


def facts_sentence(facts, rng):
    """把 required_facts 组装成一句带引导语的编号说明；facts 为空则返回空串。

    引导语里的 {num} 用中文数字替换，保证条数与实际条目对得上。
    """
    if not facts:
        return ""
    lead = rng.choice(FACT_LEADS).replace("{num}", NUM_CN[min(len(facts), 6)])
    return lead + join_numbered(facts)


def questions_sentence(questions, rng, shuffle=False):
    """把 required_questions 组装成追问句：单条用短引导语，多条用编号列表。

    shuffle=True 时打乱提问顺序，用于生成同一种子的表达变体，覆盖内容不变。
    """
    if not questions:
        return ""
    qs = list(questions)
    if shuffle:
        rng.shuffle(qs)
    if len(qs) == 1:
        return rng.choice(QUESTION_LEAD_ONE) + qs[0] + "。"
    lead = rng.choice(QUESTION_LEAD_MANY).replace("{num}", NUM_CN[min(len(qs), 6)])
    return lead + join_numbered(qs)


def actions_sentence(actions, rng):
    """把 required_actions 组装成一句带引导语的处理建议；actions 为空则返回空串。"""
    if not actions:
        return ""
    lead = rng.choice(ACTION_LEADS)
    return lead + join_numbered(actions)


def empathy_for(seed, rng):
    """按类目挑共情开场语：类目 3/6 涉安全走 EMPATHY_SAFE，类目 8 走主动关怀口径。"""
    if seed["category_id"] in (3, 6):
        return rng.choice(EMPATHY_SAFE)
    if seed["category_id"] == 8:
        return rng.choice(["您好，感谢您关注我们的提醒，我来为您说明。", "您好，很高兴为您服务，我来帮您确认。"])
    return rng.choice(EMPATHY)


def build_user_first_message(seed, rng, style: str, extra_info: str = "") -> str:
    """组装用户首条消息：按角色/类目挑开场语，style="full" 时补车辆细节与求助句。

    extra_info 用于把工具必需参数（车架号、工单号、城市）塞进用户口述，避免客服凭空编参数。
    """
    sit = situation_for(seed)
    if is_agent_role(seed.get("customer_role", "")):
        opener = rng.choice(USER_OPENERS_AGENT)
    elif seed["category_id"] == 8:
        opener = rng.choice(USER_OPENERS_CARE)
    else:
        opener = rng.choice(USER_OPENERS_SELF)
    detail = rng.choice(USER_DETAIL_POOL) if style == "full" else ""
    extra = extra_info
    body = f"{opener}{sit}。"
    if detail:
        body += detail
    if extra:
        body += extra
    if style == "full":
        body += rng.choice(["麻烦您帮我看看该怎么处理？", "想请您给我一个处理方案。", "您看这个问题应该怎么解决？"])
    return body


def build_assistant_single_turn(seed, rng, questions_order="keep") -> str:
    """组装单轮客服回复：共情 + 情境复述 + 事实 + 追问 + 行动 + 收尾。

    顺序固定，保证 required_facts/questions/actions 三类内容逐条出现，便于 02 做覆盖校验。
    """
    parts = [empathy_for(seed, rng)]
    sit = situation_for(seed)
    parts.append(f"您反映的情况是「{sit}」，我来帮您分析处理。")
    fs = facts_sentence(seed.get("required_facts") or [], rng)
    if fs:
        parts.append(fs)
    qs = questions_sentence(seed.get("required_questions") or [], rng, shuffle=(questions_order == "shuffle"))
    if qs:
        parts.append(qs)
    acts = actions_sentence(seed.get("required_actions") or [], rng)
    if acts:
        parts.append(acts)
    parts.append(rng.choice(CLOSINGS))
    return "".join(parts)


def build_multi_turn(seed, rng, variant: int):
    """信息不全 → 逐轮追问（每轮1个，超过2个则第2轮合并）→ 最终结论。"""
    qs = list(seed.get("required_questions") or [])
    messages = [{"role": "user", "content": build_user_first_message(seed, rng, style="vague")}]
    asked = []
    last_reply = None
    turn_no = 0
    while len(qs) - len(asked) > 1 and len(asked) < 2:
        q = qs[len(asked)]
        asked.append(q)
        turn_no += 1
        if turn_no == 1:
            a = empathy_for(seed, rng) + f"为了给您准确的处理方案，我先跟您确认一个信息：{q}。"
        else:
            a = f"收到。还需要跟您确认一个信息：{q}。"
        messages.append({"role": "assistant", "content": a})
        reply = user_reply_for_question(q, rng, last_reply)
        last_reply = reply
        messages.append({"role": "user", "content": reply})
    # 剩余问题（含最后一个，或超过2个时的合并）
    rest = qs[len(asked):]
    asked.extend(rest)
    # 最终结论：事实 + 剩余问题确认 + 行动 + 收尾
    parts = ["好的，情况我基本了解了。"]
    fs = facts_sentence(seed.get("required_facts") or [], rng)
    if fs:
        parts.append(fs)
    if rest:
        parts.append("最后还需要跟您核实：" + join_numbered(rest))
    acts = actions_sentence(seed.get("required_actions") or [], rng)
    if acts:
        parts.append(acts)
    if seed.get("tool_required") and seed.get("tool_name"):
        parts.append(f"等信息齐全后，我会为您在系统里正式发起相关查询，结果出来后第一时间同步给您，不会凭空下结论。")
    parts.append(rng.choice(CLOSINGS))
    messages.append({"role": "assistant", "content": "".join(parts)})
    return messages


def build_tool_call_turn(seed, rng, schemas) -> tuple:
    """单轮含工具调用：用户给足 tool 必需参数，客服给出事实+动作+<tool_call>，不伪造结果。"""
    tool_name = seed["tool_name"]
    args = fill_tool_args(tool_name, seed, rng, schemas)
    desc = schemas.get(tool_name, {}).get("description", f"执行{tool_name}").rstrip("。")
    extra = ""
    if "vehicle_id" in args:
        extra = f"我的车架号（VIN）是{args['vehicle_id']}，"
    elif "work_order_id" in args:
        extra = f"我的工单号是{args['work_order_id']}，"
    elif "city" in args:
        extra = f"我在{args['city']}，"
    user_msg = build_user_first_message(seed, rng, style="full", extra_info=extra)
    parts = [empathy_for(seed, rng)]
    fs = facts_sentence(seed.get("required_facts") or [], rng)
    if fs:
        parts.append(fs)
    parts.append(TOOL_LEAD.format(desc=desc))
    parts.append(tool_call_block(tool_name, args))
    parts.append(rng.choice(TOOL_ANNOUNCE))
    qs = questions_sentence(seed.get("required_questions") or [], rng)
    if qs:
        parts.append("在等待结果的同时，" + qs)
    acts = actions_sentence(seed.get("required_actions") or [], rng)
    if acts:
        parts.append(acts)
    parts.append(rng.choice(CLOSINGS))
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "".join(parts)},
    ]
    return messages, {"name": tool_name, "arguments": args}


# ---------------------------------------------------------------------------
# 5. 变体规划与主流程
# ---------------------------------------------------------------------------

def make_record(seed, messages, sample_type, variant, tool_call=None):
    """把 messages 和种子元信息（seed_id/类目/tool_name 等）打包成一条 JSONL 样本。"""
    rec = {
        "seed_id": seed["seed_id"],
        "split": seed["split"],
        "category_id": seed["category_id"],
        "category": seed["category"],
        "subcategory": seed.get("subcategory"),
        "scenario": seed.get("scenario"),
        "sample_type": sample_type,
        "variant": variant,
        "tool_name": seed.get("tool_name"),
        "messages": messages,
    }
    if tool_call is not None:
        rec["tool_call"] = tool_call
    return rec


def gen_variants_for_seed(seed, schemas):
    """一条种子最多产出 3 个变体：v0 单轮完整、v1 工具调用或换表达单轮、v2 多轮追问。

    v1/v2 都带条件：无 tool_required 时退化为换问题顺序的单轮，无 required_questions 时不生成多轮。
    """
    recs = []
    # v0: 单轮完整信息
    rng = rng_for(seed["seed_id"], 0)
    msgs = [
        {"role": "user", "content": build_user_first_message(seed, rng, style="full")},
        {"role": "assistant", "content": build_assistant_single_turn(seed, rng)},
    ]
    recs.append(make_record(seed, msgs, "single_turn", 0))

    # v1: tool_required → 工具调用样本；否则 → 换表达/换问题顺序的单轮变体
    if seed.get("tool_required") and seed.get("tool_name"):
        rng = rng_for(seed["seed_id"], 1)
        msgs, tc = build_tool_call_turn(seed, rng, schemas)
        recs.append(make_record(seed, msgs, "tool_call", 1, tool_call=tc))
    else:
        rng = rng_for(seed["seed_id"], 1)
        msgs = [
            {"role": "user", "content": build_user_first_message(seed, rng, style="full")},
            {"role": "assistant", "content": build_assistant_single_turn(seed, rng, questions_order="shuffle")},
        ]
        recs.append(make_record(seed, msgs, "single_turn", 1))

    # v2: 多轮追问（有 required_questions 才生成）
    if seed.get("required_questions"):
        rng = rng_for(seed["seed_id"], 2)
        msgs = build_multi_turn(seed, rng, 2)
        recs.append(make_record(seed, msgs, "multi_turn", 2))
    return recs


def dedup_and_filter(records):
    """按 messages 的 md5 去重并做长度过滤，返回 (保留样本, 剔除原因计数)。

    user 超 600 字或 assistant 不在 20~2500 字区间的样本直接丢弃，与 02 的长度校验口径一致。
    """
    seen = set()
    out, dropped = [], Counter()
    for r in records:
        h = hashlib.md5(json.dumps(r["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if h in seen:
            dropped["重复"] += 1
            continue
        # 长度过滤
        ok = True
        for m in r["messages"]:
            c = m["content"]
            if m["role"] == "user" and len(c) > 600:
                ok = False
            if m["role"] == "assistant" and not (20 <= len(c) <= 2500):
                ok = False
        if not ok:
            dropped["长度过滤"] += 1
            continue
        seen.add(h)
        out.append(r)
    return out, dropped


def write_jsonl(path, records):
    """逐行写出 JSONL（ensure_ascii=False 保留中文），自动创建父目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_seed_stats(seeds, title):
    """打印种子集的总数、tool_required 条数与各类目分布，便于生成前对账。"""
    cnt = Counter(s["category_id"] for s in seeds)
    tool = sum(1 for s in seeds if s.get("tool_required"))
    print(f"[{title}] 种子总数={len(seeds)}  tool_required={tool}")
    for cid in sorted(cnt):
        name = next(s["category"] for s in seeds if s["category_id"] == cid)
        print(f"  类目{cid} {name}: {cnt[cid]} 条")


# ---------------------------------------------------------------------------
# 6. API 生成模式（可选，需 API Key）
# ---------------------------------------------------------------------------

def api_generate(seeds, args):
    """逐种子调用 OpenAI 兼容接口。Prompt 模板见 scripts/prompts/sft_generation_prompt.md"""
    import requests

    key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("未提供 API Key（--api-key 或环境变量 DASHSCOPE_API_KEY），无法使用 API 模式。", file=sys.stderr)
        sys.exit(1)
    base_url = args.api_base.rstrip("/")
    with open(PROMPT_PATH, encoding="utf-8") as f:
        sys_prompt = f.read()
    session = requests.Session()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(seed):
        """对单条种子发一次请求，最多重试 3 次；全部失败返回 None 并跳过该种子。"""
        user_prompt = (
            f"请基于以下种子生成一条 SFT 样本，输出 JSON：{{\"messages\":[...]}}。\n"
            f"种子：{json.dumps(seed, ensure_ascii=False)}"
        )
        payload = {
            "model": args.api_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(3):
            try:
                resp = session.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                obj = json.loads(content)
                return make_record(seed, obj["messages"], obj.get("sample_type", "single_turn"), 0)
            except Exception as e:  # noqa
                if attempt == 2:
                    print(f"  [失败] {seed['seed_id']}: {e}", file=sys.stderr)
                    return None
        return None

    records = []
    for i, seed in enumerate(seeds):
        r = call(seed)
        if r:
            records.append(r)
        if (i + 1) % 50 == 0:
            print(f"  API 生成进度: {i+1}/{len(seeds)}")
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """CLI 入口：按 template/api 模式生成 train 与 validation 两份 JSONL 并打印分布统计。

    验证集只生成单轮和工具调用两种变体，不做多轮追问。
    """
    parser = argparse.ArgumentParser(description="构造汽车售后客服 SFT 数据")
    parser.add_argument("--mode", choices=["template", "api"], default="template",
                        help="template=话术池+规则组装（默认）；api=调用 OpenAI 兼容接口")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-model", default="qwen-plus")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    schemas = load_tool_schemas()

    if not args.skip_train:
        seeds = load_seeds("train")
        print_seed_stats(seeds, "train 种子分布")
        if args.mode == "api":
            records = api_generate(seeds, args)
        else:
            records = []
            for s in seeds:
                records.extend(gen_variants_for_seed(s, schemas))
        records, dropped = dedup_and_filter(records)
        out = os.path.join(OUT_DIR, "sft_train.jsonl")
        write_jsonl(out, records)
        cnt = Counter((r["category_id"], r["sample_type"]) for r in records)
        print(f"✅ 训练集写入 {out}：有效样本 {len(records)} 条，剔除 {dict(dropped)}")
        print("  每类样本数：")
        for cid in range(1, 9):
            by_type = {t: cnt[(cid, t)] for t in ("single_turn", "tool_call", "multi_turn") if cnt[(cid, t)]}
            name = next(r["category"] for r in records if r["category_id"] == cid)
            print(f"    类目{cid} {name}: 合计 {sum(by_type.values())} {by_type}")

    if not args.skip_val:
        seeds = load_seeds("validation")
        print_seed_stats(seeds, "validation 种子分布")
        records = []
        for s in seeds:
            rng = rng_for(s["seed_id"] + "::val", 0)
            msgs = [
                {"role": "user", "content": build_user_first_message(s, rng, style="full")},
                {"role": "assistant", "content": build_assistant_single_turn(s, rng)},
            ]
            records.append(make_record(s, msgs, "single_turn", 0))
            if s.get("tool_required") and s.get("tool_name"):
                rng = rng_for(s["seed_id"] + "::val", 1)
                msgs, tc = build_tool_call_turn(s, rng, schemas)
                records.append(make_record(s, msgs, "tool_call", 1, tool_call=tc))
        records, dropped = dedup_and_filter(records)
        out = os.path.join(OUT_DIR, "sft_validation.jsonl")
        write_jsonl(out, records)
        cnt = Counter(r["category_id"] for r in records)
        print(f"✅ 验证集写入 {out}：有效样本 {len(records)} 条，剔除 {dict(dropped)}")
        print("  每类样本数：", dict(sorted(cnt.items())))


if __name__ == "__main__":
    main()
