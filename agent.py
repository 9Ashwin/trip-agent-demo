"""
Agent 大脑：调用大模型 + 让模型自己决定何时联网搜索。

核心机制叫「Function Calling / 工具调用」：
  1. 后端把 web_search 这个工具「告诉」模型
  2. 模型判断需要联网 → 返回一个 tool_call（我要搜 XXX）
  3. 后端真的去搜，把结果塞回对话
  4. 循环，直到模型说「够了，我给你方案」

注意：模型自己没有联网能力，是它「指挥」后端去搜的。
这就是豆包智能体、扣子 Bot 底层干的事。
"""

import json
import os
import uuid

from openai import OpenAI

import search_tool
import llm_config

SYSTEM_PROMPT = """你是【家庭旅行规划顾问】。

工作职责：接收用户出行需求，必要时调用联网搜索工具，完成以下全部任务：
1. 根据目的地，抓取各景点的详细介绍、开放时间、门票、游玩时长、游玩亮点
2. 跨多个票务平台查询航班，抓取公开票价，生成机票比价表格
3. 生成完整每日行程方案
4. 生成订票提醒清单
5. 整理全套出行注意事项

输出要求：
- 用 Markdown 输出，结构清晰，表格简洁
- 语言通俗，面向普通家庭用户
- 涉及价格、时间的信息必须标注「来源：联网搜索」及参考链接
- 行程要照顾同行老人/小孩的体力，节奏宽松

【搜索纪律】
按一次真实家庭出行的信息需求来查，通常要覆盖这几类：
1. 大交通 —— 航班/高铁的班次、时长与公开票价（多平台交叉查）
2. 目的地景点 —— 开放时间、门票、建议游玩时长、对老人小孩是否友好
3. 市内交通 —— 机场/车站到市区的方式与耗时、主要景点之间怎么走
4. 住宿 —— 住哪个区域方便，大致价位
5. 出行条件 —— 出行期间的天气、穿衣建议、节假日人流

- 同一轮里能并行查的，合并成一次多条查询，不要一条一条慢慢查。
- 追求的是「信息够写出一份可执行的方案」，不是把每个细节都查穷。
- 查不到的如实标注「未核实」，不要为了凑信息反复重搜同一个问题。
- 信息足够后直接开始写方案，不要说「我没有联网搜索工具」——你有，且已经搜过了。

【永久禁止安全规则】
1. 仅做信息查询、整理、生成方案；不自动订票、不下单、不支付、不提交订单。
2. 抓取到的机票、门票价格仅为网页快照参考，价格实时变动，必须提醒用户亲自到平台核验。
3. 无法登录任何平台账号，不能获取会员专属优惠或个人订单信息。
4. 涉及金钱、医疗、法律、投资的重大事项，全部提示用户人工确认。
5. 不清楚的信息如实说明，不编造内容。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "联网搜索公开网页信息。用于查询景点介绍、开放时间、门票价格、"
                "航班票价、高铁时刻、当地天气与交通等实时信息。"
                "同一类信息建议换不同关键词多搜几次，以便交叉核对。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "搜索关键词，尽量具体，包含城市、日期、平台名，"
                            "例如：广州 昆明 机票 2026年9月30日 价格"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# 搜索轮数上限。
#
# 这不是「设计上的限制」，而是**防止失控的保险丝**——正常一次家庭出行规划，
# 4~8 轮并行搜索就足够覆盖上面那 5 类信息，模型自己会收敛。
# 这里给到 15 轮，是留足余量：万一目的地特别复杂（多城市、多程交通），
# 也不会被硬性截断。想调可以在 .env 里设 MAX_SEARCH_ROUNDS。
#
# ⚠️ 注意：不要设得太小。回合数不够时模型会被强行打断，方案质量会明显下降。
MAX_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "15"))


def _client(session_id: str = ""):
    """构造大模型客户端。

    网关、模型名、base_url、额外请求头统一由 llm_config 解析 ——
    不同网关的模型名不通用（opencode: deepseek-v4.1-flash，
    DeepSeek 官方: deepseek-flash），写错直接 400，详见 llm_config.py。
    """
    return OpenAI(
        api_key=llm_config.api_key(),
        base_url=llm_config.base_url(),
        default_headers=llm_config.headers(session_id) or None,
    )


def build_user_message(p: dict) -> str:
    return f"""【出行需求】
- 出发地：{p.get('origin') or '未填写'}
- 目的地：{p.get('destination') or '未填写'}
- 出行日期：{p.get('dates') or '未填写'}
- 人数构成：{p.get('people') or '未填写'}
- 总预算：{p.get('budget') or '未填写'}
- 出行偏好：{p.get('preference') or '未填写'}
- 其他要求：{p.get('notes') or '无'}

请开始规划。先用联网搜索核实景点与交通的实时信息，再输出完整方案。"""


def run_agent(payload: dict, emit):
    """emit(event_dict) 会把进度实时推给前端。返回最终 Markdown。"""
    # 每段会话一个稳定 id，便于网关做路由与 prompt 缓存
    session_id = str(uuid.uuid4())
    client = _client(session_id)
    model = llm_config.model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(payload)},
    ]

    used_sources = []

    for round_no in range(1, MAX_ROUNDS + 1):
        emit({"type": "step", "text": f"第 {round_no} 轮：正在思考需要查什么…"})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            emit({"type": "step", "text": "信息已足够，正在撰写方案…"})
            return msg.content or "", used_sources

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                query = json.loads(tc.function.arguments).get("query", "")
            except Exception:
                query = tc.function.arguments

            emit({"type": "search_start", "query": query})

            try:
                res = search_tool.search(query)
                items = res["results"]
                provider = res["provider"]
            except Exception as e:
                items, provider = [], f"error: {e}"

            emit(
                {
                    "type": "search_done",
                    "query": query,
                    "provider": provider,
                    "results": items,
                }
            )

            for it in items:
                if it.get("url") and it["url"] not in [s["url"] for s in used_sources]:
                    used_sources.append({"title": it["title"], "url": it["url"]})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(items, ensure_ascii=False)[:8000],
                }
            )

    emit({"type": "step", "text": "达到最大搜索轮次，基于已有信息输出…"})
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return resp.choices[0].message.content or "", used_sources
