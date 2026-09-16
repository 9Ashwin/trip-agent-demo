"""
Agent 大脑（LangChain 版）。

与 agent.py（原生 OpenAI SDK 版）**对外契约完全一致**：
    run_agent(payload: dict, emit) -> (markdown: str, sources: list[dict])
所以 app.py 可以无缝切换，前端一行都不用改。

架构对照（原生版 → LangChain 版）：

    手写 for 循环 + client.chat.completions.create   →   create_agent(...) 生成的 LangGraph 图
    手写 TOOLS JSON Schema                           →   @tool 装饰器自动生成 Schema
    手写 messages.append / role="tool"               →   框架自动维护 state["messages"]
    手写「没有 tool_calls 就结束」                   →   框架自动判断停止条件

核心机制没变，还是 Function Calling：
  后端把 web_search 工具「注册」给模型 → 模型决定要搜什么 → 框架真的去调这个函数
  → 结果写回对话 → 循环到模型不再调工具为止。

用的 API 是 LangChain v1 的 `create_agent`（旧版 `AgentExecutor` /
`create_tool_calling_agent` 在 v1 已移除，迁到 langchain-classic 包了）。

⚠️ 两个实测踩到的坑（已在下方处理，改动前请务必先读）：

  1. `create_agent` 没有 `max_iterations` 参数。内置 AgentExecutor 时代可以直接限轮数，
     v1 只能在外部自己数。如果不数，模型会一直搜下去，最终撞上 LangGraph 的
     recursion_limit 抛 GraphRecursionError。

  2. 兜底分支**必须把已经搜到的历史一起喂回去**。踩过的真实事故：兜底时用一条全新的
     空消息去问模型，等于把前面几十次搜索结果全扔了，模型于是回答
     「当前对话环境未提供联网搜索工具」——用户看到一个「明明搜了却说自己不能联网」的方案。
"""

import asyncio
import json
import os
import threading
import uuid

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import search_tool
import errlog
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

# 搜索轮数上限。
#
# 这不是「设计上的限制」，而是**防止失控的保险丝**——正常一次家庭出行规划，
# 4~8 轮并行搜索就足够覆盖上面那 5 类信息，模型自己会收敛。
# 这里给到 15 轮，是留足余量：万一目的地特别复杂（多城市、多程交通），
# 也不会被硬性截断。
#
# 想调可以在 .env 里设 MAX_SEARCH_ROUNDS。
# ⚠️ 注意：不要设得太小。回合数不够时模型会被强行打断，方案质量会明显下降。
MAX_ROUNDS = int(os.getenv("MAX_SEARCH_ROUNDS", "15"))

# LangGraph 递归上限。每轮约 2 个 superstep（model + tools），给点余量。
# 这是最后一道保险丝；正常情况下 MAX_ROUNDS 会先一步把我们拦住。
RECURSION_LIMIT = MAX_ROUNDS * 2 + 6

TOOL_DESCRIPTION = (
    "联网搜索公开网页信息。用于查询景点介绍、开放时间、门票价格、"
    "航班票价、高铁时刻、当地天气与交通等实时信息。"
    "同一类信息建议换不同关键词多搜几次，以便交叉核对。"
)

# 收尾指令：搜索结束时逼模型输出正文
FINALIZE_HINT = (
    "搜索阶段已经结束（工具已被收回）。"
    "请立即基于以上已经获得的全部搜索结果，直接输出完整的 Markdown 方案。"
    "不要再要求调用任何工具，也不要声明自己没有联网能力。"
)


def _build_llm(session_id: str = ""):
    """构造 LangChain 的 Chat 模型。

    网关、模型名、base_url、额外请求头全部交给 llm_config 统一解析 ——
    因为**不同网关的模型名不通用**（opencode 是 deepseek-v4.1-flash，
    DeepSeek 官方是 deepseek-flash），写错会直接 400，见 llm_config.py 顶部说明。
    """
    return ChatOpenAI(
        api_key=llm_config.api_key(),
        base_url=llm_config.base_url(),
        model=llm_config.model(),
        temperature=0.3,
        default_headers=llm_config.headers(session_id) or None,
    )


# ---------------------------------------------------------------------------
# 常驻事件循环
#
# ⚠️ 这里是本项目最容易踩、也最难查的一个坑，改动前务必读完。
#
# 症状：同一个进程里，**第 1 个请求正常，之后每个请求都在 1 秒内失败**。
#       错误是 RuntimeError: Event loop is closed（在 model 任务里抛出）。
#       表现到用户那里就是「第一批人能用，后面统统转圈」。
#
# 原因：底层 httpx / openai SDK 的**异步客户端是按参数全局缓存**的，
#       它会把连接池绑在「第一次用到它的那个事件循环」上。
#       而 asyncio.run() 的语义是「新建一个 loop，跑完就 close」。
#       于是第 2 个请求复用了那个绑在已关闭 loop 上的客户端，立刻炸。
#
# 解法：整个进程只用一个常驻事件循环（跑在后台线程里），
#       请求通过 run_coroutine_threadsafe 提交上去。loop 永不关闭，
#       缓存客户端始终有效，同时也天然支持并发。
#
# 一句话：**这里绝对不要改回 asyncio.run()。**
# ---------------------------------------------------------------------------
_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP
    with _LOOP_LOCK:
        if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, daemon=True, name="agent-async-loop"
            ).start()
            _ASYNC_LOOP = loop
        return _ASYNC_LOOP


def _run_async(coro):
    """把协程提交到常驻事件循环并等结果。替代 asyncio.run()。"""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()


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


def _make_search_tool(emit, used_sources):
    """把 search_tool.search 包装成 LangChain 工具。

    用闭包而不是模块级 @tool，是因为每个请求都要绑定自己的 emit 与来源列表，
    模块级工具无法知道「这次是哪个用户在跑」。

    函数签名 (query: str) -> str 会被 @tool 自动转成 JSON Schema，
    不再需要手写那 24 行 TOOLS 字典。
    """

    @tool("web_search", description=TOOL_DESCRIPTION)
    def web_search(query: str) -> str:
        emit({"type": "search_start", "query": query})

        try:
            res = search_tool.search(query)
            items, provider = res["results"], res["provider"]
        except Exception as e:  # 搜索失败不应该炸掉整轮规划
            items, provider = [], f"error: {e}"

        emit(
            {
                "type": "search_done",
                "query": query,
                "provider": provider,
                "results": items,
            }
        )

        seen = {s["url"] for s in used_sources}
        for it in items:
            url = it.get("url")
            if url and url not in seen:
                seen.add(url)
                used_sources.append({"title": it.get("title", ""), "url": url})

        # 工具返回值就是「喂回给模型」的内容，截断避免上下文爆炸
        return json.dumps(items, ensure_ascii=False)[:8000]

    return web_search


def _text_of(msg) -> str:
    """把 AIMessage 的 content 统一取成字符串。

    推理模型可能把 token 全花在 reasoning 上，导致 content 为空 —— 这里保守处理。
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 多模态分块形式
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _sanitize_for_final(history: list) -> list:
    """把带工具调用的历史压成「纯对话」形式，供不挂工具的收尾调用使用。

    为什么要压：历史里有 assistant.tool_calls 和 ToolMessage。如果收尾时不再传 tools，
    部分 OpenAI 兼容网关会因为「有工具调用记录却没给工具定义」直接报错。
    这里把搜索结果转成普通人类消息，最稳。
    """
    out = []
    for m in history:
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, AIMessage):
            txt = _text_of(m)
            if txt:
                out.append(AIMessage(content=txt))
        else:  # ToolMessage
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append(
                HumanMessage(content=f"【此前联网搜索到的资料】\n{content[:4000]}")
            )
    return out


async def _force_final(llm, history: list) -> str:
    """不带工具再问一次，强制产出正文。

    关键点：**必须带上已经搜到的历史**，否则模型会以为自己在真空里，回答
    「我没有联网搜索工具」。
    """
    if not history:
        return ""
    try:
        resp = await llm.ainvoke(
            _sanitize_for_final(history) + [HumanMessage(content=FINALIZE_HINT)]
        )
        return _text_of(resp)
    except Exception as e:
        errlog.record(e, where="agent_langchain._force_final 收尾调用")
        return f"生成方案失败：{type(e).__name__}: {e}"


async def _arun(llm, user_msg: str, emit, used_sources, state: dict) -> str:
    web_search = _make_search_tool(emit, used_sources)

    agent = create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        name="trip_planner",
    )

    history = state["history"]          # 与 run_agent 共享，异常时也要能拿到
    final = ""
    round_no = 0
    stop_searching = False

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": user_msg}]},
        stream_mode="updates",
        config={"recursion_limit": RECURSION_LIMIT},
    ):
        for node, payload in (chunk or {}).items():
            msgs = (payload or {}).get("messages") or []
            if node == "model":
                for m in msgs:
                    history.append(m)
                    if getattr(m, "tool_calls", None):
                        round_no += 1
                        emit(
                            {
                                "type": "step",
                                "text": f"第 {round_no} 轮：正在思考需要查什么…",
                            }
                        )
                    else:
                        txt = _text_of(m)
                        if txt:
                            emit({"type": "step", "text": "信息已足够，正在撰写方案…"})
                            final = txt
            elif node == "tools":
                history.extend(msgs)

        # 轮数到顶：跳出图，改用不带工具的收尾调用，避免撞递归上限
        if round_no >= MAX_ROUNDS and not stop_searching:
            stop_searching = True
            break

    if not final:
        emit({"type": "step", "text": "搜索完成，正在基于已有信息撰写方案…"})
        final = await _force_final(llm, history)

    return final


def run_agent(payload: dict, emit):
    """emit(event_dict) 会把进度实时推给前端。返回 (最终 Markdown, 来源列表)。

    与 agent.py 的 run_agent 签名/返回完全一致，可互换。
    """
    used_sources: list = []
    user_msg = build_user_message(payload)

    retries = int(os.getenv("AGENT_RETRIES", "1"))
    markdown = ""

    # 最多尝试 1 + retries 次。
    #
    # 为什么要重试：实测模型网关（OpenCode Go）会偶发瞬时失败 ——
    # 表现为第一轮模型调用在 1 秒内抛异常，一次搜索都没跑，
    # 然后就会输出一份「凭空写的」方案（来源 0 个）。5 次里能撞上 2 次。
    # 这种情况下**原样重试一次**几乎都能成功，比直接降级成兜底好得多。
    for attempt in range(1, retries + 2):
        # 每次尝试用新的 session id：网关要求会话隔离，卡住的会话换一个更稳
        llm = _build_llm(str(uuid.uuid4()))
        state = {"history": [HumanMessage(content=user_msg)]}

        try:
            markdown = _run_async(_arun(llm, user_msg, emit, used_sources, state))
            break
        except Exception as e:
            print(f"[agent_langchain] 主循环异常(第 {attempt} 次)：{type(e).__name__}: {e}")
            # 线上看不到日志，记进 errlog 供 /api/debug 取
            errlog.record(e, where=f"agent_langchain._arun 主循环(第 {attempt} 次)")

            # 一次搜索都还没做 → 大概率是网关抖动，原样重试
            if attempt <= retries and not used_sources:
                emit({"type": "step", "text": "模型这次没响应，正在重试…"})
                continue

            # 已经搜到东西了就不重来（重来要再花几十秒），带着已有资料收尾
            emit({"type": "step", "text": "搜索中途中断，正在基于已获得的信息输出…"})
            markdown = _run_async(_force_final(llm, state["history"]))
            break

    if not markdown:
        markdown = "（模型没有返回内容，请重试一次）"

    return markdown, used_sources
