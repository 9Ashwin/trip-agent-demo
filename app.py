"""
家庭旅行规划 Agent —— 后端服务（Flask）

职责：
  1. 托管前端页面（static/index.html）
  2. 提供 /api/plan 接口：接收出行需求，跑 Agent（联网搜索 + 大模型），
     用 SSE 把「正在搜什么」实时推给前端
  3. 保管 API Key —— Key 只存在后端，前端永远看不到

启动：
  python app.py
  打开 http://127.0.0.1:5050

注意：macOS 上 5000 端口被 AirPlay 接收器（ControlCenter）占用，
表现为 POST 请求返回 403，所以这里默认用 5050。
"""

import json
import os
import threading
import time
import traceback

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__, static_folder=STATIC_DIR)


def is_mock_mode() -> bool:
    """没配模型 Key → 自动进入演示模式，保证一定能看到效果。"""
    if os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        return True
    return not os.getenv("LLM_API_KEY")


# ---------------------------------------------------------------------------
# Agent 实现切换
#   langchain（默认）—— LangChain v1 的 create_agent，见 agent_langchain.py
#   native          —— 原生 OpenAI SDK 手写循环，见 agent.py
# 两者 run_agent(payload, emit) 签名与返回完全一致，可随时互换。
# ---------------------------------------------------------------------------
AGENT_IMPL = os.getenv("AGENT_IMPL", "langchain").lower()


# SSE 心跳间隔（秒）。0 = 关闭。
# 部署环境前面通常有反向代理，空闲太久会掐连接；这里定期发注释行保活。
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "10"))


# 最近一次失败的信息，供 /api/debug 查看。
# 线上沙箱没法看日志，把异常留在内存里是唯一能远程拿到原因的办法。
# 实现放在 errlog.py，agent 模块也会往里写，这里只做转发。
import errlog


def _record_error(e: BaseException):
    errlog.record(e, where="app.py")


def agent_module():
    if AGENT_IMPL == "native":
        import agent as mod
    else:
        import agent_langchain as mod
    return mod


# ---------------------------------------------------------------------------
# 访问频率限制
#
# **默认关闭（0）**。想开启就设 RATE_LIMIT_PER_HOUR=20。
#
# 开启后任何拿到链接的人都会被限流，适合「把带 Key 的版本公开出去」的场景；
# 给朋友试用 / 自己调试时反而碍事，所以出厂是关的。
#
# 实现说明：进程内存计数，单实例够用；多实例部署需换 Redis。
# ---------------------------------------------------------------------------
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "0"))
_rate_hits: dict = {}
_rate_lock = threading.Lock()


def client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "unknown")


def rate_limited() -> bool:
    if RATE_LIMIT <= 0:
        return False
    now = time.time()
    ip = client_ip()
    with _rate_lock:
        hits = [t for t in _rate_hits.get(ip, []) if now - t < 3600]
        if len(hits) >= RATE_LIMIT:
            _rate_hits[ip] = hits
            return True
        hits.append(now)
        _rate_hits[ip] = hits
        # 顺手清理过期条目，避免内存无限增长
        if len(_rate_hits) > 500:
            for k in [k for k, v in _rate_hits.items() if not any(now - t < 3600 for t in v)]:
                _rate_hits.pop(k, None)
    return False


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify(
        {
            "mock": is_mock_mode(),
            "model": os.getenv("LLM_MODEL", "deepseek-chat"),
            "search": os.getenv("SEARCH_PROVIDER", "bing"),
            "has_llm_key": bool(os.getenv("LLM_API_KEY")),
            "has_search_key": bool(
                os.getenv("TAVILY_API_KEY") or os.getenv("BOCHA_API_KEY")
            ),
            "search_keyless": os.getenv("SEARCH_PROVIDER", "bing").lower()
            in ("bing", "auto", "duckduckgo"),
            "agent_impl": AGENT_IMPL,
            "rate_limit": RATE_LIMIT,
        }
    )


@app.route("/api/debug")
def debug():
    """线上排查用：告出运行时环境与最近一次异常。

    只暴露版本号与异常信息，不含任何 Key。
    """
    import platform
    import sys
    from importlib.metadata import version as _pkg_version

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "agent_impl": AGENT_IMPL,
        "rate_limit": RATE_LIMIT,
    }

    for pkg in ("langchain", "langchain-core", "langgraph", "langchain-openai", "openai"):
        try:
            info[pkg] = _pkg_version(pkg)
        except Exception as e:
            info[pkg] = f"未安装/取版本失败: {type(e).__name__}"

    # 真正跑一次模块加载，这才等价于 /api/plan 的路径
    try:
        info["agent_module"] = agent_module().__name__
        info["agent_load"] = "ok"
    except Exception as e:
        info["agent_module"] = None
        info["agent_load"] = f"{type(e).__name__}: {e}"

    # 再往前一步：连 create_agent 能不能建起来也试一下
    try:
        from langchain.agents import create_agent

        info["create_agent"] = "可导入" if callable(create_agent) else "不可调用"
    except Exception as e:
        info["create_agent"] = f"{type(e).__name__}: {e}"

    info["heartbeat_seconds"] = HEARTBEAT_SECONDS
    info["max_search_rounds"] = os.getenv("MAX_SEARCH_ROUNDS", "(默认 15)")
    info["last_error"] = errlog.last()
    return jsonify(info)


@app.route("/api/plan", methods=["POST"])
def plan():
    if rate_limited():
        return (
            jsonify(
                {
                    "error": "rate_limited",
                    "message": f"请求太频繁了，每小时限 {RATE_LIMIT} 次，请稍后再试。",
                }
            ),
            429,
        )

    payload = request.get_json(force=True, silent=True) or {}

    def generate():
        queue = []
        started = time.time()
        last_ping = time.time()

        def emit(ev):
            queue.append(ev)

        result = {}

        def work():
            try:
                if is_mock_mode():
                    import mock_agent

                    result["md"], result["src"] = mock_agent.run_mock(payload, emit)
                else:
                    result["md"], result["src"] = agent_module().run_agent(payload, emit)
            except Exception as e:
                traceback.print_exc()
                result["err"] = str(e)
                # 留在内存里，便于线上排查（见 /api/debug）
                _record_error(e)
            finally:
                result["finished"] = True

        threading.Thread(target=work, daemon=True).start()

        while not result.get("finished"):
            if queue:
                out, queue[:] = list(queue), []
                for ev in out:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

            # SSE 心跳：长时间没有事件时（比如模型正在长思考 / 写最终方案），
            # 用注释行保活。注释行以 ":" 开头，按 SSE 规范会被客户端忽略，
            # 前端也只处理 "data:" 开头的行，所以不会污染 UI。
            # 没有这个的话，中间的反向代理会按「空闲超时」把连接掐掉。
            if time.time() - last_ping > HEARTBEAT_SECONDS:
                last_ping = time.time()
                yield ": ping\n\n"

            time.sleep(0.1)

        for ev in queue:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        if result.get("err"):
            yield f"data: {json.dumps({'type': 'error', 'message': result['err']}, ensure_ascii=False)}\n\n"
        else:
            done = {
                "type": "done",
                "markdown": result.get("md", ""),
                "sources": result.get("src", []),
                "elapsed": round(time.time() - started, 1),
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    # 监听 0.0.0.0，否则部署到线上后外部访问不到
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5050"))
    print("\n" + "=" * 56)
    print("  家庭旅行规划 Agent —— Demo 已启动")
    print("=" * 56)
    print(f"  本机访问：  http://127.0.0.1:{port}")
    mode = "演示模式（无需 API Key）" if is_mock_mode() else "真实模式（已接入大模型）"
    print(f"  运行模式：  {mode}")
    if not is_mock_mode():
        print(f"  模型：      {os.getenv('LLM_MODEL', 'deepseek-chat')}")
        print(f"  搜索提供方：{os.getenv('SEARCH_PROVIDER', 'bing')}")
        print(f"  Agent 实现：{AGENT_IMPL}")
    print("=" * 56 + "\n")
    app.run(host=host, port=port, debug=False, threaded=True)
