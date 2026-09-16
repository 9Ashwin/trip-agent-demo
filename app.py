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
# 访问频率限制
# 一旦把真实 Key 部署到公网，任何拿到链接的人都能消耗你的额度。
# 这里按 IP 做简单限流兜底（进程内存，单实例够用；多实例需换 Redis）。
# 设 RATE_LIMIT_PER_HOUR=0 可关闭。
# ---------------------------------------------------------------------------
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
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
            "rate_limit": RATE_LIMIT,
        }
    )


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

        def emit(ev):
            queue.append(ev)

        result = {}

        def work():
            try:
                if is_mock_mode():
                    import mock_agent

                    result["md"], result["src"] = mock_agent.run_mock(payload, emit)
                else:
                    import agent

                    result["md"], result["src"] = agent.run_agent(payload, emit)
            except Exception as e:
                traceback.print_exc()
                result["err"] = str(e)
            finally:
                result["finished"] = True

        threading.Thread(target=work, daemon=True).start()

        while not result.get("finished"):
            if queue:
                out, queue[:] = list(queue), []
                for ev in out:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
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
    print("=" * 56 + "\n")
    app.run(host=host, port=port, debug=False, threaded=True)
