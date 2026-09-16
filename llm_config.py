"""
大模型接入配置 —— 统一解析「用哪家网关、用哪个模型名、要不要额外请求头」。

为什么单独抽一个模块：
  两个 Agent 实现（agent_langchain.py / agent.py）都要建模型客户端，
  而**不同网关的模型名并不通用，同一把 Key 也不能跨网关用**。
  把解析逻辑集中在一处，避免两边各写一份、慢慢跑偏。

支持的网关（环境变量 LLM_PROVIDER）：

  opencode —— OpenCode Go 订阅网关（$10/月，编码模型套餐）
      base_url : https://opencode.ai/zen/go/v1
      模型名   : deepseek-v4.1-flash
      额外要求 : 自定义 User-Agent（不能用通用 SDK 名）
                 + 每段会话一个稳定的 x-opencode-session 头

  deepseek —— DeepSeek 官方 API（按量计费）
      base_url : https://api.deepseek.com/v1
      模型名   : deepseek-flash / deepseek-v4-pro（官方**只认这两个**）
      额外要求 : 无

⚠️ 最容易踩的坑：**两家的模型名不通用**
     opencode 叫 deepseek-v4.1-flash
     deepseek 官方叫 deepseek-flash
   名字写错不会降级、不会报错提示「你要的是不是 xxx」，
   而是直接 400 invalid_request_error：
     "The supported API model names are deepseek-flash, deepseek-v4-pro,
      but you passed deepseek-v4.1-flash."
   所以这里把「网关 → 默认模型名」绑死，别手写。

任何一项都可以用环境变量单独覆盖（留空则用上表默认值）：
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL / LLM_USER_AGENT / LLM_SESSION_HEADER
"""

import os

# 网关 → 默认参数。key_env 是按优先级尝试的 API Key 环境变量名。
PROVIDERS = {
    "opencode": {
        "label": "OpenCode Go",
        "base_url": "https://opencode.ai/zen/go/v1",
        "model": "deepseek-v4.1-flash",
        "user_agent": "trip-agent-demo/1.0",
        "session_header": "x-opencode-session",
        "key_env": ("OPENCODE_API_KEY", "LLM_API_KEY"),
    },
    "deepseek": {
        "label": "DeepSeek 官方",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "user_agent": "",
        "session_header": "",
        "key_env": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
    },
}

DEFAULT_PROVIDER = "opencode"


def provider_name() -> str:
    """当前网关名。填了不认识的值就回落到默认，不抛异常（配置错不该让页面挂掉）。"""
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    return p if p in PROVIDERS else DEFAULT_PROVIDER


def _first_env(names):
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return ""


def api_key() -> str:
    """按当前网关取 Key；网关专属变量优先，其次回落到通用的 LLM_API_KEY。"""
    return _first_env(PROVIDERS[provider_name()]["key_env"])


def has_key() -> bool:
    """没配 Key 就该走演示模式，而不是发一个必然 401 的请求出去。"""
    return bool(api_key())


def model() -> str:
    return (os.getenv("LLM_MODEL") or "").strip() or PROVIDERS[provider_name()]["model"]


def base_url() -> str:
    return (os.getenv("LLM_BASE_URL") or "").strip() or PROVIDERS[provider_name()]["base_url"]


def headers(session_id: str = "") -> dict:
    """网关要求的额外请求头。不需要的网关返回空 dict。"""
    p = PROVIDERS[provider_name()]
    out = {}
    ua = (os.getenv("LLM_USER_AGENT") or "").strip() or p["user_agent"]
    if ua:
        out["User-Agent"] = ua
    header_name = (os.getenv("LLM_SESSION_HEADER") or "").strip() or p["session_header"]
    if header_name:
        # 会话 id 为空时兜一个随机值：某些网关缺少这个头会直接拒绝
        out[header_name] = session_id or os.urandom(16).hex()
    return out


def summary() -> dict:
    """/api/status 与 /api/debug 用，不含任何 Key。"""
    name = provider_name()
    return {
        "provider": name,
        "provider_label": PROVIDERS[name]["label"],
        "model": model(),
        "base_url": base_url(),
        "has_key": has_key(),
    }
