"""
联网搜索工具层。

这是整个 Demo 里最关键的 30 行 —— 也是「网页自己抓不了数据」这句话的答案：
真正联网的动作发生在这台后端机器上，不是浏览器里。

支持五种后端，按优先级自动选择：
  1. tavily   —— 专为 AI Agent 设计的搜索 API，免费额度 1000 次/月（推荐）
  2. bocha    —— 博查搜索，国内可直连，按次计费（国内首选）
  3. bing     —— 【免 API Key】抓取 cn.bing.com 搜索结果页，国内可用（详见下方说明）
  4. duckduckgo —— 免注册兜底，但国内网络基本连不上，详见说明
  5. mock     —— 离线演示，返回假数据，让朋友零成本先看效果

⚠️ 关于「免 Key 方案」的取舍（重要）
  真正"不用搜索 API"的可行做法只有「抓搜索结果页」，本项目内置了必应：

  必应（bing）：实测可用 ✅
    - cn.bing.com/search 直接返回 HTML，国内可访问，无需任何 Key
    - 解析 <li class="b_algo"> 块可稳定拿到 标题/URL/摘要，质量不错
    - 代价：① 属于网页抓取，对方改版就可能失效，无任何 SLA
            ② 高频调用会被限流或弹验证码
            ③ 不符合对方使用条款，别用于正式产品
    - 结论：学习、Demo、个人低频使用完全够用；线上服务请用 Tavily / 博查

  DuckDuckGo（duckduckgo）：实测不可用 ❌
    1) 网络：duckduckgo.com 与 api.duckduckgo.com 在中国大陆无法访问，
       实测直接返回连接失败（curl 得到 000）
    2) 接口性质：用的是 Instant Answer API —— 它不是搜索引擎，
       而是「百科摘要」接口：只返回某词条的 Abstract 和 RelatedTopics。
       对本项目这种查询（「广州 昆明 机票 9月30日 价格」）通常返回空结果。

  百度 / 维基百科：实测不可用（百度返回反爬页，维基被网络限制）
"""

import html
import os
import re

import requests

TIMEOUT = 20

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def search(query: str, max_results: int = 5):
    """统一入口：返回 {"provider": str, "results": [{"title","url","snippet"}]}"""
    provider = os.getenv("SEARCH_PROVIDER", "auto").lower()

    if provider in ("auto", "tavily") and os.getenv("TAVILY_API_KEY"):
        try:
            return _tavily(query, max_results)
        except Exception as e:
            if provider == "tavily":
                raise
            print(f"[search] tavily 失败，降级：{e}")

    if provider in ("auto", "bocha") and os.getenv("BOCHA_API_KEY"):
        try:
            return _bocha(query, max_results)
        except Exception as e:
            if provider == "bocha":
                raise
            print(f"[search] bocha 失败，降级：{e}")

    if provider in ("auto", "bing"):
        try:
            r = _bing(query, max_results)
            if r["results"]:
                return r
        except Exception as e:
            print(f"[search] bing 失败，降级：{e}")

    if provider in ("auto", "duckduckgo"):
        try:
            r = _duckduckgo(query)
            if r["results"]:
                return r
        except Exception as e:
            print(f"[search] duckduckgo 失败，降级：{e}")

    return _mock(query)


def _bing(query, max_results):
    """免 API Key：抓取必应搜索结果页并解析。"""
    r = requests.get(
        "https://cn.bing.com/search",
        params={"q": query, "setlang": "zh-CN", "count": max(max_results, 10)},
        headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()

    out = []
    for block in re.findall(r'<li class="b_algo".*?</li>', r.text, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        p = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        out.append(
            {
                "title": _strip_tags(m.group(2))[:120],
                "url": html.unescape(m.group(1)),
                "snippet": _strip_tags(p.group(1))[:400] if p else "",
            }
        )
        if len(out) >= max_results:
            break

    return {"provider": "bing", "results": out}


def _tavily(query, max_results):
    r = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "provider": "tavily",
        "results": [
            {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "snippet": (x.get("content") or "")[:400],
            }
            for x in data.get("results", [])
        ],
    }


def _bocha(query, max_results):
    r = requests.post(
        "https://api.bochaai.com/v1/web-search",
        headers={"Authorization": f"Bearer {os.environ['BOCHA_API_KEY']}"},
        json={"query": query, "count": max_results, "summary": True},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    pages = (r.json().get("data") or {}).get("webPages", {}).get("value", [])
    return {
        "provider": "bocha",
        "results": [
            {
                "title": p.get("name", ""),
                "url": p.get("url", ""),
                "snippet": (p.get("summary") or p.get("snippet") or "")[:400],
            }
            for p in pages[:max_results]
        ],
    }


def _duckduckgo(query):
    r = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    out = []
    if d.get("AbstractText"):
        out.append(
            {
                "title": d.get("Heading") or query,
                "url": d.get("AbstractURL", ""),
                "snippet": d["AbstractText"][:400],
            }
        )
    for t in d.get("RelatedTopics", [])[:4]:
        if isinstance(t, dict) and t.get("Text"):
            out.append(
                {
                    "title": t["Text"].split(" - ")[0][:60],
                    "url": t.get("FirstURL", ""),
                    "snippet": t["Text"][:400],
                }
            )
    return {"provider": "duckduckgo", "results": out}


def _mock(query):
    """离线演示：不联网，返回结构化假数据，保证 Demo 一定能跑通。"""
    return {
        "provider": "mock",
        "results": [
            {
                "title": f"[演示数据] {query}",
                "url": "https://example.com/demo",
                "snippet": (
                    "这是演示模式返回的模拟搜索结果。配置 TAVILY_API_KEY 或 "
                    "BOCHA_API_KEY 后，这里会变成真实的网页内容。"
                ),
            }
        ],
    }
