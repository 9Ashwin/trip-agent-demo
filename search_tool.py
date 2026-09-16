"""
联网搜索工具层。

这是整个 Demo 里最关键的 30 行 —— 也是「网页自己抓不了数据」这句话的答案：
真正联网的动作发生在这台后端机器上，不是浏览器里。

★ 默认方案：必应（bing）—— 完全不需要任何 API Key，一行配置即可联网。

支持四种后端，按优先级自动选择：
  1. bing     —— 【默认·免 API Key】抓取 cn.bing.com 搜索结果页，国内可用（推荐）
  2. tavily   —— 专为 AI Agent 设计的搜索 API，免费额度 1000 次/月（进阶可选，质量更高）
  3. bocha    —— 博查搜索，国内可直连，按次计费（进阶可选）
  4. mock     —— 离线演示，返回假数据，让朋友零成本先看效果

⚠️ 关于「免 Key 方案」的取舍（重要）
  真正"不用搜索 API"的可行做法只有「抓搜索结果页」，本项目内置了必应：

  必应（bing）：实测可用 ✅
    - cn.bing.com/search 直接返回 HTML，国内可访问，无需任何 Key
    - 解析 <li class="b_algo"> 块可稳定拿到 标题/URL/摘要，质量不错
    - 代价：① 属于网页抓取，对方改版就可能失效，无任何 SLA
            ② 高频调用会被限流或弹验证码
            ③ 不符合对方使用条款，别用于正式产品
    - 结论：学习、Demo、个人低频使用完全够用；线上服务请用 Tavily / 博查

  ❌ DuckDuckGo 已彻底移除（连降级分支一起删）：
     国内连不上（curl 得到 000），且免 Key 只能走 Instant Answer API ——
     那是「百科摘要」接口不是搜索引擎，对「广州 昆明 机票价格」这类查询返回空。
     留着它只会让人配了 duckduckgo 拿到空结果，还以为是模型不会搜。

  百度 / 维基百科：实测不可用（百度返回反爬页，维基被网络限制）

⚠️ 填了不认识的 SEARCH_PROVIDER（如残留的 duckduckgo）不会报错，
   而是**静默降级成 mock 演示假数据** —— 改完配置请用 /api/status
   确认 "search" 字段是不是 "bing"。
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
    """统一入口：返回 {"provider": str, "results": [{"title","url","snippet"}]}

    默认走免 Key 的必应；配置了 Tavily / 博查的 Key 时会优先用它们（质量更高）。
    """
    provider = os.getenv("SEARCH_PROVIDER", "bing").lower()

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


def _mock(query):
    """离线演示：不联网，返回结构化假数据，保证 Demo 一定能跑通。"""
    return {
        "provider": "mock",
        "results": [
            {
                "title": f"[演示数据] {query}",
                "url": "https://example.com/demo",
                "snippet": (
                    "这是演示模式返回的模拟搜索结果。默认方案（免 Key 必应）无需任何配置，"
                    "只要能联网就会自动返回真实网页内容。"
                ),
            }
        ],
    }
