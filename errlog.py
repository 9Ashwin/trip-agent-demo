"""
极简的错误记录器（进程内存）。

为什么需要它：线上部署后**看不到沙箱日志**，出了异常只能靠猜。
这个模块把最近一次异常连堆栈一起留在内存里，再用 `GET /api/debug` 取出来。

只保留最近一次就够了 —— 排查线上问题时，你最关心的就是「刚才那次为什么失败」。

用法：
    import errlog
    try:
        ...
    except Exception as e:
        errlog.record(e)      # 记下来
        raise / 兜底
"""

import time
import traceback

_LAST: dict = {}


def record(e: BaseException, where: str = ""):
    """记录一次异常（覆盖上一次）。只存类型/信息/时间/堆栈，不存任何 Key。"""
    _LAST.clear()
    _LAST.update(
        {
            "where": where or "unknown",
            "type": type(e).__name__,
            "message": str(e),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "traceback": "".join(traceback.format_exception(e))[-3000:],
        }
    )


def last() -> dict | None:
    """取最近一次异常，没有则返回 None。"""
    return dict(_LAST) or None


def clear():
    _LAST.clear()
