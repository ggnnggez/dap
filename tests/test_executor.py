from __future__ import annotations

import time

import pytest

from dap.runtime.executor import DirectExecutor, SubprocessExecutor
from dap.runtime.tools import Tool


# Module-level funcs so the child process can import them (pickleability).
def _add(a: int, b: int) -> int:
    return a + b


def _sleep_forever(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _boom() -> None:
    raise ValueError("kaboom")


def _tool(func, name: str = "t") -> Tool:
    return Tool(name=name, description="", parameters={"type": "object"}, func=func)


def test_direct_executor_calls_func():
    assert DirectExecutor().execute(_tool(_add), {"a": 2, "b": 3}) == 5


def test_subprocess_executor_success():
    ex = SubprocessExecutor(timeout_s=5.0)
    assert ex.execute(_tool(_add), {"a": 10, "b": 32}) == 42


def test_subprocess_executor_timeout_kills_child():
    ex = SubprocessExecutor(timeout_s=0.3)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded"):
        ex.execute(_tool(_sleep_forever), {"seconds": 10})
    # must actually return soon after timeout, not hang on the full sleep
    assert time.monotonic() - t0 < 3.0


def test_subprocess_executor_propagates_exception():
    ex = SubprocessExecutor(timeout_s=5.0)
    with pytest.raises(RuntimeError, match="ValueError: kaboom"):
        ex.execute(_tool(_boom), {})
