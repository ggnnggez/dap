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


def test_subprocess_executor_defaults_to_spawn():
    """Lock in the default context so a future change doesn't silently
    revert to fork and reintroduce cross-platform behavior drift."""
    ex = SubprocessExecutor()
    assert ex._ctx.get_start_method() == "spawn"


def test_subprocess_executor_spawn_explicit_success():
    """Even with explicit spawn, a top-level tool func round-trips cleanly."""
    ex = SubprocessExecutor(timeout_s=5.0, mp_context="spawn")
    assert ex.execute(_tool(_add), {"a": 10, "b": 32}) == 42


def test_subprocess_executor_spawn_timeout():
    ex = SubprocessExecutor(timeout_s=0.3, mp_context="spawn")
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        ex.execute(_tool(_sleep_forever), {"seconds": 10})
    # spawn has higher startup cost than fork; 4s leaves room without
    # masking a hang.
    assert time.monotonic() - t0 < 4.0


def test_subprocess_executor_closes_pipes_on_start_failure():
    """Pickling an unpicklable closure fails inside proc.start() under spawn.
    The executor must close both pipe ends before re-raising, and remain
    reusable for subsequent calls (proxy for 'no fd leak')."""
    ex = SubprocessExecutor(timeout_s=5.0, mp_context="spawn")

    def _local_closure() -> int:  # unpicklable under spawn
        return 1

    bad_tool = Tool(
        name="bad", description="", parameters={"type": "object"}, func=_local_closure
    )

    with pytest.raises(Exception):  # AttributeError / PicklingError / TypeError
        ex.execute(bad_tool, {})

    # Hammer it: if pipes leaked, fd exhaustion would eventually bite.
    for _ in range(20):
        with pytest.raises(Exception):
            ex.execute(bad_tool, {})

    # Executor must still be healthy for valid calls.
    assert ex.execute(_tool(_add), {"a": 1, "b": 2}) == 3
