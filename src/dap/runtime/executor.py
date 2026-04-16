"""Tool execution substrates.

An executor is **not** a constraint — constraints observe and decide, whereas
an executor is the medium of execution itself. Sandbox-style isolation lives
here. A constraint at pre_tool can swap the executor for a given tool call
by emitting a MODIFY with payload `{"executor": <ToolExecutor>}`; the loop
then routes that one call through it.

Executors should be long-lived — instantiate once, mount once, reuse. Cheap
per-call executors (DirectExecutor) and expensive ones (container-backed)
share the same interface so the call site doesn't care.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Protocol

from dap.runtime.tools import Tool


class ToolExecutor(Protocol):
    def execute(self, tool: Tool, arguments: dict) -> Any: ...


class DirectExecutor:
    """In-process execution. Default. Zero isolation."""

    def execute(self, tool: Tool, arguments: dict) -> Any:
        return tool.func(**arguments)


class SubprocessExecutor:
    """Runs each tool call in a child process with a wall-clock timeout
    and optional virtual-memory cap.

    Defaults to the "spawn" multiprocessing context so behavior is identical
    on Linux, macOS, and Windows, and the child does not silently inherit
    parent state (open fds, loggers, threads, RNG seeds, ...). This imposes a
    pickle-first contract rather than depending on whatever the platform
    default happens to be.

    Constraints:
      - tool.func and arguments must be picklable. Top-level module
        functions qualify; lambdas, nested functions, closures, and
        interactive-`__main__` definitions do not. Spawn will raise at
        start() for unpicklable inputs; the executor closes both pipe
        ends and re-raises so callers can recover without fd leaks.
      - memory_mb (Linux/macOS only) applies via resource.setrlimit(RLIMIT_AS)
        inside the child.
      - On timeout raises TimeoutError; on child crash or uncaught
        exception raises RuntimeError. These propagate up to the
        AgentLoop's per-tool try/except and become string error results
        the model can see.
    """

    def __init__(
        self,
        timeout_s: float = 10.0,
        memory_mb: int | None = None,
        mp_context: str = "spawn",
    ) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self._ctx = mp.get_context(mp_context)

    def execute(self, tool: Tool, arguments: dict) -> Any:
        parent_conn, child_conn = self._ctx.Pipe(duplex=False)
        proc = self._ctx.Process(
            target=_sandbox_entry,
            args=(child_conn, tool.func, arguments, self.memory_mb),
        )
        try:
            proc.start()
        except BaseException:
            # start() can fail on pickling errors (spawn), resource limits,
            # or platform issues. Close both pipe ends before propagating
            # so repeated failures don't exhaust fds.
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()  # only the child holds the write end now

        try:
            if not parent_conn.poll(self.timeout_s):
                _kill(proc)
                raise TimeoutError(
                    f"tool {tool.name!r} exceeded {self.timeout_s}s wall time"
                )
            try:
                status, payload = parent_conn.recv()
            except EOFError as e:
                raise RuntimeError(
                    f"tool {tool.name!r} crashed before sending result "
                    f"(exitcode={proc.exitcode})"
                ) from e
        finally:
            parent_conn.close()
            proc.join(1.0)
            if proc.is_alive():
                _kill(proc)

        if status == "err":
            raise RuntimeError(f"sandboxed tool {tool.name!r} raised: {payload}")
        return payload


def _sandbox_entry(conn, func, arguments: dict, memory_mb: int | None) -> None:
    """Runs in the child process."""
    try:
        if memory_mb is not None:
            import resource

            byte_limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))
        result = func(**arguments)
        conn.send(("ok", result))
    except BaseException as e:  # include KeyboardInterrupt / SystemExit
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _kill(proc) -> None:
    proc.terminate()
    proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join()
