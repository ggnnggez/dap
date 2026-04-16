from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from dap.trace.events import EventKind, TraceEvent


class Tracer(Protocol):
    def emit(self, event: TraceEvent) -> None: ...
    def close(self) -> None: ...


class JsonlTracer:
    """Append-only JSONL trace. One event per line, flushed on every emit."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def emit(self, event: TraceEvent) -> None:
        self._fp.write(json.dumps(event.to_dict(), default=_json_default) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "JsonlTracer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class NullTracer:
    def emit(self, event: TraceEvent) -> None:
        return

    def close(self) -> None:
        return


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return repr(obj)


__all__ = ["Tracer", "JsonlTracer", "NullTracer", "TraceEvent", "EventKind"]
