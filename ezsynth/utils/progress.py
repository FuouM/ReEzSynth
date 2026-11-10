"""Progress reporting hook for long-running synthesis stages.

Hosts such as ComfyUI install a callback with :func:`progress_hook`; library code
calls :func:`report_progress` at natural checkpoints (one pyramid level, one frame).
Without an installed hook :func:`report_progress` does nothing.

Exceptions raised by the hook propagate to the caller unchanged; a host uses that to
abort a run (e.g. ComfyUI's interrupt).

The hook is stored in a :class:`contextvars.ContextVar`, so it applies to the
current thread / async context only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressCallback = Callable[[str, int, int], None]

_hook: ContextVar[ProgressCallback | None] = ContextVar(
    "ezsynth_progress_hook", default=None
)


@contextmanager
def progress_hook(callback: ProgressCallback) -> Iterator[None]:
    """Install ``callback(stage, current, total)`` for the current context."""
    token = _hook.set(callback)
    try:
        yield
    finally:
        _hook.reset(token)


def report_progress(stage: str, current: int, total: int) -> None:
    """Call the installed hook with ``current`` of ``total`` units of ``stage`` done."""
    callback = _hook.get()
    if callback is not None:
        callback(stage, current, total)
