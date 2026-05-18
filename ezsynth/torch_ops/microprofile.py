"""Optional fine-grained timing for PyTorch synthesis hot paths."""

from __future__ import annotations

import atexit
import contextlib
import time
from collections import defaultdict
from typing import Optional

import torch

from ezsynth import consts as _ez_consts

_STATS: defaultdict[str, float] = defaultdict(float)
_COUNTS: defaultdict[str, int] = defaultdict(int)
_ATEXIT_REGISTERED = False


def _mode_value() -> str:
    return str(_ez_consts.TORCH_MICROPROFILE).strip().lower()


def enabled() -> bool:
    v = _mode_value()
    return v in ("1", "true", "yes", "sync", "2", "gpu")


def sync_mode() -> bool:
    return _mode_value() in ("sync", "2", "gpu")


def _sync_device(device: Optional[torch.device]) -> None:
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def reset() -> None:
    _STATS.clear()
    _COUNTS.clear()


def _print_report() -> None:
    if not _STATS:
        return
    total = sum(_STATS.values())
    print("\n--- TORCH_MICROPROFILE (wall seconds) ---", flush=True)
    if total <= 0:
        return
    sm = sync_mode()
    print(
        f"  mode: {'device-sync per region' if sm else 'perf_counter only (GPU times misleading without sync)'}",
        flush=True,
    )
    print(
        "  note: nested regions overlap; percentages are relative to summed row times.",
        flush=True,
    )
    rows = sorted(_STATS.items(), key=lambda kv: kv[1], reverse=True)
    for name, sec in rows:
        n = _COUNTS[name]
        pct = 100.0 * sec / total
        print(f"  {name:<40} {sec:10.4f}s  ({pct:5.1f}%)  n={n}", flush=True)
    print(f"  {'TOTAL':<40} {total:10.4f}s", flush=True)


def _ensure_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_print_report)
        _ATEXIT_REGISTERED = True


@contextlib.contextmanager
def region(name: str, device: Optional[torch.device] = None):
    if not enabled():
        yield
        return
    _ensure_atexit()
    do_sync = sync_mode()
    if do_sync:
        _sync_device(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if do_sync:
            _sync_device(device)
        dt = time.perf_counter() - t0
        _STATS[name] += dt
        _COUNTS[name] += 1
