"""Process-wide CUDA memory budget.

ReEzSynth shares the GPU with the rest of the machine (browser, desktop compositor,
other apps). Before the first CUDA work on a device, the PyTorch caching allocator is
capped to a budget derived from the memory that is *free at that moment*, minus a
reserve left for everyone else. Exceeding the budget raises a regular CUDA OOM in this
process instead of starving the whole system.

Environment overrides (read when the budget is applied):

- ``EZSYNTH_GPU_MEMORY_LIMIT``: ``0``/``off``/``false`` disables the cap.
- ``EZSYNTH_GPU_MEMORY_LIMIT_MB``: absolute budget in MiB (still never above free - reserve).
- ``EZSYNTH_GPU_MEMORY_FRACTION``: fraction of free memory to use (default 0.9).
- ``EZSYNTH_GPU_MEMORY_RESERVE_MB``: MiB always left free for other processes (default 1536).

Scope: allocations made through PyTorch (including Taichi/CUDA-extension kernels that
operate on torch tensors). CUDA context overhead and Taichi/CuPy private pools are not
counted; the reserve covers them.
"""

from __future__ import annotations

import os
import threading

import torch

DEFAULT_FRACTION = 0.9
DEFAULT_RESERVE_MB = 1536
MIN_BUDGET_MB = 256
_MIB = 1 << 20

_lock = threading.Lock()
_applied: dict[int, int] = {}
_enabled = True


def disable_gpu_memory_limit() -> None:
    """Never cap the allocator in this process (for hosts that manage VRAM, e.g. ComfyUI)."""
    global _enabled
    _enabled = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def compute_budget_bytes(
    free_bytes: int,
    total_bytes: int,
    already_reserved_bytes: int = 0,
    *,
    fraction: float = DEFAULT_FRACTION,
    reserve_bytes: int = DEFAULT_RESERVE_MB * _MIB,
    limit_bytes: int | None = None,
) -> int:
    """Budget for this process: never more than what others leave free minus ``reserve_bytes``.

    Memory this process already holds counts as available to it.
    Raises ``RuntimeError`` when the result is below ``MIN_BUDGET_MB``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"GPU memory fraction must be in (0, 1], got {fraction}")
    available = free_bytes + already_reserved_bytes
    ceiling = available - reserve_bytes
    wanted = limit_bytes if limit_bytes is not None else int(available * fraction)
    budget = min(wanted, ceiling, total_bytes)
    if budget < MIN_BUDGET_MB * _MIB:
        raise RuntimeError(
            f"Only {available // _MIB} MiB GPU memory available "
            f"({reserve_bytes // _MIB} MiB kept free for other applications); "
            f"need at least {MIN_BUDGET_MB} MiB. Close other GPU applications, lower "
            "EZSYNTH_GPU_MEMORY_RESERVE_MB, or set EZSYNTH_GPU_MEMORY_LIMIT=0."
        )
    return budget


def apply_gpu_memory_limit(device: torch.device | str | None = None) -> int | None:
    """Cap the caching allocator on ``device`` once; returns the budget in bytes or None.

    No-op for non-CUDA devices, when CUDA is unavailable, or when disabled.
    """
    if not _enabled or not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if os.environ.get("EZSYNTH_GPU_MEMORY_LIMIT", "").strip().lower() in {
        "0",
        "off",
        "false",
        "no",
    }:
        return None

    with _lock:
        if index in _applied:
            return _applied[index]
        free, total = torch.cuda.mem_get_info(index)
        limit_mb = os.environ.get("EZSYNTH_GPU_MEMORY_LIMIT_MB", "").strip()
        budget = compute_budget_bytes(
            free,
            total,
            torch.cuda.memory_reserved(index),
            fraction=_env_float("EZSYNTH_GPU_MEMORY_FRACTION", DEFAULT_FRACTION),
            reserve_bytes=int(
                _env_float("EZSYNTH_GPU_MEMORY_RESERVE_MB", DEFAULT_RESERVE_MB) * _MIB
            ),
            limit_bytes=int(float(limit_mb) * _MIB) if limit_mb else None,
        )
        torch.cuda.set_per_process_memory_fraction(budget / total, index)
        _applied[index] = budget
        print(
            f"[GPU memory] cuda:{index} budget {budget // _MIB} MiB "
            f"(free {free // _MIB} / total {total // _MIB} MiB)"
        )
        return budget


def current_budget_bytes(device: torch.device | str | None = None) -> int | None:
    """Budget recorded by :func:`apply_gpu_memory_limit` for ``device``, else None.

    Read-only: never applies a cap and never initializes CUDA.
    """
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        return _applied.get(index)
