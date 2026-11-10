"""Single owner of the process-wide Taichi runtime.

ezsynth, FastBlend and FaceBlit Taichi backends all initialize Taichi through
:func:`ensure_ti_init`, so the runtime is initialized once per process and only
reset when a caller explicitly requests a different arch (e.g. CPU vs CUDA).

``taichi`` is imported lazily so importing this module stays cheap.

Verbosity: set ``EZSYNTH_TAICHI_INIT_VERBOSE=1`` (``consts.TAICHI_INIT_VERBOSE``) or
``FASTBLEND_TAICHI_INIT_VERBOSE=1`` to log the chosen arch at init time.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from typing import Any

import torch

from .. import consts

logger = logging.getLogger(__name__)

TAICHI_RANDOM_SEED = 42

_lock = threading.Lock()
_ti_arch: Any = None


def _device_type(device: torch.device | str | None) -> str | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    return str(device).lower().split(":", maxsplit=1)[0]


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_ti_arch(device: torch.device | str | None = None):
    """Taichi arch for a PyTorch-style ``device`` (``cpu`` / ``cuda`` / ``mps``).

    ``None`` picks the default: Metal on Apple Silicon, else CUDA when available,
    else CPU.
    """
    import taichi as ti

    kind = _device_type(device)
    if kind is None:
        if _is_apple_silicon():
            return ti.metal
        return ti.cuda if torch.cuda.is_available() else ti.cpu
    if kind == "cuda" and torch.cuda.is_available():
        return ti.cuda
    if kind == "mps" and _is_apple_silicon():
        return ti.metal
    return ti.cpu


def _verbose() -> bool:
    fastblend_flag = os.environ.get("FASTBLEND_TAICHI_INIT_VERBOSE", "")
    return consts.TAICHI_INIT_VERBOSE or fastblend_flag.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def ensure_ti_init(device: torch.device | str | None = None):
    """Initialize Taichi once for the process and return the active arch.

    With ``device=None`` an already-initialized runtime is kept as is; otherwise the
    default arch from :func:`get_ti_arch` is used. A ``device`` that maps to a
    different arch than the active one resets the runtime (``ti.reset()``) and
    re-initializes it on the new arch; Taichi fields allocated before the switch
    are released.
    """
    global _ti_arch
    import taichi as ti

    with _lock:
        if device is None and _ti_arch is not None:
            return _ti_arch
        arch = get_ti_arch(device)
        if _ti_arch == arch:
            return arch
        if _ti_arch is not None:
            ti.reset()
            _ti_arch = None
        verbose = _verbose()
        ti.init(
            arch=arch,
            log_level=ti.INFO if verbose else ti.ERROR,
            random_seed=TAICHI_RANDOM_SEED,
        )
        if verbose:
            logger.info(f"[Taichi] Initialized with arch: {arch}")
        _ti_arch = arch
        return arch


def get_taichi_torch_device() -> str:
    """PyTorch device string matching the active (or default) Taichi arch."""
    import taichi as ti

    arch = _ti_arch if _ti_arch is not None else get_ti_arch()
    if arch == ti.cuda:
        return "cuda"
    if (
        arch == ti.metal
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"
