"""Which compute engines the cross-backend parity tests can run on this machine.

``engine_param(name)`` is a ``pytest.param`` that skips with the reason when the
engine is unavailable. The JIT-built ``cuda`` extensions take minutes to compile on
the first run, so they run only with ``EZSYNTH_TEST_JIT=1``.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache

import pytest


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@cache
def _cuda_device() -> bool:
    import torch

    return torch.cuda.is_available()


@cache
def _wgpu_unavailable() -> str | None:
    if not _installed("ezsynth_wgpu"):
        return "ezsynth_wgpu not installed"
    import ezsynth_wgpu

    try:
        ezsynth_wgpu.adapter_info()
    except RuntimeError as exc:
        return f"no WebGPU adapter: {exc}"
    return None


def unavailable(engine: str) -> str | None:
    """Why ``engine`` cannot run here, or ``None``."""
    if engine in ("opencv", "torch", "numpy"):
        return None
    if engine == "taichi":
        return None if _installed("taichi") else "taichi not installed"
    if engine == "wgpu":
        return _wgpu_unavailable()
    if engine == "cupy":
        if not _installed("cupy"):
            return "cupy not installed"
        return None if _cuda_device() else "no CUDA device"
    if engine == "cuda":
        if not _cuda_device():
            return "no CUDA device"
        if os.environ.get("EZSYNTH_TEST_JIT") != "1":
            return (
                "JIT-built CUDA extension; set EZSYNTH_TEST_JIT=1 to build and test it"
            )
        return None
    raise ValueError(f"unknown engine {engine!r}")


def engine_param(engine: str, *marks) -> pytest.param:
    reason = unavailable(engine)
    skip = [pytest.mark.skip(reason=reason)] if reason else []
    return pytest.param(engine, marks=[*skip, *marks], id=engine)


def engine_params(*engines: str) -> list:
    return [engine_param(e) for e in engines]
