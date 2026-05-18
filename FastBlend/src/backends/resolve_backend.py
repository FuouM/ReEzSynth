"""Pick a concrete patch-matching backend without loading unused stacks."""

import platform

import torch

from .cuda_extension import is_available
from .fb_availability import cupy_available, taichi_available


def _cuda_extension_usable() -> bool:
    """True if CUDA is present and the JIT extension loaded; False if JIT failed."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(is_available())
    except RuntimeError:
        return False


def resolve_fastblend_backend(requested: str) -> str:
    """
    Return ``cuda``, ``cupy``, or ``taichi``. JIT-compiles the CUDA extension only
    when the resolved path may use CUDA (``cuda`` or ``auto`` when CUDA wins).
    """
    req = (requested or "auto").lower().strip()
    if req not in ("auto", "cuda", "cupy", "taichi"):
        raise ValueError(
            f"Unknown FastBlend backend {requested!r}; expected auto, cuda, cupy, or taichi"
        )
    if req == "cupy":
        if not cupy_available:
            raise ImportError("CuPy backend requested but CuPy is not available")
        return "cupy"
    if req == "taichi":
        if not taichi_available:
            raise ImportError("Taichi backend requested but Taichi is not available")
        return "taichi"
    if req == "cuda":
        if not is_available():
            raise ImportError(
                "CUDA backend requested but the FastBlend CUDA extension is not available"
            )
        return "cuda"
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and taichi_available
    ):
        return "taichi"
    if _cuda_extension_usable():
        return "cuda"
    if taichi_available:
        return "taichi"
    if cupy_available:
        return "cupy"
    raise ImportError("No FastBlend backend available")
