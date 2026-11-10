# FastBlend: JIT loader for the small CUDA extension under ``../fastblend_extension/``
# (distinct from ``ezsynth/ebsynth_extension`` used by ``ezsynth`` / ``ebsynth_torch``).

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.utils.cpp_extension

from ezsynth.utils.ebsynth_jit_platform import ensure_ninja_on_path

logger = logging.getLogger(__name__)

MODULE_NAME = "fastblend_extension_cuda"

_ext_dir = Path(__file__).resolve().parent.parent / "fastblend_extension"
_source_files_str = [
    str(_ext_dir / "extension.cpp"),
    str(_ext_dir / "dispatch.cpp"),
    str(_ext_dir / "kernels.cu"),
]
# kernels.cu includes no PyTorch headers, so nvcc can compile it as C++17 (PyTorch
# adds -std=c++20 first; nvcc keeps the last -std). Host .cpp files keep PyTorch's
# own standard. CUDA architectures come from TORCH_CUDA_ARCH_LIST or the local GPU.
_EXTRA_CUDA_CFLAGS = ["-std=c++17"]

_cached: Any | None = None
_load_attempted = False


def load_fastblend_extension():
    """
    JIT-load the CUDA extension once.

    Returns ``None`` when PyTorch has no CUDA device (no compilation attempted).
    Raises :class:`RuntimeError` if compilation or load fails (fail-fast).
    """
    global _cached, _load_attempted
    if _load_attempted:
        return _cached
    if not torch.cuda.is_available():
        _load_attempted = True
        _cached = None
        return None
    verbose = os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes")
    if verbose:
        logger.info(
            f"Attempting to JIT compile and load CUDA extension '{MODULE_NAME}'..."
        )
    _load_attempted = True
    ensure_ninja_on_path()
    try:
        _cached = torch.utils.cpp_extension.load(
            name=MODULE_NAME,
            sources=_source_files_str,
            extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
            verbose=verbose,
        )
    except Exception as e:
        _cached = None
        if verbose or os.getenv("FASTBLEND_LOG_CUDA_JIT_FAILURE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.error("=" * 50)
            logger.error(
                f"[ERROR] Failed to JIT compile the CUDA extension '{MODULE_NAME}'."
            )
            logger.error(
                "Please ensure you have a compatible C++ compiler and CUDA Toolkit."
            )
            logger.error(f"Error details: {e}")
            logger.error("=" * 50)
        raise RuntimeError(
            "FastBlend CUDA extension JIT compile/load failed "
            "(set JIT_VERBOSE=1 or FASTBLEND_LOG_CUDA_JIT_FAILURE=1 for logs)."
        ) from e
    if verbose:
        logger.info("FastBlend CUDA extension loaded successfully via JIT compilation.")
    return _cached
