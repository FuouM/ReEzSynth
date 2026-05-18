# FastBlend: JIT loader for the small CUDA extension under ``../fastblend_extension/``
# (distinct from repo-root ``ebsynth_extension`` used by ``ezsynth`` / ``ebsynth_torch``).

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.utils.cpp_extension

MODULE_NAME = "fastblend_extension_cuda"

_ext_dir = Path(__file__).resolve().parent.parent / "fastblend_extension"
_source_files_str = [
    str(_ext_dir / "extension.cpp"),
    str(_ext_dir / "dispatch.cu"),
    str(_ext_dir / "kernels.cu"),
]

_cached: Optional[Any] = None
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
        print(f"Attempting to JIT compile and load CUDA extension '{MODULE_NAME}'...")
    _load_attempted = True
    try:
        _cached = torch.utils.cpp_extension.load(
            name=MODULE_NAME,
            sources=_source_files_str,
            verbose=verbose,
        )
    except Exception as e:
        _cached = None
        if verbose or os.getenv("FASTBLEND_LOG_CUDA_JIT_FAILURE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            print("=" * 50)
            print(f"[ERROR] Failed to JIT compile the CUDA extension '{MODULE_NAME}'.")
            print("Please ensure you have a compatible C++ compiler and CUDA Toolkit.")
            print(f"Error details: {e}")
            print("=" * 50)
        raise RuntimeError(
            "FastBlend CUDA extension JIT compile/load failed "
            "(set JIT_VERBOSE=1 or FASTBLEND_LOG_CUDA_JIT_FAILURE=1 for logs)."
        ) from e
    if verbose:
        print("FastBlend CUDA extension loaded successfully via JIT compilation.")
    return _cached
