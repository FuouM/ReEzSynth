"""FastBlend on the JIT-built CUDA extension (sources in ``cuda_src/``).

This is FastBlend's own small extension (remap / patch_error / pairwise_patch_error),
separate from the ``kernels/synth/cuda_src`` synthesis extension.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.utils.cpp_extension

from ...utils.ebsynth_jit_platform import ensure_ninja_on_path
from .core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore

logger = logging.getLogger(__name__)

MODULE_NAME = "fastblend_extension_cuda"
_SRC = Path(__file__).resolve().parent / "cuda_src"
_SOURCES = [
    str(_SRC / name) for name in ("extension.cpp", "dispatch.cpp", "kernels.cu")
]
# kernels.cu includes no PyTorch headers, so nvcc can compile it as C++17 (PyTorch
# adds -std=c++20 first; nvcc keeps the last -std). Host .cpp files keep PyTorch's
# own standard. CUDA architectures come from TORCH_CUDA_ARCH_LIST or the local GPU.
_EXTRA_CUDA_CFLAGS = ["-std=c++17"]

_extension: Any | None = None
_extension_error: Exception | None = None
_load_attempted = False


def _load_extension() -> Any | None:
    """JIT-build and load the extension once; ``None`` without a CUDA device.

    A failed build is cached and re-raised as RuntimeError by every kernel call.
    """
    global _extension, _extension_error, _load_attempted
    if _load_attempted:
        return _extension
    _load_attempted = True
    if not torch.cuda.is_available():
        return None
    verbose = os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes")
    ensure_ninja_on_path()
    try:
        _extension = torch.utils.cpp_extension.load(
            name=MODULE_NAME,
            sources=_SOURCES,
            extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
            verbose=verbose,
        )
    except Exception as e:
        if verbose or os.getenv("FASTBLEND_LOG_CUDA_JIT_FAILURE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.error(
                f"Failed to JIT compile the CUDA extension {MODULE_NAME!r}: {e}"
            )
        _extension_error = RuntimeError(
            "FastBlend CUDA extension JIT compile/load failed "
            "(set JIT_VERBOSE=1 or FASTBLEND_LOG_CUDA_JIT_FAILURE=1 for logs)."
        )
        _extension_error.__cause__ = e
    return _extension


def is_available() -> bool:
    """True only if a CUDA device exists and the extension built and loaded."""
    return _load_extension() is not None


def _ext() -> Any:
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(
            "FastBlend CUDA extension not available"
        ) from _extension_error
    return ext


def staging_device(gpu_id: int) -> torch.device:
    return torch.device("cuda", gpu_id)


class PatchMatcher(FaceBlitPatchMatcherCore):
    def __init__(self, *args, **kwargs):
        _ext()
        super().__init__(*args, **kwargs)

    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return _ext().remap(
            source.contiguous(),
            nnf.contiguous(),
            self.patch_size,
            self.pad_size,
        )

    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return _ext().patch_error(
            source.contiguous(),
            nnf.contiguous(),
            target.contiguous(),
            self.patch_size,
            self.pad_size,
        )

    def _kernel_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
        source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
        err = _ext().pairwise_patch_error(
            source_a,
            nnf_a,
            source_b,
            nnf_b,
            self.patch_size,
            self.pad_size,
        )
        return err.repeat_interleave(2, dim=0)

    def _level_execution_context(self):
        return torch.cuda.device(self.gpu_id)


class PyramidPatchMatcher(PyramidPatchMatcherCore):
    level_matcher = PatchMatcher

    def __init__(self, *args, **kwargs):
        _ext()
        super().__init__(*args, **kwargs)

    @property
    def _pyramid_torch_device(self) -> torch.device:
        return staging_device(self.gpu_id)

    def _pyramid_run_context(self):
        return torch.cuda.device(self.gpu_id)
