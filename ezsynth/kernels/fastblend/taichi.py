"""FastBlend on Taichi (kernels in ``taichi_kernels.py``)."""

from __future__ import annotations

import platform

import torch

from ...utils.taichi_runtime import ensure_ti_init
from . import taichi_kernels as tk
from .core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore


def staging_device(gpu_id: int) -> torch.device:
    """The torch device whose tensors the active Taichi arch reads."""
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    return torch.device("cpu")


class PatchMatcher(FaceBlitPatchMatcherCore):
    def __init__(self, *args, **kwargs):
        ensure_ti_init()
        super().__init__(*args, **kwargs)

    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        ensure_ti_init()
        p = self.pad_size
        target = torch.zeros(
            (source.shape[0], self.height + p * 2, self.width + p * 2, self.channel),
            dtype=torch.float32,
            device=source.device,
        )
        tk.remap_kernel(
            source,
            nnf,
            target,
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            p,
        )
        return target

    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        ensure_ti_init()
        error = torch.zeros(
            (source.shape[0], self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        tk.patch_error_kernel(
            source,
            nnf,
            target,
            error,
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
        )
        return error

    def _kernel_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        ensure_ti_init()
        source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
        source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
        error = torch.zeros(
            (source_a.shape[0], self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        tk.pairwise_patch_error_kernel(
            source_a,
            nnf_a,
            source_b,
            nnf_b,
            error,
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
        )
        return error.repeat_interleave(2, dim=0)


class PyramidPatchMatcher(PyramidPatchMatcherCore):
    level_matcher = PatchMatcher

    @property
    def _pyramid_torch_device(self) -> torch.device:
        return staging_device(self.gpu_id)
