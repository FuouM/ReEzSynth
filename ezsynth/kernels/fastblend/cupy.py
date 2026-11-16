"""FastBlend on CuPy: ``cupy_kernels.py`` holds ``cuda_src/kernels.cu`` as RawKernels."""

from __future__ import annotations

import cupy as cp
import torch

from .core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore
from .cupy_kernels import (
    pairwise_patch_error_kernel,
    patch_error_kernel,
    remapping_kernel,
)


def staging_device(gpu_id: int) -> torch.device:
    return torch.device("cuda", gpu_id)


class PatchMatcher(FaceBlitPatchMatcherCore):
    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        batch_size = source.shape[0]
        target = torch.zeros(
            (
                batch_size,
                self.height + self.pad_size * 2,
                self.width + self.pad_size * 2,
                self.channel,
            ),
            dtype=torch.float32,
            device=source.device,
        )
        source_cp = cp.asarray(source.detach())
        nnf_cp = cp.asarray(nnf.detach())
        target_cp = cp.asarray(target.detach())
        remapping_kernel(
            (*self.grid, batch_size),
            self.block,
            (
                self.height,
                self.width,
                self.channel,
                self.patch_size,
                self.pad_size,
                source_cp,
                nnf_cp,
                target_cp,
            ),
        )
        target.copy_(torch.as_tensor(target_cp))
        return target

    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        batch_size = source.shape[0]
        error = torch.zeros(
            (batch_size, self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        source_cp = cp.asarray(source.detach())
        nnf_cp = cp.asarray(nnf.detach())
        target_cp = cp.asarray(target.detach())
        error_cp = cp.asarray(error.detach())
        patch_error_kernel(
            (*self.grid, batch_size),
            self.block,
            (
                self.height,
                self.width,
                self.channel,
                self.patch_size,
                self.pad_size,
                source_cp,
                nnf_cp,
                target_cp,
                error_cp,
            ),
        )
        error.copy_(torch.as_tensor(error_cp))
        return error

    def _kernel_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        batch_size = source.shape[0] // 2
        source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
        source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
        error = torch.zeros(
            (batch_size, self.height, self.width),
            dtype=torch.float32,
            device=source.device,
        )
        source_a_cp = cp.asarray(source_a.detach())
        nnf_a_cp = cp.asarray(nnf_a.detach())
        source_b_cp = cp.asarray(source_b.detach())
        nnf_b_cp = cp.asarray(nnf_b.detach())
        error_cp = cp.asarray(error.detach())
        pairwise_patch_error_kernel(
            (*self.grid, batch_size),
            self.block,
            (
                self.height,
                self.width,
                self.channel,
                self.patch_size,
                self.pad_size,
                source_a_cp,
                nnf_a_cp,
                source_b_cp,
                nnf_b_cp,
                error_cp,
            ),
        )
        error.copy_(torch.as_tensor(error_cp))
        return error.repeat_interleave(2, dim=0)

    def _level_execution_context(self):
        return torch.cuda.device(self.gpu_id)


class PyramidPatchMatcher(PyramidPatchMatcherCore):
    level_matcher = PatchMatcher

    @property
    def _pyramid_torch_device(self) -> torch.device:
        return staging_device(self.gpu_id)

    def _pyramid_run_context(self):
        return torch.cuda.device(self.gpu_id)
