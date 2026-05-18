"""FaceBlit-style patch match via CuPy RawKernels (same math as ``kernels.cu``)."""

from __future__ import annotations

import cupy as cp
import torch

from .cupy_kernels import (
    pairwise_patch_error_kernel,
    patch_error_kernel,
    remapping_kernel,
)
from .patch_matcher_core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore


class PatchMatcherCupy(FaceBlitPatchMatcherCore):
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
            self.grid + (batch_size,),
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
            self.grid + (batch_size,),
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
            self.grid + (batch_size,),
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


class PyramidPatchMatcherCupy(PyramidPatchMatcherCore):
    @property
    def _pyramid_torch_device(self) -> torch.device:
        return torch.device(f"cuda:{self.gpu_id}")

    def _pyramid_run_context(self):
        return torch.cuda.device(self.gpu_id)

    def _new_level_matcher(self, height: int, width: int, channel: int):
        return PatchMatcherCupy(
            height,
            width,
            channel,
            minimum_patch_size=self.minimum_patch_size,
            threads_per_block=self.threads_per_block,
            num_iter=self.num_iter,
            gpu_id=self.gpu_id,
            guide_weight=self.guide_weight,
            use_mean_target_style=self.use_mean_target_style,
            use_pairwise_patch_error=self.use_pairwise_patch_error,
            tracking_window_size=self.tracking_window_size,
        )

    def apply_nnf_to_image(
        self, nnf: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        with torch.cuda.device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            return self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
