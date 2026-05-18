"""FaceBlit-style patch match via JIT ``fastblend_extension`` CUDA kernels."""

from __future__ import annotations

import torch

from .cuda_extension import is_available, pairwise_patch_error, patch_error, remap
from .patch_matcher_core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore


class PatchMatcherCUDA(FaceBlitPatchMatcherCore):
    def __init__(self, *args, **kwargs):
        if not is_available():
            raise RuntimeError("FastBlend CUDA extension not available")
        super().__init__(*args, **kwargs)

    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return remap(
            source.contiguous(),
            nnf.contiguous(),
            self.patch_size,
            self.pad_size,
        )

    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return patch_error(
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
        err = pairwise_patch_error(
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


class PyramidPatchMatcherCUDA(PyramidPatchMatcherCore):
    def __init__(self, *args, **kwargs):
        if not is_available():
            raise RuntimeError("FastBlend CUDA extension not available")
        super().__init__(*args, **kwargs)

    @property
    def _pyramid_torch_device(self) -> torch.device:
        return torch.device(f"cuda:{self.gpu_id}")

    def _pyramid_run_context(self):
        return torch.cuda.device(self.gpu_id)

    def _new_level_matcher(self, height: int, width: int, channel: int):
        return PatchMatcherCUDA(
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
