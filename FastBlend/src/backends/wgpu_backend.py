"""WebGPU FastBlend backend: the pyramid matcher of the ``ezsynth_wgpu`` Rust module.

The whole pyramid (resampling, propagation, random search, tracking, remap) runs on the
GPU through wgpu; inputs and results are CPU tensors.
"""

from __future__ import annotations

import numpy as np
import torch

from ezsynth.engines.backends.wgpu_backend import import_wgpu


def _array(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


class PyramidPatchMatcherWgpu:
    """Same constructor and ``estimate_nnf`` contract as ``PyramidPatchMatcherCore``."""

    def __init__(
        self,
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block=16,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity",
    ):
        self._matcher = import_wgpu().FastBlendMatcher(
            int(image_height),
            int(image_width),
            int(channel),
            int(minimum_patch_size),
            num_iter=int(num_iter),
            guide_weight=float(guide_weight),
            use_mean_target_style=bool(use_mean_target_style),
            use_pairwise_patch_error=bool(use_pairwise_patch_error),
            tracking_window_size=int(tracking_window_size),
            initialize=str(initialize),
        )
        self.pyramid_level = self._matcher.pyramid_level

    def estimate_nnf(
        self,
        source_guide: np.ndarray | torch.Tensor,
        target_guide: np.ndarray | torch.Tensor,
        source_style: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nnf, target_style = self._matcher.estimate_nnf(
            _array(source_guide), _array(target_guide), _array(source_style)
        )
        return torch.from_numpy(nnf), torch.from_numpy(target_style)
