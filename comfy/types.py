"""Wire types passed between ReEzSynth ComfyUI nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ezsynth.guide import GuideObject


@dataclass
class ReEzGuideList:
    """Ordered list of guides for :class:`~ezsynth.api.ImageSynth`."""

    guides: List[GuideObject] = field(default_factory=list)

    def append(self, guide: GuideObject) -> "ReEzGuideList":
        return ReEzGuideList(guides=[*self.guides, guide])


@dataclass(frozen=True)
class ReEzImgSynthConfig:
    """Synthesis parameters (mirrors ``run_img_synth.py`` / ``RunConfig``)."""

    backend: str = "torch"
    cost_function: str = "ssd"
    use_residual_transfer: bool = True
    use_optimization: bool = True
    use_bilateral: bool = False
    sigma_spatial: float = 4.0
    sigma_color: float = 10.0
    n_size_step: int = 1
    image_weight: float = 6.0
    uniformity: float = 3500.0
    patch_size: int = 7
    pyramid_levels: int = 6
    search_vote_iters: int = 12
    patch_match_iters: int = 6


@dataclass(frozen=True)
class ReEzVideoStyleKeyframes:
    """Style keyframes and their corresponding content frame indices."""

    frames: List[np.ndarray]
    indices: List[int]


@dataclass(frozen=True)
class ReEzVideoSynthConfig:
    """Video synthesis parameters for the full EbSynth pipeline."""

    flow_engine: str = "RAFT"
    flow_model: str = "sintel"
    edge_method: str = "Classic"
    opencv_flow_method: str = "DIS"
    torchvision_flow_model: str = "raft_large"
    pyramid_levels: int = 6
    alpha: float = 0.75
    use_residual_transfer: bool = False
    use_temporal_nnf_propagation: bool = False
    use_sparse_feature_guide: bool = False
    use_pseudo_endpoint_styles: bool = False
    use_flow_occlusion_masks: bool = False
    use_forward_warping: bool = False
    poisson_solver: str = "disabled"
    poisson_maxiter: int = 0
    poisson_grad_weight_l: float = 2.5
    poisson_grad_weight_ab: float = 0.5
    use_taichi_ops: bool = False
    backend: str = "taichi"
    cost_function: str = "ncc"
    vote_mode: str = "weighted"
    uniformity: float = 3500.0
    patch_size: int = 7
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    stop_threshold: int = 5
    search_pruning_threshold: float = 50.0
    use_bilateral: bool = False
    sigma_spatial: float = 4.0
    sigma_color: float = 10.0
    n_size_step: int = 1
    extra_pass_3x3: bool = False
    image_weight: float = 6.0
    edge_weight: float = 1.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5
    sparse_anchor_weight: float = 50.0
    use_optimization: bool = True
    save_flow_viz: bool = False
