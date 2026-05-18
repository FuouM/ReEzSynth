from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .flow.types import (
    NEUFLOW_FLOW_MODELS,
    RAFT_FLOW_MODELS,
    FlowEngineName,
    FlowModelName,
    OpenCvFlowMethod,
    TorchVisionRaftModel,
)


class ProjectConfig(BaseModel):
    name: str = "DefaultProject"
    # --- REQUIRED PATHS ---
    content_dir: str
    style_path: Union[str, List[str]]
    style_indices: List[int]
    output_dir: str
    # --- OPTIONAL PATHS ---
    mask_dir: Optional[str] = None
    modulation_dir: Optional[str] = None
    # --- CACHING ---
    cache_dir: str = "cache/DefaultProject"
    force_recompute_flow: bool = False
    force_recompute_edge: bool = False
    force_style_size: bool = True


class PrecomputationConfig(BaseModel):
    flow_engine: FlowEngineName = "NeuFlow"
    # Interpretation depends on ``flow_engine`` (RAFT vs NeuFlow checkpoint names).
    flow_model: FlowModelName = "neuflow_mixed"
    edge_method: Literal["Classic", "PAGE", "PST"] = "Classic"
    opencv_flow_method: OpenCvFlowMethod = "DIS"
    torchvision_flow_model: TorchVisionRaftModel = "raft_large"

    @model_validator(mode="after")
    def _flow_model_matches_engine(self):
        if self.flow_engine == "RAFT" and self.flow_model not in RAFT_FLOW_MODELS:
            raise ValueError(
                f"flow_model {self.flow_model!r} is not a RAFT checkpoint name {RAFT_FLOW_MODELS}"
            )
        if self.flow_engine == "NeuFlow" and self.flow_model not in NEUFLOW_FLOW_MODELS:
            raise ValueError(
                f"flow_model {self.flow_model!r} is not a NeuFlow checkpoint name {NEUFLOW_FLOW_MODELS}"
            )
        return self


class PipelineConfig(BaseModel):
    pyramid_levels: int = 1
    use_residual_transfer: bool = True
    alpha: float = Field(0.75, ge=0.0, le=1.0)
    use_temporal_nnf_propagation: bool = False
    use_sparse_feature_guide: bool = False
    use_flow_occlusion_masks: bool = False
    use_flow_occlusion_modulation: bool = False
    use_flow_occlusion_fill: bool = False
    use_flow_occlusion_refine: bool = False
    occlusion_consistency_alpha: float = 0.01
    occlusion_consistency_beta: float = 0.5
    occlusion_mask_dilate: int = 1
    occlusion_use_coverage_mask: bool = True
    occlusion_modulation_floor: int = Field(64, ge=0, le=255)
    occlusion_fill_radius: int = 3
    occlusion_fill_feather: int = 5
    occlusion_refine_feather: int = 5


class BlendingConfig(BaseModel):
    poisson_solver: Literal[
        "lsqr",
        "lsmr",
        "cg",
        "amg",
        "seamless",
        "taichi-cg",
        "disabled",
    ] = "lsqr"
    poisson_maxiter: Optional[int] = None
    poisson_grad_weight_l: float = 2.5  # Gradient weight for L channel
    poisson_grad_weight_ab: float = 0.5  # Gradient weight for a/b channels
    use_taichi_ops: bool = False


class EbsynthParamsConfig(BaseModel):
    uniformity: float = 3500.0
    patch_size: int = 7
    vote_mode: Literal["weighted", "plain"] = "weighted"
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    stop_threshold: int = 5
    # New: Skip random search for patches with SSD error below this. 0.0 disables.
    search_pruning_threshold: float = 50.0
    # Bilateral parameters
    use_bilateral: bool = False
    sigma_spatial: float = 4.0
    sigma_color: float = 10.0
    n_size_step: int = 1
    # New: Cost function for patch matching.
    cost_function: Literal["ssd", "ncc"] = "ssd"
    # New: Backend for synthesis operations.
    backend: Literal["cuda", "torch", "taichi"] = "cuda"
    # New: Device for synthesis (allows CPU with C++ extension)
    device: Optional[str] = None  # None for auto-detect, or "cpu"/"cuda"
    extra_pass_3x3: bool = False
    edge_weight: float = 1.0
    image_weight: float = 6.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5
    sparse_anchor_weight: float = 10.0
    # New: Use optimized index_vector CPU backend (if available)
    use_optimization: bool = True


class DebugConfig(BaseModel):
    save_flow_viz: bool = False
    flow_viz_dir: str = "debug/flow_viz"
    save_occlusion_debug: bool = False
    occlusion_debug_dir: str = "debug/occlusion"


class MainConfig(BaseModel):
    project: ProjectConfig
    precomputation: PrecomputationConfig
    pipeline: PipelineConfig
    blending: BlendingConfig = Field(default_factory=BlendingConfig)
    ebsynth_params: EbsynthParamsConfig
    debug: DebugConfig = Field(default_factory=DebugConfig)
