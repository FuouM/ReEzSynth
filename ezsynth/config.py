from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator


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
    force_precomputation: bool = False
    force_style_size: bool = True


class PrecomputationConfig(BaseModel):
    flow_engine: str = "RAFT"
    flow_model: str = "sintel"
    edge_method: str = "Classic"


class FinalPassConfig(BaseModel):
    enabled: bool = False
    strength: float = Field(1.0, ge=0.0)


class PipelineConfig(BaseModel):
    pyramid_levels: int = 1
    use_residual_transfer: bool = False
    final_pass: FinalPassConfig = Field(default_factory=FinalPassConfig)
    alpha: float = Field(0.75, ge=0.0, le=1.0)
    max_iter: int = 200
    flip_aug: bool = False
    content_loss: bool = False
    colorize: bool = True


class BlendingConfig(BaseModel):
    use_lsqr: bool = True
    poisson_maxiter: Optional[int] = None


class EbsynthParamsConfig(BaseModel):
    uniformity: float = 3500.0
    patch_size: int = 7
    vote_mode: str = "weighted"  # 'weighted' or 'plain'
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    stop_threshold: int = 5
    # New: Skip random search for patches with SSD error below this. 0.0 disables.
    search_pruning_threshold: float = 50.0 
    extra_pass_3x3: bool = False
    edge_weight: float = 1.0
    image_weight: float = 6.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5

    @validator("vote_mode")
    def vote_mode_must_be_valid(cls, v):
        if v not in ["weighted", "plain"]:
            raise ValueError("vote_mode must be 'weighted' or 'plain'")
        return v


class DebugConfig(BaseModel):
    save_flow_viz: bool = False
    flow_viz_dir: str = "debug/flow_viz"


class MainConfig(BaseModel):
    project: ProjectConfig
    precomputation: PrecomputationConfig
    pipeline: PipelineConfig
    blending: BlendingConfig = Field(default_factory=BlendingConfig)
    ebsynth_params: EbsynthParamsConfig
    debug: DebugConfig = Field(default_factory=DebugConfig)