from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "DefaultProject"
    content_dir: str
    style_path: Union[str, List[str]]
    style_indices: List[int]
    output_dir: str
    mask_dir: Optional[str] = None
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
    final_pass: FinalPassConfig = Field(default_factory=FinalPassConfig)  # New
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
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    edge_weight: float = 1.0
    image_weight: float = 6.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5


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
