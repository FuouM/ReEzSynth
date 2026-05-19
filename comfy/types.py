"""Wire types passed between ReEzSynth ComfyUI nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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
