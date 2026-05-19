"""ComfyUI nodes for single-image EbSynth synthesis (``run_img_synth`` workflow)."""

from __future__ import annotations

import gc
from typing import Tuple

import torch

from .image_io import (
    bgr_uint8_to_image_tensor,
    error_map_to_image_tensor,
    image_tensor_to_bgr_uint8,
)
from .types import ReEzGuideList, ReEzImgSynthConfig

REEZ_GUIDES = "REEZ_GUIDES"
REEZ_IMG_SYNTH_CONFIG = "REEZ_IMG_SYNTH_CONFIG"

_ALLOWED_BACKENDS = ("torch", "taichi")


def _build_run_config(cfg: ReEzImgSynthConfig):
    from ezsynth.api import RunConfig

    if cfg.backend not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"Backend {cfg.backend!r} is not supported in ComfyUI nodes. "
            f"Use one of: {', '.join(_ALLOWED_BACKENDS)}."
        )

    return RunConfig(
        backend=cfg.backend,
        uniformity=cfg.uniformity,
        patch_size=cfg.patch_size,
        pyramid_levels=cfg.pyramid_levels,
        search_vote_iters=cfg.search_vote_iters,
        patch_match_iters=cfg.patch_match_iters,
        use_residual_transfer=cfg.use_residual_transfer,
        cost_function=cfg.cost_function,
        device=None,
        use_optimization=cfg.use_optimization,
        use_bilateral=cfg.use_bilateral,
        sigma_spatial=cfg.sigma_spatial,
        sigma_color=cfg.sigma_color,
        n_size_step=cfg.n_size_step,
        image_weight=cfg.image_weight,
    )


class ReEzSynthGuide:
    """Pair a source guide with a target guide and a weight."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = (REEZ_GUIDES,)
    RETURN_NAMES = ("guide",)
    FUNCTION = "create"
    CATEGORY = "ReEzSynth"

    def create(self, source, target, weight):
        from ezsynth.guide import GuideObject

        guide = GuideObject(
            keyframe=image_tensor_to_bgr_uint8(source),
            target=image_tensor_to_bgr_uint8(target),
            weight=float(weight),
        )
        return (ReEzGuideList(guides=[guide]),)


class ReEzSynthGuidesAppend:
    """Append a guide to an existing guide list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": (REEZ_GUIDES,),
            },
            "optional": {
                "guides": (REEZ_GUIDES,),
            },
        }

    RETURN_TYPES = (REEZ_GUIDES,)
    RETURN_NAMES = ("guides",)
    FUNCTION = "append"
    CATEGORY = "ReEzSynth"

    def append(self, guide: ReEzGuideList, guides: ReEzGuideList | None = None):
        base = guides if guides is not None else ReEzGuideList()
        if len(guide.guides) != 1:
            raise ValueError("Expected a single guide from ReEzSynth Guide node.")
        merged = ReEzGuideList(guides=[*base.guides, guide.guides[0]])
        return (merged,)


class ReEzSynthImageSynthConfig:
    """Parameters for :class:`ReEzSynthImageSynth` (matches ``run_img_synth`` CLI)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (list(_ALLOWED_BACKENDS), {"default": "torch"}),
                "cost_function": (["ssd", "ncc"], {"default": "ssd"}),
                "use_residual_transfer": ("BOOLEAN", {"default": True}),
                "use_optimization": ("BOOLEAN", {"default": True}),
                "use_bilateral": ("BOOLEAN", {"default": False}),
                "sigma_spatial": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 64.0, "step": 0.1},
                ),
                "sigma_color": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 255.0, "step": 0.1},
                ),
                "n_size_step": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "image_weight": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "uniformity": (
                    "FLOAT",
                    {"default": 3500.0, "min": 0.0, "max": 20000.0, "step": 10.0},
                ),
                "patch_size": ("INT", {"default": 7, "min": 3, "max": 31, "step": 2}),
                "pyramid_levels": ("INT", {"default": 6, "min": 1, "max": 12, "step": 1}),
                "search_vote_iters": (
                    "INT",
                    {"default": 12, "min": 1, "max": 64, "step": 1},
                ),
                "patch_match_iters": (
                    "INT",
                    {"default": 6, "min": 1, "max": 32, "step": 1},
                ),
            },
        }

    RETURN_TYPES = (REEZ_IMG_SYNTH_CONFIG,)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "ReEzSynth"

    def build(
        self,
        backend,
        cost_function,
        use_residual_transfer,
        use_optimization,
        use_bilateral,
        sigma_spatial,
        sigma_color,
        n_size_step,
        image_weight,
        uniformity,
        patch_size,
        pyramid_levels,
        search_vote_iters,
        patch_match_iters,
    ):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
        cfg = ReEzImgSynthConfig(
            backend=backend,
            cost_function=cost_function,
            use_residual_transfer=use_residual_transfer,
            use_optimization=use_optimization,
            use_bilateral=use_bilateral,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            n_size_step=n_size_step,
            image_weight=image_weight,
            uniformity=uniformity,
            patch_size=patch_size,
            pyramid_levels=pyramid_levels,
            search_vote_iters=search_vote_iters,
            patch_match_iters=patch_match_iters,
        )
        return (cfg,)


class ReEzSynthImageSynth:
    """Run single-image EbSynth synthesis (``ImageSynth`` API)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "guides": (REEZ_GUIDES,),
                "config": (REEZ_IMG_SYNTH_CONFIG,),
            },
            "optional": {
                "benchmark": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "error_map")
    FUNCTION = "synthesize"
    CATEGORY = "ReEzSynth"

    def synthesize(
        self,
        style_image,
        guides: ReEzGuideList,
        config: ReEzImgSynthConfig,
        benchmark: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not guides.guides:
            raise ValueError("At least one guide is required. Chain ReEzSynth Guide nodes.")

        from ezsynth.api import ImageSynth

        style_bgr = image_tensor_to_bgr_uint8(style_image)
        run_config = _build_run_config(config)

        guide_tuples = [(g.keyframe, g.target, g.weight) for g in guides.guides]
        synth = ImageSynth(style_image=style_bgr, config=run_config)
        result_img, result_err = synth.run(
            guides=guide_tuples,
            benchmark=benchmark,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        out_image = bgr_uint8_to_image_tensor(result_img)
        out_error = error_map_to_image_tensor(result_err)
        return (out_image, out_error)
