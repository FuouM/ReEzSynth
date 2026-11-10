"""ComfyUI nodes for single-image EbSynth synthesis (``ezsynth.api.ImageSynth``)."""

from __future__ import annotations

import torch

from .image_io import (
    bgr_uint8_to_image_tensor,
    error_map_to_image_tensor,
    image_tensor_to_bgr_uint8,
)
from .options import (
    COST_FUNCTIONS,
    MULTICAND_K_RANGE,
    PRECISIONS,
    SEARCH_MODES,
    SYNTH_BACKENDS,
    check_patch_size,
    check_synthesis_options,
    validate_synthesis_options,
)
from .types import REEZ_GUIDES, REEZ_IMG_SYNTH_CONFIG, ReEzGuideList, ReEzImgSynthConfig

CATEGORY = "ReEzSynth/image"


def build_run_config(cfg: ReEzImgSynthConfig, device: str | None):
    """Translate the node config into :class:`ezsynth.api.RunConfig`."""
    from ezsynth.api import RunConfig

    return RunConfig(
        backend=cfg.backend,
        cost_function=cfg.cost_function,
        search_mode=cfg.search_mode,
        multicand_k=cfg.multicand_k,
        precision=cfg.precision,
        uniformity=cfg.uniformity,
        patch_size=cfg.patch_size,
        pyramid_levels=cfg.pyramid_levels,
        search_vote_iters=cfg.search_vote_iters,
        patch_match_iters=cfg.patch_match_iters,
        use_residual_transfer=cfg.use_residual_transfer,
        device=device,
        use_optimization=cfg.use_optimization,
        use_bilateral=cfg.use_bilateral,
        sigma_spatial=cfg.sigma_spatial,
        sigma_color=cfg.sigma_color,
        n_size_step=cfg.n_size_step,
        image_weight=cfg.image_weight,
    )


class ReEzSynthGuide:
    DESCRIPTION = (
        "One EbSynth guide: a source guide (aligned with the style image) and the "
        "matching target guide (aligned with the output), plus its weight."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (
                    "IMAGE",
                    {"tooltip": "Guide for the style image (same size as the style)."},
                ),
                "target": (
                    "IMAGE",
                    {"tooltip": "Guide for the output (defines the output size)."},
                ),
                "weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Relative importance of this guide in patch matching.",
                    },
                ),
            },
        }

    RETURN_TYPES = (REEZ_GUIDES,)
    RETURN_NAMES = ("guide",)
    OUTPUT_TOOLTIPS = ("A one-guide list; chain several with Guides Append.",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, source, target, weight):
        from ezsynth.guide import GuideObject

        guide = GuideObject(
            keyframe=image_tensor_to_bgr_uint8(source),
            target=image_tensor_to_bgr_uint8(target),
            weight=float(weight),
        )
        return (ReEzGuideList(guides=(guide,)),)


class ReEzSynthGuidesAppend:
    DESCRIPTION = "Append guides to a guide list (order does not matter for synthesis)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": (
                    REEZ_GUIDES,
                    {"tooltip": "Guide(s) to add, e.g. from a ReEzSynth Guide node."},
                ),
            },
            "optional": {
                "guides": (
                    REEZ_GUIDES,
                    {
                        "tooltip": "Existing guide list to extend (leave empty to start one)."
                    },
                ),
            },
        }

    RETURN_TYPES = (REEZ_GUIDES,)
    RETURN_NAMES = ("guides",)
    OUTPUT_TOOLTIPS = ("The combined guide list.",)
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(self, guide: ReEzGuideList, guides: ReEzGuideList | None = None):
        base = guides.guides if guides is not None else ()
        return (ReEzGuideList(guides=(*base, *guide.guides)),)


def _float(default, lo, hi, step, tooltip):
    return (
        "FLOAT",
        {"default": default, "min": lo, "max": hi, "step": step, "tooltip": tooltip},
    )


def _int(default, lo, hi, step, tooltip):
    return (
        "INT",
        {"default": default, "min": lo, "max": hi, "step": step, "tooltip": tooltip},
    )


def _bool(default, tooltip):
    return ("BOOLEAN", {"default": default, "tooltip": tooltip})


def _combo(options, default, tooltip):
    return (list(options), {"default": default, "tooltip": tooltip})


BACKEND_TOOLTIP = (
    "Synthesis backend. taichi (default): compiled GPU/CPU kernels, installed with "
    "requirements.txt. torch: pure PyTorch, slower, for platforms without Taichi. "
    "cuda: native extension (JIT build needs a CUDA toolkit + compiler matching "
    "ComfyUI's PyTorch); supports cost ssd/ncc and search_mode patchmatch only."
)
COST_TOOLTIP = (
    "Patch distance: ssd (sum of squared differences, default), ncc (normalized "
    "cross-correlation, robust to brightness changes), sad (sum of absolute "
    "differences; torch/taichi only)."
)
SEARCH_MODE_TOOLTIP = (
    "Nearest-neighbour search: patchmatch (classic), jfa (jump flooding, fast on GPU), "
    "multicand (keeps multicand_k candidates per pixel; slower, fewer artifacts). "
    "jfa/multicand need the torch or taichi backend."
)
MULTICAND_TOOLTIP = "Candidates kept per pixel when search_mode is multicand."
PRECISION_TOOLTIP = (
    "Compute precision of the patch-cost math. float16/bfloat16 roughly halve memory "
    "traffic on GPUs with fast half precision; float32 is the reference."
)


def synthesis_core_inputs() -> dict:
    """EbSynth core parameters shared by the image and video config nodes."""
    return {
        "uniformity": _float(
            3500.0,
            0.0,
            20000.0,
            10.0,
            "Penalty for reusing the same style patch; higher spreads the style more evenly.",
        ),
        "patch_size": _int(7, 3, 31, 2, "Patch side length in pixels (odd)."),
        "search_vote_iters": _int(
            12, 1, 64, 1, "Search + vote iterations per pyramid level."
        ),
        "patch_match_iters": _int(
            6, 1, 32, 1, "PatchMatch propagation / random-search iterations per search."
        ),
    }


class ReEzSynthImageSynthConfig:
    DESCRIPTION = "Parameters for ReEzSynth Image Synth."

    @classmethod
    def INPUT_TYPES(cls):
        core = synthesis_core_inputs()
        return {
            # Widget order is part of saved workflows: append new inputs at the end.
            "required": {
                "backend": _combo(SYNTH_BACKENDS, "taichi", BACKEND_TOOLTIP),
                "cost_function": _combo(COST_FUNCTIONS, "ssd", COST_TOOLTIP),
                "use_residual_transfer": _bool(
                    True,
                    "Transfer the high-frequency residual of the style for crisper detail.",
                ),
                "use_optimization": _bool(
                    True, "Use the optimized search/vote kernels (same result, faster)."
                ),
                "use_bilateral": _bool(
                    False, "Edge-aware (bilateral) weighting of patch votes."
                ),
                "sigma_spatial": _float(
                    4.0, 0.1, 64.0, 0.1, "Bilateral spatial sigma (use_bilateral only)."
                ),
                "sigma_color": _float(
                    10.0, 0.1, 255.0, 0.1, "Bilateral color sigma (use_bilateral only)."
                ),
                "n_size_step": _int(
                    1, 1, 8, 1, "Neighbour step used by the bilateral filter."
                ),
                "image_weight": _float(
                    6.0,
                    0.0,
                    100.0,
                    0.1,
                    "Weight of the style image term in the patch cost.",
                ),
                "uniformity": core["uniformity"],
                "patch_size": core["patch_size"],
                "pyramid_levels": _int(
                    6,
                    1,
                    12,
                    1,
                    "Coarse-to-fine pyramid levels; more helps large structures.",
                ),
                "search_vote_iters": core["search_vote_iters"],
                "patch_match_iters": core["patch_match_iters"],
                "search_mode": _combo(SEARCH_MODES, "patchmatch", SEARCH_MODE_TOOLTIP),
                "multicand_k": _int(3, *MULTICAND_K_RANGE, 1, MULTICAND_TOOLTIP),
                "precision": _combo(PRECISIONS, "float32", PRECISION_TOOLTIP),
            },
        }

    RETURN_TYPES = (REEZ_IMG_SYNTH_CONFIG,)
    RETURN_NAMES = ("config",)
    OUTPUT_TOOLTIPS = ("Config for ReEzSynth Image Synth.",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        backend=None,
        cost_function=None,
        search_mode=None,
        precision=None,
        multicand_k=None,
    ):
        values = (backend, cost_function, search_mode, precision, multicand_k)
        if any(v is None for v in values):
            return True  # linked inputs: validated in build()
        return validate_synthesis_options(*values) or True

    def build(self, **kwargs):
        cfg = ReEzImgSynthConfig(**kwargs)
        check_patch_size(cfg.patch_size)
        check_synthesis_options(
            cfg.backend,
            cfg.cost_function,
            cfg.search_mode,
            cfg.precision,
            cfg.multicand_k,
        )
        return (cfg,)


class ReEzSynthImageSynth:
    DESCRIPTION = (
        "Stylize one image with EbSynth: transfers the style image onto the target "
        "described by the guides."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": (
                    "IMAGE",
                    {
                        "tooltip": "Style exemplar; same size as the guides' source images."
                    },
                ),
                "guides": (
                    REEZ_GUIDES,
                    {"tooltip": "Guide list from ReEzSynth Guide / Guides Append."},
                ),
                "config": (
                    REEZ_IMG_SYNTH_CONFIG,
                    {"tooltip": "From ReEzSynth Image Synth Config."},
                ),
            },
            "optional": {
                "benchmark": _bool(False, "Log per-stage timings to the console."),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "error_map")
    OUTPUT_TOOLTIPS = (
        "Stylized image (size of the target guides).",
        "Final patch error, normalized to its maximum (bright = poor match).",
    )
    FUNCTION = "synthesize"
    CATEGORY = CATEGORY

    def synthesize(
        self,
        style_image,
        guides: ReEzGuideList,
        config: ReEzImgSynthConfig,
        benchmark: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not guides.guides:
            raise ValueError(
                "At least one guide is required. Chain ReEzSynth Guide nodes."
            )
        check_synthesis_options(
            config.backend,
            config.cost_function,
            config.search_mode,
            config.precision,
            config.multicand_k,
        )

        from ezsynth.api import ImageSynth

        from . import comfy_runtime as rt

        style_bgr = image_tensor_to_bgr_uint8(style_image)
        device = rt.torch_device()
        height, width = guides.guides[0].target.shape[:2]
        channels = sum(g.target.shape[2] for g in guides.guides)
        rt.prepare_vram(device, rt.estimate_synthesis_bytes(height, width, channels))
        try:
            with rt.comfy_progress():
                synth = ImageSynth(
                    style_image=style_bgr, config=build_run_config(config, str(device))
                )
                result_img, result_err = synth.run(
                    guides=list(guides.guides), benchmark=benchmark
                )
        finally:
            rt.release_vram()

        return (
            bgr_uint8_to_image_tensor(result_img),
            error_map_to_image_tensor(result_err),
        )
