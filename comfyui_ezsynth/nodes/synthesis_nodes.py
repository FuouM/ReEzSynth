# comfyui_ezsynth/nodes/synthesis_nodes.py
"""
Synthesis nodes for image and video style transfer.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..types import EbsynthParams
from .base import EZBaseNode


class ImageSynthNode(EZBaseNode):
    """
    Simple single-image synthesis using ImageSynth.
    Lightweight, no NNF propagation support.
    """

    CATEGORY = "ReEzSynth/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guides": ("GUIDE_LIST",),
                "target_guides": ("GUIDE_LIST",),
            },
            "optional": {
                "uniformity": (
                    "FLOAT",
                    {"default": 3500.0, "min": 100.0, "max": 10000.0},
                ),
                "patch_size": ("INT", {"default": 7, "min": 3, "max": 21, "step": 2}),
                "pyramid_levels": ("INT", {"default": 6, "min": 1, "max": 10}),
                "search_vote_iters": ("INT", {"default": 12, "min": 1, "max": 50}),
                "patch_match_iters": ("INT", {"default": 6, "min": 1, "max": 20}),
                "backend": (["cuda", "torch"], {"default": "cuda"}),
                "extra_pass_3x3": ("BOOLEAN", {"default": False}),
                "cost_function": (["ssd", "ncc"], {"default": "ssd"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("stylized", "error_map")
    FUNCTION = "synthesize"

    def synthesize(
        self,
        style_image: torch.Tensor,
        source_guides: List[Tuple[torch.Tensor, float]],
        target_guides: List[Tuple[torch.Tensor, float]],
        uniformity: float = 3500.0,
        patch_size: int = 7,
        pyramid_levels: int = 6,
        search_vote_iters: int = 12,
        patch_match_iters: int = 6,
        backend: str = "cuda",
        extra_pass_3x3: bool = False,
        cost_function: str = "ssd",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run simple image synthesis.
        """
        style_image = self._validate_image_input(style_image, "style_image")

        # Create engine with params
        params = EbsynthParams(
            uniformity=uniformity,
            patch_size=patch_size,
            pyramid_levels=pyramid_levels,
            search_vote_iters=search_vote_iters,
            patch_match_iters=patch_match_iters,
            backend=backend,
            extra_pass_3x3=extra_pass_3x3,
            cost_function=cost_function,
        )

        from ..core.engine_wrapper import ImageSynthNodeEngine

        engine = ImageSynthNodeEngine(params)

        # Convert guides to proper format (source, target, weight)
        guide_tuples = []
        for src, tgt in zip(source_guides, target_guides):
            src_img = src[0]
            tgt_img = tgt[0]
            
            # Debug: Log shapes
            print(f"DEBUG: src_img shape: {src_img.shape if hasattr(src_img, 'shape') else 'no shape'}")
            print(f"DEBUG: tgt_img shape: {tgt_img.shape if hasattr(tgt_img, 'shape') else 'no shape'}")
            
            # Validate guide dimensions - source and target must have same dimensions
            src_shape = tuple(src_img.shape) if hasattr(src_img, 'shape') else src_img.size
            tgt_shape = tuple(tgt_img.shape) if hasattr(tgt_img, 'shape') else tgt_img.size
            
            print(f"DEBUG: src_shape: {src_shape}, tgt_shape: {tgt_shape}")
            
            if src_shape != tgt_shape:
                raise ValueError(
                    f"Source and target guide dimensions must match. "
                    f"Got source: {src_shape}, target: {tgt_shape}"
                )
            
            guide_tuples.append((src_img, tgt_img, src[1] * tgt[1]))

        print(f"DEBUG: Number of guides: {len(guide_tuples)}")
        if guide_tuples:
            print(f"DEBUG: First guide src shape: {guide_tuples[0][0].shape}")
            print(f"DEBUG: First guide tgt shape: {guide_tuples[0][1].shape}")

        stylized, error_map = engine.synthesize(style_image, guide_tuples)

        engine.cleanup()

        return (
            self._format_output(stylized),
            self._format_output(error_map),
        )


class EbsynthNode(EZBaseNode):
    """
    Advanced synthesis using EbsynthEngine.
    Supports NNF propagation for temporal coherence.
    """

    CATEGORY = "ReEzSynth/Synthesis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_guides": ("GUIDE_LIST",),
                "target_guides": ("GUIDE_LIST",),
            },
            "optional": {
                "initial_nnf": ("NNF",),
                "uniformity": (
                    "FLOAT",
                    {"default": 3500.0, "min": 100.0, "max": 10000.0},
                ),
                "patch_size": ("INT", {"default": 7, "min": 3, "max": 21, "step": 2}),
                "pyramid_levels": ("INT", {"default": 6, "min": 1, "max": 10}),
                "search_vote_iters": ("INT", {"default": 12, "min": 1, "max": 50}),
                "patch_match_iters": ("INT", {"default": 6, "min": 1, "max": 20}),
                "backend": (["cuda", "torch"], {"default": "cuda"}),
                "extra_pass_3x3": ("BOOLEAN", {"default": False}),
                "cost_function": (["ssd", "ncc"], {"default": "ssd"}),
                "output_nnf": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "NNF")
    RETURN_NAMES = ("stylized", "error_map", "nnf")
    FUNCTION = "synthesize"

    def synthesize(
        self,
        style_image: torch.Tensor,
        source_guides: List[Tuple[torch.Tensor, float]],
        target_guides: List[Tuple[torch.Tensor, float]],
        initial_nnf: Optional[torch.Tensor] = None,
        uniformity: float = 3500.0,
        patch_size: int = 7,
        pyramid_levels: int = 6,
        search_vote_iters: int = 12,
        patch_match_iters: int = 6,
        backend: str = "cuda",
        extra_pass_3x3: bool = False,
        cost_function: str = "ssd",
        output_nnf: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run advanced synthesis with NNF support.
        """
        style_image = self._validate_image_input(style_image, "style_image")

        params = EbsynthParams(
            uniformity=uniformity,
            patch_size=patch_size,
            pyramid_levels=pyramid_levels,
            search_vote_iters=search_vote_iters,
            patch_match_iters=patch_match_iters,
            backend=backend,
            extra_pass_3x3=extra_pass_3x3,
            cost_function=cost_function,
        )

        from ..core.engine_wrapper import EbsynthNodeEngine

        engine = EbsynthNodeEngine(params)

        # Build guide tuples (source, target, weight) with validation
        guides = []
        for src, tgt in zip(source_guides, target_guides):
            src_img = src[0]
            tgt_img = tgt[0]
            
            # Validate guide dimensions
            src_shape = src_img.shape if hasattr(src_img, 'shape') else src_img.size
            tgt_shape = tgt_img.shape if hasattr(tgt_img, 'shape') else tgt_img.size
            
            if src_shape != tgt_shape:
                raise ValueError(
                    f"Source and target guide dimensions must match. "
                    f"Got source: {src_shape}, target: {tgt_shape}"
                )
            
            guides.append((src_img, tgt_img, src[1] * tgt[1]))

        stylized, error_map, nnf = engine.synthesize(
            style_image,
            guides,
            initial_nnf=initial_nnf,
            output_nnf=output_nnf,
        )

        engine.cleanup()

        return (
            self._format_output(stylized),
            self._format_output(error_map),
            self._format_output(nnf) if nnf is not None else torch.zeros(1, 1, 1),
        )
