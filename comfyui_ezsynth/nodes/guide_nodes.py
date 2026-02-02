# comfyui_ezsynth/nodes/guide_nodes.py
"""
Guide preparation nodes for synthesis.
"""

from typing import List, Optional, Tuple

import torch

from .base import EZBaseNode


class PositionalGuideNode(EZBaseNode):
    """
    Generate positional guide from image dimensions.
    """

    CATEGORY = "ReEzSynth/Guides"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
            },
            "optional": {
                "draw_grid": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pos_guide",)
    FUNCTION = "create_guide"

    def create_guide(
        self, width: int, height: int, draw_grid: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Create positional gradient guide.
        """
        from ezsynth.utils.warp_utils import PositionalGuide

        from ..core.tensor_utils import numpy_to_tensor

        pos_guide = PositionalGuide(height, width)
        guide_np = pos_guide.get_pristine_guide_uint8()

        return (self._format_output(numpy_to_tensor(guide_np)),)


class WarpGuideNode(EZBaseNode):
    """
    Warp a guide using optical flow.
    """

    CATEGORY = "ReEzSynth/Guides"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide": ("IMAGE",),
                "flow": ("FLOW",),
            },
            "optional": {
                "inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_guide",)
    FUNCTION = "warp_guide"

    def warp_guide(
        self, guide: torch.Tensor, flow: torch.Tensor, inverse: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Warp guide using flow field.
        """
        from ezsynth.utils.warp_utils import Warp

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_flow, tensor_to_numpy

        guide_np = tensor_to_numpy(guide)
        flow_np = tensor_to_flow(flow)

        h, w = guide_np.shape[:2]
        warp = Warp(h, w)

        direction = -1 if inverse else 1
        warped_np = warp.run_warping(guide_np, flow_np * direction)

        return (self._format_output(numpy_to_tensor(warped_np)),)


class CombineGuidesNode(EZBaseNode):
    """
    Combine multiple guides into a guide list for synthesis.
    """

    CATEGORY = "ReEzSynth/Guides"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide1_source": ("IMAGE",),
                "guide1_target": ("IMAGE",),
                "guide1_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "guide2_source": ("IMAGE",),
                "guide2_target": ("IMAGE",),
                "guide2_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "guide3_source": ("IMAGE",),
                "guide3_target": ("IMAGE",),
                "guide3_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            },
        }

    RETURN_TYPES = ("GUIDE_LIST", "GUIDE_LIST")
    RETURN_NAMES = ("source_guides", "target_guides")
    FUNCTION = "combine_guides"

    def combine_guides(
        self,
        guide1_source: torch.Tensor,
        guide1_target: torch.Tensor,
        guide1_weight: float = 1.0,
        guide2_source: Optional[torch.Tensor] = None,
        guide2_target: Optional[torch.Tensor] = None,
        guide2_weight: float = 1.0,
        guide3_source: Optional[torch.Tensor] = None,
        guide3_target: Optional[torch.Tensor] = None,
        guide3_weight: float = 1.0,
    ) -> Tuple[List[Tuple[torch.Tensor, float]], List[Tuple[torch.Tensor, float]]]:
        """
        Combine multiple guides into guide lists.
        """
        def validate_guide_pair(src: torch.Tensor, tgt: torch.Tensor, name: str):
            """Validate that source and target have matching dimensions."""
            src_shape = src.shape if src.dim() >= 3 else tuple(src.shape)
            tgt_shape = tgt.shape if tgt.dim() >= 3 else tuple(tgt.shape)
            if len(src_shape) >= 3:
                src_shape = src_shape[-3:]  # Get (H, W, C)
            if len(tgt_shape) >= 3:
                tgt_shape = tgt_shape[-3:]
            if src_shape != tgt_shape:
                raise ValueError(
                    f"{name}: Source and target guide dimensions must match. "
                    f"Got source: {src_shape}, target: {tgt_shape}"
                )
        
        # Validate guide1
        validate_guide_pair(guide1_source, guide1_target, "Guide 1")
        
        source_guides = [(guide1_source, guide1_weight)]
        target_guides = [(guide1_target, guide1_weight)]

        if guide2_source is not None and guide2_target is not None:
            validate_guide_pair(guide2_source, guide2_target, "Guide 2")
            source_guides.append((guide2_source, guide2_weight))
            target_guides.append((guide2_target, guide2_weight))

        if guide3_source is not None and guide3_target is not None:
            validate_guide_pair(guide3_source, guide3_target, "Guide 3")
            source_guides.append((guide3_source, guide3_weight))
            target_guides.append((guide3_target, guide3_weight))

        return (source_guides, target_guides)
