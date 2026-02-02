# comfyui_ezsynth/nodes/utility_nodes.py
"""
Utility nodes for image manipulation.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import EZBaseNode


class WarpImageNode(EZBaseNode):
    """
    Warp an image using optical flow.
    """

    CATEGORY = "ReEzSynth/Utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flow": ("FLOW",),
            },
            "optional": {
                "inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped_image",)
    FUNCTION = "warp_image"

    def warp_image(
        self,
        image: torch.Tensor,
        flow: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Warp image using flow field.
        """
        from ezsynth.utils.warp_utils import Warp

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_flow, tensor_to_numpy

        image_np = tensor_to_numpy(image)
        flow_np = tensor_to_flow(flow)

        h, w = image_np.shape[:2]
        warp = Warp(h, w)

        direction = -1 if inverse else 1
        warped_np = warp.run_warping(image_np, flow_np * direction)

        return (self._format_output(numpy_to_tensor(warped_np)),)


class MaskNode(EZBaseNode):
    """
    Apply a mask to an image.
    """

    CATEGORY = "ReEzSynth/Utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("masked_image",)
    FUNCTION = "apply_mask"

    def apply_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        invert_mask: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Apply mask to image.
        """
        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        image_np = tensor_to_numpy(image)
        mask_np = tensor_to_numpy(mask)

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        if invert_mask:
            mask_np = 255 - mask_np

        mask_binary = (mask_np > 127).astype(np.uint8) * 255
        masked = image_np * (mask_binary[..., np.newaxis] / 255.0)

        return (self._format_output(numpy_to_tensor(masked.astype(np.uint8))),)


class ColorTransferNode(EZBaseNode):
    """
    Transfer color from source to target image.
    """

    CATEGORY = "ReEzSynth/Utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "target": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color_matched",)
    FUNCTION = "transfer_color"

    def transfer_color(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        Transfer color statistics from source to target.
        """
        from FaceBlit.faceblit_pytorch.src.api import gray_hist_matching

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        source_np = tensor_to_numpy(source)
        target_np = tensor_to_numpy(target)

        if source_np.ndim == 3:
            result = target_np.copy()
            for c in range(3):
                result[:, :, c] = gray_hist_matching(
                    target_np[:, :, c], source_np[:, :, c]
                )
        else:
            result = gray_hist_matching(target_np, source_np)

        return (self._format_output(numpy_to_tensor(result)),)
