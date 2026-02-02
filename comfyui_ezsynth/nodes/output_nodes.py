# comfyui_ezsynth/nodes/output_nodes.py
"""
Output nodes for saving and previewing results.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import EZBaseNode


class SaveVideoNode(EZBaseNode):
    """
    Save video frames to a directory.
    """

    CATEGORY = "ReEzSynth/Output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
                "output_dir": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "prefix": ("STRING", {"default": "frame_", "multiline": False}),
                "format": (["png", "jpg"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_video"

    def save_video(
        self,
        frames: List[torch.Tensor],
        output_dir: str,
        prefix: str = "frame_",
        format: str = "png",
    ) -> Tuple[str]:
        """
        Save frames to output directory.
        """
        from ezsynth.utils.io_utils import write_image

        from ..core.tensor_utils import tensor_to_numpy

        if not frames:
            return ("No frames to save",)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_np = tensor_to_numpy(frame)
            output_path = Path(output_dir) / f"{prefix}{i:05d}.{format}"
            write_image(output_path, frame_np)

        return (f"Saved {len(frames)} frames to {output_dir}",)


class PreviewImageNode(EZBaseNode):
    """
    Preview an image in the ComfyUI UI.
    """

    CATEGORY = "ReEzSynth/Output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview_image"

    def preview_image(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Preview an image (passes through unchanged).
        """
        if image.dim() == 4:
            image = image[0]
        return (image,)


class SaveImageNode(EZBaseNode):
    """
    Save a single image to file.
    """

    CATEGORY = "ReEzSynth/Output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_image"

    def save_image(self, image: torch.Tensor, output_path: str) -> Tuple[str]:
        """
        Save a single image to file.
        """
        from ezsynth.utils.io_utils import write_image

        from ..core.tensor_utils import tensor_to_numpy

        image_np = tensor_to_numpy(image)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_image(Path(output_path), image_np)

        return (f"Saved image to {output_path}",)
