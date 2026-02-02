# comfyui_ezsynth/nodes/input_nodes.py
"""
Input nodes for loading images and videos.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..types import IMAGE_LIST
from .base import EZBaseNode


class LoadImageNode(EZBaseNode):
    """
    Load a single image from file path.
    """

    CATEGORY = "ReEzSynth/Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, image_path: str) -> Tuple[torch.Tensor]:
        """
        Load an image from the specified path.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple containing the loaded image tensor
        """
        from ..core.tensor_utils import numpy_to_tensor

        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        # Try cv2 first, fall back to PIL
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image

            img = np.array(Image.open(image_path).convert("RGB"))

        return (self._format_output(numpy_to_tensor(img)),)


class LoadVideoNode(EZBaseNode):
    """
    Load video frames from a directory.
    """

    CATEGORY = "ReEzSynth/Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "frame_extension": (["png", "jpg", "jpeg"], {"default": "png"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "load_video"

    def load_video(
        self, frames_dir: str, frame_extension: str = "png", max_frames: int = 0
    ) -> Tuple[List[torch.Tensor]]:
        """
        Load video frames from a directory.

        Args:
            frames_dir: Directory containing frame images
            frame_extension: Extension of frame files
            max_frames: Maximum number of frames to load (0 = all)

        Returns:
            Tuple containing list of frame tensors
        """
        from ezsynth.utils.io_utils import load_frames_from_dir

        from ..core.tensor_utils import numpy_to_tensor

        if not frames_dir or not os.path.exists(frames_dir):
            raise ValueError(f"Frames directory does not exist: {frames_dir}")

        frames = load_frames_from_dir(Path(frames_dir))

        if max_frames > 0 and len(frames) > max_frames:
            frames = frames[:max_frames]

        tensors = [numpy_to_tensor(f) for f in frames]

        return (tensors,)


class LoadFlowNode(EZBaseNode):
    """
    Load precomputed optical flow from files.
    """

    CATEGORY = "ReEzSynth/Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_dir": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "max_flows": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOW_LIST",)
    RETURN_NAMES = ("flows",)
    FUNCTION = "load_flow"

    def load_flow(self, flow_dir: str, max_flows: int = 0) -> Tuple[List[torch.Tensor]]:
        """
        Load optical flow fields from .npy files.

        Args:
            flow_dir: Directory containing .npy flow files
            max_flows: Maximum number of flows to load (0 = all)

        Returns:
            Tuple containing list of flow tensors
        """
        from ..core.tensor_utils import flow_to_tensor

        if not flow_dir or not os.path.exists(flow_dir):
            raise ValueError(f"Flow directory does not exist: {flow_dir}")

        flow_files = sorted(Path(flow_dir).glob("*.npy"))

        if max_flows > 0:
            flow_files = flow_files[:max_flows]

        flows = []
        for f in flow_files:
            flow_np = np.load(str(f))
            flows.append(flow_to_tensor(flow_np))

        return (flows,)
