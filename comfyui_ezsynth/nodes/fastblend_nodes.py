# comfyui_ezsynth/nodes/fastblend_nodes.py
"""
FastBlend post-processing nodes for temporal smoothing.
"""

from typing import List, Optional, Tuple

import torch

from .base import EZBaseNode


class FastBlendNode(EZBaseNode):
    """
    Apply temporal smoothing to stylized frames using FastBlend.
    """

    CATEGORY = "ReEzSynth/FastBlend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_frames": ("IMAGE_LIST",),
                "style_frames": ("IMAGE_LIST",),
            },
            "optional": {
                "accuracy": ([1, 2, 3], {"default": 2}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 21}),
                "batch_size": ("INT", {"default": 16, "min": 1}),
                "guide_weight": ("FLOAT", {"default": 10.0}),
                "backend": (["auto", "cuda", "cupy"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("smoothed_frames",)
    FUNCTION = "run_fastblend"

    def run_fastblend(
        self,
        content_frames: List[torch.Tensor],
        style_frames: List[torch.Tensor],
        accuracy: int = 2,
        window_size: int = 5,
        batch_size: int = 16,
        guide_weight: float = 10.0,
        backend: str = "auto",
    ) -> Tuple[List[torch.Tensor]]:
        """
        Apply FastBlend temporal smoothing.
        """
        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        content_np = [tensor_to_numpy(f) for f in content_frames]
        style_np = [tensor_to_numpy(f) for f in style_frames]

        try:
            from FastBlend import FastBlendRunner, create_config

            config = create_config(
                accuracy=accuracy,
                window_size=window_size,
                batch_size=batch_size,
                minimum_patch_size=5,
                num_iter=5,
                guide_weight=guide_weight,
                backend=backend,
            )

            runner = FastBlendRunner(config)
            result = runner.run(content_np, style_np)

            return ([numpy_to_tensor(f) for f in result],)

        except ImportError:
            raise ImportError("FastBlend is not installed.")


class FastBlendInterpolateNode(EZBaseNode):
    """
    Interpolate frames between keyframes using FastBlend.
    """

    CATEGORY = "ReEzSynth/FastBlend"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_frames": ("IMAGE_LIST",),
                "keyframes": ("IMAGE_LIST",),
                "keyframe_indices": ("STRING", {"default": "0,10,20"}),
            },
            "optional": {
                "accuracy": ([1, 2, 3], {"default": 2}),
                "window_size": ("INT", {"default": 5, "min": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    RETURN_NAMES = ("interpolated_frames",)
    FUNCTION = "run_interpolation"

    def run_interpolation(
        self,
        guide_frames: List[torch.Tensor],
        keyframes: List[torch.Tensor],
        keyframe_indices: str = "0,10,20",
        accuracy: int = 2,
        window_size: int = 5,
    ) -> Tuple[List[torch.Tensor]]:
        """
        Interpolate between keyframes.
        """
        indices = [int(x.strip()) for x in keyframe_indices.split(",")]

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        guide_np = [tensor_to_numpy(f) for f in guide_frames]
        keyframe_np = [tensor_to_numpy(f) for f in keyframes]

        try:
            from FastBlend import InterpolationModeRunner, create_interp_config

            config = create_interp_config(
                accuracy=accuracy,
                window_size=window_size,
                batch_size=8,
                minimum_patch_size=15,
            )

            runner = InterpolationModeRunner(config)
            result = runner.run(guide_np, keyframe_np, indices)

            return ([numpy_to_tensor(f) for f in result],)

        except ImportError:
            raise ImportError("FastBlend is not installed.")
