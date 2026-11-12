"""Which implementation runs the video pipeline's image operations.

Backward warping, forward splatting, histogram blending and occlusion masks run on
OpenCV/NumPy by default. Forward splatting and ``use_taichi_ops`` need a GPU engine:
Taichi, or WebGPU when the synthesis backend is ``wgpu`` (then every op runs there).
"""

from __future__ import annotations

from typing import Literal

ImageOpsEngine = Literal["opencv", "taichi", "wgpu"]


def resolve_image_ops(backend: str, *, gpu_ops: bool = False) -> ImageOpsEngine:
    """Engine for the image ops of a run with synthesis ``backend``.

    ``gpu_ops``: Taichi ops or forward splatting were requested.
    """
    if backend == "wgpu":
        return "wgpu"
    return "taichi" if gpu_ops else "opencv"
