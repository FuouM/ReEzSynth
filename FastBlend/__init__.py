"""
FastBlend - Temporal smoothing for frame-by-frame stylization results.

This package provides tools to smooth temporal inconsistencies in stylized video frames
by finding and blending similar patches across frames.
"""

from .api import smooth_video, create_config, check_frames_compatibility, interpolate_video
from .config import FastBlendConfig, get_default_config
from .runner import FastBlendRunner

__version__ = "1.0.0"
__all__ = [
    "smooth_video",
    "create_config",
    "check_frames_compatibility",
    "interpolate_video",
    "FastBlendConfig",
    "get_default_config",
    "FastBlendRunner",
]