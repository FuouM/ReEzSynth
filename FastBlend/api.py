"""
FastBlend API - Clean interface for video frame post-processing.

This module provides a simple API for applying FastBlend to stylized video frames.
FastBlend smooths temporal inconsistencies in frame-by-frame stylization results.
"""

from typing import Callable, List, Optional

import numpy as np

from .config import FastBlendConfig, get_default_config
from .interpolation_runner import InterpolationModeRunner
from .runner import FastBlendRunner


def smooth_video(
    frames_guide: List[np.ndarray],
    frames_style: List[np.ndarray],
    accuracy: int = 2,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backend: str = "auto",
    config: Optional[FastBlendConfig] = None,
) -> List[np.ndarray]:
    """
    Apply FastBlend post-processing to stylized video frames.

    Args:
        frames_guide: List of guide frames (content frames) as numpy arrays (H, W, 3) uint8
        frames_style: List of stylized frames to smooth as numpy arrays (H, W, 3) uint8
        accuracy: Accuracy level (1=Fast, 2=Balanced, 3=Accurate). Ignored if config is provided.
        window_size: Temporal window size for blending. Uses default if None.
        batch_size: Batch size for processing. Uses default if None.
        progress_callback: Optional callback function called with (current_frame, total_frames)
        backend: Backend to use ("auto", "cuda", "cupy")
        config: Optional FastBlendConfig. If provided, other parameters are ignored.

    Returns:
        List of smoothed frames as numpy arrays (H, W, 3) uint8

    Example:
        >>> import numpy as np
        >>> from fastblend import smooth_video
        >>>
        >>> # Create dummy frames (replace with your actual frames)
        >>> guide_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        >>> style_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
        >>>
        >>> # Apply FastBlend
        >>> smoothed = smooth_video(guide_frames, style_frames, accuracy=2)
    """
    # Create configuration
    if config is None:
        config = get_default_config(accuracy)
        # Override specific parameters if provided
        if window_size is not None:
            config.window_size = window_size
        if batch_size is not None:
            config.batch_size = batch_size

    # Create and run the processor
    runner = FastBlendRunner(config)
    return runner.run(frames_guide, frames_style, progress_callback, backend)


def create_config(
    accuracy: int = 2,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    minimum_patch_size: Optional[int] = None,
    num_iter: Optional[int] = None,
    guide_weight: Optional[float] = None,
    backend: str = "auto",
    gpu_id: int = 0,
) -> FastBlendConfig:
    """
    Create a FastBlend configuration with custom parameters.

    Args:
        accuracy: Base accuracy level (1=Fast, 2=Balanced, 3=Accurate)
        window_size: Temporal window size for blending
        batch_size: Batch size for processing
        minimum_patch_size: Minimum patch size for matching
        num_iter: Number of patch matching iterations
        guide_weight: Weight for guide matching
        backend: Backend to use ("auto", "cuda", "cupy")
        gpu_id: GPU device ID

    Returns:
        FastBlendConfig instance

    Example:
        >>> from fastblend import create_config, smooth_video
        >>>
        >>> config = create_config(
        ...     accuracy=2,
        ...     window_size=10,
        ...     batch_size=8,
        ...     guide_weight=12.0
        ... )
        >>> smoothed = smooth_video(guide_frames, style_frames, config=config)
    """
    config = get_default_config(accuracy)

    # Override parameters if provided
    if window_size is not None:
        config.window_size = window_size
    if batch_size is not None:
        config.batch_size = batch_size
    if minimum_patch_size is not None:
        config.minimum_patch_size = minimum_patch_size
    if num_iter is not None:
        config.num_iter = num_iter
    if guide_weight is not None:
        config.guide_weight = guide_weight

    config.backend = backend
    config.gpu_id = gpu_id

    return config


def check_frames_compatibility(
    frames_guide: List[np.ndarray], frames_style: List[np.ndarray]
) -> str:
    """
    Check compatibility of guide and style frames.

    Args:
        frames_guide: Guide frames
        frames_style: Style frames

    Returns:
        Warning message if issues found, empty string otherwise
    """
    message = ""

    if len(frames_guide) != len(frames_style):
        message += f"Frame count mismatch: guide has {len(frames_guide)} frames, style has {len(frames_style)} frames. "

    if len(frames_guide) > 0 and len(frames_style) > 0:
        h_guide, w_guide = frames_guide[0].shape[:2]
        h_style, w_style = frames_style[0].shape[:2]

        if h_guide != h_style or w_guide != w_style:
            message += f"Frame size mismatch: guide is {h_guide}x{w_guide}, style is {h_style}x{w_style}. "

    return message


def interpolate_video(
    frames_guide: List[np.ndarray],
    keyframes_style: List[np.ndarray],
    keyframe_indices: List[int],
    accuracy: int = 2,
    window_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    backend: str = "auto",
    config: Optional[FastBlendConfig] = None,
) -> List[np.ndarray]:
    """
    Interpolate frames between keyframes using FastBlend.

    Args:
        frames_guide: List of all guide frames (content frames) as numpy arrays (H, W, 3) uint8
        keyframes_style: List of stylized keyframe frames as numpy arrays (H, W, 3) uint8
        keyframe_indices: Indices in frames_guide that correspond to keyframes
        accuracy: Accuracy level (1=Fast, 2=Balanced, 3=Accurate). Default: 2
        window_size: Temporal window size for blending. Uses default if None.
        batch_size: Batch size for processing. Uses default if None.
        progress_callback: Optional callback function called with (current_frame, total_frames)
        backend: Backend to use ("auto", "cuda", "cupy"). Default: "auto"
        config: Optional FastBlendConfig. If provided, other parameters are ignored.

    Returns:
        List of interpolated frames as numpy arrays (H, W, 3) uint8

    Example:
        >>> import numpy as np
        >>> from fastblend import interpolate_video
        >>>
        >>> # Load all frames and keyframes
        >>> all_frames = load_frames_from_dir("path/to/all/frames")
        >>> keyframe_frames = load_frames_from_dir("path/to/keyframes")
        >>> keyframe_indices = [0, 10, 20, 30]  # keyframes at frames 0, 10, 20, 30
        >>>
        >>> # Interpolate
        >>> interpolated = interpolate_video(all_frames, keyframe_frames, keyframe_indices)
    """
    # Handle special cases
    if len(keyframe_indices) == 1:
        # Single keyframe - use it for all frames
        single_keyframe = keyframes_style[0]
        return [single_keyframe] * len(frames_guide)

    # Check if keyframes cover all frames (no interpolation needed)
    if len(keyframe_indices) == len(frames_guide):
        # All frames are keyframes - just return them in order
        result_frames = [None] * len(frames_guide)
        for i, kf_idx in enumerate(keyframe_indices):
            if i < len(keyframes_style):
                result_frames[kf_idx] = keyframes_style[i]
        # Fill any None values
        for i in range(len(result_frames)):
            if result_frames[i] is None:
                # Find nearest keyframe
                distances = [(abs(i - kf_idx), kf_idx) for kf_idx in keyframe_indices]
                nearest_kf_idx = min(distances)[1]
                kf_style_idx = keyframe_indices.index(nearest_kf_idx)
                result_frames[i] = keyframes_style[kf_style_idx]
        return result_frames

    # Create configuration optimized for interpolation
    if config is None:
        config = get_default_config(accuracy)
        # Override with interpolation-optimized defaults
        config.minimum_patch_size = 15  # Larger patch size for interpolation
        config.batch_size = min(config.batch_size, 16)  # Smaller batch size

        # Override specific parameters if provided
        if window_size is not None:
            config.window_size = window_size
        if batch_size is not None:
            config.batch_size = batch_size

    config.backend = backend

    # Convert frames to float32 for patch matching engine
    guide_frames_float = [frame.astype(np.float32) for frame in frames_guide]
    keyframes_style_float = [frame.astype(np.float32) for frame in keyframes_style]

    # Create interpolation runner
    interpolation_runner = InterpolationModeRunner()

    # Get ebsynth config from FastBlend config
    ebsynth_config = config.get_ebsynth_config()

    # Run interpolation
    result_frames_float = interpolation_runner.run(
        guide_frames_float,
        keyframes_style_float,
        keyframe_indices,
        batch_size=config.batch_size,
        ebsynth_config=ebsynth_config,
        progress_callback=progress_callback,
    )

    # Convert back to uint8
    result_frames = [
        np.clip(frame, 0, 255).astype(np.uint8) if frame is not None else None
        for frame in result_frames_float
    ]

    # Fill any None values (shouldn't happen with proper implementation)
    for i in range(len(result_frames)):
        if result_frames[i] is None:
            # Find nearest keyframe
            distances = [(abs(i - kf_idx), kf_idx) for kf_idx in keyframe_indices]
            nearest_kf_idx = min(distances)[1]
            kf_style_idx = keyframe_indices.index(nearest_kf_idx)
            result_frames[i] = keyframes_style[kf_style_idx]

    return result_frames
