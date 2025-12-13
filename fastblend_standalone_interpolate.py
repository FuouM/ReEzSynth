#!/usr/bin/env python3
"""
Standalone FastBlend Interpolation Script

This script allows you to interpolate between keyframes using FastBlend.
Instead of processing every frame, you provide keyframes (stylized frames at specific intervals)
and FastBlend will interpolate the missing frames between them.

Usage:
    python fastblend_standalone_interpolate.py --frames_dir PATH_TO_ALL_FRAMES \
                                             --keyframes_dir PATH_TO_KEYFRAMES \
                                             --output_dir PATH_TO_SAVE_INTERPOLATED_FRAMES \
                                             --keyframe_interval N  # Process every Nth frame as keyframe
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional
import re

import numpy as np
from tqdm import tqdm

# Import after setting up environment
import torch

from FastBlend import FastBlendRunner, FastBlendConfig, create_config
from FastBlend.interpolation_runner import InterpolationModeRunner
from ezsynth.utils.io_utils import load_frames_from_dir, write_image


def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename using regex."""
    # Match common frame numbering patterns like frame_00001.png, 00001.png, frame00001.jpg, etc.
    patterns = [
        r'(\d{5,})\.png$',  # 00001.png
        r'(\d{5,})\.jpg$',  # 00001.jpg
        r'(\d{5,})\.jpeg$', # 00001.jpeg
        r'frame[_]?(\d{5,})\.png$',  # frame_00001.png or frame00001.png
        r'frame[_]?(\d{5,})\.jpg$',  # frame_00001.jpg or frame00001.jpg
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: try to find any sequence of 5+ digits
    match = re.search(r'(\d{5,})', filename)
    if match:
        return int(match.group(1))

    return None


def match_keyframes_to_frames(frame_files: List[str], keyframe_files: List[str]) -> Tuple[List[Optional[str]], List[int]]:
    """
    Match keyframes to frames based on frame numbers.

    Returns:
        Tuple of (matched_keyframes, keyframe_indices)
        - matched_keyframes: List where each element is either a keyframe filename or None
        - keyframe_indices: List of frame indices that have corresponding keyframes
    """
    # Extract frame numbers
    frame_numbers = []
    for filename in frame_files:
        frame_num = extract_frame_number(filename)
        frame_numbers.append(frame_num)

    keyframe_numbers = []
    for filename in keyframe_files:
        frame_num = extract_frame_number(filename)
        keyframe_numbers.append(frame_num)

    # Create mapping from frame number to keyframe filename
    keyframe_map = {num: filename for num, filename in zip(keyframe_numbers, keyframe_files) if num is not None}

    # Match keyframes to frames
    matched_keyframes = []
    keyframe_indices = []

    for i, frame_num in enumerate(frame_numbers):
        if frame_num in keyframe_map:
            matched_keyframes.append(keyframe_map[frame_num])
            keyframe_indices.append(i)
        else:
            matched_keyframes.append(None)

    return matched_keyframes, keyframe_indices


def create_interpolation_config(fastblend_params: dict = None) -> FastBlendConfig:
    """Create a FastBlend configuration optimized for interpolation."""

    # Default parameters optimized for interpolation (similar to third-party defaults)
    default_params = {
        "enabled": True,
        "accuracy": 2,  # Balanced mode
        "window_size": 5,  # Smaller window for interpolation
        "batch_size": 8,   # Smaller batch size for interpolation
        "minimum_patch_size": 15,  # Larger patch size for interpolation (as recommended)
        "num_iter": 5,
        "guide_weight": 10.0,
        "backend": "auto",
    }

    if fastblend_params:
        default_params.update(fastblend_params)

    # Create FastBlend config using the helper function
    return create_config(
        accuracy=default_params["accuracy"],
        window_size=default_params["window_size"],
        batch_size=default_params["batch_size"],
        minimum_patch_size=default_params["minimum_patch_size"],
        num_iter=default_params["num_iter"],
        guide_weight=default_params["guide_weight"],
        backend=default_params["backend"],
    )


def interpolate_frames(
    guide_frames: List[np.ndarray],
    style_frames: List[np.ndarray],
    keyframe_indices: List[int],
    config: FastBlendConfig,
    progress_callback: Optional[callable] = None
) -> List[np.ndarray]:
    """
    Interpolate frames between keyframes using FastBlend patch matching.

    Args:
        guide_frames: All guide frames (original content)
        style_frames: Keyframe style frames (stylized keyframes)
        keyframe_indices: Indices in guide_frames that correspond to keyframes
        config: FastBlend configuration
        progress_callback: Optional progress callback

    Returns:
        List of interpolated frames for all positions
    """

    if len(keyframe_indices) == 0:
        raise ValueError("No keyframes found")

    if len(keyframe_indices) == 1:
        # Single keyframe - use it for all frames
        single_keyframe = style_frames[0]
        return [single_keyframe] * len(guide_frames)

    # Check if keyframes cover all frames (no interpolation needed)
    if len(keyframe_indices) == len(guide_frames):
        # All frames are keyframes - just return them in order
        result_frames = [None] * len(guide_frames)
        for i, kf_idx in enumerate(keyframe_indices):
            if i < len(style_frames):
                result_frames[kf_idx] = style_frames[i]
        return result_frames

    # Convert frames to float32 for patch matching engine
    guide_frames_float = [frame.astype(np.float32) for frame in guide_frames]
    style_frames_float = [frame.astype(np.float32) for frame in style_frames]

    # Create interpolation runner
    interpolation_runner = InterpolationModeRunner()

    # Get ebsynth config from FastBlend config
    ebsynth_config = config.get_ebsynth_config()

    # Run interpolation
    result_frames_float = interpolation_runner.run(
        guide_frames_float,
        style_frames_float,
        keyframe_indices,
        batch_size=config.batch_size,
        ebsynth_config=ebsynth_config,
        progress_callback=progress_callback
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
            result_frames[i] = style_frames[kf_style_idx]

    return result_frames


def load_frames_sorted(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load frames from directory, sorted by filename."""
    path = Path(directory)

    # Use the existing load_frames_from_dir function which handles sorting and loading
    frames = load_frames_from_dir(path)
    print(f"Loaded {len(frames)} frames from {directory}")

    # Also get filenames for matching
    frame_files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    return frames, frame_files


def save_frames(frames: List[np.ndarray], output_dir: str, prefix: str = "interp_"):
    """Save frames to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(frames)} frames to {output_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        output_filename = f"{prefix}{i:05d}.png"
        write_image(output_path / output_filename, frame)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone FastBlend Interpolation for ReEzSynth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing all guide frames (original video frames)",
    )

    parser.add_argument(
        "--keyframes_dir",
        type=str,
        required=True,
        help="Directory containing stylized keyframe frames",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save interpolated frames",
    )

    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Interval between keyframes (e.g., 2 means use every 2nd frame as keyframe). If specified, uses frames at regular intervals from keyframes_dir.",
    )

    parser.add_argument(
        "--accuracy",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="FastBlend accuracy level (1=Fast, 2=Balanced, 3=Accurate)",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Window size for temporal blending",
    )

    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for processing"
    )

    parser.add_argument(
        "--minimum_patch_size",
        type=int,
        default=None,
        help="Minimum patch size for matching (odd numbers only)",
    )

    parser.add_argument(
        "--prefix", type=str, default="interp_", help="Prefix for output filenames"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "cuda", "cupy"],
        default="auto",
        help="Backend to use for FastBlend processing",
    )

    args = parser.parse_args()

    # Welcome message
    print("=" * 60)
    print("    Standalone FastBlend Interpolation")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (slower but works without CUDA)")

    # Load configuration
    fastblend_params = {}

    # Override with command line arguments
    if args.accuracy:
        fastblend_params["accuracy"] = args.accuracy
    if args.window_size:
        fastblend_params["window_size"] = args.window_size
    if args.batch_size:
        fastblend_params["batch_size"] = args.batch_size
    if args.minimum_patch_size:
        fastblend_params["minimum_patch_size"] = args.minimum_patch_size
    fastblend_params["backend"] = args.backend  # Always set backend

    print(f"FastBlend parameters: {fastblend_params}")

    # Create FastBlend configuration
    fastblend_config = create_interpolation_config(fastblend_params)

    # Load frames
    print("\nLoading input frames...")
    guide_frames, guide_filenames = load_frames_sorted(args.frames_dir)
    style_frames, style_filenames = load_frames_sorted(args.keyframes_dir)

    # Determine if we're using interval-based keyframes and whether we have
    # a style frame for every guide frame.
    interval_mode = args.keyframe_interval is not None and args.keyframe_interval > 1
    full_style_coverage = len(style_frames) == len(guide_frames)

    # Determine keyframes based on interval
    if interval_mode:
        # Use keyframes at regular intervals from available style frames
        #
        # Two typical cases:
        # 1) Only keyframes are present in keyframes_dir (style_frames shorter than guide_frames)
        # 2) All frames are present in keyframes_dir (style_frames same length as guide_frames)
        #
        # In case (2) we want to ignore the extra style frames and only use frames
        # at the desired interval positions, matching the behavior of case (1).
        if full_style_coverage:
            max_keyframes = (len(guide_frames) + args.keyframe_interval - 1) // args.keyframe_interval
        else:
            max_keyframes = min(
                len(style_frames),
                (len(guide_frames) + args.keyframe_interval - 1) // args.keyframe_interval,
            )
        keyframe_indices = [i * args.keyframe_interval for i in range(max_keyframes)]
        keyframe_indices = [idx for idx in keyframe_indices if idx < len(guide_frames)]

        if len(keyframe_indices) == 0:
            print(f"Error: No keyframes available for interval {args.keyframe_interval}")
            return

        print(f"\nUsing keyframes at interval {args.keyframe_interval}: positions {keyframe_indices}")
        print(f"Using {len(keyframe_indices)} keyframes from {len(style_frames)} available style frames")
    else:
        # Match keyframes to frames by filename (fallback behavior)
        print("\nMatching keyframes to frames...")
        matched_keyframes, keyframe_indices = match_keyframes_to_frames(guide_filenames, style_filenames)

        # Print matching summary
        matched_count = sum(1 for kf in matched_keyframes if kf is not None)
        print(f"Found {matched_count} keyframes out of {len(guide_frames)} total frames")
        print(f"Keyframe indices: {keyframe_indices}")

    print(f"Will interpolate between {len(keyframe_indices)} keyframes")

    if len(keyframe_indices) == 0:
        print("Error: No keyframes matched to frames")
        return

    # Only warn when we have fewer style frames than keyframe positions.
    # Having *more* style frames is fine: we just ignore the extras.
    if len(style_frames) < len(keyframe_indices):
        print(
            f"Warning: Only {len(style_frames)} style frames available for "
            f"{len(keyframe_indices)} keyframe positions"
        )
        print("Some keyframes may be reused and interpolation quality could be reduced")

    # Validate frame compatibility
    if len(guide_frames) > 0 and len(style_frames) > 0:
        h_guide, w_guide = guide_frames[0].shape[:2]
        h_style, w_style = style_frames[0].shape[:2]

        if h_guide != h_style or w_guide != w_style:
            print(f"Warning: Frame size mismatch - guide: {h_guide}x{w_guide}, style: {h_style}x{w_style}")

    print(f"\nInterpolating {len(guide_frames)} frames using {len(keyframe_indices)} keyframes...")
    print(f"FastBlend Configuration:")
    print(f"  - Accuracy: {fastblend_config.accuracy}")
    print(f"  - Window size: {fastblend_config.window_size}")
    print(f"  - Batch size: {fastblend_config.batch_size}")
    print(f"  - Minimum patch size: {fastblend_config.minimum_patch_size}")
    print(f"  - Number of iterations: {fastblend_config.num_iter}")
    print(f"  - Guide weight: {fastblend_config.guide_weight}")
    print(f"  - Backend: {fastblend_config.backend}")

    # Extract keyframes in the correct order
    if interval_mode and full_style_coverage:
        # We have a style frame for every guide frame; select only frames at the
        # computed keyframe positions so that excess style frames are ignored.
        keyframes_to_use = [style_frames[idx] for idx in keyframe_indices]
    else:
        # Fallback: assume style_frames are already ordered to match the
        # keyframe_indices sequence (original behavior).
        keyframes_to_use = style_frames[:len(keyframe_indices)]

    # Interpolate frames
    interpolated_frames = interpolate_frames(
        guide_frames, keyframes_to_use, keyframe_indices, fastblend_config
    )

    # Save results
    print(f"\nSaving interpolated results...")
    save_frames(interpolated_frames, args.output_dir, args.prefix)

    print(f"\nInterpolation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total frames processed: {len(interpolated_frames)}")
    print(f"Keyframes used: {len(keyframe_indices)}")


if __name__ == "__main__":
    main()