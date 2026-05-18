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

import torch

from ezsynth.utils.fastblend_cli import (
    create_interpolation_config,
    interpolate_frames,
    load_frames_sorted_with_names,
    save_frames,
)
from ezsynth.utils.frame_numbers import match_keyframes_to_frames


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
        choices=["auto", "cuda", "cupy", "taichi"],
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
    guide_frames, guide_filenames = load_frames_sorted_with_names(args.frames_dir)
    style_frames, style_filenames = load_frames_sorted_with_names(args.keyframes_dir)

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
            max_keyframes = (
                len(guide_frames) + args.keyframe_interval - 1
            ) // args.keyframe_interval
        else:
            max_keyframes = min(
                len(style_frames),
                (len(guide_frames) + args.keyframe_interval - 1)
                // args.keyframe_interval,
            )
        keyframe_indices = [i * args.keyframe_interval for i in range(max_keyframes)]
        keyframe_indices = [idx for idx in keyframe_indices if idx < len(guide_frames)]

        if len(keyframe_indices) == 0:
            print(
                f"Error: No keyframes available for interval {args.keyframe_interval}"
            )
            return

        print(
            f"\nUsing keyframes at interval {args.keyframe_interval}: positions {keyframe_indices}"
        )
        print(
            f"Using {len(keyframe_indices)} keyframes from {len(style_frames)} available style frames"
        )
    else:
        # Match keyframes to frames by filename (fallback behavior)
        print("\nMatching keyframes to frames...")
        matched_keyframes, keyframe_indices = match_keyframes_to_frames(
            guide_filenames, style_filenames
        )

        # Print matching summary
        matched_count = sum(1 for kf in matched_keyframes if kf is not None)
        print(
            f"Found {matched_count} keyframes out of {len(guide_frames)} total frames"
        )
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
            print(
                f"Warning: Frame size mismatch - guide: {h_guide}x{w_guide}, style: {h_style}x{w_style}"
            )

    print(
        f"\nInterpolating {len(guide_frames)} frames using {len(keyframe_indices)} keyframes..."
    )
    print("FastBlend Configuration:")
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
        keyframes_to_use = style_frames[: len(keyframe_indices)]

    # Interpolate frames
    interpolated_frames = interpolate_frames(
        guide_frames, keyframes_to_use, keyframe_indices, fastblend_config
    )

    # Save results
    print("\nSaving interpolated results...")
    save_frames(interpolated_frames, args.output_dir, args.prefix)

    print("\nInterpolation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total frames processed: {len(interpolated_frames)}")
    print(f"Keyframes used: {len(keyframe_indices)}")


if __name__ == "__main__":
    main()
