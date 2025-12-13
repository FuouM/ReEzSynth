#!/usr/bin/env python3
"""
Standalone FastBlend Post-Processing Script

This script allows you to apply FastBlend to existing stylized frames independently,
without running the full ReEzSynth pipeline. It's useful for:
- Post-processing existing outputs
- Testing different FastBlend parameters
- Applying FastBlend to outputs from other stylization methods

Usage:
    python fastblend_standalone.py --content_dir PATH_TO_CONTENT_FRAMES \
                                  --style_dir PATH_TO_STYLIZED_FRAMES \
                                  --output_dir PATH_TO_SAVE_FASTBLENDED_FRAMES
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

# Import after setting up environment
import torch
from tqdm import tqdm

from FastBlend import FastBlendRunner, FastBlendConfig, create_config
from ezsynth.utils.io_utils import load_frames_from_dir, write_image


def create_fastblend_config(fastblend_params: dict = None) -> FastBlendConfig:
    """Create a FastBlend configuration for standalone processing."""

    # Default FastBlend parameters
    default_params = {
        "enabled": True,
        "accuracy": 2,  # Balanced mode
        "window_size": 5,  # Increased default for better temporal smoothing
        "batch_size": 16,
        "minimum_patch_size": 5,
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


def load_frames_sorted(directory: str) -> List[np.ndarray]:
    """Load frames from directory, sorted by filename."""
    path = Path(directory)

    # Use the existing load_frames_from_dir function which handles sorting and loading
    frames = load_frames_from_dir(path)
    print(f"Loaded {len(frames)} frames from {directory}")

    return frames


def save_frames(frames: List[np.ndarray], output_dir: str, prefix: str = "fastblend_"):
    """Save frames to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(frames)} frames to {output_dir}")
    for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
        output_filename = f"{prefix}{i:05d}.png"
        write_image(output_path / output_filename, frame)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone FastBlend Post-Processing for ReEzSynth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--content_dir",
        type=str,
        required=True,
        help="Directory containing content/guide frames (original video frames)",
    )

    parser.add_argument(
        "--style_dir",
        type=str,
        required=True,
        help="Directory containing stylized frames to be processed with FastBlend",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save FastBlend-processed frames",
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
        "--prefix", type=str, default="fastblend_", help="Prefix for output filenames"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "cuda", "cupy"],
        default="auto",
        help="Backend to use for FastBlend processing: auto=prefer CUDA, fallback to cupy; cuda=CUDA only; cupy=CuPy only",
    )

    args = parser.parse_args()

    # Welcome message
    print("=" * 60)
    print("    Standalone FastBlend Post-Processing")
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
    fastblend_params["backend"] = args.backend  # Always set backend

    print(f"FastBlend parameters: {fastblend_params}")

    # Create FastBlend configuration
    fastblend_config = create_fastblend_config(fastblend_params)

    # Load frames
    print("\nLoading input frames...")
    content_frames = load_frames_sorted(args.content_dir)
    style_frames = load_frames_sorted(args.style_dir)

    # Validate frame counts
    if len(content_frames) != len(style_frames):
        print(
            f"Warning: Content frames ({len(content_frames)}) and style frames ({len(style_frames)}) have different counts"
        )
        print(f"Using minimum count: {min(len(content_frames), len(style_frames))}")
        min_count = min(len(content_frames), len(style_frames))
        content_frames = content_frames[:min_count]
        style_frames = style_frames[:min_count]

    if len(content_frames) == 0:
        print("Error: No frames to process")
        return

    print(f"Processing {len(content_frames)} frames with FastBlend...")
    print(f"FastBlend Configuration:")
    print(f"  - Accuracy: {fastblend_config.accuracy}")
    print(f"  - Window size: {fastblend_config.window_size}")
    print(f"  - Batch size: {fastblend_config.batch_size}")
    print(f"  - Minimum patch size: {fastblend_config.minimum_patch_size}")
    print(f"  - Number of iterations: {fastblend_config.num_iter}")
    print(f"  - Guide weight: {fastblend_config.guide_weight}")
    print(f"  - Backend: {fastblend_config.backend}")

    # Create and run FastBlend
    fastblend_runner = FastBlendRunner(fastblend_config)

    if not fastblend_config.enabled:
        print("FastBlend is disabled in configuration")
        return

    # Progress callback for FastBlend
    def progress_callback(current_frame, total_frames):
        progress_percent = (current_frame / total_frames) * 100
        # print(f"FastBlend Progress: {current_frame}/{total_frames} frames completed ({progress_percent:.1f}%)")

    # Run FastBlend
    fastblend_frames = fastblend_runner.run(
        content_frames, style_frames, progress_callback=progress_callback
    )

    # Save results
    print(f"\nSaving FastBlend results...")
    save_frames(fastblend_frames, args.output_dir, args.prefix)

    print(f"\nFastBlend processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total frames processed: {len(fastblend_frames)}")


if __name__ == "__main__":
    main()
