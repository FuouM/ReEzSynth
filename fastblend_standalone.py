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

    Optional MP4 (same pattern as run.py): add ``--export-mp4``; use ``--mp4-fps`` and
    ``--mp4-output`` to override defaults (requires ffmpeg on PATH).
"""

import argparse
from pathlib import Path

import torch
from FastBlend.src.engine import FastBlendEngine, FastBlendInput_Sequence

from ezsynth.utils.fastblend_cli import (
    create_fastblend_config,
    load_frames_sorted,
    save_frames,
)
from ezsynth.utils.video import export_frames_to_browser_h264_mp4


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
        choices=["auto", "cuda", "cupy", "taichi"],
        default="auto",
        help="Backend to use for FastBlend processing: auto=prefer CUDA, fallback to cupy; cuda=CUDA only; cupy=CuPy only",
    )

    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="After saving frames, export H.264 MP4 (yuv420p, browser-friendly; requires ffmpeg).",
    )
    parser.add_argument(
        "--mp4-fps",
        type=float,
        default=30.0,
        help="Frame rate for --export-mp4.",
    )
    parser.add_argument(
        "--mp4-output",
        type=str,
        default=None,
        help="Output .mp4 path. Default: next to the frame folder (<output_dir parent>/<output_dir name>.mp4).",
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
    print("FastBlend Configuration:")
    print(f"  - Accuracy: {fastblend_config.accuracy}")
    print(f"  - Window size: {fastblend_config.window_size}")
    print(f"  - Batch size: {fastblend_config.batch_size}")
    print(f"  - Minimum patch size: {fastblend_config.minimum_patch_size}")
    print(f"  - Number of iterations: {fastblend_config.num_iter}")
    print(f"  - Guide weight: {fastblend_config.guide_weight}")
    print(f"  - Backend: {fastblend_config.backend}")

    if not fastblend_config.enabled:
        print("FastBlend is disabled in configuration")
        return

    def progress_callback(_current_frame, _total_frames):
        pass

    engine = FastBlendEngine()
    fastblend_frames = engine.smooth_sequence(
        FastBlendInput_Sequence(content_frames, style_frames),
        fastblend_config,
        progress_callback=progress_callback,
        backend=fastblend_config.backend,
    )

    # Save results
    print("\nSaving FastBlend results...")
    save_frames(fastblend_frames, args.output_dir, args.prefix)

    if args.export_mp4:
        try:
            if args.mp4_output:
                mp4_path = Path(args.mp4_output)
            else:
                od = Path(args.output_dir)
                mp4_path = od.parent / f"{od.name}.mp4"
            export_frames_to_browser_h264_mp4(
                fastblend_frames, mp4_path, args.mp4_fps
            )
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            if "ffmpeg" not in str(e).lower():
                print("Please check --output_dir, --mp4-output, and related paths.")
        except Exception as e:
            print(f"\n[ERROR] MP4 export failed: {e}")
            import traceback

            traceback.print_exc()

    print("\nFastBlend processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total frames processed: {len(fastblend_frames)}")


if __name__ == "__main__":
    main()
