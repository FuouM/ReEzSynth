"""``reezsynth-video``: run a YAML video project (``python -m ezsynth.cli.video``)."""

import argparse
import logging
import os
import time
from pathlib import Path

from ezsynth.consts import RUNPY_STARTUP_ENV
from ezsynth.logging_setup import configure_cli_logging

for _key, _value in RUNPY_STARTUP_ENV.items():
    os.environ[_key] = _value

import torch

from ezsynth.project import Project
from ezsynth.utils.video import export_frames_to_browser_h264_mp4

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for running the Ezsynth v2 pipeline.
    Parses command-line arguments, initializes a Project, and runs it.
    """
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Run the Ezsynth v2 video-to-video synthesis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the project configuration YAML file.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cuda", "torch", "taichi"],
        default=None,
        help="Backend for synthesis operations (overrides config). Options: cuda, torch, taichi.",
    )
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="After synthesis, export H.264 MP4 (yuv420p, browser-friendly; requires ffmpeg).",
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
        help="Output .mp4 path. Default: next to the frame folder (<output_dir parent>/<name>.mp4).",
    )
    parser.add_argument(
        "--forward-warp",
        action="store_true",
        default=None,
        help="Use forward-warping paths for synthesis guide and blend-mask propagation.",
    )
    args = parser.parse_args()

    # --- Welcome Message & Environment Check ---
    logger.info("========================================")
    logger.info("          Starting Ezsynth v2           ")
    logger.info("========================================")

    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not found. Running on CPU. This will be very slow.")

    start_time = time.time()

    try:
        # --- Project Initialization ---
        logger.info(f"\nLoading project with configuration: {args.config}")
        project = Project(
            config_path=args.config,
            backend=args.backend,
            forward_warp=args.forward_warp,
        )

        # --- Pipeline Execution ---
        logger.info("\nStarting Ezsynth v2 pipeline...")
        final_frames = project.run()

        if args.export_mp4:
            if args.mp4_output:
                mp4_path = Path(args.mp4_output)
            else:
                od = project.data.output_dir
                mp4_path = od.parent / f"{od.name}.mp4"
            export_frames_to_browser_h264_mp4(final_frames, mp4_path, args.mp4_fps)

    except FileNotFoundError as e:
        logger.error(f"\n[ERROR] {e}")
        if "ffmpeg" not in str(e).lower():
            logger.error("Please check the paths in your configuration file.")
    except Exception as e:
        logger.exception(f"\n[ERROR] An unexpected error occurred: {e}")

    finally:
        end_time = time.time()
        logger.info("\n----------------------------------------")
        logger.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")
        logger.info("========================================")


if __name__ == "__main__":
    main()
