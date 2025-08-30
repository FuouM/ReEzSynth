# ruff: noqa: E402
import argparse
import os
import time

import torch

# This addresses the OpenMP runtime conflict.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ezsynth.project import Project


def main():
    """
    Main entry point for running the Ezsynth v2 pipeline.
    Parses command-line arguments, initializes a Project, and runs it.
    """
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
    args = parser.parse_args()

    # --- Welcome Message & Environment Check ---
    print("========================================")
    print("          Starting Ezsynth v2           ")
    print("========================================")

    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not found. Running on CPU. This will be very slow.")

    start_time = time.time()

    try:
        # --- Project Initialization ---
        print(f"\nLoading project with configuration: {args.config}")
        project = Project(config_path=args.config)

        # --- Pipeline Execution ---
        print("\nStarting Ezsynth v2 pipeline...")
        project.run()

    except FileNotFoundError as e:
        print(f"\n[ERROR] A required file or directory was not found: {e}")
        print("Please check the paths in your configuration file.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()

    finally:
        end_time = time.time()
        print("\n----------------------------------------")
        print(f"Pipeline finished in {end_time - start_time:.2f} seconds.")
        print("========================================")


if __name__ == "__main__":
    main()
