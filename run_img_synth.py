import gc
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch

from ezsynth.api import ImageSynth, RunConfig, load_guide
from ezsynth.utils.io_utils import write_image

# Parse command line arguments
parser = ArgumentParser(description="Run EBSynth image synthesis examples")
parser.add_argument(
    "--backend",
    choices=["cuda", "torch"],
    default="cuda",
    help="Backend to use for synthesis (default: cuda)",
)
parser.add_argument(
    "--full-params",
    action="store_true",
    help="Run with full synthesis parameters (default: fast parameters)",
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Enable detailed benchmarking and timing output",
)

args = parser.parse_args()

print(f"Using backend: {args.backend}")
print(f"Using full parameters: {args.full_params}")


def save_synth_result(output_dir, base_name, result_img, result_err):
    """Saves the synthesis result image and its error map."""
    write_image(f"{output_dir}/{base_name}_out.png", result_img)

    # Error maps are float32, need to normalize for saving as visible image
    # Avoid division by zero if error map is all zeros
    max_err = result_err.max()
    if max_err > 1e-6:
        result_err_vis = (255 * (result_err / max_err)).astype(np.uint8)
    else:
        result_err_vis = result_err.astype(np.uint8)

    write_image(f"{output_dir}/{base_name}_err.png", result_err_vis)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


st = time.time()

# --- Setup Paths ---
EXAMPLES_DIR = "examples"
OUTPUT_DIR = (
    f"output_synth_api_{args.backend}_{'full' if args.full_params else 'minimal'}"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving output to: {OUTPUT_DIR}")

# --- Example 1: Segment Retargeting ---
print("\n--- Running: Segment Retargeting ---")
start_example_time = time.time()
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/texbynum/source_photo.png",
    config=RunConfig(backend=args.backend, image_weight=1.0)
    if args.full_params
    else RunConfig(
        backend=args.backend, image_weight=1.0, pyramid_levels=1, search_vote_iters=1
    ),
)

# The main source/target pair is now passed as a guide to run()
result_img, result_err = ezsynner.run(
    guides=[
        load_guide(
            f"{EXAMPLES_DIR}/texbynum/source_segment.png",
            f"{EXAMPLES_DIR}/texbynum/target_segment.png",
            weight=1.0,  # The weight is now explicit per-guide
        )
    ],
    benchmark=args.benchmark,
)
save_synth_result(OUTPUT_DIR, "retarget", result_img, result_err)
print(f"Segment Retargeting took: {time.time() - start_example_time:.4f} s")


# --- Example 2: Stylit ---
print("\n--- Running: Stylit ---")
start_example_time = time.time()
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/stylit/source_style.png",
    config=RunConfig(backend=args.backend)
    if args.full_params
    else RunConfig(backend=args.backend, pyramid_levels=1, search_vote_iters=1),
)

result_img, result_err = ezsynner.run(
    guides=[
        load_guide(
            f"{EXAMPLES_DIR}/stylit/source_fullgi.png",
            f"{EXAMPLES_DIR}/stylit/target_fullgi.png",
            weight=0.66,
        ),
        load_guide(
            f"{EXAMPLES_DIR}/stylit/source_dirdif.png",
            f"{EXAMPLES_DIR}/stylit/target_dirdif.png",
            weight=0.66,
        ),
        load_guide(
            f"{EXAMPLES_DIR}/stylit/source_indirb.png",
            f"{EXAMPLES_DIR}/stylit/target_indirb.png",
            weight=0.66,
        ),
    ],
    benchmark=args.benchmark,
)
save_synth_result(OUTPUT_DIR, "stylit", result_img, result_err)
print(f"Stylit took: {time.time() - start_example_time:.4f} s")

clear_memory()


# --- Example 3: Face Style ---
print("\n--- Running: Face Style ---")
start_example_time = time.time()
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/facestyle/source_painting.png",
    config=RunConfig(backend=args.backend)
    if args.full_params
    else RunConfig(backend=args.backend, pyramid_levels=1, search_vote_iters=1),
)

result_img, result_err = ezsynner.run(
    guides=[
        load_guide(
            f"{EXAMPLES_DIR}/facestyle/source_Gapp.png",
            f"{EXAMPLES_DIR}/facestyle/target_Gapp.png",
            weight=2.0,
        ),
        load_guide(
            f"{EXAMPLES_DIR}/facestyle/source_Gseg.png",
            f"{EXAMPLES_DIR}/facestyle/target_Gseg.png",
            weight=1.5,
        ),
        load_guide(
            f"{EXAMPLES_DIR}/facestyle/source_Gpos.png",
            f"{EXAMPLES_DIR}/facestyle/target_Gpos.png",
            weight=1.5,
        ),
    ],
    benchmark=args.benchmark,
)
save_synth_result(OUTPUT_DIR, "facestyle", result_img, result_err)
print(f"Face Style took: {time.time() - start_example_time:.4f} s")

clear_memory()


# --- Cleanup ---
clear_memory()

print(f"\nTotal time taken: {time.time() - st:.4f} s")
