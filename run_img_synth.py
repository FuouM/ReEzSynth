import gc
import os
import time

import numpy as np
import torch

from ezsynth.api import ImageSynth, RunConfig, load_guide
from ezsynth.utils.io_utils import write_image

st = time.time()

# --- Setup Paths ---
EXAMPLES_DIR = "examples"
OUTPUT_DIR = "output_synth_api"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Saving output to: {OUTPUT_DIR}")

# --- Example 1: Segment Retargeting ---
print("\n--- Running: Segment Retargeting ---")
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/texbynum/source_photo.png",
    config=RunConfig(image_weight=1.0),  # Note: new RunConfig doesn't use guide weights
)

# The main source/target pair is now passed as a guide to run()
result_img, result_err = ezsynner.run(
    guides=[
        load_guide(
            f"{EXAMPLES_DIR}/texbynum/source_segment.png",
            f"{EXAMPLES_DIR}/texbynum/target_segment.png",
            weight=1.0,  # The weight is now explicit per-guide
        )
    ]
)

write_image(f"{OUTPUT_DIR}/retarget_out.png", result_img)
# Error maps are float32, need to normalize for saving as visible image
result_err_vis = (255 * (result_err / result_err.max())).astype(np.uint8)
write_image(f"{OUTPUT_DIR}/retarget_err.png", result_err_vis)

# --- Example 2: Stylit ---
print("\n--- Running: Stylit ---")
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/stylit/source_style.png",
    config=RunConfig(),  # Use default config and specify weights in guides
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
    ]
)

write_image(f"{OUTPUT_DIR}/stylit_out.png", result_img)
result_err_vis = (255 * (result_err / result_err.max())).astype(np.uint8)
write_image(f"{OUTPUT_DIR}/stylit_err.png", result_err_vis)

# --- Example 3: Face Style ---
print("\n--- Running: Face Style ---")
ezsynner = ImageSynth(
    style_image=f"{EXAMPLES_DIR}/facestyle/source_painting.png",
    config=RunConfig(),
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
    ]
)

write_image(f"{OUTPUT_DIR}/facestyle_out.png", result_img)
result_err_vis = (255 * (result_err / result_err.max())).astype(np.uint8)
write_image(f"{OUTPUT_DIR}/facestyle_err.png", result_err_vis)


# --- Cleanup ---
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"\nTotal time taken: {time.time() - st:.4f} s")
