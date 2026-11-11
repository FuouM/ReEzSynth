import gc
import logging
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from ezsynth.api import ImageSynth, RunConfig
from ezsynth.logging_setup import configure_cli_logging
from ezsynth.utils.io_utils import write_image

logger = logging.getLogger(__name__)

# Constants for synthesis parameters
USE_RESIDUAL_TRANSFER = True
EXAMPLES_DIR = "examples"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run EBSynth image synthesis examples")
    parser.add_argument(
        "--backend",
        choices=["cuda", "torch", "cpu", "taichi", "wgpu"],
        default="cuda",
        help="Backend to use for synthesis (default: cuda). 'cpu' uses the C++ extension on CPU. 'taichi' uses the Taichi backend. 'wgpu' uses the WebGPU (Rust) backend.",
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
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Use optimized CPU backend (default: True)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Disable optimized CPU backend",
    )
    parser.add_argument(
        "--use-bilateral",
        action="store_true",
        help="Enable bilateral weighting (edge-aware matching)",
    )
    parser.add_argument(
        "--sigma-spatial",
        type=float,
        default=4.0,
        help="Bilateral spatial sigma (default: 4.0)",
    )
    parser.add_argument(
        "--sigma-color",
        type=float,
        default=10.0,
        help="Bilateral color sigma (default: 10.0)",
    )
    parser.add_argument(
        "--n-size-step",
        type=int,
        default=1,
        help="Patch comparison step size (default: 1, >1 for sparse/faster matching)",
    )
    parser.add_argument(
        "--cost-function",
        choices=["ssd", "ncc", "sad"],
        default="ssd",
        help="Patch cost: sum of squared / absolute differences or normalized "
        "cross-correlation (default: ssd; the cuda/cpu backends support ssd, ncc)",
    )
    parser.add_argument(
        "--search-mode",
        choices=["patchmatch", "jfa", "multicand"],
        default="patchmatch",
        help="Correspondence search (torch/taichi backends; default: patchmatch)",
    )
    parser.add_argument(
        "--multicand-k",
        type=int,
        default=3,
        help="Matches kept per pixel with --search-mode multicand (2-8, default: 3)",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Patch storage precision of the torch backend (default: float32)",
    )
    return parser.parse_args()


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


def resolve_backend(cli_backend: str) -> tuple[str, str | None]:
    """Map the CLI backend to ``(RunConfig.backend, RunConfig.device)``.

    ``cpu`` runs the C++ extension (``cuda`` backend) on the CPU; every other choice
    auto-detects the device.
    """
    if cli_backend == "cpu":
        return "cuda", "cpu"
    return cli_backend, None


def build_config(args: Namespace, **overrides) -> RunConfig:
    """RunConfig for one example; minimal mode uses one pyramid level and iteration."""
    backend, device = resolve_backend(args.backend)
    params = {
        "backend": backend,
        "use_residual_transfer": USE_RESIDUAL_TRANSFER,
        "cost_function": args.cost_function,
        "search_mode": args.search_mode,
        "multicand_k": args.multicand_k,
        "precision": args.precision,
        "device": device,
        "use_optimization": args.optimize,
        "use_bilateral": args.use_bilateral,
        "sigma_spatial": args.sigma_spatial,
        "sigma_color": args.sigma_color,
        "n_size_step": args.n_size_step,
        **overrides,
    }
    if not args.full_params:
        params.update(pyramid_levels=1, search_vote_iters=1)
    return RunConfig(**params)


def run_example(args, output_dir, title, base_name, style_image, guides, **overrides):
    logger.info(f"\n--- Running: {title} ---")
    start_example_time = time.time()
    ezsynner = ImageSynth(
        style_image=style_image, config=build_config(args, **overrides)
    )
    # Guides are (source, target, weight) tuples; paths or arrays are accepted.
    result_img, result_err = ezsynner.run(guides=guides, benchmark=args.benchmark)
    save_synth_result(output_dir, base_name, result_img, result_err)
    logger.info(
        f"{title} took: {time.time() - start_example_time:.4f} s, "
        f"mean error: {float(result_err.mean()):.2f}"
    )


def main():
    configure_cli_logging()
    args = parse_args()
    logger.info(f"Using backend: {args.backend}")
    logger.info(f"Using full parameters: {args.full_params}")

    st = time.time()
    mode = "full" if args.full_params else "minimal"
    variant = f"{args.cost_function}_{args.search_mode}"
    if args.search_mode == "multicand":
        variant += f"_k{args.multicand_k}"
    if args.precision != "float32":
        variant += f"_{args.precision}"
    output_dir = f"output_synth_api_{args.backend}_{mode}_{variant}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving output to: {output_dir}")

    run_example(
        args,
        output_dir,
        "Segment Retargeting",
        "retarget",
        f"{EXAMPLES_DIR}/texbynum/source_photo.png",
        [
            (
                f"{EXAMPLES_DIR}/texbynum/source_segment.png",
                f"{EXAMPLES_DIR}/texbynum/target_segment.png",
                1.0,
            )
        ],
        image_weight=1.0,
    )

    run_example(
        args,
        output_dir,
        "Stylit",
        "stylit",
        f"{EXAMPLES_DIR}/stylit/source_style.png",
        [
            (
                f"{EXAMPLES_DIR}/stylit/source_fullgi.png",
                f"{EXAMPLES_DIR}/stylit/target_fullgi.png",
                0.66,
            ),
            (
                f"{EXAMPLES_DIR}/stylit/source_dirdif.png",
                f"{EXAMPLES_DIR}/stylit/target_dirdif.png",
                0.66,
            ),
            (
                f"{EXAMPLES_DIR}/stylit/source_indirb.png",
                f"{EXAMPLES_DIR}/stylit/target_indirb.png",
                0.66,
            ),
        ],
    )
    clear_memory()

    run_example(
        args,
        output_dir,
        "Face Style",
        "facestyle",
        f"{EXAMPLES_DIR}/facestyle/source_painting.png",
        [
            (
                f"{EXAMPLES_DIR}/facestyle/source_Gapp.png",
                f"{EXAMPLES_DIR}/facestyle/target_Gapp.png",
                2.0,
            ),
            (
                f"{EXAMPLES_DIR}/facestyle/source_Gseg.png",
                f"{EXAMPLES_DIR}/facestyle/target_Gseg.png",
                1.5,
            ),
            (
                f"{EXAMPLES_DIR}/facestyle/source_Gpos.png",
                f"{EXAMPLES_DIR}/facestyle/target_Gpos.png",
                1.5,
            ),
        ],
    )
    clear_memory()

    logger.info(f"\nTotal time taken: {time.time() - st:.4f} s")


if __name__ == "__main__":
    main()
