"""Windowed flow warping: propagate a single style frame through a video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from tqdm import tqdm

from ..flow.run import optical_flow_engine
from ..flow.types import FlowEngineName, FlowModelName
from ..utils.video import export_frames_to_browser_h264_mp4
from ..utils.warp_utils import Warp
from .common import (
    load_video_frames,
    precomputation_config_for_engine,
    write_png_sequence,
)


def run(args: argparse.Namespace) -> None:
    frames = load_video_frames(args.video, args.num_frames)
    n = len(frames)
    h, w = frames[0].shape[:2]
    warper = Warp(h, w, use_taichi=True)

    style0 = cv2.imread(args.style)
    if style0 is None:
        raise FileNotFoundError(f"Style image not found: {args.style}")
    style0 = cv2.resize(style0, (w, h))

    engine = cast(FlowEngineName, args.engine)
    flow_model = cast(FlowModelName | None, args.flow_model)
    cfg = precomputation_config_for_engine(engine, flow_model)

    out_path = Path(args.output)
    seq_dir = out_path.parent / f"{out_path.stem}_seq"
    montage_frames: list[np.ndarray] = []
    stylized_out = [style0]
    psnrs: list[float] = []

    current_source = style0
    current_source_idx = 0

    print(
        f"Windowed warping (window={args.window_size}, method={args.method}) on {n} frames..."
    )
    with optical_flow_engine(cfg) as compute_flow:
        for i in tqdm(range(1, n)):
            target_frame = frames[i]

            if (i - current_source_idx) > args.window_size:
                current_source = stylized_out[-1]
                current_source_idx = i - 1

            flow = compute_flow([frames[current_source_idx], target_frame])[0]

            if args.method == "forward":
                warped, weight = warper.run_forward_warping(
                    current_source,
                    flow,
                    return_weight=True,
                    src_guide=frames[current_source_idx],
                    tgt_guide=target_frame,
                )
            else:
                warped = warper.run_warping(current_source, -flow)
                weight = np.ones((h, w), dtype=np.float32)

            stylized_out.append(warped)

            diff_img = cv2.absdiff(warped, target_frame)
            mse = np.mean(diff_img.astype(np.float32) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
            psnrs.append(psnr)

            conf_viz = cv2.applyColorMap(
                (np.clip(weight, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
            )
            err_viz = cv2.applyColorMap(
                cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT
            )
            montage = np.concatenate([warped, conf_viz, err_viz], axis=1)
            cv2.putText(
                montage,
                f"Warp {current_source_idx}->{i} (window={args.window_size})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            montage_frames.append(montage)

    if psnrs:
        print(f"Average PSNR ({args.method}): {np.mean(psnrs):.2f} dB")

    write_png_sequence(stylized_out, seq_dir)
    print(f"PNG sequence -> {seq_dir}")
    export_frames_to_browser_h264_mp4(montage_frames, out_path, args.fps)
    print(f"Saved montage video to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stylize video by warping a style frame through optical flow."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--style", type=str, required=True, help="Style image for frame 0")
    parser.add_argument("--output", type=str, default="output/cat_warped.mp4")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument(
        "--engine", type=str, default="NeuFlow", choices=["RAFT", "NeuFlow", "OpenCV"]
    )
    parser.add_argument("--flow-model", type=str, default=None)
    parser.add_argument(
        "--method", type=str, default="forward", choices=["forward", "backward"]
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=1,
        help="Max frames before re-anchoring source (1=iterative)",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
