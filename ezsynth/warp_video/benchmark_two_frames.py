"""Benchmark forward vs backward warping over frame pairs in a video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from ..config import PrecomputationConfig
from ..flow.run import compute_optical_flow_sequence
from ..flow.types import FlowEngineName, FlowModelName
from ..utils.viz_utils import flow_to_image
from ..utils.warp_utils import Warp


def _compute_metrics(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    diff = (img1.astype(np.float32) - img2.astype(np.float32)) ** 2
    mse = float(np.mean(diff))
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0
    return mse, psnr


def _load_frame(cap: cv2.VideoCapture, idx: int) -> tuple[bool, np.ndarray | None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return cap.read()


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    psnrs_back: list[float] = []
    psnrs_fwd: list[float] = []
    max_gap = -1.0
    best_montage = None
    best_test_info = ""
    h = w = 0
    warper = None

    engine = cast(FlowEngineName, args.engine)
    flow_model = cast(FlowModelName, args.flow_model)
    cfg = PrecomputationConfig(flow_engine=engine, flow_model=flow_model)

    print(
        f"Benchmarking {args.engine} flow, skip={args.frame_skip}, "
        f"{args.num_tests} tests..."
    )

    try:
        for test_i in range(args.num_tests):
            idx1 = args.frame_idx + test_i * args.stride
            idx2 = idx1 + args.frame_skip
            if idx2 >= total_frames:
                print(f"Reached end of video at test {test_i}.")
                break

            ok1, frame1 = _load_frame(cap, idx1)
            ok2, frame2 = _load_frame(cap, idx2)
            if not (ok1 and ok2):
                print(f"Error reading frames {idx1}, {idx2}")
                continue

            if warper is None:
                h, w = frame1.shape[:2]
                warper = Warp(h, w, use_taichi=args.use_taichi)

            style_img = None
            if args.style and os.path.exists(args.style):
                style_img = cv2.imread(args.style)
                if style_img is not None:
                    style_img = cv2.resize(style_img, (w, h))

            flow = compute_optical_flow_sequence([frame1, frame2], cfg)[0]
            warp_src = style_img if style_img is not None else frame1

            warped_backward = warper.run_warping(warp_src, -flow)
            warped_forward, weight = warper.run_forward_warping(
                warp_src, flow, return_weight=True
            )

            _, psnr_back = _compute_metrics(warped_backward, frame2)
            _, psnr_fwd = _compute_metrics(warped_forward, frame2)
            psnrs_back.append(psnr_back)
            psnrs_fwd.append(psnr_fwd)

            gap = psnr_fwd - psnr_back
            print(
                f"Test {test_i} ({idx1}->{idx2}): "
                f"backward={psnr_back:.2f} dB, forward={psnr_fwd:.2f} dB (gap={gap:.2f})"
            )

            if gap > max_gap:
                max_gap = gap
                flow_viz = flow_to_image(flow, convert_to_bgr=True)
                conf_map = cv2.applyColorMap(
                    (np.clip(weight, 0, 1) * 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS,
                )
                top_row = np.concatenate([frame1, frame2, flow_viz], axis=1)
                bottom_row = np.concatenate(
                    [warped_backward, warped_forward, conf_map], axis=1
                )
                best_montage = np.concatenate([top_row, bottom_row], axis=0)
                title = "Style" if style_img is not None else "Content"
                best_test_info = f"{title} F{idx1}->F{idx2}, gap={gap:.2f}"
    finally:
        cap.release()

    if best_montage is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), best_montage)
        print(f"Saved best comparison ({best_test_info}) to {output_path}")

    if psnrs_back:
        avg_back = float(np.mean(psnrs_back))
        avg_fwd = float(np.mean(psnrs_fwd))
        print("\n" + "=" * 30)
        print(f"AVERAGE (skip={args.frame_skip}):")
        print(f"Backward PSNR: {avg_back:.2f} dB")
        print(f"Forward PSNR:  {avg_fwd:.2f} dB")
        print(f"Improvement:   {avg_fwd - avg_back:.2f} dB")
        print("=" * 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark forward vs backward warping on video frame pairs."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_tests", type=int, default=5)
    parser.add_argument(
        "--engine", type=str, default="NeuFlow", choices=["RAFT", "NeuFlow", "OpenCV"]
    )
    parser.add_argument("--flow-model", type=str, default="neuflow_mixed")
    parser.add_argument("--use_taichi", action="store_true", default=True)
    parser.add_argument(
        "--output",
        type=str,
        default="output/warp_comparison_best.png",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
