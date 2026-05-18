"""Multi-keyframe bidirectional forward-splat warping for video stylization."""

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
    get_style_keyframes,
    load_video_frames,
    precomputation_config_for_engine,
    write_png_sequence,
)


def run(args: argparse.Namespace) -> None:
    frames = load_video_frames(args.video, args.num_frames)
    n = len(frames)
    h, w = frames[0].shape[:2]

    keyframes = get_style_keyframes(args.style_dir)
    if not keyframes:
        raise FileNotFoundError(f"No style keyframes in {args.style_dir}")

    sorted_keys = sorted(keyframes.keys())
    style_imgs = {
        idx: cv2.resize(cv2.imread(path), (w, h))
        for idx, path in keyframes.items()
    }

    warper = Warp(h, w, use_taichi=True)
    engine = cast(FlowEngineName, args.engine)
    flow_model = cast(FlowModelName | None, args.flow_model)
    cfg = precomputation_config_for_engine(engine, flow_model)

    out_path = Path(args.output)
    seq_dir = out_path.parent / f"{out_path.stem}_seq"
    montage_frames: list[np.ndarray] = []
    stylized_out: list[np.ndarray] = []

    print(f"Multi-keyframe bidirectional warping ({len(sorted_keys)} keyframes)...")
    with optical_flow_engine(cfg) as compute_flow:
        for i in tqdm(range(n)):
            target_content = frames[i]
            pre_idx = max([k for k in sorted_keys if k <= i], default=min(sorted_keys))
            post_idx = min([k for k in sorted_keys if k >= i], default=max(sorted_keys))

            if pre_idx == post_idx:
                final_uint8 = style_imgs[pre_idx]
                total_weight = np.ones((h, w), dtype=np.float32)
            else:
                dist = post_idx - pre_idx
                t = (i - pre_idx) / dist

                f0 = compute_flow([frames[pre_idx], target_content])[0]
                w0, weight0 = warper.run_forward_warping(
                    style_imgs[pre_idx],
                    f0,
                    fill_holes=False,
                    return_weight=True,
                    src_guide=frames[pre_idx],
                    tgt_guide=target_content,
                )

                f1 = compute_flow([frames[post_idx], target_content])[0]
                w1, weight1 = warper.run_forward_warping(
                    style_imgs[post_idx],
                    f1,
                    fill_holes=False,
                    return_weight=True,
                    src_guide=frames[post_idx],
                    tgt_guide=target_content,
                )

                c0 = weight0 * (1.0 - t)
                c1 = weight1 * t
                tw = c0 + c1
                mask = tw > 1e-6
                safe_tw = np.where(mask, tw, 1.0)

                final_float = np.zeros_like(w0, dtype=np.float32)
                for c in range(3):
                    final_float[..., c] = (w0[..., c] * c0 + w1[..., c] * c1) / safe_tw

                final_uint8 = final_float.clip(0, 255).astype(np.uint8)
                if np.any(~mask):
                    final_uint8 = warper.run_forward_warping(
                        final_uint8, np.zeros_like(f0), fill_holes=True
                    )
                total_weight = tw

            stylized_out.append(final_uint8)

            conf_viz = cv2.applyColorMap(
                (np.clip(total_weight, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_VIRIDIS,
            )
            err_viz = cv2.applyColorMap(
                cv2.cvtColor(
                    cv2.absdiff(final_uint8, target_content), cv2.COLOR_BGR2GRAY
                ),
                cv2.COLORMAP_HOT,
            )
            montage = np.concatenate([final_uint8, conf_viz, err_viz], axis=1)
            cv2.putText(
                montage,
                f"Frame {i} (window {pre_idx}<->{post_idx})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            montage_frames.append(montage)

    write_png_sequence(stylized_out, seq_dir)
    print(f"PNG sequence -> {seq_dir}")
    export_frames_to_browser_h264_mp4(montage_frames, out_path, args.fps)
    print(f"Saved montage video to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-keyframe bidirectional flow warping for video stylization."
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument(
        "--engine", type=str, default="NeuFlow", choices=["RAFT", "NeuFlow", "OpenCV"]
    )
    parser.add_argument("--flow-model", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/cat_multi_bidirectional.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
