"""Adaptive multi-keyframe bidirectional flow warping for stylized video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import cv2
import numpy as np
from tqdm import tqdm

from ..api import ImageSynth, RunConfig
from ..flow.run import optical_flow_engine
from ..flow.types import FlowEngineName, FlowModelName
from ..guide import GuideObject
from ..utils.video import export_frames_to_browser_h264_mp4
from ..utils.warp_utils import PositionalGuide, Warp
from .common import (
    blended_splat_fill_holes,
    compute_warp_score_from_flow,
    flow_between_frames,
    get_style_keyframes,
    load_video_frames,
    precompute_adjacent_flows,
    precomputation_config_for_engine,
    write_png_sequence,
)


def _padded_bbox_from_mask(
    mask_u8: np.ndarray, pad: int, min_side: int, h: int, w: int
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_u8 > 0)
    if len(ys) == 0:
        return 0, h, 0, w
    y0, y1 = int(ys.min()) - pad, int(ys.max()) + 1 + pad
    x0, x1 = int(xs.min()) - pad, int(xs.max()) + 1 + pad
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y1), min(w, x1)
    ch, cw = y1 - y0, x1 - x0
    if ch < min_side:
        d = min_side - ch
        y0 = max(0, y0 - d // 2)
        y1 = min(h, y0 + min_side)
        y0 = max(0, y1 - min_side)
    if cw < min_side:
        d = min_side - cw
        x0 = max(0, x0 - d // 2)
        x1 = min(w, x0 + min_side)
        x0 = max(0, x1 - min_side)
    return y0, y1, x0, x1


def _mean_sq_error_on_mask(
    warper: Warp, src: np.ndarray, tgt: np.ndarray, flow: np.ndarray, mask: np.ndarray
) -> float:
    wc = warper.run_warping(src, -flow).astype(np.float32)
    d = (wc - tgt.astype(np.float32)) ** 2
    m = mask > 0
    if not np.any(m):
        return float("inf")
    return float(np.mean(d[m]))


def _feather_alpha(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.float32)
    if sigma <= 0:
        return np.clip(m, 0.0, 1.0)
    k = max(1, int(sigma * 6)) | 1
    return np.clip(cv2.GaussianBlur(m, (k, k), sigma), 0.0, 1.0)


def _bgr_to_gray3(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.stack([g, g, g], axis=-1).astype(np.uint8)


def _build_hole_refine_guides(
    frames_k: np.ndarray,
    frames_i: np.ndarray,
    pristine_pos: np.ndarray,
    target_pos: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    app_weight: float,
    pos_weight: float,
) -> list[GuideObject]:
    gk = _bgr_to_gray3(frames_k)
    gi = _bgr_to_gray3(frames_i)
    return [
        GuideObject(gk[y0:y1, x0:x1], gi[y0:y1, x0:x1], app_weight),
        GuideObject(
            pristine_pos[y0:y1, x0:x1], target_pos[y0:y1, x0:x1], pos_weight
        ),
    ]


def _refine_records_with_imagesynth(
    records: list[dict[str, Any]],
    frames: list[np.ndarray],
    style_imgs: dict[int, np.ndarray],
    warper: Warp,
    h: int,
    w: int,
    get_flow_pair: Callable[[int, int], np.ndarray],
    pos_guider: PositionalGuide,
    hole_min_pixels: int,
    hole_pad: int,
    hole_min_crop: int,
    feather_sigma: float,
    synth_config: RunConfig,
) -> None:
    synth_cache: dict[int, ImageSynth] = {}

    def synth_for_anchor(anchor: int) -> ImageSynth:
        if anchor not in synth_cache:
            synth_cache[anchor] = ImageSynth(style_imgs[anchor], config=synth_config)
        return synth_cache[anchor]

    for rec in records:
        if rec["is_kf"]:
            continue
        hm = rec["hole_mask"]
        if int(np.count_nonzero(hm)) < hole_min_pixels:
            continue
        k_dilate = cv2.dilate(hm, np.ones((5, 5), np.uint8), iterations=1)
        pre, post = rec["best_pre"], rec["best_post"]
        f_pre = get_flow_pair(pre, rec["index"])
        f_post = get_flow_pair(post, rec["index"])
        err_pre = _mean_sq_error_on_mask(warper, frames[pre], frames[rec["index"]], f_pre, k_dilate)
        err_post = _mean_sq_error_on_mask(
            warper, frames[post], frames[rec["index"]], f_post, k_dilate
        )
        anchor = pre if err_pre <= err_post else post
        f_ak = f_pre if anchor == pre else f_post

        y0, y1, x0, x1 = _padded_bbox_from_mask(
            k_dilate, hole_pad, hole_min_crop, h, w
        )
        pristine = pos_guider.get_pristine_guide_uint8()
        tgt_pos = pos_guider.create_from_flow(f_ak)
        guides = _build_hole_refine_guides(
            frames[anchor],
            frames[rec["index"]],
            pristine,
            tgt_pos,
            y0,
            y1,
            x0,
            x1,
            app_weight=2.0,
            pos_weight=1.5,
        )
        synth = synth_for_anchor(anchor)
        patch, _ = synth.run(
            [(g.keyframe, g.target, g.weight) for g in guides], benchmark=False
        )
        base = rec["stylized"]
        roi = base[y0:y1, x0:x1].copy()
        alpha = _feather_alpha(hm[y0:y1, x0:x1], feather_sigma)[..., np.newaxis]
        blended = (
            roi.astype(np.float32) * (1.0 - alpha) + patch.astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)
        out = base.copy()
        out[y0:y1, x0:x1] = blended
        rec["stylized"] = out


def _propagate_hole_fills(
    records: list[dict[str, Any]],
    frames: list[np.ndarray],
    warper: Warp,
    h: int,
    w: int,
    adj_fwd: list[np.ndarray],
    adj_to_prev: list[np.ndarray],
    direct_long_flow: bool,
    compute_flow: Callable[[list[np.ndarray]], list[np.ndarray]],
    feather_sigma: float,
    bidirectional: bool,
) -> None:
    n = len(records)

    def flow_fwd(a: int, b: int) -> np.ndarray:
        if direct_long_flow or not adj_fwd:
            return compute_flow([frames[a], frames[b]])[0]
        return flow_between_frames(a, b, adj_fwd, adj_to_prev, h, w)

    styl = [rec["stylized"].copy() for rec in records]
    hole = [rec["hole_mask"].copy() for rec in records]

    for j in range(n - 1):
        f_j_to_jp1 = flow_fwd(j, j + 1)
        prop = warper.run_warping(styl[j], -f_j_to_jp1).astype(np.float32)
        m_prop = warper.run_warping((hole[j] > 0).astype(np.float32), -f_j_to_jp1).astype(
            np.float32
        )
        k = _feather_alpha(hole[j + 1], feather_sigma)
        alpha = np.clip(k[..., np.newaxis] + 0.35 * m_prop[..., np.newaxis], 0.0, 1.0)
        styl[j + 1] = (
            styl[j + 1].astype(np.float32) * (1.0 - alpha) + prop * alpha
        ).clip(0, 255).astype(np.uint8)

    if bidirectional:
        for j in range(n - 1, 0, -1):
            f_jm1_to_j = flow_fwd(j - 1, j)
            prop = warper.run_warping(styl[j], f_jm1_to_j).astype(np.float32)
            k = _feather_alpha(hole[j - 1], feather_sigma)
            alpha = np.clip(k[..., np.newaxis], 0.0, 1.0)
            styl[j - 1] = (
                styl[j - 1].astype(np.float32) * (1.0 - alpha) + prop * alpha
            ).clip(0, 255).astype(np.uint8)

    for rec, s in zip(records, styl, strict=True):
        rec["stylized"] = s


def run(args: argparse.Namespace) -> None:
    frames = load_video_frames(args.video, args.num_frames)
    n = len(frames)
    h, w = frames[0].shape[:2]

    keyframes = get_style_keyframes(args.style_dir)
    sorted_keys = sorted(k for k in keyframes if k < n)
    if not sorted_keys:
        raise FileNotFoundError(
            f"No style keyframes found in range 0-{n - 1} under {args.style_dir}"
        )

    style_imgs = {
        idx: cv2.resize(cv2.imread(keyframes[idx]), (w, h))
        for idx in sorted_keys
    }
    warper = Warp(h, w, use_taichi=True)
    engine = cast(FlowEngineName, args.engine)
    flow_model = cast(FlowModelName | None, args.flow_model)
    cfg = precomputation_config_for_engine(engine, flow_model)

    out_path = Path(args.output)
    seq_dir = out_path.parent / f"{out_path.stem}_seq"
    montage_frames: list[np.ndarray] = []
    prev_pre = prev_post = None

    print(f"Stylizing {n} frames with adaptive window selection...")
    with optical_flow_engine(cfg) as compute_flow:
        adj_fwd: list[np.ndarray] = []
        adj_to_prev: list[np.ndarray] = []
        if not args.direct_long_flow:
            adj_fwd, adj_to_prev = precompute_adjacent_flows(frames, compute_flow)
            print(
                f"Chained adjacent flow: {len(adj_fwd)} fwd + {len(adj_to_prev)} rev segments."
            )

        def get_flow_pair(k_src: int, k_tgt: int) -> np.ndarray:
            if args.direct_long_flow or not adj_fwd:
                return compute_flow([frames[k_src], frames[k_tgt]])[0]
            return flow_between_frames(k_src, k_tgt, adj_fwd, adj_to_prev, h, w)

        records: list[dict[str, Any]] = []

        for i in tqdm(range(n)):
            target_content = frames[i]

            if i in style_imgs:
                final_uint8 = style_imgs[i].copy()
                best_pre, best_post = i, i
                hole_mask = np.zeros((h, w), dtype=np.uint8)
            else:
                pre_candidates = [k for k in sorted_keys if k < i] or [min(sorted_keys)]
                post_candidates = [k for k in sorted_keys if k > i] or [max(sorted_keys)]

                best_pre_score = -1.0
                best_pre = pre_candidates[-1]
                scores_pre: dict[int, float] = {}
                for k in pre_candidates:
                    flow_k = get_flow_pair(k, i)
                    score = compute_warp_score_from_flow(
                        warper, frames[k], target_content, flow_k
                    )
                    scores_pre[k] = score
                    if score > best_pre_score:
                        best_pre_score = score
                        best_pre = k

                best_post_score = -1.0
                best_post = post_candidates[0]
                scores_post: dict[int, float] = {}
                for k in post_candidates:
                    flow_k = get_flow_pair(k, i)
                    score = compute_warp_score_from_flow(
                        warper, frames[k], target_content, flow_k
                    )
                    scores_post[k] = score
                    if score > best_post_score:
                        best_post_score = score
                        best_post = k

                hz = float(args.anchor_hysteresis_db)
                if hz > 0 and i > 0 and prev_pre is not None and prev_post is not None:
                    if prev_pre in scores_pre and prev_pre != best_pre:
                        if best_pre_score - scores_pre[prev_pre] <= hz:
                            best_pre = prev_pre
                    if prev_post in scores_post and prev_post != best_post:
                        if best_post_score - scores_post[prev_post] <= hz:
                            best_post = prev_post

                best_f0 = get_flow_pair(best_pre, i)
                best_f1 = get_flow_pair(best_post, i)

                w0, weight0 = warper.run_forward_warping(
                    style_imgs[best_pre],
                    best_f0,
                    fill_holes=True,
                    return_weight=True,
                    src_guide=frames[best_pre],
                    tgt_guide=target_content,
                )
                w1, weight1 = warper.run_forward_warping(
                    style_imgs[best_post],
                    best_f1,
                    fill_holes=True,
                    return_weight=True,
                    src_guide=frames[best_post],
                    tgt_guide=target_content,
                )

                s0 = np.exp(best_pre_score / 10.0)
                s1 = np.exp(best_post_score / 10.0)
                c0 = weight0 * s0
                c1 = weight1 * s1
                tw = c0 + c1
                final_uint8, hole_mask = blended_splat_fill_holes(
                    warper, w0, w1, c0, c1, tw
                )

            records.append(
                {
                    "index": i,
                    "stylized": final_uint8,
                    "hole_mask": hole_mask,
                    "best_pre": best_pre,
                    "best_post": best_post,
                    "target": target_content,
                    "is_kf": i in style_imgs,
                }
            )
            prev_pre, prev_post = best_pre, best_post

        if args.hole_refine:
            print("Hole refine: ImageSynth on splat-hole ROIs...")
            pos_guider = PositionalGuide(h, w, use_taichi=False)
            synth_config = RunConfig(
                backend=args.hole_synth_backend,
                search_vote_iters=12 if args.hole_full_params else 1,
                patch_match_iters=6,
                pyramid_levels=6 if args.hole_full_params else 1,
                use_residual_transfer=True,
                use_temporal_nnf_propagation=False,
                use_sparse_feature_guide=False,
            )
            _refine_records_with_imagesynth(
                records,
                frames,
                style_imgs,
                warper,
                h,
                w,
                get_flow_pair,
                pos_guider,
                args.hole_min_pixels,
                args.hole_pad,
                args.hole_min_crop,
                args.hole_feather,
                synth_config,
            )

        if args.hole_refine and not args.no_hole_propagate and n > 1:
            print("Hole propagate: flow-warp hole fills...")
            _propagate_hole_fills(
                records,
                frames,
                warper,
                h,
                w,
                adj_fwd,
                adj_to_prev,
                args.direct_long_flow,
                compute_flow,
                args.hole_feather,
                bidirectional=not args.no_hole_prop_bidirectional,
            )

        stylized_out: list[np.ndarray] = []
        for i, rec in enumerate(records):
            final_uint8 = rec["stylized"]
            target_content = rec["target"]
            best_pre, best_post = rec["best_pre"], rec["best_post"]

            diff_img = cv2.absdiff(final_uint8, target_content)
            err_viz = cv2.applyColorMap(
                cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT
            )
            montage = np.concatenate([final_uint8, err_viz], axis=1)
            cv2.putText(
                montage,
                f"Adaptive Window: {best_pre} <-> {best_post}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            montage_frames.append(montage)
            stylized_out.append(final_uint8)

    write_png_sequence(stylized_out, seq_dir)
    print(f"PNG sequence -> {seq_dir}")
    export_frames_to_browser_h264_mp4(montage_frames, out_path, args.fps)
    print(f"Saved montage video to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive multi-keyframe bidirectional flow warping for video stylization."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--style_dir", type=str, required=True, help="Directory with styleNNN images"
    )
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument(
        "--engine", type=str, default="NeuFlow", choices=["RAFT", "NeuFlow", "OpenCV"]
    )
    parser.add_argument("--flow-model", type=str, default=None)
    parser.add_argument(
        "--direct-long-flow",
        action="store_true",
        help="One-shot keyframe->current flow (drifts on long gaps)",
    )
    parser.add_argument("--anchor-hysteresis-db", type=float, default=1.5)
    parser.add_argument("--output", type=str, default="output/cat_adaptive.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--hole-refine", action="store_true")
    parser.add_argument("--no-hole-propagate", action="store_true")
    parser.add_argument("--no-hole-prop-bidirectional", action="store_true")
    parser.add_argument("--hole-min-pixels", type=int, default=200)
    parser.add_argument("--hole-pad", type=int, default=24)
    parser.add_argument("--hole-min-crop", type=int, default=96)
    parser.add_argument("--hole-feather", type=float, default=12.0)
    parser.add_argument(
        "--hole-synth-backend",
        type=str,
        default="cuda",
        choices=["cuda", "torch", "taichi"],
    )
    parser.add_argument("--hole-full-params", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
