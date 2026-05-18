"""Pseudo endpoint style anchor generation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import DebugConfig
from .engines.pass_runner import SynthesisPassRunner
from .engines.synthesis_engine import EbsynthEngine
from .precompute import PrecomputeState
from .utils.io_utils import write_image
from .utils.occlusion import (
    accumulate_target_to_source_coords,
    build_dfs_pseudo_style,
    mask_to_bgr,
)
from .utils.sequence_utils import SynthesisSequence


class PseudoEndpointGenerator:
    """Create synthetic style anchors at sequence endpoints."""

    def __init__(
        self,
        *,
        engine: EbsynthEngine,
        precompute_state: PrecomputeState,
        debug_cfg: DebugConfig,
    ) -> None:
        self.engine = engine
        self.precompute_state = precompute_state
        self.debug_cfg = debug_cfg

    def add_endpoint_styles(
        self,
        *,
        content_frames: List[np.ndarray],
        style_frames: List[np.ndarray],
        style_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[int]]:
        if not style_indices:
            return style_frames, style_indices

        n = len(content_frames)
        sorted_pairs = sorted(zip(style_indices, style_frames), key=lambda p: p[0])
        augmented: dict[int, np.ndarray] = {
            idx: frame for idx, frame in sorted_pairs if 0 <= idx < n
        }
        if not augmented:
            return style_frames, style_indices

        print("\n--- Building pseudo endpoint style anchors ---")
        first_idx = min(augmented)
        last_idx = max(augmented)

        if first_idx > 0:
            print(f"Generating pseudo start style at frame 0 from style frame {first_idx}...")
            augmented[0] = self.generate(
                target_idx=0,
                source_idx=first_idx,
                source_style=augmented[first_idx],
                content_frames=content_frames,
            )

        if last_idx < n - 1:
            print(
                f"Generating pseudo end style at frame {n - 1} from style frame {last_idx}..."
            )
            augmented[n - 1] = self.generate(
                target_idx=n - 1,
                source_idx=last_idx,
                source_style=augmented[last_idx],
                content_frames=content_frames,
            )

        augmented_indices = sorted(augmented)
        augmented_frames = [augmented[idx] for idx in augmented_indices]
        print(f"Using style anchors at frames: {augmented_indices}")
        return augmented_frames, augmented_indices

    def generate(
        self,
        *,
        target_idx: int,
        source_idx: int,
        source_style: np.ndarray,
        content_frames: List[np.ndarray],
    ) -> np.ndarray:
        pc = self.engine.pipeline_config
        if pc.pseudo_endpoint_mode == "synthesis":
            seq = SynthesisSequence(
                min(target_idx, source_idx),
                max(target_idx, source_idx),
                (
                    SynthesisSequence.MODE_FWD
                    if target_idx > source_idx
                    else SynthesisSequence.MODE_REV
                ),
                [0],
            )
            pseudo_sequence, _, _, _ = SynthesisPassRunner(
                engine=self.engine,
                precompute_state=self.precompute_state,
                debug_cfg=self.debug_cfg,
            ).run(
                seq=seq,
                style_img=source_style,
                is_forward=target_idx > source_idx,
                content_frames=content_frames,
            )
            return pseudo_sequence[-1] if target_idx > source_idx else pseudo_sequence[0]

        h, w = content_frames[target_idx].shape[:2]
        source_coords, valid_coords = accumulate_target_to_source_coords(
            height=h,
            width=w,
            target_idx=target_idx,
            source_idx=source_idx,
            fwd_flows=self.precompute_state.fwd_flows,
            bwd_flows=self.precompute_state.bwd_flows,
        )
        pseudo, confidence = build_dfs_pseudo_style(
            style_img=source_style,
            source_content=content_frames[source_idx],
            target_content=content_frames[target_idx],
            source_coords=source_coords,
            valid_coords=valid_coords,
            content_error_threshold=pc.pseudo_endpoint_dfs_content_threshold,
            offset_error_threshold=pc.pseudo_endpoint_dfs_offset_threshold,
            min_region_size=pc.pseudo_endpoint_dfs_min_region_size,
            inpaint_radius=pc.pseudo_endpoint_dfs_inpaint_radius,
        )

        if self.debug_cfg.save_occlusion_debug:
            out_dir = self._occlusion_debug_output_dir() / "pseudo_endpoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_image(out_dir / f"{target_idx:05d}_style.png", pseudo)
            write_image(out_dir / f"{target_idx:05d}_confidence.png", mask_to_bgr(confidence))

        assigned_ratio = float((confidence > 0).mean())
        print(
            f"Pseudo endpoint frame {target_idx}: DFS assigned {assigned_ratio:.1%} before fill"
        )
        return pseudo

    def _occlusion_debug_output_dir(self) -> Path:
        p = Path(self.debug_cfg.occlusion_debug_dir)
        return p if p.is_absolute() else Path.cwd() / p
