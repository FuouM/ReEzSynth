"""Precompute orchestration for flow, edges, and sparse guides."""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .config import DebugConfig, PipelineConfig, PrecomputationConfig, ProjectConfig
from .flow.run import (
    compute_backward_optical_flow_sequence,
    compute_bidirectional_optical_flow_sequence,
    compute_optical_flow_sequence,
)
from .precompute import PrecomputeState, compute_guide
from .utils.feature_utils import generate_tracked_features, render_gaussian_guide
from .utils.io_utils import load_frames_from_dir, write_image
from .utils.occlusion import compute_flow_occlusion_masks, mask_to_bgr
from .utils.pipeline_utils import (
    has_exact_cache_files,
    load_cached_arrays,
    load_cached_flow,
    save_array_cache,
    save_edge_map_cache,
    save_flow_cache,
)
from .utils.viz_utils import flow_to_image


class PrecomputeRunner:
    """Run and cache all data needed before synthesis passes."""

    def __init__(
        self,
        *,
        project_cfg: ProjectConfig,
        precomputation_cfg: PrecomputationConfig,
        pipeline_cfg: PipelineConfig,
        debug_cfg: DebugConfig,
        state: Optional[PrecomputeState] = None,
    ) -> None:
        self.project_cfg = project_cfg
        self.precomputation_cfg = precomputation_cfg
        self.pipeline_cfg = pipeline_cfg
        self.debug_cfg = debug_cfg
        self.state = state if state is not None else PrecomputeState()

    def run(self, content_frames: List[np.ndarray]) -> PrecomputeState:
        if self.uses_flow_occlusion():
            self._compute_bidirectional_optical_flow(content_frames)
            self._compute_occlusion_masks()
        else:
            self._compute_optical_flow(content_frames)

        self._compute_edge_maps(content_frames)

        if self.pipeline_cfg.use_sparse_feature_guide:
            print("\nGenerating sparse feature guides...")
            tracked_points = generate_tracked_features(
                content_frames[0], self.state.fwd_flows
            )
            h, w, _ = content_frames[0].shape
            self.state.sparse_guides = [
                render_gaussian_guide(h, w, pts) for pts in tracked_points
            ]

        print("\nAll pre-computation finished.")
        return self.state

    def uses_flow_occlusion(self) -> bool:
        pc = self.pipeline_cfg
        return (
            pc.use_flow_occlusion_masks
            or pc.use_flow_occlusion_modulation
            or pc.use_flow_occlusion_fill
            or pc.use_flow_occlusion_refine
            or self.debug_cfg.save_occlusion_debug
        )

    def _compute_optical_flow(self, content_frames: List[np.ndarray]) -> None:
        self.state.fwd_flows = compute_guide(
            content_frames=content_frames,
            cache_dir=self.project_cfg.cache_dir,
            prefix="flow",
            extension="npy",
            force_recompute=self.project_cfg.force_recompute_flow,
            num_expected=len(content_frames) - 1,
            title="Optical flow",
            load_cache_fn=load_cached_flow,
            compute_fn=compute_optical_flow_sequence,
            save_fn=save_flow_cache,
            compute_kwargs={"precomputation_cfg": self.precomputation_cfg},
        )

        if self.debug_cfg.save_flow_viz and self.state.fwd_flows:
            self._save_flow_visualizations(
                self.state.fwd_flows,
                subdir="fwd" if self.uses_flow_occlusion() else None,
            )

    def _compute_backward_optical_flow(self, content_frames: List[np.ndarray]) -> None:
        self.state.bwd_flows = compute_guide(
            content_frames=content_frames,
            cache_dir=self.project_cfg.cache_dir,
            prefix="flow_bwd",
            extension="npy",
            force_recompute=self.project_cfg.force_recompute_flow,
            num_expected=len(content_frames) - 1,
            title="Backward optical flow",
            load_cache_fn=load_cached_flow,
            compute_fn=compute_backward_optical_flow_sequence,
            save_fn=save_flow_cache,
            compute_kwargs={"precomputation_cfg": self.precomputation_cfg},
        )

        if self.debug_cfg.save_flow_viz and self.state.bwd_flows:
            self._save_flow_visualizations(self.state.bwd_flows, subdir="bwd")

    def _compute_bidirectional_optical_flow(self, content_frames: List[np.ndarray]) -> None:
        cache_root = Path(self.project_cfg.cache_dir)
        num_expected = len(content_frames) - 1
        fwd_cache_ready = has_exact_cache_files(cache_root / "flow", "npy", num_expected)
        bwd_cache_ready = has_exact_cache_files(
            cache_root / "flow_bwd",
            "npy",
            num_expected,
        )

        if (
            not self.project_cfg.force_recompute_flow
            and fwd_cache_ready
            and bwd_cache_ready
        ):
            self._compute_optical_flow(content_frames)
            self._compute_backward_optical_flow(content_frames)
            return

        if self.project_cfg.force_recompute_flow or (
            not fwd_cache_ready and not bwd_cache_ready
        ):
            self.state.fwd_flows, self.state.bwd_flows = (
                compute_bidirectional_optical_flow_sequence(
                    content_frames,
                    self.precomputation_cfg,
                )
            )
            save_flow_cache(self.state.fwd_flows, cache_root / "flow")
            save_flow_cache(self.state.bwd_flows, cache_root / "flow_bwd")
            if self.debug_cfg.save_flow_viz and self.state.fwd_flows:
                self._save_flow_visualizations(self.state.fwd_flows, subdir="fwd")
            if self.debug_cfg.save_flow_viz and self.state.bwd_flows:
                self._save_flow_visualizations(self.state.bwd_flows, subdir="bwd")
        else:
            self._compute_optical_flow(content_frames)
            self._compute_backward_optical_flow(content_frames)

    def _compute_occlusion_masks(self) -> None:
        num_expected = len(self.state.fwd_flows)
        cache_root = Path(self.project_cfg.cache_dir) / self._occlusion_cache_key()
        fwd_dir = cache_root / "occlusion_fwd"
        bwd_dir = cache_root / "occlusion_bwd"

        can_load = (
            not self.project_cfg.force_recompute_flow
            and has_exact_cache_files(fwd_dir, "npy", num_expected)
            and has_exact_cache_files(bwd_dir, "npy", num_expected)
        )
        if can_load:
            self.state.fwd_occlusion_masks = load_cached_arrays(fwd_dir)
            self.state.bwd_occlusion_masks = load_cached_arrays(bwd_dir)
        else:
            self.state.fwd_occlusion_masks, self.state.bwd_occlusion_masks = (
                compute_flow_occlusion_masks(
                    self.state.fwd_flows,
                    self.state.bwd_flows,
                    alpha=self.pipeline_cfg.occlusion_consistency_alpha,
                    beta=self.pipeline_cfg.occlusion_consistency_beta,
                    dilate=self.pipeline_cfg.occlusion_mask_dilate,
                    coverage=self.pipeline_cfg.occlusion_use_coverage_mask,
                )
            )
            save_array_cache(self.state.fwd_occlusion_masks, fwd_dir)
            save_array_cache(self.state.bwd_occlusion_masks, bwd_dir)

        if self.debug_cfg.save_occlusion_debug:
            self._save_occlusion_debug()

    def _compute_edge_maps(self, content_frames: List[np.ndarray]) -> None:
        self.state.edge_maps = compute_guide(
            content_frames=content_frames,
            cache_dir=self.project_cfg.cache_dir,
            prefix="edges",
            extension="png",
            force_recompute=self.project_cfg.force_recompute_edge,
            num_expected=len(content_frames),
            title="Edge Maps",
            load_cache_fn=load_frames_from_dir,
            compute_fn=compute_edge_maps,
            save_fn=save_edge_map_cache,
            compute_kwargs={"edge_method": self.precomputation_cfg.edge_method},
        )

    def _flow_viz_output_dir(self) -> Path:
        p = Path(self.debug_cfg.flow_viz_dir)
        return p if p.is_absolute() else Path.cwd() / p

    def _save_flow_visualizations(
        self,
        flows: List[np.ndarray],
        *,
        subdir: Optional[str] = None,
    ) -> None:
        out_dir = self._flow_viz_output_dir()
        if subdir:
            out_dir = out_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing flow color maps to: {out_dir}")
        for i, flow in enumerate(tqdm(flows, desc="Flow visualization")):
            f32 = flow if flow.dtype == np.float32 else flow.astype(np.float32)
            bgr = flow_to_image(f32, convert_to_bgr=True)
            write_image(out_dir / f"{i:05d}.png", bgr)

    def _occlusion_cache_key(self) -> str:
        params = {
            "alpha": self.pipeline_cfg.occlusion_consistency_alpha,
            "beta": self.pipeline_cfg.occlusion_consistency_beta,
            "dilate": self.pipeline_cfg.occlusion_mask_dilate,
            "coverage": self.pipeline_cfg.occlusion_use_coverage_mask,
        }
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"occlusion_{digest}"

    def _occlusion_debug_output_dir(self) -> Path:
        p = Path(self.debug_cfg.occlusion_debug_dir)
        return p if p.is_absolute() else Path.cwd() / p

    def _save_occlusion_debug(self) -> None:
        out_dir = self._occlusion_debug_output_dir()
        fwd_dir = out_dir / "fwd"
        bwd_dir = out_dir / "bwd"
        fwd_dir.mkdir(parents=True, exist_ok=True)
        bwd_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing occlusion masks to: {out_dir}")
        for i, mask in enumerate(
            tqdm(self.state.fwd_occlusion_masks, desc="Fwd occlusion debug")
        ):
            write_image(fwd_dir / f"{i:05d}.png", mask_to_bgr(mask))
        for i, mask in enumerate(
            tqdm(self.state.bwd_occlusion_masks, desc="Bwd occlusion debug")
        ):
            write_image(bwd_dir / f"{i:05d}.png", mask_to_bgr(mask))

def compute_edge_maps(
    content_frames: List[np.ndarray], edge_method: str
) -> List[np.ndarray]:
    from .engines.edge_engine import EdgeEngine

    engine = EdgeEngine(method=edge_method)
    try:
        return engine.compute(content_frames)
    finally:
        del engine
