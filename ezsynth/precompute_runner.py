"""Precompute orchestration for flow, edges, and sparse guides."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .config import DebugConfig, PipelineConfig, PrecomputationConfig, ProjectConfig
from .flow.run import compute_optical_flow_sequence
from .precompute import PrecomputeState, compute_guide
from .utils.feature_utils import generate_tracked_features, render_gaussian_guide
from .utils.io_utils import load_frames_from_dir, write_image
from .utils.pipeline_utils import (
    load_cached_flow,
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
            self._save_flow_visualizations(self.state.fwd_flows)

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

    def _save_flow_visualizations(self, flows: List[np.ndarray]) -> None:
        out_dir = self._flow_viz_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing flow color maps to: {out_dir}")
        for i, flow in enumerate(tqdm(flows, desc="Flow visualization")):
            f32 = flow if flow.dtype == np.float32 else flow.astype(np.float32)
            bgr = flow_to_image(f32, convert_to_bgr=True)
            write_image(out_dir / f"{i:05d}.png", bgr)

def compute_edge_maps(
    content_frames: List[np.ndarray], edge_method: str
) -> List[np.ndarray]:
    from .engines.edge_engine import EdgeEngine

    engine = EdgeEngine(method=edge_method)
    try:
        return engine.compute(content_frames)
    finally:
        del engine
