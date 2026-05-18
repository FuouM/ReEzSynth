from pathlib import Path

import numpy as np

from ezsynth.config import DebugConfig, PipelineConfig, PrecomputationConfig, ProjectConfig
from ezsynth.precompute import compute_guide
from ezsynth.precompute_runner import PrecomputeRunner


def test_compute_guide_uses_cache_when_exact_files_exist(tmp_path: Path):
    cache_root = tmp_path / "cache"
    flow_dir = cache_root / "flow"
    flow_dir.mkdir(parents=True)
    np.save(flow_dir / "00000.npy", np.zeros((2, 2, 2), dtype=np.float32))

    result = compute_guide(
        content_frames=[np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        cache_dir=str(cache_root),
        prefix="flow",
        extension="npy",
        force_recompute=False,
        num_expected=1,
        title="Flow",
        load_cache_fn=lambda path: ["loaded", path],
        compute_fn=lambda frames: ["computed"],
        save_fn=lambda arrays, path: None,
    )

    assert result == ["loaded", flow_dir]


def test_precompute_runner_populates_state_without_heavy_engines(tmp_path, monkeypatch):
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    edge = np.zeros((2, 2, 3), dtype=np.uint8)
    sparse = np.ones((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "ezsynth.precompute_runner.compute_optical_flow_sequence",
        lambda frames, precomputation_cfg: [flow],
    )
    monkeypatch.setattr(
        "ezsynth.precompute_runner.compute_edge_maps",
        lambda frames, edge_method: [edge, edge.copy()],
    )
    monkeypatch.setattr(
        "ezsynth.precompute_runner.generate_tracked_features",
        lambda initial_frame, flows: [np.zeros((1, 2)), np.ones((1, 2))],
    )
    monkeypatch.setattr(
        "ezsynth.precompute_runner.render_gaussian_guide",
        lambda h, w, pts: sparse.copy(),
    )

    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
    runner = PrecomputeRunner(
        project_cfg=ProjectConfig(
            content_dir="content",
            style_path="style.png",
            style_indices=[0],
            output_dir=str(tmp_path / "output"),
            cache_dir=str(tmp_path / "cache"),
            force_recompute_flow=True,
            force_recompute_edge=True,
        ),
        precomputation_cfg=PrecomputationConfig(),
        pipeline_cfg=PipelineConfig(use_sparse_feature_guide=True),
        debug_cfg=DebugConfig(),
    )

    state = runner.run(frames)

    assert state.fwd_flows[0] is flow
    assert len(state.edge_maps) == 2
    assert len(state.sparse_guides) == 2
    np.testing.assert_array_equal(state.edge_maps[0], edge)
    np.testing.assert_array_equal(state.sparse_guides[0], sparse)
