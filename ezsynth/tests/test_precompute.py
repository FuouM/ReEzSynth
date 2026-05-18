from pathlib import Path

import numpy as np

from ezsynth.config import (
    DebugConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from ezsynth.flow.run import (
    compute_backward_optical_flow_sequence,
    compute_bidirectional_optical_flow_sequence,
    compute_optical_flow_sequence,
    optical_flow_engine,
)
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


def test_precompute_runner_populates_occlusion_masks(tmp_path, monkeypatch):
    fwd_flow = np.zeros((3, 3, 2), dtype=np.float32)
    bwd_flow = np.zeros((3, 3, 2), dtype=np.float32)
    edge = np.zeros((3, 3, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "ezsynth.precompute_runner.compute_bidirectional_optical_flow_sequence",
        lambda frames, precomputation_cfg: ([fwd_flow], [bwd_flow]),
    )
    monkeypatch.setattr(
        "ezsynth.precompute_runner.compute_edge_maps",
        lambda frames, edge_method: [edge, edge.copy()],
    )

    frames = [np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(2)]
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
        pipeline_cfg=PipelineConfig(
            use_flow_occlusion_modulation=True,
            occlusion_mask_dilate=0,
            occlusion_use_coverage_mask=False,
        ),
        debug_cfg=DebugConfig(),
    )

    state = runner.run(frames)

    assert state.fwd_flows[0] is fwd_flow
    assert state.bwd_flows[0] is bwd_flow
    assert len(state.fwd_occlusion_masks) == 1
    assert len(state.bwd_occlusion_masks) == 1
    assert not np.any(state.fwd_occlusion_masks[0])
    assert not np.any(state.bwd_occlusion_masks[0])


def test_compute_optical_flow_sequence_supports_opencv_engine(monkeypatch):
    monkeypatch.setattr(
        "ezsynth.flow.run.compute_opencv_flow_sequence",
        lambda frames, method: [np.zeros((2, 2, 2), dtype=np.float32)],
    )

    flows = compute_optical_flow_sequence(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        PrecomputationConfig(flow_engine="OpenCV", opencv_flow_method="FARNEBACK"),
    )

    assert len(flows) == 1


def test_backward_flow_reverses_reversed_sequence(monkeypatch):
    monkeypatch.setattr(
        "ezsynth.flow.run.compute_optical_flow_sequence",
        lambda frames, precomputation_cfg: [
            frame[..., :2].astype(np.float32) for frame in frames[:-1]
        ],
    )
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]

    flows = compute_backward_optical_flow_sequence(
        frames,
        PrecomputationConfig(flow_engine="OpenCV"),
    )

    assert [int(flow[0, 0, 0]) for flow in flows] == [1, 2]


def test_bidirectional_flow_reuses_session(monkeypatch):
    calls = []

    def _fake_engine(precomputation_cfg):
        class _Session:
            def __enter__(self):
                def _compute(frames):
                    calls.append([int(frame[0, 0, 0]) for frame in frames])
                    return [np.zeros((2, 2, 2), dtype=np.float32) for _ in frames[:-1]]

                return _compute

            def __exit__(self, exc_type, exc, tb):
                return None

        return _Session()

    monkeypatch.setattr("ezsynth.flow.run.optical_flow_engine", _fake_engine)
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]

    fwd, bwd = compute_bidirectional_optical_flow_sequence(
        frames,
        PrecomputationConfig(flow_engine="OpenCV"),
    )

    assert len(fwd) == 2
    assert len(bwd) == 2
    assert calls == [[0, 1, 2], [2, 1, 0]]


def test_optical_flow_engine_session_supports_opencv():
    frames = [
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8),
    ]

    with optical_flow_engine(PrecomputationConfig(flow_engine="OpenCV")) as compute:
        flows = compute(frames)

    assert len(flows) == 1
