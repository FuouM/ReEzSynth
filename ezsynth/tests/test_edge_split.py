import numpy as np

from ezsynth.edge.batch import compute_edge, compute_edge_frame
from ezsynth.engines.edge_engine import EdgeEngine


def test_classic_edge_frame_returns_three_channel_uint8():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[:, 2:] = 255

    edge = compute_edge_frame(frame, "Classic")

    assert edge.shape == frame.shape
    assert edge.dtype == np.uint8


def test_edge_engine_wraps_split_edge_batch(monkeypatch):
    expected = [np.zeros((2, 2, 3), dtype=np.uint8)]
    calls = []

    def _fake_compute_edge(frames, edge_method):
        calls.append((len(frames), edge_method))
        return expected

    monkeypatch.setattr("ezsynth.engines.edge_engine.compute_edge", _fake_compute_edge)

    result = EdgeEngine(method="Classic").compute(
        [np.zeros((2, 2, 3), dtype=np.uint8)]
    )

    assert result is expected
    assert calls == [(1, "Classic")]


def test_compute_edge_batch_classic():
    frames = [np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(2)]

    edges = compute_edge(frames, "Classic")

    assert len(edges) == 2
    assert all(edge.shape == frame.shape for edge, frame in zip(edges, frames))
