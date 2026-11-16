import numpy as np

from ezsynth.edge.batch import compute_edge, compute_edge_frame


def test_classic_edge_frame_returns_three_channel_uint8():
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[:, 2:] = 255

    edge = compute_edge_frame(frame, "Classic")

    assert edge.shape == frame.shape
    assert edge.dtype == np.uint8


def test_compute_edge_batch_classic():
    frames = [np.zeros((3, 3, 3), dtype=np.uint8) for _ in range(2)]

    edges = compute_edge(frames, "Classic")

    assert len(edges) == 2
    assert all(edge.shape == frame.shape for edge, frame in zip(edges, frames))
