import numpy as np

from ezsynth.warp_video.common import (
    compose_adjacent_flows,
    flow_between_frames,
    get_style_keyframes,
)


def test_compose_adjacent_flows_identity_segments():
    h, w = 4, 4
    zero = np.zeros((h, w, 2), dtype=np.float32)
    out = compose_adjacent_flows([zero, zero])
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_flow_between_frames_same_index_is_zero():
    h, w = 3, 3
    adj_fwd = [np.ones((h, w, 2), dtype=np.float32)]
    adj_rev = [np.ones((h, w, 2), dtype=np.float32)]
    out = flow_between_frames(1, 1, adj_fwd, adj_rev, h, w)
    np.testing.assert_allclose(out, 0.0)


def test_get_style_keyframes_parses_names(tmp_path):
    (tmp_path / "style0.jpg").write_bytes(b"x")
    (tmp_path / "style10.png").write_bytes(b"x")
    (tmp_path / "other.txt").write_bytes(b"x")
    keys = get_style_keyframes(tmp_path)
    assert keys == {0: str(tmp_path / "style0.jpg"), 10: str(tmp_path / "style10.png")}
