import numpy as np

from ezsynth.flow.pad import pad_bgr_to_stride
from ezsynth.flow.run import compute_optical_flow_sequence
from ezsynth.config import PrecomputationConfig


def test_pad_bgr_to_stride_pads_bottom_and_right_with_edge_values():
    frame = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)

    padded, (h, w, pb, pr) = pad_bgr_to_stride(frame, stride=4)

    assert (h, w, pb, pr) == (3, 5, 1, 3)
    assert padded.shape == (4, 8, 3)
    np.testing.assert_array_equal(padded[:3, :5], frame)
    np.testing.assert_array_equal(padded[-1, -1], frame[-1, -1])


def test_flow_dispatch_uses_split_raft_module(monkeypatch):
    calls = []

    def _fake_raft(frames, model_name, device):
        calls.append((model_name, device, len(frames)))
        return [np.zeros((2, 2, 2), dtype=np.float32)]

    monkeypatch.setattr("ezsynth.flow.run.compute_custom_raft_sequence", _fake_raft)

    flows = compute_optical_flow_sequence(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        PrecomputationConfig(flow_engine="RAFT", flow_model="sintel"),
    )

    assert len(flows) == 1
    assert calls == [("sintel", "auto", 2)]
