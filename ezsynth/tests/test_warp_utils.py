import cv2
import numpy as np
import pytest

from ezsynth.utils.blend_utils import Blender
from ezsynth.utils.warp_utils import PositionalGuide, Warp


def test_warp_float_map_preserves_numeric_dtype_for_identity_flow():
    warp = Warp(2, 2)
    data = np.arange(4, dtype=np.int32).reshape(2, 2)
    flow = np.zeros((2, 2, 2), dtype=np.float32)

    warped = warp.run_warping_float_map(data, flow, interpolation=cv2.INTER_NEAREST)

    assert warped.dtype == np.float32
    np.testing.assert_array_equal(warped, data.astype(np.float32))


def test_forward_warping_needs_a_splatting_engine():
    warp = Warp(2, 2)
    image = np.full((2, 2, 3), 10, dtype=np.uint8)
    flow = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="taichi"):
        warp.run_forward_warping(image, flow)


def test_warp_float_map_replaces_non_finite_values():
    warp = Warp(2, 2)
    data = np.array([[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float32)
    flow = np.zeros((2, 2, 2), dtype=np.float32)

    warped = warp.run_warping_float_map(data, flow)

    np.testing.assert_array_equal(
        warped,
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_positional_guide_can_use_forward_warping(monkeypatch):
    called = {"forward": False}

    def _fake_forward(self, img, flow, **kwargs):
        called["forward"] = True
        return img.copy()

    monkeypatch.setattr(Warp, "run_forward_warping", _fake_forward)

    guide = PositionalGuide(2, 2, use_forward_warp=True)
    result = guide.create_from_flow(np.zeros((2, 2, 2), dtype=np.float32))

    assert called["forward"] is True
    assert result.dtype == np.uint8


def test_blender_uses_forward_warping_for_mask_propagation(monkeypatch):
    calls = []

    def _fake_forward(self, img, flow, **kwargs):
        calls.append((img.copy(), flow.copy()))
        return img

    monkeypatch.setattr(Warp, "run_forward_warping", _fake_forward)

    blender = Blender(2, 2, poisson_solver="disabled", use_forward_warping=True)
    masks = [
        np.zeros((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
    ]
    flows = [np.zeros((2, 2, 2), dtype=np.float32) for _ in masks]

    warped = blender.warp_masks(flows, masks)

    assert len(warped) == 2
    assert len(calls) == 1
