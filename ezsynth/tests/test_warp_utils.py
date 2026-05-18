import cv2
import numpy as np

from ezsynth.utils.blend_utils import Blender
from ezsynth.utils.warp_utils import PositionalGuide, Warp


def test_warp_float_map_preserves_numeric_dtype_for_identity_flow():
    warp = Warp(2, 2)
    data = np.arange(4, dtype=np.int32).reshape(2, 2)
    flow = np.zeros((2, 2, 2), dtype=np.float32)

    warped = warp.run_warping_float_map(data, flow, interpolation=cv2.INTER_NEAREST)

    assert warped.dtype == np.float32
    np.testing.assert_array_equal(warped, data.astype(np.float32))


def test_forward_warping_fallback_returns_weight_when_requested():
    warp = Warp(2, 2, use_taichi=False)
    image = np.full((2, 2, 3), 10, dtype=np.uint8)
    flow = np.zeros((2, 2, 2), dtype=np.float32)

    warped, weight = warp.run_forward_warping(image, flow, return_weight=True)

    np.testing.assert_array_equal(warped, image)
    np.testing.assert_array_equal(weight, np.ones((2, 2), dtype=np.float32))


def test_forward_warping_uses_splat_ops_when_available():
    class _FakeOps:
        def soft_splat_kernel(
            self,
            img,
            flow,
            dst_color,
            dst_weight,
            src_guide,
            tgt_guide,
            use_bilateral,
        ):
            dst_color[...] = img
            dst_weight[...] = 1.0

        def normalize_splat_kernel(self, dst_color, dst_weight, out, was_uint8):
            del dst_weight, was_uint8
            out[...] = dst_color

    warp = Warp(2, 2, use_taichi=False)
    warp.use_taichi = True
    warp._taichi_available = True
    warp.ops = _FakeOps()
    image = np.full((2, 2, 3), 10, dtype=np.uint8)
    flow = np.ones((2, 2, 2), dtype=np.float32)

    warped, weight = warp.run_forward_warping(
        image,
        flow,
        fill_holes=False,
        return_weight=True,
    )

    np.testing.assert_array_equal(warped, image)
    np.testing.assert_array_equal(weight, np.ones((2, 2), dtype=np.float32))


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
