"""Image-op engines (``kernels.imageops``) against their reference: OpenCV for warping,
occlusion masks and histogram blending, Taichi for forward splatting and pull-push.
Warp parity assumes OpenCV 5's remap (exact float bilinear for 1/3/4 channels,
1/32-pixel fixed point for 2)."""

import cv2
import numpy as np
import pytest

from ezsynth.tests.engines import engine_param, unavailable
from ezsynth.utils.blend_logic import hist_blender
from ezsynth.utils.occlusion import compute_flow_occlusion_masks
from ezsynth.utils.warp_utils import Warp

# Odd sizes exercise the last row/column of the pull-push pyramid.
H, W = 45, 61


def _require(engine: str) -> None:
    if reason := unavailable(engine):
        pytest.skip(f"{engine} reference: {reason}")


def _smooth_image(rng, h=H, w=W):
    base = rng.uniform(0, 255, (h // 6 + 2, w // 6 + 2, 3)).astype(np.float32)
    up = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 10, (h, w, 3)).astype(np.float32)
    return np.clip(up + noise, 0, 255).astype(np.uint8)


def _flow(rng, scale, h=H, w=W, noise=0.4):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    smooth = np.stack([np.sin(yy / 9.0) * scale, np.cos(xx / 13.0) * scale], -1)
    return (smooth + rng.normal(0, noise, (h, w, 2))).astype(np.float32)


# --- warp (reference: OpenCV) ---------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("border", ["reflect", "constant"])
@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_raw_warp_matches_cv2_remap(engine, channels, border):
    import ezsynth_wgpu

    rng = np.random.default_rng(channels)
    src = rng.random((H, W, channels)).astype(np.float32)
    flow = _flow(rng, 8.0)  # reaches well past the borders
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mode = cv2.BORDER_REFLECT if border == "reflect" else cv2.BORDER_CONSTANT
    expected = cv2.remap(
        src, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR, borderMode=mode
    ).reshape(H, W, channels)

    got = ezsynth_wgpu.warp(src, flow, border=border)

    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("engine", "whole_frame"),
    [
        pytest.param(
            *engine_param("wgpu").values, True, marks=engine_param("wgpu").marks
        ),
        # Taichi clamps samples outside the frame where OpenCV reflects.
        pytest.param(
            *engine_param("taichi").values, False, marks=engine_param("taichi").marks
        ),
    ],
    ids=["wgpu", "taichi"],
)
def test_linear_warp_matches_opencv(engine, whole_frame):
    rng = np.random.default_rng(6)
    data = rng.random((H, W, 3)).astype(np.float32)
    flow = _flow(rng, 3.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = xx + flow[..., 0], yy + flow[..., 1]
    inside = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    region = np.ones_like(inside) if whole_frame else inside

    got = Warp(H, W, engine=engine).run_warping_float_map(data, flow)
    expected = Warp(H, W).run_warping_float_map(data, flow)

    np.testing.assert_allclose(got[region], expected[region], atol=1e-5)


@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_uint8_warping_matches_opencv(engine):
    rng = np.random.default_rng(6)
    img = _smooth_image(rng)
    flow = _flow(rng, 3.0)

    got = Warp(H, W, engine=engine).run_warping(img, flow)
    expected = Warp(H, W).run_warping(img, flow)

    assert np.abs(got.astype(int) - expected.astype(int)).max() <= 1


@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_nearest_warp_of_int_map_is_exact(engine):
    rng = np.random.default_rng(5)
    nnf = rng.integers(0, 400, (H, W, 2)).astype(np.int32)
    flow = _flow(rng, 20.0)

    got = Warp(H, W, engine=engine).run_warping_float_map(
        nnf, flow, interpolation=cv2.INTER_NEAREST
    )
    expected = Warp(H, W).run_warping_float_map(
        nnf, flow, interpolation=cv2.INTER_NEAREST
    )

    np.testing.assert_array_equal(got, expected)


# --- occlusion masks and histogram blending (reference: OpenCV) ---------------------


@pytest.mark.parametrize(("dilate", "coverage"), [(0, False), (1, True), (2, True)])
@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_occlusion_masks_match_opencv(engine, dilate, coverage):
    rng = np.random.default_rng(dilate)
    fwd = [_flow(rng, 4.0, noise=0.05), _flow(rng, 2.0, noise=0.05)]
    bwd = [-f + rng.normal(0, 0.25, f.shape).astype(np.float32) for f in fwd]
    kwargs = {"dilate": dilate, "coverage": coverage}

    exp_fwd, exp_bwd = compute_flow_occlusion_masks(fwd, bwd, **kwargs)
    got_fwd, got_bwd = compute_flow_occlusion_masks(fwd, bwd, engine=engine, **kwargs)

    for got, exp in zip(got_fwd + got_bwd, exp_fwd + exp_bwd):
        assert 0 < np.count_nonzero(exp) < exp.size
        np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize(
    ("engine", "max_mean_diff", "min_within_3"),
    [
        # OpenCV's 8-bit BGR2Lab interpolates a LUT; the GPU converts in float, so Lab
        # values differ by up to 2 levels and the blend amplifies a few of them.
        pytest.param(
            *engine_param("wgpu").values, 0.6, 0.99, marks=engine_param("wgpu").marks
        ),
        # Taichi's blend is an approximation of the OpenCV path.
        pytest.param(
            *engine_param("taichi").values,
            1.0,
            0.98,
            marks=engine_param("taichi").marks,
        ),
    ],
    ids=["wgpu", "taichi"],
)
def test_hist_blend_is_close_to_opencv(engine, max_mean_diff, min_within_3):
    rng = np.random.default_rng(7)
    a = _smooth_image(rng, 120, 160)
    b = (_smooth_image(rng, 120, 160) * 0.6 + 60).astype(np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    mask = ((xx // 20 + yy // 20) % 2).astype(np.uint8)

    expected = hist_blender(a, b, mask)
    got = hist_blender(a, b, mask, engine=engine)

    diff = np.abs(got.astype(int) - expected.astype(int))
    assert diff.mean() < max_mean_diff
    assert np.mean(diff <= 3) > min_within_3


# --- forward splatting and pull-push (reference: Taichi) -------------------------


@pytest.mark.parametrize("fill_holes", [False, True])
@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_forward_splat_matches_taichi(engine, fill_holes):
    _require("taichi")
    rng = np.random.default_rng(8)
    img = _smooth_image(rng)
    flow = _flow(rng, 3.0)

    exp, exp_w = Warp(H, W, engine="taichi").run_forward_warping(
        img, flow, fill_holes=fill_holes, return_weight=True
    )
    got, got_w = Warp(H, W, engine=engine).run_forward_warping(
        img, flow, fill_holes=fill_holes, return_weight=True
    )

    assert got.dtype == np.uint8
    # Float atomics add in a different order: truncation may flip by one level.
    assert np.abs(got.astype(int) - exp.astype(int)).max() <= 1
    np.testing.assert_allclose(got_w, exp_w, atol=1e-5)


@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_bilateral_forward_splat_of_float_map_matches_taichi(engine):
    _require("taichi")
    rng = np.random.default_rng(9)
    data = rng.random((H, W)).astype(np.float32)
    flow = _flow(rng, 2.0)
    src_guide = _smooth_image(rng).astype(np.float32)
    tgt_guide = _smooth_image(rng).astype(np.float32)
    kwargs = {"src_guide": src_guide, "tgt_guide": tgt_guide}

    exp = Warp(H, W, engine="taichi").run_forward_warping(data, flow, **kwargs)
    got = Warp(H, W, engine=engine).run_forward_warping(data, flow, **kwargs)

    np.testing.assert_allclose(got, exp, atol=1e-5)


@pytest.mark.parametrize("engine", [engine_param("wgpu")])
def test_pull_push_matches_taichi(engine):
    _require("taichi")
    rng = np.random.default_rng(10)
    weight = np.where(rng.random((H, W)) < 0.4, 0.0, rng.random((H, W)) * 2)
    weight = weight.astype(np.float32)
    color = (rng.random((H, W, 3)) * 255 * weight[..., None]).astype(np.float32)

    exp_c, exp_w = Warp(H, W, engine="taichi").run_pull_push(
        color.copy(), weight.copy()
    )
    got_c, got_w = Warp(H, W, engine=engine).run_pull_push(color, weight)

    np.testing.assert_allclose(got_c, exp_c, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(got_w, exp_w, atol=1e-6)
