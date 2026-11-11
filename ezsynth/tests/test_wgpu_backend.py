"""WebGPU backend: kernel parity with Taichi, search validity, FastBlend matching."""

import itertools

import numpy as np
import pytest
import torch

from ezsynth.config import EbsynthParamsConfig, PipelineConfig
from ezsynth.consts import (
    COST_FUNCTION_NCC,
    COST_FUNCTION_SAD,
    COST_FUNCTION_SSD,
    EBSYNTH_VOTEMODE_WEIGHTED,
)
from ezsynth.engines.backends.common import random_init_nnf
from ezsynth.engines.backends.wgpu_backend import WgpuBackend

pytest.importorskip("ezsynth_wgpu")

try:
    import ezsynth_wgpu

    ezsynth_wgpu.adapter_info()
except RuntimeError as exc:  # no WebGPU adapter on this machine
    pytest.skip(f"no WebGPU adapter: {exc}", allow_module_level=True)

try:
    import taichi  # noqa: F401

    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

SH, SW, TH, TW, PS = 32, 40, 28, 36, 5


def _smooth(rng, h, w, c):
    base = torch.from_numpy(
        rng.uniform(0, 255, (1, c, h // 4 + 2, w // 4 + 2)).astype(np.float32)
    )
    up = torch.nn.functional.interpolate(base, size=(h, w), mode="bilinear")[0].permute(
        1, 2, 0
    )
    noise = torch.from_numpy(rng.normal(0, 8, (h, w, c)).astype(np.float32))
    return (up + noise).clamp(0, 255).to(torch.uint8).contiguous()


@pytest.fixture(scope="module")
def level_inputs():
    rng = np.random.default_rng(3)
    nnf = random_init_nnf(
        "cpu", TH, TW, SH, SW, PS, generator=torch.Generator().manual_seed(1)
    )
    return {
        "style": _smooth(rng, SH, SW, 3),
        "source_guide": _smooth(rng, SH, SW, 6),
        "target_guide": _smooth(rng, TH, TW, 6),
        "modulation": torch.from_numpy(
            rng.integers(0, 256, (TH, TW, 6)).astype(np.uint8)
        ),
        "nnf": nnf,
        "style_weights": torch.full((3,), 1 / 3),
        "guide_weights": torch.tensor([2 / 3] * 3 + [0.5] * 3, dtype=torch.float32),
    }


def _run(backend, x, *, cost, iters, modulation=False):
    mod = x["modulation"] if modulation else torch.empty(0, dtype=torch.uint8)
    image, error, nnf = backend.run_level(
        x["style"],
        x["source_guide"],
        x["target_guide"],
        mod,
        x["nnf"].clone(),
        x["style_weights"],
        x["guide_weights"],
        3500.0,
        PS,
        EBSYNTH_VOTEMODE_WEIGHTED,
        iters,
        4,
        5,
        None,
        cost,
    )
    return image.cpu().numpy(), error.cpu().numpy(), nnf.cpu().numpy()


@pytest.mark.skipif(not HAS_TAICHI, reason="taichi not installed")
@pytest.mark.parametrize(
    ("cost", "bilateral", "n_size_step", "modulation"),
    list(
        itertools.product(
            [COST_FUNCTION_SSD, COST_FUNCTION_SAD, COST_FUNCTION_NCC],
            [False, True],
            [1, 2],
            [False, True],
        )
    ),
)
def test_vote_and_error_map_match_taichi(
    level_inputs, cost, bilateral, n_size_step, modulation
):
    # search_vote_iters=0 runs only the deterministic kernels: initial vote and error map.
    from ezsynth.engines.backends.taichi_backend import TaichiBackend

    cfg = EbsynthParamsConfig(use_bilateral=bilateral, n_size_step=n_size_step)
    ti_img, ti_err, _ = _run(
        TaichiBackend(cfg, PipelineConfig()),
        level_inputs,
        cost=cost,
        iters=0,
        modulation=modulation,
    )
    wg_img, wg_err, wg_nnf = _run(
        WgpuBackend(cfg, PipelineConfig()),
        level_inputs,
        cost=cost,
        iters=0,
        modulation=modulation,
    )
    np.testing.assert_array_equal(wg_img, ti_img)
    np.testing.assert_allclose(wg_err, ti_err, rtol=1e-4, atol=1e-2)
    np.testing.assert_array_equal(wg_nnf, level_inputs["nnf"].numpy())


@pytest.mark.parametrize(
    ("search_mode", "cost"),
    list(
        itertools.product(
            ["patchmatch", "jfa", "multicand"], [COST_FUNCTION_SSD, COST_FUNCTION_NCC]
        )
    ),
)
def test_search_lowers_error_and_keeps_matches_valid(level_inputs, search_mode, cost):
    cfg = EbsynthParamsConfig(search_mode=search_mode, multicand_k=3)
    backend = WgpuBackend(cfg, PipelineConfig())
    _, start_err, _ = _run(backend, level_inputs, cost=cost, iters=0)
    image, err, nnf = _run(backend, level_inputs, cost=cost, iters=4)
    r = PS // 2
    assert image.shape == (TH, TW, 3)
    assert np.isfinite(err).all()
    assert (nnf[..., 0] >= 0).all() and (nnf[..., 0] < SW).all()
    assert (nnf[..., 1] >= 0).all() and (nnf[..., 1] < SH).all()
    assert (nnf[..., 0] >= r).mean() > 0.99 and (nnf[..., 1] < SH - r).mean() > 0.99
    assert err.mean() < start_err.mean()


def test_invalid_patch_size_raises_value_error(level_inputs):
    backend = WgpuBackend(EbsynthParamsConfig(), PipelineConfig())
    x = level_inputs
    with pytest.raises(ValueError, match="patch_size"):
        backend.run_level(
            x["style"],
            x["source_guide"],
            x["target_guide"],
            torch.empty(0, dtype=torch.uint8),
            x["nnf"].clone(),
            x["style_weights"],
            x["guide_weights"],
            3500.0,
            4,
            EBSYNTH_VOTEMODE_WEIGHTED,
            1,
            1,
            5,
            None,
            COST_FUNCTION_SSD,
        )


@pytest.mark.parametrize(
    "mode",
    [
        {},
        {"use_mean_target_style": True},
        {"use_pairwise_patch_error": True},
        {"tracking_window_size": 1},
    ],
    ids=["plain", "mean", "pairwise", "track"],
)
def test_fastblend_recovers_a_shift(mode):
    from FastBlend.src.backends.patch_matcher_dispatch import (
        build_pyramid_patch_matcher,
    )

    rng = np.random.default_rng(0)
    h, w, b, dy, dx = 96, 128, 2, 3, 5
    base = torch.nn.functional.interpolate(
        torch.from_numpy(
            rng.uniform(0, 255, (b, 3, h // 8, w // 8)).astype(np.float32)
        ),
        size=(h + 16, w + 16),
        mode="bicubic",
    ).clamp(0, 255)
    source = base[:, :, 8 : 8 + h, 8 : 8 + w].permute(0, 2, 3, 1).contiguous()
    target = (
        base[:, :, 8 + dy : 8 + dy + h, 8 + dx : 8 + dx + w]
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    matcher = build_pyramid_patch_matcher(
        h, w, 3, 5, num_iter=5, backend="wgpu", **mode
    )

    nnf, style = matcher.estimate_nnf(source, source, source)
    rows = np.arange(h)[None, :, None]
    if not mode.get("use_pairwise_patch_error"):
        # Equal frames: the identity match has zero error and must survive. (Pairwise
        # errors compare the two different frames of a pair, so identity is not optimal.)
        assert (nnf.numpy()[..., 0] == rows).mean() > 0.95
    assert style.shape == source.shape

    nnf, _ = matcher.estimate_nnf(source, target, source)
    inner = nnf.numpy()[:, 12:-12, 12:-12]
    yy, xx = np.mgrid[12 : h - 12, 12 : w - 12]
    hit = (inner[..., 0] == yy + dy) & (inner[..., 1] == xx + dx)
    assert hit.mean() > (0.7 if mode.get("use_pairwise_patch_error") else 0.95)
