"""Synthesis backends (``kernels.synth``) on one pyramid level: the deterministic kernels
(initial vote and error map) match the Taichi reference, and every search mode lowers
the error and keeps matches inside the source."""

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
from ezsynth.kernels import synth
from ezsynth.kernels.synth.capabilities import (
    BACKEND_COST_FUNCTIONS,
    BACKEND_SEARCH_MODES,
)
from ezsynth.kernels.synth.common import random_init_nnf
from ezsynth.tests.engines import engine_param, unavailable

SH, SW, TH, TW, PS = 32, 40, 28, 36, 5
COSTS = {"ssd": COST_FUNCTION_SSD, "ncc": COST_FUNCTION_NCC, "sad": COST_FUNCTION_SAD}


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


def _run(backend, x, *, cost, iters, modulation=False, patch_size=PS):
    """``run_level`` with the inputs on the backend's device, like the engine."""
    device = backend.device
    rand_states = torch.empty(TH * TW * 48, dtype=torch.uint8, device=device)
    if hasattr(backend, "init_rand_states"):
        backend.init_rand_states(rand_states)
    mod = x["modulation"] if modulation else torch.empty(0, dtype=torch.uint8)
    image, error, nnf = backend.run_level(
        x["style"].to(device),
        x["source_guide"].to(device),
        x["target_guide"].to(device),
        mod.to(device),
        x["nnf"].clone().to(device),
        x["style_weights"].to(device),
        x["guide_weights"].to(device),
        3500.0,
        patch_size,
        EBSYNTH_VOTEMODE_WEIGHTED,
        iters,
        4,
        5,
        rand_states,
        cost,
    )
    return image.cpu().numpy(), error.cpu().numpy(), nnf.cpu().numpy()


def _known_divergence(backend: str, bilateral: bool, n_size_step: int) -> str | None:
    # The torch and cuda backends implement these options differently from the
    # Taichi/wgpu pair (which agree bit for bit); kept as strict xfails so a fix shows.
    if backend in ("torch", "cuda") and bilateral:
        return "bilateral weights differ from Taichi (votes and errors)"
    if backend in ("torch", "cuda") and n_size_step > 1:
        return "n_size_step > 1 error map differs from Taichi (votes agree)"
    return None


def _kernel_params():
    params = []
    for backend, cost, bilateral, n_size_step, modulation in itertools.product(
        ("wgpu", "torch", "cuda"),
        COSTS,
        (False, True),
        (1, 2),
        (False, True),
    ):
        if cost not in BACKEND_COST_FUNCTIONS[backend]:
            continue
        marks = []
        reason = _known_divergence(backend, bilateral, n_size_step)
        if reason:
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
        param = engine_param(backend, *marks)
        params.append(
            pytest.param(
                *param.values,
                cost,
                bilateral,
                n_size_step,
                modulation,
                marks=param.marks,
                id=f"{backend}-{cost}-bil{bilateral:d}-n{n_size_step}-mod{modulation:d}",
            )
        )
    return params


@pytest.mark.parametrize(
    ("backend", "cost", "bilateral", "n_size_step", "modulation"), _kernel_params()
)
def test_vote_and_error_map_match_taichi(
    level_inputs, backend, cost, bilateral, n_size_step, modulation
):
    if reason := unavailable("taichi"):
        pytest.skip(f"Taichi reference: {reason}")
    # search_vote_iters=0 runs only the deterministic kernels: initial vote and error map.
    cfg = EbsynthParamsConfig(
        backend=backend,
        cost_function=cost,
        use_bilateral=bilateral,
        n_size_step=n_size_step,
    )
    run = {"cost": COSTS[cost], "iters": 0, "modulation": modulation}
    ref_img, ref_err, _ = _run(
        synth.create("taichi", cfg, PipelineConfig()), level_inputs, **run
    )
    img, err, nnf = _run(
        synth.create(backend, cfg, PipelineConfig()), level_inputs, **run
    )
    np.testing.assert_array_equal(img, ref_img)
    # wgpu repeats Taichi's float operation order; torch/cuda sum in another order.
    rtol = 1e-4 if backend == "wgpu" else 1e-3
    np.testing.assert_allclose(err, ref_err, rtol=rtol, atol=1e-2)
    np.testing.assert_array_equal(nnf, level_inputs["nnf"].numpy())


def _search_params():
    return [
        pytest.param(
            *engine_param(backend).values,
            mode,
            cost,
            marks=engine_param(backend).marks,
            id=f"{backend}-{mode}-{cost}",
        )
        for backend in ("wgpu", "taichi", "torch", "cuda")
        for mode in BACKEND_SEARCH_MODES[backend]
        for cost in ("ssd", "ncc")
    ]


@pytest.mark.parametrize(("backend", "search_mode", "cost"), _search_params())
def test_search_lowers_error_and_keeps_matches_valid(
    level_inputs, backend, search_mode, cost
):
    cfg = EbsynthParamsConfig(
        backend=backend, cost_function=cost, search_mode=search_mode, multicand_k=3
    )
    level = synth.create(backend, cfg, PipelineConfig())
    _, start_err, _ = _run(level, level_inputs, cost=COSTS[cost], iters=0)
    image, err, nnf = _run(level, level_inputs, cost=COSTS[cost], iters=4)
    r = PS // 2
    assert image.shape == (TH, TW, 3)
    assert np.isfinite(err).all()
    assert (nnf[..., 0] >= 0).all() and (nnf[..., 0] < SW).all()
    assert (nnf[..., 1] >= 0).all() and (nnf[..., 1] < SH).all()
    assert (nnf[..., 0] >= r).mean() > 0.99 and (nnf[..., 1] < SH - r).mean() > 0.99
    assert err.mean() < start_err.mean()


@pytest.mark.parametrize("backend", [engine_param("wgpu")])
def test_even_patch_size_raises_value_error(level_inputs, backend):
    level = synth.create(backend, EbsynthParamsConfig(), PipelineConfig())
    with pytest.raises(ValueError, match="patch_size"):
        _run(level, level_inputs, cost=COST_FUNCTION_SSD, iters=1, patch_size=4)
