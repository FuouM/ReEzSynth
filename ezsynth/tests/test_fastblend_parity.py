"""FastBlend engines (``kernels.fastblend``): every engine recovers a known shift, and
the single-level matchers' kernels match a reference implementation."""

import numpy as np
import pytest
import torch

from ezsynth.kernels import fastblend
from ezsynth.tests.engines import engine_params

ENGINES = engine_params("cuda", "cupy", "taichi", "wgpu")
# Engines with a single-level ``PatchMatcher`` (cuda/cupy/taichi kernels + core.py).
LEVEL_ENGINES = engine_params("cuda", "cupy", "taichi")


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
@pytest.mark.parametrize("engine", ENGINES)
def test_pyramid_matcher_recovers_a_shift(engine, mode):
    device = fastblend.load(engine).staging_device(0)
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
    source, target = source.to(device), target.to(device)
    matcher = fastblend.build_pyramid_matcher(engine, h, w, 3, 5, num_iter=5, **mode)

    nnf, style = matcher.estimate_nnf(source, source, source)
    rows = np.arange(h)[None, :, None]
    if not mode.get("use_pairwise_patch_error"):
        # Equal frames: the identity match has zero error and must survive. (Pairwise
        # errors compare the two different frames of a pair, so identity is not optimal.)
        assert (nnf.cpu().numpy()[..., 0] == rows).mean() > 0.95
    assert nnf.shape == (b, h, w, 2) and style.shape == source.shape
    assert int(nnf.min()) >= 0 and int(nnf[..., 0].max()) < h
    assert int(nnf[..., 1].max()) < w

    nnf, _ = matcher.estimate_nnf(source, target, source)
    inner = nnf.cpu().numpy()[:, 12:-12, 12:-12]
    yy, xx = np.mgrid[12 : h - 12, 12 : w - 12]
    hit = (inner[..., 0] == yy + dy) & (inner[..., 1] == xx + dx)
    assert hit.mean() > (0.7 if mode.get("use_pairwise_patch_error") else 0.95)


B, H, W, C = 1, 16, 16, 3
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2


def _level_inputs(device: torch.device):
    torch.manual_seed(42)
    shape = (B, H + 2 * PAD, W + 2 * PAD, C)
    source = torch.rand(shape).to(device)
    target = torch.rand(shape).to(device)
    ys = torch.arange(H).view(H, 1).expand(H, W)
    xs = torch.arange(W).view(1, W).expand(H, W)
    identity_nnf = torch.stack([ys, xs], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    return source, target, identity_nnf.to(torch.int32).to(device)


def _reference_ssd(source, nnf, target):
    source, nnf, target = source.cpu(), nnf.cpu(), target.cpu()
    r = PATCH_SIZE // 2
    err = torch.zeros((B, H, W))
    for b in range(B):
        for y in range(H):
            for x in range(W):
                sy, sx = (int(v) for v in nnf[b, y, x])
                tp = target[
                    b, y + PAD - r : y + PAD + r + 1, x + PAD - r : x + PAD + r + 1
                ]
                sp = source[
                    b, sy + PAD - r : sy + PAD + r + 1, sx + PAD - r : sx + PAD + r + 1
                ]
                err[b, y, x] = ((tp - sp) ** 2).sum()
    return err


def _level(engine):
    device = fastblend.load(engine).staging_device(0)
    matcher = fastblend.build_matcher(engine, H, W, C, PATCH_SIZE, num_iter=1)
    return matcher, _level_inputs(device)


@pytest.mark.parametrize("engine", LEVEL_ENGINES)
def test_identity_nnf_remap_reconstructs_source(engine):
    matcher, (source, _, nnf) = _level(engine)
    # Every patch covering (y, x) votes for source[y, x] under an identity NNF.
    remapped = matcher.unpad_image(matcher.apply_nnf_to_image(nnf, source))
    expected = source[:, PAD:-PAD, PAD:-PAD, :]
    assert (remapped - expected).abs().max().item() <= 1e-5


@pytest.mark.parametrize("engine", LEVEL_ENGINES)
def test_patch_error_matches_reference_ssd(engine):
    matcher, (source, target, nnf) = _level(engine)
    error = matcher.get_patch_error(source, nnf, target)
    reference = _reference_ssd(source, nnf, target)
    assert (error.cpu() - reference).abs().max().item() <= 1e-4


@pytest.mark.parametrize("engine", LEVEL_ENGINES)
def test_estimate_nnf_keeps_coordinates_in_bounds(engine):
    matcher, (source, target, nnf) = _level(engine)
    # estimate_nnf pads its (H, W) frames itself.
    source, target = matcher.unpad_image(source), matcher.unpad_image(target)
    nnf_new, style = matcher.estimate_nnf(source, target, source, nnf)
    assert nnf_new.shape == nnf.shape and style.shape == source.shape
    assert int(nnf_new[..., 0].min()) >= 0 and int(nnf_new[..., 0].max()) < H
    assert int(nnf_new[..., 1].min()) >= 0 and int(nnf_new[..., 1].max()) < W
