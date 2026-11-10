import pytest
import torch

pytest.importorskip("taichi")

from FastBlend.src.backends import taichi_backend

B, H, W, C = 1, 16, 16, 3
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _inputs(device: torch.device):
    torch.manual_seed(42)
    shape = (B, H + 2 * PAD, W + 2 * PAD, C)
    source = torch.rand(shape, device=device)
    target = torch.rand(shape, device=device)
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)
    identity_nnf = torch.stack([ys, xs], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    return source, target, identity_nnf.to(torch.int32)


def _reference_ssd(source, nnf, target):
    r = PATCH_SIZE // 2
    err = torch.zeros((B, H, W), device=source.device)
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


@pytest.fixture
def matcher():
    return taichi_backend.PatchMatcherTaichi(H, W, C, PATCH_SIZE, num_iter=1)


def test_identity_nnf_remap_reconstructs_source(matcher):
    source, _, nnf = _inputs(_device())
    # Every patch covering (y, x) votes for source[y, x] under an identity NNF.
    remapped = matcher.unpad_image(matcher.apply_nnf_to_image(nnf, source))
    expected = source[:, PAD:-PAD, PAD:-PAD, :]
    assert (remapped - expected).abs().max().item() <= 1e-5


def test_patch_error_matches_reference_ssd(matcher):
    source, target, nnf = _inputs(_device())
    error = matcher.get_patch_error(source, nnf, target)
    reference = _reference_ssd(source, nnf, target)
    assert (error - reference).abs().max().item() <= 1e-4


def test_estimate_nnf_keeps_coordinates_in_bounds(matcher):
    source, target, nnf = _inputs(_device())
    nnf_new, _ = matcher.estimate_nnf(source, target, source, nnf)
    assert nnf_new.shape == nnf.shape
    assert int(nnf_new[..., 0].min()) >= 0 and int(nnf_new[..., 0].max()) < H
    assert int(nnf_new[..., 1].min()) >= 0 and int(nnf_new[..., 1].max()) < W
