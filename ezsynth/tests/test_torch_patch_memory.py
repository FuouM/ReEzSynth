import pytest
import torch

from ezsynth.config import EbsynthParamsConfig, PipelineConfig
from ezsynth.consts import (
    COST_FUNCTION_NCC,
    COST_FUNCTION_SAD,
    COST_FUNCTION_SSD,
    EBSYNTH_VOTEMODE_WEIGHTED,
)
from ezsynth.engines.backends.pytorch_backend import PyTorchBackend
from ezsynth.torch_ops import omega_ops
from ezsynth.torch_ops.patch_memory import (
    BUDGET_ENV,
    STRATEGY_ENV,
    LevelShape,
    bounded_bytes_per_pixel,
    estimate_materialized_bytes,
    plan_level,
)
from ezsynth.torch_ops.patch_ops import ImagePatches, extract_patches
from ezsynth.utils.gpu_memory import apply_gpu_memory_limit

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PATCH_SIZE = 5


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        apply_gpu_memory_limit(device)
    return device


def _omega_reference(nnf: torch.Tensor, h_s: int, w_s: int) -> torch.Tensor:
    omega = torch.zeros((h_s, w_s), dtype=torch.int32)
    r = PATCH_SIZE // 2
    for x, y in nnf.reshape(-1, 2).tolist():
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                omega[min(max(y + dy, 0), h_s - 1), min(max(x + dx, 0), w_s - 1)] += 1
    return omega


@pytest.mark.parametrize("device_name", DEVICES)
def test_omega_ops_chunked_counts_are_exact(monkeypatch, device_name):
    device = _device(device_name)
    # Three centers per chunk, so populate/update span several chunks.
    monkeypatch.setattr(
        omega_ops, "vote_chunk_budget_bytes", lambda: 3 * PATCH_SIZE**2 * 16
    )
    gen = torch.Generator().manual_seed(5)
    h_s, w_s = 9, 11
    nnf = torch.stack(
        (
            torch.randint(-1, w_s + 1, (4, 5), generator=gen),
            torch.randint(-1, h_s + 1, (4, 5), generator=gen),
        ),
        dim=-1,
    ).to(torch.int32)
    omega = omega_ops.populate_omega_map(nnf.to(device), (h_s, w_s), PATCH_SIZE)
    assert torch.equal(omega.cpu(), _omega_reference(nnf, h_s, w_s))

    new_nnf = nnf.clone()
    new_nnf[1:3] = torch.flip(nnf[1:3], dims=(1,))
    old_coords = nnf[1:3].reshape(-1, 2).to(device)
    new_coords = new_nnf[1:3].reshape(-1, 2).to(device)
    omega_ops.update_omega_map(omega, old_coords, new_coords, PATCH_SIZE)
    assert torch.equal(omega.cpu(), _omega_reference(new_nnf, h_s, w_s))


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_image_patches_match_unfold_grid(device_name, dtype):
    device = _device(device_name)
    gen = torch.Generator().manual_seed(3)
    image = torch.randint(0, 256, (9, 11, 4), generator=gen).to(dtype).to(device)
    grid = extract_patches(image, PATCH_SIZE, as_float=True)
    view = ImagePatches(image, PATCH_SIZE, chunk_pixels=7)
    assert view.shape == grid.shape
    assert view.numel() == grid.numel()
    ys = torch.tensor([0, 0, 4, 8, 8, 3], device=device)
    xs = torch.tensor([0, 10, 5, 0, 10, 1], device=device)
    assert torch.equal(view.gather(ys, xs), grid[ys, xs])


def test_plan_level_switches_on_budget(monkeypatch):
    shape = LevelShape((64, 64), (64, 64), 3, 12, 0, 7)
    estimate = estimate_materialized_bytes(shape, iterative=True)
    monkeypatch.delenv(STRATEGY_ENV, raising=False)

    monkeypatch.setenv(BUDGET_ENV, str(estimate * 2 / (1 << 20)))
    assert plan_level(shape, "cpu", iterative=True).chunk_pixels is None

    monkeypatch.setenv(BUDGET_ENV, str(estimate / 2 / (1 << 20)))
    plan = plan_level(shape, "cpu", iterative=True)
    assert plan.strategy == "bounded"
    expected = int(estimate / 2 * 0.5) // bounded_bytes_per_pixel(shape)
    assert abs(plan.chunk_pixels - expected) <= 1

    monkeypatch.setenv(STRATEGY_ENV, "materialized")
    assert plan_level(shape, "cpu", iterative=True).chunk_pixels is None
    monkeypatch.delenv(BUDGET_ENV)
    monkeypatch.setenv(STRATEGY_ENV, "bounded")
    assert plan_level(shape, "cpu", iterative=True).chunk_pixels >= 1
    monkeypatch.setenv(STRATEGY_ENV, "bogus")
    with pytest.raises(ValueError):
        plan_level(shape, "cpu", iterative=True)


def test_plan_level_rejects_alternatives_over_budget(monkeypatch):
    # 7 alternatives x 4096 pixels need ~1 MiB of candidate state alone.
    oversize = LevelShape((64, 64), (64, 64), 3, 12, 0, 7, candidate_slots=7)
    monkeypatch.delenv(STRATEGY_ENV, raising=False)
    monkeypatch.setenv(BUDGET_ENV, "0.5")
    with pytest.raises(MemoryError, match="multicand_k"):
        plan_level(oversize, "cpu", iterative=True)

    # The same level without alternatives fits in bounded chunks.
    plain = LevelShape((64, 64), (64, 64), 3, 12, 0, 7)
    assert plan_level(plain, "cpu", iterative=True).strategy == "bounded"


def _inputs(device: torch.device, with_modulation: bool):
    gen = torch.Generator().manual_seed(11)
    h_s, w_s, h_t, w_t = 12, 14, 10, 12
    r = PATCH_SIZE // 2

    def img(h, w, c):
        return torch.randint(0, 256, (h, w, c), generator=gen, dtype=torch.uint8)

    style = img(h_s, w_s, 3)
    source_guide = img(h_s, w_s, 6)
    target_guide = img(h_t, w_t, 6)
    modulation = (
        img(h_t, w_t, 6) if with_modulation else torch.empty(0, dtype=torch.uint8)
    )
    nnf = torch.stack(
        (
            torch.randint(r, w_s - r, (h_t, w_t), generator=gen),
            torch.randint(r, h_s - r, (h_t, w_t), generator=gen),
        ),
        dim=-1,
    ).to(torch.int32)
    style_weights = torch.rand(3, generator=gen) + 0.5
    guide_weights = torch.rand(6, generator=gen) + 0.5
    tensors = (
        style,
        source_guide,
        target_guide,
        modulation,
        nnf,
        style_weights,
        guide_weights,
    )
    return tuple(t.to(device) for t in tensors)


def _run(device, residual, cost, bilateral, inputs, **params):
    backend = PyTorchBackend(
        EbsynthParamsConfig(
            use_bilateral=bilateral, search_pruning_threshold=40.0, **params
        ),
        PipelineConfig(use_residual_transfer=residual),
    )
    backend.device = device
    style, sg, tg, mod, nnf, sw, gw = inputs
    return backend.run_level(
        style,
        sg,
        tg,
        mod,
        nnf.clone(),
        sw,
        gw,
        uniformity_weight=3500.0,
        patch_size=PATCH_SIZE,
        vote_mode=EBSYNTH_VOTEMODE_WEIGHTED,
        search_vote_iters=2,
        patch_match_iters=2,
        stop_threshold=5.0,
        rand_states=None,
        cost_function_mode=cost,
    )


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
@pytest.mark.parametrize(
    ("cost", "bilateral", "with_modulation"),
    [
        (COST_FUNCTION_SSD, False, False),
        (COST_FUNCTION_SSD, False, True),
        (COST_FUNCTION_SSD, True, True),
        (COST_FUNCTION_NCC, False, False),
        (COST_FUNCTION_NCC, False, True),
    ],
    ids=["ssd", "ssd-mod", "ssd-bilateral-mod", "ncc", "ncc-mod"],
)
def test_bounded_strategy_matches_materialized(
    monkeypatch, device_name, residual, cost, bilateral, with_modulation
):
    device = _device(device_name)
    inputs = _inputs(device, with_modulation)

    monkeypatch.setenv(STRATEGY_ENV, "materialized")
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    ref_image, ref_error, ref_nnf = _run(device, residual, cost, bilateral, inputs)

    # A ~200 KiB budget gives chunks of ~17 pixels, so every cost evaluation over
    # the 10x12 target spans several chunks.
    monkeypatch.setenv(STRATEGY_ENV, "bounded")
    monkeypatch.setenv(BUDGET_ENV, "0.2")
    image, error, nnf = _run(device, residual, cost, bilateral, inputs)

    assert torch.equal(nnf, ref_nnf)
    image_diff = (image.int() - ref_image.int()).abs()
    if device.type == "cpu":
        assert int(image_diff.max()) == 0
        torch.testing.assert_close(error, ref_error, rtol=1e-5, atol=1e-3)
    else:
        # CUDA reductions over (1, chunk, D) and (H, W, D) layouts may round the
        # last float bit differently; weighted votes then truncate to uint8, and the
        # residual variant re-scores its errors against that voted image.
        assert int(image_diff.max()) <= 1
        assert float((image_diff > 0).float().mean()) < 0.05
        torch.testing.assert_close(error, ref_error, rtol=5e-3, atol=1e-3)


@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
@pytest.mark.parametrize(
    ("search_mode", "multicand_k", "cost", "with_modulation"),
    [
        ("jfa", 3, COST_FUNCTION_SSD, False),
        ("jfa", 3, COST_FUNCTION_SAD, True),
        ("multicand", 2, COST_FUNCTION_SSD, False),
        ("multicand", 3, COST_FUNCTION_SSD, True),
        ("multicand", 3, COST_FUNCTION_SAD, False),
    ],
    ids=["jfa", "jfa-sad-mod", "multicand2", "multicand3-mod", "multicand3-sad"],
)
def test_bounded_strategy_matches_materialized_search_modes(
    monkeypatch, residual, search_mode, multicand_k, cost, with_modulation
):
    device = torch.device("cpu")
    inputs = _inputs(device, with_modulation)
    params = {"search_mode": search_mode, "multicand_k": multicand_k}

    monkeypatch.setenv(STRATEGY_ENV, "materialized")
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    ref = _run(device, residual, cost, False, inputs, **params)

    monkeypatch.setenv(STRATEGY_ENV, "bounded")
    monkeypatch.setenv(BUDGET_ENV, "0.2")
    image, error, nnf = _run(device, residual, cost, False, inputs, **params)

    assert torch.equal(nnf, ref[2])
    assert torch.equal(image, ref[0])
    assert torch.equal(error, ref[1])
