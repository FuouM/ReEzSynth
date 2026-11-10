"""SAD cost, JFA / multicand search, patch precision and the ImageSynth API surface."""

import numpy as np
import pytest
import torch

from ezsynth.api import ImageSynth, RunConfig
from ezsynth.config import EbsynthParamsConfig, PipelineConfig
from ezsynth.consts import COST_FUNCTION_SAD
from ezsynth.engines.backends import pytorch_backend
from ezsynth.engines.backends.cuda_backend import validate_native_options
from ezsynth.engines.synthesis_engine import EbsynthEngine
from ezsynth.guide import GuideObject, load_guide
from ezsynth.torch_ops.patch_ops import compute_patch_ssd_vectorized, extract_patches
from ezsynth.utils import io_utils
from ezsynth.utils.progress import progress_hook

try:
    import taichi  # noqa: F401

    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

needs_taichi = pytest.mark.skipif(not HAS_TAICHI, reason="taichi not installed")

PATCH_SIZE = 5
SEARCH_MODES = [("jfa", 3), ("multicand", 2), ("multicand", 3)]


@pytest.fixture(autouse=True)
def _torch_backend_on_cpu(monkeypatch):
    # Exact-equality assertions below hold for CPU float math.
    monkeypatch.setattr(pytorch_backend, "get_auto_torch_device", lambda: "cpu")


def _images(seed: int = 7):
    rng = np.random.default_rng(seed)
    style = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
    source = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
    target = rng.integers(0, 256, (36, 44, 3), dtype=np.uint8)
    return style, source, target


def _config(**overrides) -> RunConfig:
    params = {
        "backend": "torch",
        "patch_size": PATCH_SIZE,
        "pyramid_levels": 2,
        "search_vote_iters": 2,
        "patch_match_iters": 2,
    }
    params.update(overrides)
    return RunConfig(**params)


def _synth(guides=None, **overrides):
    style, source, target = _images()
    if guides is None:
        guides = [(source, target, 1.0)]
    return ImageSynth(style, _config(**overrides)).run(guides)


# --- SAD cost -------------------------------------------------------------------


@pytest.mark.parametrize("with_modulation", [False, True], ids=["plain", "mod"])
def test_torch_sad_matches_numpy_reference(with_modulation):
    gen = torch.Generator().manual_seed(0)
    h_s, w_s, h_t, w_t, channels = 9, 11, 7, 8, 3
    source = torch.randint(0, 256, (h_s, w_s, channels), generator=gen).to(torch.uint8)
    target = torch.randint(0, 256, (h_t, w_t, channels), generator=gen).to(torch.uint8)
    modulation = torch.randint(0, 256, (h_t, w_t, channels), generator=gen).to(
        torch.uint8
    )
    nnf = torch.stack(
        (
            torch.randint(0, w_s, (h_t, w_t), generator=gen),
            torch.randint(0, h_s, (h_t, w_t), generator=gen),
        ),
        dim=-1,
    ).to(torch.int32)
    weights = torch.rand(channels, generator=gen) + 0.5

    source_patches = extract_patches(source, PATCH_SIZE, as_float=True)
    target_patches = extract_patches(target, PATCH_SIZE, as_float=True)
    modulation_patches = (
        extract_patches(modulation, PATCH_SIZE, as_float=True)
        if with_modulation
        else None
    )
    sad = compute_patch_ssd_vectorized(
        source_patches, target_patches, nnf, weights, modulation_patches, l1=True
    )
    ssd = compute_patch_ssd_vectorized(
        source_patches, target_patches, nnf, weights, modulation_patches
    )

    nnf_np = nnf.numpy()
    matched = source_patches.double().numpy()[nnf_np[..., 1], nnf_np[..., 0]]
    diff = (matched - target_patches.double().numpy()).reshape(
        h_t, w_t, channels, PATCH_SIZE * PATCH_SIZE
    )
    scale = np.ones_like(diff)
    if with_modulation:
        scale = modulation_patches.double().numpy().reshape(diff.shape) / 255.0
    w = weights.double().numpy()
    sad_ref = ((np.abs(diff) * scale).sum(-1) * w).sum(-1)
    ssd_ref = ((diff * diff * scale).sum(-1) * w).sum(-1)

    np.testing.assert_allclose(sad.numpy(), sad_ref, rtol=1e-5)
    np.testing.assert_allclose(ssd.numpy(), ssd_ref, rtol=1e-5)


@needs_taichi
@pytest.mark.parametrize("with_modulation", [False, True], ids=["plain", "mod"])
def test_taichi_sad_error_map_matches_reference(with_modulation):
    from ezsynth.engines.backends import taichi_kernels as tk
    from ezsynth.utils.taichi_runtime import ensure_ti_init, get_taichi_torch_device

    ensure_ti_init()
    device = get_taichi_torch_device()
    rng = np.random.default_rng(1)
    h_s, w_s, h_t, w_t = 9, 11, 7, 8
    nsc, ngc = 3, 2

    def img(h, w, c):
        return rng.integers(0, 256, (h, w, c), dtype=np.uint8)

    s_style, t_style = img(h_s, w_s, nsc), img(h_t, w_t, nsc)
    s_guide, t_guide = img(h_s, w_s, ngc), img(h_t, w_t, ngc)
    modulation = img(h_t, w_t, ngc)
    nnf = np.stack(
        (rng.integers(0, w_s, (h_t, w_t)), rng.integers(0, h_s, (h_t, w_t))), axis=-1
    ).astype(np.int32)
    style_w = (rng.random(nsc) + 0.5).astype(np.float32)
    guide_w = (rng.random(ngc) + 0.5).astype(np.float32)

    def dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device)

    error_map = torch.zeros((h_t, w_t), dtype=torch.float32, device=device)
    s_sat = torch.zeros((h_s, w_s), dtype=torch.float32, device=device)
    t_sat = torch.zeros((h_t, w_t), dtype=torch.float32, device=device)
    tk.compute_error_map_kernel(
        dev(nnf),
        error_map,
        dev(s_style),
        dev(t_style),
        dev(s_guide),
        dev(t_guide),
        dev(modulation),
        1 if with_modulation else 0,
        PATCH_SIZE,
        dev(style_w),
        dev(guide_w),
        COST_FUNCTION_SAD,
        s_sat,
        s_sat.clone(),
        t_sat,
        t_sat.clone(),
        0,
        4.0,
        10.0,
        1,
    )

    # Reference: L1 over the patch with edge-clamped coordinates.
    r = PATCH_SIZE // 2
    ref = np.zeros((h_t, w_t))
    for ty in range(h_t):
        for tx in range(w_t):
            sx, sy = nnf[ty, tx]
            total = 0.0
            for py in range(-r, r + 1):
                for px in range(-r, r + 1):
                    syy, sxx = (
                        min(max(sy + py, 0), h_s - 1),
                        min(max(sx + px, 0), w_s - 1),
                    )
                    tyy, txx = (
                        min(max(ty + py, 0), h_t - 1),
                        min(max(tx + px, 0), w_t - 1),
                    )
                    d_style = np.abs(
                        s_style[syy, sxx].astype(np.float64) - t_style[tyy, txx]
                    )
                    d_guide = np.abs(
                        s_guide[syy, sxx].astype(np.float64) - t_guide[tyy, txx]
                    )
                    if with_modulation:
                        d_guide = d_guide * modulation[tyy, txx] / 255.0
                    total += (d_style * style_w).sum() + (d_guide * guide_w).sum()
            ref[ty, tx] = total

    np.testing.assert_allclose(error_map.cpu().numpy(), ref, rtol=1e-5)


# --- JFA / multicand search ------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["torch", pytest.param("taichi", marks=needs_taichi)]
)
@pytest.mark.parametrize(("search_mode", "multicand_k"), SEARCH_MODES)
def test_search_modes_produce_valid_output(backend, search_mode, multicand_k):
    _, _, target = _images()
    image, error = _synth(
        backend=backend, search_mode=search_mode, multicand_k=multicand_k
    )
    assert image.dtype == np.uint8
    assert image.shape == target.shape
    assert error.shape[:2] == target.shape[:2]
    assert np.isfinite(error).all()


@pytest.mark.parametrize(
    ("search_mode", "multicand_k"), [("patchmatch", 3), *SEARCH_MODES]
)
def test_torch_search_modes_are_deterministic(search_mode, multicand_k):
    first = _synth(search_mode=search_mode, multicand_k=multicand_k)
    second = _synth(search_mode=search_mode, multicand_k=multicand_k)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


# --- precision -------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
@pytest.mark.parametrize("cost_function", ["ssd", "sad"])
def test_low_precision_patches_match_float32_for_uint8_inputs(cost_function, precision):
    ref_image, ref_error = _synth(cost_function=cost_function)
    image, error = _synth(cost_function=cost_function, precision=precision)
    np.testing.assert_array_equal(image, ref_image)
    np.testing.assert_array_equal(error, ref_error)


# --- native backend option validation --------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"cost_function": "sad"},
        {"search_mode": "jfa"},
        {"search_mode": "multicand"},
    ],
    ids=["sad", "jfa", "multicand"],
)
def test_native_backend_rejects_unsupported_options(options):
    config = EbsynthParamsConfig(backend="cuda", **options)
    with pytest.raises(ValueError, match="native 'cuda' backend"):
        validate_native_options(config)
    # The engine rejects them before loading (or JIT-building) the extension.
    with pytest.raises(ValueError, match="native 'cuda' backend"):
        EbsynthEngine(config, PipelineConfig())


@pytest.mark.parametrize("cost_function", ["ssd", "ncc"])
def test_native_backend_accepts_supported_options(cost_function):
    validate_native_options(
        EbsynthParamsConfig(
            backend="cuda", cost_function=cost_function, search_mode="patchmatch"
        )
    )


# --- ImageSynth API ----------------------------------------------------------------


def test_image_synth_accepts_guide_objects_tuples_and_paths(tmp_path):
    _, source, target = _images()
    source_path, target_path = tmp_path / "source.png", tmp_path / "target.png"
    io_utils.write_image(source_path, source)
    io_utils.write_image(target_path, target)

    ref_image, ref_error = _synth([(source, target, 1.0)])
    variants = [
        [GuideObject(source, target, 1.0)],
        [load_guide(source, target)],
        [[source, target, 1.0]],
        [(str(source_path), str(target_path), 1.0)],
    ]
    for guides in variants:
        image, error = _synth(guides)
        np.testing.assert_array_equal(image, ref_image)
        np.testing.assert_array_equal(error, ref_error)


def test_image_synth_rejects_bad_guides():
    style, source, target = _images()
    synth = ImageSynth(style, _config())
    with pytest.raises(TypeError):
        synth.run([object()])
    with pytest.raises(TypeError):
        synth.run([(source, target)])
    with pytest.raises(ValueError):
        synth.run([])


# --- progress hook -----------------------------------------------------------------


def test_progress_hook_reports_every_pyramid_level():
    reports = []
    with progress_hook(lambda *report: reports.append(report)):
        _synth(pyramid_levels=2)
    assert reports == [("level", 1, 2), ("level", 2, 2)]

    # Uninstalled after the context: later runs do not report.
    _synth(pyramid_levels=2)
    assert len(reports) == 2


def test_progress_hook_exception_aborts_run():
    class Interrupted(Exception):
        pass

    reports = []

    def hook(*report):
        reports.append(report)
        raise Interrupted

    with pytest.raises(Interrupted), progress_hook(hook):
        _synth(pyramid_levels=2)
    assert reports == [("level", 1, 2)]


# --- image IO ----------------------------------------------------------------------


def test_io_round_trips_non_ascii_path(tmp_path):
    folder = tmp_path / "\u30b9\u30bf\u30a4\u30eb_\u00fcn\u00efcode"
    folder.mkdir()
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    mask = rng.integers(0, 256, (6, 7), dtype=np.uint8)

    io_utils.write_image(folder / "\u5e27_0001.png", image)
    io_utils.write_image(folder / "m\u00e4ske.png", mask)

    np.testing.assert_array_equal(
        io_utils.read_image(folder / "\u5e27_0001.png"), image
    )
    np.testing.assert_array_equal(io_utils.read_mask(folder / "m\u00e4ske.png"), mask)
