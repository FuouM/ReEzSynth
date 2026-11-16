import torch

from ezsynth import consts
from ezsynth.kernels.synth import taichi_kernels, torch_ops
from ezsynth.kernels.synth.common import random_init_nnf, resample_tensor
from ezsynth.kernels.synth.torch_ops import microprofile
from ezsynth.kernels.synth.torch_ops.device_cache import clear_torch_device_cache


def test_resample_tensor_preserves_uint8_dtype():
    tensor = torch.arange(4 * 5 * 3, dtype=torch.uint8).reshape(4, 5, 3)

    resized = resample_tensor(tensor, 2, 3)

    assert resized.shape == (2, 3, 3)
    assert resized.dtype == torch.uint8
    assert resized.is_contiguous()


def test_resample_tensor_returns_same_object_when_size_matches():
    tensor = torch.zeros((4, 5, 3), dtype=torch.float32)

    resized = resample_tensor(tensor, 4, 5)

    assert resized is tensor


def test_random_init_nnf_shape_dtype_and_bounds():
    nnf = random_init_nnf(
        device="cpu",
        target_h=4,
        target_w=5,
        source_h=8,
        source_w=9,
        patch_size=3,
    )

    assert nnf.shape == (4, 5, 2)
    assert nnf.dtype == torch.int32
    assert nnf.is_contiguous()
    assert int(nnf[..., 0].min()) >= 1
    assert int(nnf[..., 0].max()) < 8
    assert int(nnf[..., 1].min()) >= 1
    assert int(nnf[..., 1].max()) < 7


def test_torch_ops_package_is_lightweight_and_device_cache_accepts_none():
    assert not hasattr(torch_ops, "extract_patches")
    clear_torch_device_cache(None)


def test_consts_expose_extension_and_tuning_controls(monkeypatch):
    from ezsynth.utils import ebsynth_jit_platform as jit

    assert consts.vote_chunk_budget_bytes() >= 8 << 20
    assert hasattr(consts, "FORCE_EBSYNTH_WHEEL")
    assert hasattr(consts, "TORCH_MICROPROFILE")
    assert hasattr(consts, "TAICHI_INIT_VERBOSE")

    # Host flags are portable unless EZSYNTH_NATIVE_ARCH opts in.
    monkeypatch.delenv(jit.NATIVE_ARCH_ENV, raising=False)
    portable = jit.ebsynth_jit_cflags()
    assert not any("native" in f or f.startswith("/arch:") for f in portable)
    std = jit.cxx_standard()
    assert any(f in (f"/std:{std}", f"-std={std}") for f in portable)
    # Never pass arch flags to nvcc: torch picks TORCH_CUDA_ARCH_LIST / local GPU.
    assert not any("arch" in f for f in jit.ebsynth_jit_cuda_cflags())


def test_microprofile_records_regions_when_enabled(monkeypatch):
    monkeypatch.setattr(consts, "TORCH_MICROPROFILE", "1")
    microprofile.reset()

    with microprofile.region("unit-test", torch.device("cpu")):
        pass

    assert microprofile._COUNTS["unit-test"] == 1
    microprofile.reset()
    monkeypatch.setattr(consts, "TORCH_MICROPROFILE", "")


def test_taichi_backend_imports_kernel_module():
    assert callable(taichi_kernels.compute_patch_ssd)
    assert callable(taichi_kernels.random_search_kernel)
