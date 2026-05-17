import torch

from ezsynth.engines.backends.common import random_init_nnf, resample_tensor


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
