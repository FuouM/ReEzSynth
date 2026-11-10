"""Pure helpers of the ComfyUI node pack (no ComfyUI process needed)."""

import numpy as np
import pytest
import torch

from reezsynth_nodes import NODE_CLASS_MAPPINGS
from reezsynth_nodes.image_io import (
    bgr_uint8_sequence_to_image_tensor,
    bgr_uint8_to_image_tensor,
    error_map_to_image_tensor,
    image_batch_to_bgr_uint8,
    image_tensor_to_bgr_uint8,
)
from reezsynth_nodes.nodes_video_synth import make_style_keyframes
from reezsynth_nodes.options import (
    check_indices_in_range,
    parse_indices,
    parse_landmarks,
    validate_synthesis_options,
)


def _random_bgr(shape=(5, 7, 3), seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_bgr_roundtrip_is_exact():
    bgr = _random_bgr()
    tensor = bgr_uint8_to_image_tensor(bgr)
    assert tensor.shape == (1, 5, 7, 3)
    assert tensor.dtype == torch.float32
    assert np.array_equal(image_tensor_to_bgr_uint8(tensor), bgr)


def test_image_tensor_channel_order_and_clamp():
    rgb = torch.tensor([[[[1.0, 0.5, -0.2]], [[2.0, 0.0, 0.25]]]])  # [1,2,1,3]
    bgr = image_tensor_to_bgr_uint8(rgb)
    assert bgr.tolist() == [[[0, 128, 255]], [[64, 0, 255]]]


def test_batch_conversion_drops_alpha_and_expands_gray():
    rgba = torch.rand(3, 4, 6, 4)
    out = image_batch_to_bgr_uint8(rgba)
    assert out.shape == (3, 4, 6, 3) and out.flags.c_contiguous
    expected = (rgba[..., [2, 1, 0]] * 255).round().to(torch.uint8).numpy()
    assert np.array_equal(out, expected)
    gray = image_batch_to_bgr_uint8(torch.full((1, 2, 2, 1), 1.0))
    assert np.all(gray == 255) and gray.shape == (1, 2, 2, 3)


def test_single_image_rejects_batches():
    with pytest.raises(ValueError, match="batch size 1"):
        image_tensor_to_bgr_uint8(torch.zeros(2, 4, 4, 3))
    with pytest.raises(ValueError, match=r"\[B,H,W,C\]"):
        image_tensor_to_bgr_uint8(torch.zeros(4, 4, 3))


def test_sequence_to_tensor_checks_shapes():
    frames = [_random_bgr(seed=i) for i in range(3)]
    batch = bgr_uint8_sequence_to_image_tensor(frames)
    assert batch.shape == (3, 5, 7, 3)
    assert np.array_equal(image_batch_to_bgr_uint8(batch), np.stack(frames))
    with pytest.raises(ValueError, match="share one shape"):
        bgr_uint8_sequence_to_image_tensor([frames[0], _random_bgr((6, 7, 3))])


def test_error_map_normalized_gray():
    err = np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    vis = error_map_to_image_tensor(err)
    assert vis.shape == (1, 2, 2, 3)
    assert torch.allclose(vis[0, :, :, 0], torch.tensor([[0.0, 0.5], [0.25, 1.0]]))
    assert torch.equal(vis[..., 0], vis[..., 2])
    assert torch.count_nonzero(error_map_to_image_tensor(np.zeros((2, 2, 1)))) == 0


@pytest.mark.parametrize(
    ("backend", "cost", "mode", "ok"),
    [
        ("torch", "sad", "multicand", True),
        ("taichi", "ncc", "jfa", True),
        ("cuda", "ncc", "patchmatch", True),
        ("cuda", "sad", "patchmatch", False),
        ("cuda", "ssd", "jfa", False),
        ("metal", "ssd", "patchmatch", False),
    ],
)
def test_backend_capabilities(backend, cost, mode, ok):
    assert (validate_synthesis_options(backend, cost, mode) is None) is ok


def test_validation_messages_name_the_problem():
    assert "search_mode='multicand'" in validate_synthesis_options(
        "cuda", "ssd", "multicand"
    )
    assert "precision" in validate_synthesis_options(
        "torch", "ssd", "patchmatch", "int8"
    )
    assert "multicand_k" in validate_synthesis_options(
        "torch", "ssd", "multicand", "float32", 9
    )


def test_parse_indices():
    assert parse_indices("0, 24,48\n 60") == [0, 24, 48, 60]
    for bad in ("", "a, 1", "-1", "3, 3"):
        with pytest.raises(ValueError):
            parse_indices(bad)
    with pytest.raises(ValueError, match="out of range"):
        check_indices_in_range([0, 10], 10, "style_indices")


def test_parse_landmarks_formats():
    pts = [(i, 2 * i) for i in range(68)]
    txt = "68\n" + "\n".join(f"{x} {y}" for x, y in pts)
    flat = ",".join(f"{x}.4,{y}" for x, y in pts)
    assert parse_landmarks(txt, "lm") == pts
    assert parse_landmarks(flat, "lm") == pts
    assert parse_landmarks("  \n", "lm") is None
    with pytest.raises(ValueError, match="68"):
        parse_landmarks("1 2\n3 4", "lm")


def test_style_keyframes_sorted_and_validated():
    a, b = _random_bgr(seed=1), _random_bgr(seed=2)
    styles = make_style_keyframes([a, b], [30, 0])
    assert styles.indices == (0, 30)
    assert styles.frames[0] is b
    with pytest.raises(ValueError, match="unique"):
        make_style_keyframes([a, b], [3, 3])


def test_every_input_has_a_tooltip_and_category():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY.startswith("ReEzSynth/"), name
        assert len(cls.OUTPUT_TOOLTIPS) == len(cls.RETURN_TYPES), name
        for section in cls.INPUT_TYPES().values():
            for input_name, spec in section.items():
                assert len(spec) == 2 and spec[1].get("tooltip"), (name, input_name)
