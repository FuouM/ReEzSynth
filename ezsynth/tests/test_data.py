import numpy as np
import pytest

from ezsynth.utils.image_utils import assert_uniform_image_shapes


def test_assert_uniform_image_shapes_accepts_matching_shapes():
    frames = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3, 3), dtype=np.uint8),
    ]

    assert_uniform_image_shapes(frames)


def test_assert_uniform_image_shapes_rejects_mismatched_shapes():
    frames = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((3, 2, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="resolution mismatch"):
        assert_uniform_image_shapes(frames)
