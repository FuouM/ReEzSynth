import numpy as np
import pytest

from ezsynth.guide import GuideObject


def test_guide_object_validates_hwc_uint8_channels():
    good = np.zeros((4, 5, 3), dtype=np.uint8)
    GuideObject(good, good.copy(), 1.0)

    with pytest.raises(ValueError, match="HWC"):
        GuideObject(np.zeros((4, 5), dtype=np.uint8), good, 1.0)

    with pytest.raises(ValueError, match="channel"):
        GuideObject(good, np.zeros((4, 5, 1), dtype=np.uint8), 1.0)

    with pytest.raises(ValueError, match="uint8"):
        GuideObject(good.astype(np.float32), good, 1.0)
