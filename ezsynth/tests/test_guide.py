import numpy as np
import pytest

from ezsynth.engines.synthesis_engine import PreparedSynthesisContext
from ezsynth.guide import GuideObject


class _FakeContextEngine:
    device = "cpu"

    def _determine_num_pyramid_levels(self, sh, sw, th, tw):
        return 1

    def _timed_operation(self, operation_name, operation_func):
        return operation_func()


def test_guide_object_validates_hwc_uint8_channels():
    good = np.zeros((4, 5, 3), dtype=np.uint8)
    GuideObject(good, good.copy(), 1.0)

    with pytest.raises(ValueError, match="HWC"):
        GuideObject(np.zeros((4, 5), dtype=np.uint8), good, 1.0)

    with pytest.raises(ValueError, match="channel"):
        GuideObject(good, np.zeros((4, 5, 1), dtype=np.uint8), 1.0)

    with pytest.raises(ValueError, match="uint8"):
        GuideObject(good.astype(np.float32), good, 1.0)


def test_prepared_context_validates_target_shape_stability():
    style = np.zeros((4, 5, 3), dtype=np.uint8)
    source = np.zeros((4, 5, 3), dtype=np.uint8)
    target = np.zeros((4, 5, 3), dtype=np.uint8)
    context = PreparedSynthesisContext(
        _FakeContextEngine(),
        style,
        [GuideObject(source, target, 1.0)],
    )

    with pytest.raises(ValueError, match="target shape changed"):
        context.run_frame(
            [GuideObject(source, np.zeros((5, 4, 3), dtype=np.uint8), 1.0)]
        )
