from dataclasses import dataclass
from typing import Union

import numpy as np

from .utils import io_utils


@dataclass
class GuideObject:
    keyframe: np.ndarray
    target: np.ndarray
    weight: float

    def __post_init__(self) -> None:
        _validate_guide_array(self.keyframe, "keyframe")
        _validate_guide_array(self.target, "target")
        if self.keyframe.shape[2] != self.target.shape[2]:
            raise ValueError(
                "Guide keyframe and target must have the same channel count."
            )

    @staticmethod
    def load_guide(
        source: Union[str, np.ndarray],
        target: Union[str, np.ndarray],
        weight: float = 1.0,
    ) -> "GuideObject":
        src_img = io_utils.read_image(source) if isinstance(source, str) else source
        tgt_img = io_utils.read_image(target) if isinstance(target, str) else target
        return GuideObject(keyframe=src_img, target=tgt_img, weight=weight)


def _validate_guide_array(array: np.ndarray, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Guide {name} must be a NumPy array.")
    if array.ndim != 3:
        raise ValueError(f"Guide {name} must be an HWC image, got {array.shape}.")
    if array.shape[2] <= 0:
        raise ValueError(f"Guide {name} must have at least one channel.")
    if array.dtype != np.uint8:
        raise ValueError(f"Guide {name} must be uint8, got {array.dtype}.")
