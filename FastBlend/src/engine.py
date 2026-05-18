"""FastBlend entry points in the same shape as ``FaceBlit/src/engine.py``."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .balanced_runner import BalancedModeRunner
from .config import FastBlendConfig
from .interpolation_runner import InterpolationModeRunner


class FastBlendInput_Sequence:
    """Per-sequence inputs: content guides and stylized frames (uint8 ``HWC``)."""

    def __init__(
        self,
        frames_guide: List[np.ndarray],
        frames_style: List[np.ndarray],
    ) -> None:
        self.frames_guide = frames_guide
        self.frames_style = frames_style


class FastBlendInput_Keyframes:
    """Keyframe interpolation: full guide sequence, stylized keyframes, and indices."""

    def __init__(
        self,
        frames_guide: List[np.ndarray],
        keyframes_style: List[np.ndarray],
        keyframe_indices: List[int],
    ) -> None:
        self.frames_guide = frames_guide
        self.keyframes_style = keyframes_style
        self.keyframe_indices = keyframe_indices


class FastBlendEngine:
    def __init__(self) -> None:
        pass

    def smooth_sequence(
        self,
        input_sequence: FastBlendInput_Sequence,
        config: Optional[FastBlendConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        backend: str = "auto",
    ) -> List[np.ndarray]:
        cfg = config or FastBlendConfig()
        if not cfg.enabled:
            return input_sequence.frames_style

        print(f"Starting FastBlend (Accuracy={cfg.accuracy})...")

        actual_backend = cfg.backend if backend == "auto" else backend
        patch_matcher_config = cfg.get_pyramid_patch_matcher_config()

        frames_guide_float = [f.astype(np.float32) for f in input_sequence.frames_guide]
        frames_style_float = [f.astype(np.float32) for f in input_sequence.frames_style]

        balanced_runner = BalancedModeRunner()
        result_frames = balanced_runner.run(
            frames_guide_float,
            frames_style_float,
            batch_size=cfg.batch_size,
            window_size=cfg.window_size,
            patch_matcher_config=patch_matcher_config,
            desc="FastBlend Processing",
            progress_callback=progress_callback,
            backend=actual_backend,
        )

        final_frames = [
            np.clip(frame, 0, 255).astype(np.uint8) for frame in result_frames
        ]
        print("FastBlend complete.")
        return final_frames

    def interpolate_keyframes(
        self,
        input_keyframes: FastBlendInput_Keyframes,
        config: FastBlendConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[np.ndarray]:
        frames_guide = input_keyframes.frames_guide
        keyframes_style = input_keyframes.keyframes_style
        keyframe_indices = input_keyframes.keyframe_indices

        guide_float = [f.astype(np.float32) for f in frames_guide]
        keyframes_float = [f.astype(np.float32) for f in keyframes_style]

        interpolation_runner = InterpolationModeRunner()
        patch_matcher_config = config.get_pyramid_patch_matcher_config()

        result_float = interpolation_runner.run(
            guide_float,
            keyframes_float,
            keyframe_indices,
            batch_size=config.batch_size,
            patch_matcher_config=patch_matcher_config,
            progress_callback=progress_callback,
        )

        result_frames_u8: List[Optional[np.ndarray]] = [
            np.clip(f, 0, 255).astype(np.uint8) if f is not None else None
            for f in result_float
        ]

        for i in range(len(result_frames_u8)):
            if result_frames_u8[i] is None:
                distances = [(abs(i - k), k) for k in keyframe_indices]
                nearest_kf_idx = min(distances)[1]
                kf_style_idx = keyframe_indices.index(nearest_kf_idx)
                result_frames_u8[i] = keyframes_style[kf_style_idx]

        return result_frames_u8  # type: ignore[return-value]
