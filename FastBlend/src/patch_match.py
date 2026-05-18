"""Public patch-match API: availability flags, builders, and interpolation wrapper."""

from __future__ import annotations

from typing import Any

from .backends.fb_availability import cupy_available, taichi_available
from .backends.patch_matcher_dispatch import (
    build_patch_matcher,
    build_pyramid_patch_matcher,
)
from .backends.resolve_backend import resolve_fastblend_backend


def __getattr__(name: str) -> Any:
    """Lazy imports so optional backends (e.g. CuPy) are not loaded unless accessed."""
    if name in ("PatchMatcherCUDA", "PyramidPatchMatcherCUDA"):
        from .backends import cuda_patch_match as m

        return getattr(m, name)
    if name in ("PatchMatcherCupy", "PyramidPatchMatcherCupy"):
        from .backends import cupy_patch_match as m

        return getattr(m, name)
    if name in (
        "PatchMatcherTaichi",
        "PyramidPatchMatcherTaichi",
        "FastBlendTaichiBackend",
    ):
        from .backends import taichi_backend as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "cupy_available",
    "taichi_available",
    "resolve_fastblend_backend",
    "build_patch_matcher",
    "build_pyramid_patch_matcher",
    "PyramidPatchMatcher",
]


class PyramidPatchMatcher:
    """
    Pyramid patch matcher for keyframe interpolation (stable third-party shape).

    Dispatches to CUDA, CuPy, or Taichi pyramid engines based on ``backend``.
    """

    def __init__(
        self,
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block=8,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity",
        backend="auto",
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.initialize = initialize
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        resolved = resolve_fastblend_backend(backend)
        self.patch_matcher = build_pyramid_patch_matcher(
            image_height,
            image_width,
            channel,
            minimum_patch_size,
            threads_per_block=threads_per_block,
            num_iter=num_iter,
            gpu_id=gpu_id,
            guide_weight=guide_weight,
            use_mean_target_style=use_mean_target_style,
            use_pairwise_patch_error=use_pairwise_patch_error,
            tracking_window_size=tracking_window_size,
            initialize=initialize,
            backend=resolved,
        )
        self.backend = resolved

    def estimate_nnf(self, source_guide, target_guide, source_style):
        return self.patch_matcher.estimate_nnf(source_guide, target_guide, source_style)
