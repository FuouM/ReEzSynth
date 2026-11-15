"""Interpolation wrapper over the FastBlend engines of ``ezsynth.kernels.fastblend``."""

from __future__ import annotations

from ezsynth.kernels.fastblend import build_pyramid_matcher, resolve


class PyramidPatchMatcher:
    """
    Pyramid patch matcher for keyframe interpolation (stable third-party shape).

    Dispatches to the pyramid matcher of the resolved ``ezsynth.kernels.fastblend`` engine.
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

        resolved = resolve(backend)
        self.patch_matcher = build_pyramid_matcher(
            resolved,
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
        )
        self.backend = resolved

    def estimate_nnf(self, source_guide, target_guide, source_style):
        return self.patch_matcher.estimate_nnf(source_guide, target_guide, source_style)
