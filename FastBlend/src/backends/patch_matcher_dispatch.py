"""Backend selection and concrete patch-matcher construction.

This module wires ``resolve_fastblend_backend``-style choices to the actual
CUDA, CuPy, or Taichi classes. There is no abstract factory type—only plain
functions that return the appropriate implementation.
"""

from __future__ import annotations

import platform

from .cuda_extension import is_available
from .cuda_patch_match import PatchMatcherCUDA, PyramidPatchMatcherCUDA
from .fb_availability import taichi_available
from .resolve_backend import _cuda_extension_usable


def _pyramid_args(
    image_height,
    image_width,
    channel,
    minimum_patch_size,
    threads_per_block,
    num_iter,
    gpu_id,
    guide_weight,
    use_mean_target_style,
    use_pairwise_patch_error,
    tracking_window_size,
    initialize,
):
    return (
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
        initialize,
    )


def _flat_args(
    height,
    width,
    channel,
    minimum_patch_size,
    threads_per_block,
    num_iter,
    gpu_id,
    guide_weight,
    random_search_steps,
    random_search_range,
    use_mean_target_style,
    use_pairwise_patch_error,
    tracking_window_size,
):
    return (
        height,
        width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        random_search_steps,
        random_search_range,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
    )


def _make_pyramid_cuda(args):
    if not is_available():
        raise RuntimeError("CUDA extension not available")
    return PyramidPatchMatcherCUDA(*args)


def _make_pyramid_cupy(args):
    from . import cupy_patch_match

    return cupy_patch_match.PyramidPatchMatcherCupy(*args)


def _make_pyramid_taichi(args):
    from . import taichi_backend

    return taichi_backend.PyramidPatchMatcherTaichi(*args)


def _make_flat_cuda(args):
    if not is_available():
        raise RuntimeError("CUDA extension not available")
    return PatchMatcherCUDA(*args)


def _make_flat_cupy(args):
    from . import cupy_patch_match

    return cupy_patch_match.PatchMatcherCupy(*args)


def _make_flat_taichi(args):
    from . import taichi_backend

    return taichi_backend.PatchMatcherTaichi(*args)


def build_pyramid_patch_matcher(
    image_height,
    image_width,
    channel,
    minimum_patch_size,
    threads_per_block=16,
    num_iter=5,
    gpu_id=0,
    guide_weight=10.0,
    use_mean_target_style=False,
    use_pairwise_patch_error=False,
    tracking_window_size=0,
    initialize="identity",
    backend="auto",
):
    """Return a pyramid patch matcher for ``backend`` (``auto`` picks a default).

    If you unpack ``**get_pyramid_patch_matcher_config()`` (or any dict that
    already contains ``minimum_patch_size``), pass all parameters only
    once—e.g. all-keyword from ``FastBlend.src.config.merge_pyramid_patch_matcher_kwargs``.
    """
    args = _pyramid_args(
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
        initialize,
    )
    if backend == "cuda":
        return _make_pyramid_cuda(args)
    if backend == "cupy":
        return _make_pyramid_cupy(args)
    if backend == "taichi":
        return _make_pyramid_taichi(args)
    if backend == "auto":
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and taichi_available
        ):
            return _make_pyramid_taichi(args)
        if _cuda_extension_usable():
            return _make_pyramid_cuda(args)
        return _make_pyramid_cupy(args)
    raise ValueError(f"Unknown backend: {backend}")


def build_patch_matcher(
    height,
    width,
    channel,
    minimum_patch_size,
    threads_per_block=16,
    num_iter=5,
    gpu_id=0,
    guide_weight=10.0,
    random_search_steps=3,
    random_search_range=4,
    use_mean_target_style=False,
    use_pairwise_patch_error=False,
    tracking_window_size=0,
    backend="auto",
):
    """Return a single-scale patch matcher for ``backend``."""
    args = _flat_args(
        height,
        width,
        channel,
        minimum_patch_size,
        threads_per_block,
        num_iter,
        gpu_id,
        guide_weight,
        random_search_steps,
        random_search_range,
        use_mean_target_style,
        use_pairwise_patch_error,
        tracking_window_size,
    )
    if backend == "cuda":
        return _make_flat_cuda(args)
    if backend == "cupy":
        return _make_flat_cupy(args)
    if backend == "taichi":
        return _make_flat_taichi(args)
    if backend == "auto":
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and taichi_available
        ):
            return _make_flat_taichi(args)
        if _cuda_extension_usable():
            return _make_flat_cuda(args)
        return _make_flat_cupy(args)
    raise ValueError(f"Unknown backend: {backend}")
