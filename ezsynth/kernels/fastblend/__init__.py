"""FastBlend patch matching (FaceBlit-style PatchMatch over an image pyramid).

One module per engine; each defines ``PyramidPatchMatcher`` with the same constructor
(``image_height, image_width, channel, minimum_patch_size, threads_per_block=16,
num_iter=5, gpu_id=0, guide_weight=10.0, use_mean_target_style=False,
use_pairwise_patch_error=False, tracking_window_size=0, initialize="identity"``) and
``estimate_nnf(source_guide, target_guide, source_style) -> (nnf, target_style)``,
plus ``staging_device(gpu_id)``: the torch device for frames handed to it.

| engine   | kernels                                               | ``PatchMatcher`` |
|----------|-------------------------------------------------------|------------------|
| ``cuda`` | ``cuda_src/kernels.cu`` (JIT-built PyTorch extension) | yes              |
| ``cupy`` | ``cupy_kernels.py`` (RawKernels of the same source)   | yes              |
| ``taichi`` | ``taichi_kernels.py``                               | yes              |
| ``wgpu`` | ``rust/ebsynth-gpu/src/fastblend.rs``, ``shaders/fastblend_*.wgsl`` | -  |

``cuda``, ``cupy`` and ``taichi`` share the PyTorch propagation, random search,
tracking and pyramid loop of ``core.py`` and only supply the ``remap``,
``patch_error`` and ``pairwise_patch_error`` kernels; ``PatchMatcher`` is their
single-level matcher. ``wgpu`` runs the whole pyramid in Rust.
"""

from __future__ import annotations

import platform
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Literal, get_args

FastBlendEngine = Literal["cuda", "cupy", "taichi", "wgpu"]
ENGINES: tuple[str, ...] = get_args(FastBlendEngine)


def _installed(module: str) -> bool:
    return find_spec(module) is not None


def resolve(requested: str) -> FastBlendEngine:
    """Engine for ``requested`` (an engine name or ``auto``).

    An explicit engine must be usable or ImportError is raised. ``auto`` prefers Taichi
    on Apple Silicon, then the CUDA extension (JIT-built here when a CUDA device
    exists), then wgpu, Taichi and CuPy.
    """
    req = (requested or "auto").lower().strip()
    if req != "auto" and req not in ENGINES:
        raise ValueError(
            f"Unknown FastBlend backend {requested!r}; expected auto or one of {ENGINES}"
        )
    if req == "wgpu":
        if not _installed("ezsynth_wgpu"):
            from ...engines.backends.wgpu_backend import INSTALL_HINT

            raise ImportError(INSTALL_HINT)
        return "wgpu"
    if req == "cupy":
        if not _installed("cupy"):
            raise ImportError("CuPy backend requested but CuPy is not installed")
        return "cupy"
    if req == "taichi":
        if not _installed("taichi"):
            raise ImportError("Taichi backend requested but Taichi is not installed")
        return "taichi"
    if req == "cuda":
        if not load("cuda").is_available():
            raise ImportError(
                "CUDA backend requested but the FastBlend CUDA extension is not available"
            )
        return "cuda"
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and _installed("taichi")
    ):
        return "taichi"
    if load("cuda").is_available():
        return "cuda"
    if _installed("ezsynth_wgpu"):
        return "wgpu"
    if _installed("taichi"):
        return "taichi"
    if _installed("cupy"):
        return "cupy"
    raise ImportError("No FastBlend backend available")


def load(engine: str) -> ModuleType:
    """Import the module of FastBlend ``engine``."""
    if engine not in ENGINES:
        raise ValueError(
            f"Unknown FastBlend engine {engine!r}; expected one of {ENGINES}"
        )
    return import_module(f"{__name__}.{engine}")


def build_pyramid_matcher(engine: str, *args, **kwargs):
    """``PyramidPatchMatcher`` of ``engine``; arguments as in the module docstring."""
    return load(engine).PyramidPatchMatcher(*args, **kwargs)


def build_matcher(engine: str, *args, **kwargs):
    """Single-level ``PatchMatcher`` of ``engine`` (``core.FaceBlitPatchMatcherCore``)."""
    module = load(engine)
    if not hasattr(module, "PatchMatcher"):
        raise ValueError(f"The {engine!r} FastBlend engine has no single-level matcher")
    return module.PatchMatcher(*args, **kwargs)
