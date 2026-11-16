"""Image operations of the video pipeline, one module per engine.

Every engine module exposes its operations under the same names and signatures and
lists them in ``OPS``:

| op                  | opencv | taichi      | wgpu |
|---------------------|--------|-------------|------|
| ``warp``            | yes    | linear only | yes  |
| ``forward_splat``   | -      | yes         | yes  |
| ``pull_push``       | -      | yes         | yes  |
| ``hist_blend``      | yes    | approximate | yes  |
| ``occlusion_masks`` | yes    | -           | yes  |
| ``poisson_cg``      | -      | yes         | -    |
| ``poisson_lsq``     | yes    | -           | -    |

An engine raises ValueError for an op or mode it lacks; nothing is substituted.

Inputs are contiguous NumPy arrays: float32 images (H, W, C) and flows (H, W, 2) of
the same size, uint8 BGR frames for ``hist_blend``. The callers
(``utils.warp_utils.Warp``, ``utils.blend_logic``, ``utils.occlusion``) do the dtype
conversion, resizing and identity-flow shortcuts.

Two engines are picked per run:

- ``resolve_image_ops``: warping and splatting. OpenCV by default; forward splatting
  and ``use_taichi_ops`` need Taichi, or WebGPU with the ``wgpu`` synthesis backend.
- ``resolve_exact_ops``: ops that must match OpenCV (histogram blending, occlusion
  masks, nearest-neighbour NNF warps). OpenCV, or WebGPU with the ``wgpu`` backend.

Rust/WGSL sources: ``rust/ebsynth-gpu/src/imageops.rs``, ``shaders/imageops_*.wgsl``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Literal, get_args

ImageOpsEngine = Literal["opencv", "taichi", "wgpu"]
ENGINES: tuple[str, ...] = get_args(ImageOpsEngine)


def resolve_image_ops(backend: str, *, gpu_ops: bool = False) -> ImageOpsEngine:
    """Engine for the image ops of a run with synthesis ``backend``.

    ``gpu_ops``: Taichi ops or forward splatting were requested.
    """
    if backend == "wgpu":
        return "wgpu"
    return "taichi" if gpu_ops else "opencv"


def resolve_exact_ops(backend: str) -> Literal["opencv", "wgpu"]:
    """Engine for the ops that must match OpenCV, for synthesis ``backend``."""
    return "wgpu" if backend == "wgpu" else "opencv"


def load(engine: str) -> ModuleType:
    """Import the module of image ops ``engine``; it initializes its runtime."""
    if engine not in ENGINES:
        raise ValueError(
            f"Unknown image ops engine {engine!r}; expected one of {ENGINES}"
        )
    return import_module(f"{__name__}.{engine}")


def load_op(engine: str, op: str):
    """Operation ``op`` of image ops ``engine``; ValueError if the engine lacks it."""
    module = load(engine)
    if op not in module.OPS:
        raise ValueError(f"The {engine!r} image ops engine has no {op!r}")
    return getattr(module, op)
