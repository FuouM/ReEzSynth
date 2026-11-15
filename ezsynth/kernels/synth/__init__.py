"""EbSynth level synthesis (PatchMatch + voting on one pyramid level), one module per
backend.

Each backend module defines ``Backend(ebsynth_config, pipeline_config)`` with
``device``, ``timer``, ``enable_benchmarking()`` and ``run_level(...) -> (image,
error, nnf)``; ``cuda`` also has ``init_rand_states()``.

| backend    | kernels                                                     |
|------------|-------------------------------------------------------------|
| ``cuda``   | ``cuda_src/`` (C++/CUDA extension, also CPU), ``cuda_loader.py`` |
| ``torch``  | ``torch_ops/``                                              |
| ``taichi`` | ``taichi_kernels.py``                                       |
| ``wgpu``   | ``rust/ebsynth-gpu/src/ebsynth.rs``, ``shaders/ebsynth.wgsl`` |

``STAGES.md`` maps each stage of the algorithm to its function in every backend.
``common.py``: pyramid resampling and random NNF initialization shared by the engine.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Literal, get_args

SynthBackend = Literal["cuda", "torch", "taichi", "wgpu"]
BACKENDS: tuple[str, ...] = get_args(SynthBackend)


def load(backend: str) -> ModuleType:
    """Import the module of synthesis ``backend``."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Unsupported synthesis backend {backend!r}; expected one of {BACKENDS}"
        )
    return import_module(f"{__name__}.{backend}")


def create(backend: str, ebsynth_config, pipeline_config):
    """``Backend`` instance of synthesis ``backend``."""
    return load(backend).Backend(ebsynth_config, pipeline_config)
