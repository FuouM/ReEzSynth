# ezsynth/consts.py
"""
Constants and configuration for the ezsynth library.

All tunables live here as module-level values. ``ezsynth.cli.video`` copies
``RUNPY_STARTUP_ENV`` into ``os.environ`` before importing the project.
"""

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


# --- Ebsynth Vote Mode Constants ---
EBSYNTH_VOTEMODE_PLAIN = 0x0001
EBSYNTH_VOTEMODE_WEIGHTED = 0x0002

# --- Cost Function Constants ---
COST_FUNCTION_SSD = 0
COST_FUNCTION_NCC = 1
COST_FUNCTION_SAD = 2

# --- Search Mode Constants (torch / taichi backends) ---
# PatchMatch: neighbour propagation + exponentially shrinking random search.
SEARCH_MODE_PATCHMATCH = 0
# Jump flooding: propagation from neighbours at halving step sizes, then a fine
# random search.
SEARCH_MODE_JFA = 1
# PatchMatch that also keeps ``multicand_k - 1`` alternative matches per pixel;
# alternatives seed propagation/random search and blend into weighted voting.
SEARCH_MODE_MULTICAND = 2
SEARCH_MODE_BY_NAME = {
    "patchmatch": SEARCH_MODE_PATCHMATCH,
    "jfa": SEARCH_MODE_JFA,
    "multicand": SEARCH_MODE_MULTICAND,
}

# --- Native extension loader controls ---
# The loader itself lives in ``ezsynth.kernels.synth.cuda_loader``. When False
# (default): try the pip-installed `ebsynth_torch` wheel first, then JIT-build
# ``kernels/synth/cuda_src/``.
# Set FORCE_EBSYNTH_JIT_LOADER=1 in the environment for JIT-only workflows.
FORCE_EBSYNTH_JIT_LOADER = _env_bool("FORCE_EBSYNTH_JIT_LOADER", False)
# When True, only a precompiled wheel is allowed; JIT compilation is disabled.
FORCE_EBSYNTH_WHEEL = _env_bool("FORCE_EBSYNTH_WHEEL", False)
JIT_VERBOSE = False

# --- Torch Ops Cache Clearing ---
# This is useful to reduce memory usage
TORCH_CUDA_CLEAR_CACHE = True
TORCH_MPS_CLEAR_CACHE = True

# --- Torch microprofiler (``ezsynth.kernels.synth.torch_ops.microprofile``) ---
# "" = off. "1", "true", "yes" = wall time. "sync", "2", "gpu" = device sync per region.
TORCH_MICROPROFILE = os.environ.get("EZSYNTH_TORCH_MICROPROFILE", "")

# --- Taichi backend ---
# If True, print arch when ``ensure_ti_init()`` runs.
TAICHI_INIT_VERBOSE = _env_bool("EZSYNTH_TAICHI_INIT_VERBOSE", False)

# --- PyTorch compile toggles ---
TORCH_COMPILE_FUSED_SSD = _env_bool("EZSYNTH_TORCH_COMPILE_FUSED_SSD", False)
TORCH_COMPILE_PATCH_SSD = _env_bool("EZSYNTH_TORCH_COMPILE_PATCH_SSD", False)

# --- Voting chunk budget (megabytes, minimum 8 MiB effective) ---
VOTE_CHUNK_BUDGET_MB = 72.0

# --- ezsynth.cli.video Startup Environment ---
# Applied before importing heavy runtime modules in the CLI entrypoint.
RUNPY_STARTUP_ENV = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "EZSYNTH_SKIP_METAL": "1",
    "EZSYNTH_SKIP_METAL_VERBOSE": "0",
}


def vote_chunk_budget_bytes() -> int:
    """
    Soft cap for stacked offset work in vectorized voting.

    Controlled by ``VOTE_CHUNK_BUDGET_MB`` in megabytes with an 8 MiB floor.
    """
    return max(8 << 20, int(VOTE_CHUNK_BUDGET_MB * (1 << 20)))
