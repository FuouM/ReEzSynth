# ezsynth/consts.py
"""
Constants and configuration for the ezsynth library.

All tunables live here as module-level values. ``run.py`` copies
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

# --- Extension Availability ---
# Updated when ``ensure_extension_loaded()`` successfully loads the native module.
EXTENSION_AVAILABLE = False
EXTENSION_CUDA_AVAILABLE = False
ebsynth_torch = None  # Will be set if extension is available

# --- Native extension loader controls ---
# When False (default): try pip-installed `ebsynth_torch` first, then JIT.
# Set FORCE_EBSYNTH_JIT_LOADER=1 in the environment for legacy JIT-only workflows.
FORCE_EBSYNTH_JIT_LOADER = _env_bool("FORCE_EBSYNTH_JIT_LOADER", False)
# When True, only a precompiled wheel is allowed; JIT fallback is disabled.
FORCE_EBSYNTH_WHEEL = _env_bool("FORCE_EBSYNTH_WHEEL", False)
JIT_VERBOSE = False

# Load native extension only when the CUDA/extension backend actually needs it
# (avoids JIT/import side effects when using torch/taichi backends).
_extension_load_attempted = False

# --- Torch Ops Cache Clearing ---
# This is useful to reduce memory usage
TORCH_CUDA_CLEAR_CACHE = True
TORCH_MPS_CLEAR_CACHE = True

# --- Torch microprofiler (``ezsynth.torch_ops.microprofile``) ---
# "" = off. "1", "true", "yes" = wall time. "sync", "2", "gpu" = device sync per region.
TORCH_MICROPROFILE = os.environ.get("EZSYNTH_TORCH_MICROPROFILE", "")

# --- Taichi backend ---
# If True, print arch when ``ensure_ti_init()`` runs.
TAICHI_INIT_VERBOSE = _env_bool("EZSYNTH_TAICHI_INIT_VERBOSE", False)

# --- PyTorch compile toggles ---
TORCH_COMPILE_FUSED_SSD = _env_bool("EZSYNTH_TORCH_COMPILE_FUSED_SSD", False)
TORCH_COMPILE_PATCH_SSD = _env_bool("EZSYNTH_TORCH_COMPILE_PATCH_SSD", False)
TORCH_COMPILE_MASK_OPS = _env_bool("EZSYNTH_TORCH_COMPILE_MASK_OPS", False)

# --- Voting chunk budget (megabytes, minimum 8 MiB effective) ---
VOTE_CHUNK_BUDGET_MB = 72.0

# --- run.py Startup Environment ---
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
    try:
        mb = float(VOTE_CHUNK_BUDGET_MB)
    except (TypeError, ValueError):
        mb = 72.0
    return max(8 << 20, int(mb * (1 << 20)))


def ensure_extension_loaded() -> bool:
    """Load native ``ebsynth_torch`` once and return whether it is available."""
    global _extension_load_attempted
    if EXTENSION_AVAILABLE and ebsynth_torch is not None:
        return True
    if _extension_load_attempted:
        return False
    _extension_load_attempted = True
    _load_extension()
    return EXTENSION_AVAILABLE and ebsynth_torch is not None


def _load_extension():
    """
    Dynamically load the ebsynth extension and set EXTENSION_AVAILABLE.
    Invoked from ``ensure_extension_loaded()`` before first use by the CUDA backend.
    Supports both CPU and CUDA backends.
    """
    global EXTENSION_AVAILABLE, EXTENSION_CUDA_AVAILABLE, ebsynth_torch

    if EXTENSION_AVAILABLE and ebsynth_torch is not None:
        return

    if FORCE_EBSYNTH_JIT_LOADER:
        if JIT_VERBOSE:
            print("Forcing JIT loader for ebsynth_torch (direct import disabled).")

    # Initialize variables
    force_jit = True

    if not FORCE_EBSYNTH_JIT_LOADER:
        # First, try direct import of ebsynth_torch (if installed via pip)
        try:
            import importlib

            ebsynth_torch = importlib.import_module("ebsynth_torch")

            EXTENSION_AVAILABLE = True
            EXTENSION_CUDA_AVAILABLE = True  # Assume CUDA if direct import works
            if JIT_VERBOSE:
                print("Extension loaded successfully (direct import).")
            return  # Success, no need to continue
        except ImportError:
            force_jit = True  # Fall back to JIT if direct import fails

    if force_jit:
        try:
            import importlib.util
            from pathlib import Path

            repo_root = Path(__file__).resolve().parent.parent
            loader_path = repo_root / "ebsynth_torch_loader.py"
            spec = importlib.util.spec_from_file_location(
                "ebsynth_torch_loader_repo",
                str(loader_path),
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load {loader_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            jit_ebsynth_torch = getattr(mod, "ebsynth_torch", None)
            is_cuda_available = getattr(mod, "is_cuda_available")

            EXTENSION_AVAILABLE = jit_ebsynth_torch is not None
            EXTENSION_CUDA_AVAILABLE = (
                is_cuda_available() if EXTENSION_AVAILABLE else False
            )
            ebsynth_torch = jit_ebsynth_torch
            if EXTENSION_AVAILABLE:
                if JIT_VERBOSE:
                    backend_type = (
                        "CPU+CUDA" if EXTENSION_CUDA_AVAILABLE else "CPU-only"
                    )
                    print(
                        f"Extension loaded successfully (via JIT loader) - {backend_type}."
                    )
            else:
                print("JIT loader found but extension not available.")
        except ImportError as e:
            print(f"\nCould not find the JIT loader module: {e}")
            ebsynth_torch = None
            EXTENSION_AVAILABLE = False
            EXTENSION_CUDA_AVAILABLE = False

    if not EXTENSION_AVAILABLE:
        print("\n[WARNING] ebsynth_torch extension not available.")
        print("Only PyTorch / Taichi backends can be used without it.")
        print("To enable the C++ extension, ensure a C++ compiler is installed.")
        print("For CUDA support, also ensure the NVIDIA CUDA Toolkit is installed.\n")
