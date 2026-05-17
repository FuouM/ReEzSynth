# ezsynth/consts.py
"""
Constants and configuration for the ezsynth library.
Centralized location for all constants to avoid duplication and reliance on environment variables.
"""

import os

# --- Ebsynth Vote Mode Constants ---
EBSYNTH_VOTEMODE_PLAIN = 0x0001
EBSYNTH_VOTEMODE_WEIGHTED = 0x0002

# --- Cost Function Constants ---
COST_FUNCTION_SSD = 0
COST_FUNCTION_NCC = 1

# --- Extension Availability ---
# Updated when ensure_ebsynth_extension() successfully loads the native module (CUDA backend).
EXTENSION_AVAILABLE = False
EXTENSION_CUDA_AVAILABLE = False
ebsynth_torch = None  # Will be set if extension is available

# Backward compatibility aliases
CUDA_EXTENSION_AVAILABLE = False  # Deprecated, use EXTENSION_AVAILABLE instead

# --- Environment Variable Defaults ---
# Instead of relying on env vars, we use these defaults
# Can be overridden by setting environment variables before import
# When False (default): try pip-installed `ebsynth_torch` first, then JIT.
# Set FORCE_EBSYNTH_JIT_LOADER=1 in the environment for legacy JIT-only workflows.
FORCE_EBSYNTH_JIT_LOADER = os.environ.get(
    "FORCE_EBSYNTH_JIT_LOADER",
    "",
).strip().lower() in ("1", "true", "yes")

JIT_VERBOSE = False

# Load native extension only when the CUDA/extension backend actually needs it
# (avoids JIT/import side effects when using torch/taichi backends).
_extension_load_attempted = False

# --- Torch Ops Cache Clearing ---
# This is useful to reduce memory usage
TORCH_CUDA_CLEAR_CACHE = True
TORCH_MPS_CLEAR_CACHE = True


def ensure_ebsynth_extension() -> None:
    """Load native ebsynth_torch once when the CUDA C++ backend is requested."""
    global _extension_load_attempted
    if _extension_load_attempted:
        return
    _extension_load_attempted = True
    _load_extension()


def _load_extension():
    """
    Dynamically load the ebsynth extension and set EXTENSION_AVAILABLE.
    Called from ensure_ebsynth_extension() before first use by CudaBackend.
    Supports both CPU and CUDA backends.
    """
    global \
        EXTENSION_AVAILABLE, \
        EXTENSION_CUDA_AVAILABLE, \
        CUDA_EXTENSION_AVAILABLE, \
        ebsynth_torch

    if FORCE_EBSYNTH_JIT_LOADER:
        if JIT_VERBOSE:
            print("Forcing JIT loader for ebsynth_torch (direct import disabled).")

    # Initialize variables
    force_jit = True

    if not FORCE_EBSYNTH_JIT_LOADER:
        # First, try direct import of ebsynth_torch (if installed via pip)
        try:
            import ebsynth_torch

            EXTENSION_AVAILABLE = True
            EXTENSION_CUDA_AVAILABLE = True  # Assume CUDA if direct import works
            CUDA_EXTENSION_AVAILABLE = True
            if JIT_VERBOSE:
                print("Extension loaded successfully (direct import).")
            return  # Success, no need to continue
        except ImportError:
            force_jit = True  # Fall back to JIT if direct import fails

    if force_jit:
        # Try the JIT loader
        try:
            from ebsynth_torch_loader import ebsynth_torch as jit_ebsynth_torch
            from ebsynth_torch_loader import is_cuda_available

            EXTENSION_AVAILABLE = jit_ebsynth_torch is not None
            EXTENSION_CUDA_AVAILABLE = (
                is_cuda_available() if EXTENSION_AVAILABLE else False
            )
            CUDA_EXTENSION_AVAILABLE = EXTENSION_AVAILABLE  # Backward compatibility
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
            CUDA_EXTENSION_AVAILABLE = False

    if not EXTENSION_AVAILABLE:
        print("\n[WARNING] ebsynth_torch extension not available.")
        print("Only PyTorch / Taichi backends can be used without it.")
        print("To enable the C++ extension, ensure a C++ compiler is installed.")
        print("For CUDA support, also ensure the NVIDIA CUDA Toolkit is installed.\n")