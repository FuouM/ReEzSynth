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

# --- CUDA Extension Availability ---
# This is determined dynamically at import time
CUDA_EXTENSION_AVAILABLE = False
ebsynth_torch = None  # Will be set if extension is available

# --- Environment Variable Defaults ---
# Instead of relying on env vars, we use these defaults
# Can be overridden by setting environment variables before import
FORCE_EBSYNTH_JIT_LOADER = False
JIT_VERBOSE = False

# --- Torch Cache Clearing ---
# This is useful to reduce memory usage
TORCH_CUDA_CLEAR_CACHE = False
TORCH_MPS_CLEAR_CACHE = True


def _load_cuda_extension():
    """
    Dynamically load the CUDA extension and set CUDA_EXTENSION_AVAILABLE.
    This function is called at module import time.
    """
    global CUDA_EXTENSION_AVAILABLE, ebsynth_torch

    if FORCE_EBSYNTH_JIT_LOADER:
        if JIT_VERBOSE:
            print("Forcing JIT loader for ebsynth_torch (direct import disabled).")

    # Initialize variables
    force_jit = FORCE_EBSYNTH_JIT_LOADER

    if not FORCE_EBSYNTH_JIT_LOADER:
        # First, try direct import of ebsynth_torch (if installed via pip)
        try:
            import ebsynth_torch

            CUDA_EXTENSION_AVAILABLE = True
            if JIT_VERBOSE:
                print("CUDA extension loaded successfully (direct import).")
            return  # Success, no need to continue
        except ImportError:
            force_jit = True  # Fall back to JIT if direct import fails

    if force_jit:
        # Try the JIT loader
        try:
            from ebsynth_torch_loader import ebsynth_torch as jit_ebsynth_torch

            CUDA_EXTENSION_AVAILABLE = jit_ebsynth_torch is not None
            ebsynth_torch = jit_ebsynth_torch
            if CUDA_EXTENSION_AVAILABLE:
                if JIT_VERBOSE:
                    print("CUDA extension loaded successfully (via JIT loader).")
            else:
                print("JIT loader found but extension not available.")
        except ImportError as e:
            print(f"\nCould not find the JIT loader module: {e}")
            ebsynth_torch = None
            CUDA_EXTENSION_AVAILABLE = False

    if not CUDA_EXTENSION_AVAILABLE:
        print("\n[WARNING] PyTorch CUDA extension not available.")
        print("CUDA backend will not be available. Only PyTorch backend can be used.")
        print(
            "To enable CUDA backend, ensure a C++ compiler and CUDA Toolkit are installed.\n"
        )


# Load CUDA extension at import time
_load_cuda_extension()
