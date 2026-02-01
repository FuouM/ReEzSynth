# ezsynth/ebsynth_torch_loader.py

import os
from pathlib import Path

import torch
import torch.utils.cpp_extension

# This is the name the compiled module will have in Python
MODULE_NAME = "ebsynth_torch_jit"

# Find the directory where the C++/CUDA source files are located
_ext_dir = Path(__file__).parent / "ebsynth_extension"

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# List all the source files for the extension
# CPU-only files are always included
_source_files = [
    _ext_dir / "ext.cpp",
    _ext_dir / "dispatch.cpp",
    _ext_dir / "cpu" / "cost_functions_cpu.cpp",
    _ext_dir / "cpu" / "omega_ops_cpu.cpp",
    _ext_dir / "cpu" / "voting_cpu.cpp",
    _ext_dir / "cpu" / "mask_ops_cpu.cpp",
    _ext_dir / "cpu" / "integral_image_cpu.cpp",
    _ext_dir / "cpu" / "patchmatch_cpu.cpp",
    _ext_dir / "cpu" / "dispatch_cpu.cpp",
]

# Add CUDA files only if CUDA is available
if cuda_available:
    _source_files.extend([
        _ext_dir / "dispatch.cu",
        _ext_dir / "kernels.cu",           # Includes all modular .cu files internally
        _ext_dir / "integral_image.cu",
    ])

# Convert Path objects to strings for the compiler
_source_files_str = [str(p) for p in _source_files]

# JIT compilation using torch.utils.cpp_extension.load()
# This will be executed only once, the first time this module is imported.
# PyTorch caches the compiled library in a build directory.
try:
    if True:
        backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
        print(f"Attempting to JIT compile and load {backend_type} extension '{MODULE_NAME}'...")
        print(f"Source files: {_source_files_str}")

    # Set extra compile args based on backend
    extra_cflags = []
    extra_cuda_cflags = []

    if not cuda_available:
        # Define CPU_ONLY for CPU-only builds
        extra_cflags.append("-DCPU_ONLY")

    # Add OpenMP for parallel CPU execution (Windows MSVC)
    extra_cflags.append("/openmp")
    extra_cflags.append("/O2")
    extra_cflags.append("/fp:fast")

    ebsynth_torch = torch.utils.cpp_extension.load(
        name=MODULE_NAME,
        sources=_source_files_str,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        # Use verbose=True to see the compiler commands and debug issues
        verbose=True,
    )

    if True:
        backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
        print(f"{backend_type} extension loaded successfully via JIT compilation.")

except Exception as e:
    print("=" * 50)
    backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
    print(f"[ERROR] Failed to JIT compile the {backend_type} extension '{MODULE_NAME}'.")
    print("Please ensure you have a compatible C++ compiler (MSVC on Windows)")
    if cuda_available:
        print("and the NVIDIA CUDA Toolkit installed.")
    print(f"Error details: {e}")
    print("=" * 50)
    ebsynth_torch = None


def is_cuda_available():
    """Check if the compiled extension has CUDA support."""
    return cuda_available


def get_backend_type():
    """Get the type of backend (CPU-only or CPU+CUDA)."""
    return "CPU+CUDA" if cuda_available else "CPU-only"
