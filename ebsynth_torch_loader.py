# ezsynth/ebsynth_torch_loader.py

import os
from pathlib import Path

import torch.utils.cpp_extension

# This is the name the compiled module will have in Python
MODULE_NAME = "ebsynth_torch_jit"

# Find the directory where the C++/CUDA source files are located
_ext_dir = Path(__file__).parent / "ebsynth_extension"

# List all the source files for the extension
# IMPORTANT: Only compile the main entry points. The modular .cu files are
# included via #include directives in kernels.cu, so they should NOT be
# listed as separate source files to avoid duplicate compilation and slowdowns.
_source_files = [
    _ext_dir / "ext.cpp",
    _ext_dir / "dispatch.cu",
    _ext_dir / "kernels.cu",           # Includes all modular .cu files internally
    _ext_dir / "integral_image.cu",
]

# Convert Path objects to strings for the compiler
_source_files_str = [str(p) for p in _source_files]

# JIT compilation using torch.utils.cpp_extension.load()
# This will be executed only once, the first time this module is imported.
# PyTorch caches the compiled library in a build directory.
try:
    if os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"):
        print(f"Attempting to JIT compile and load CUDA extension '{MODULE_NAME}'...")
        print(f"Source files: {_source_files_str}")
    ebsynth_torch = torch.utils.cpp_extension.load(
        name=MODULE_NAME,
        sources=_source_files_str,
        # Use verbose=True to see the compiler commands and debug issues
        verbose=os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"),
    )
    if os.getenv("JIT_VERBOSE", "").lower() in ("1", "true", "yes"):
        print("CUDA extension loaded successfully via JIT compilation.")
except Exception as e:
    print("=" * 50)
    print(f"[ERROR] Failed to JIT compile the CUDA extension '{MODULE_NAME}'.")
    print("Please ensure you have a compatible C++ compiler (MSVC on Windows)")
    print("and the NVIDIA CUDA Toolkit installed.")
    print(f"Error details: {e}")
    print("=" * 50)
    ebsynth_torch = None
