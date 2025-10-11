# ezsynth/ebsynth_torch_loader.py

from pathlib import Path

import torch.utils.cpp_extension

# This is the name the compiled module will have in Python
MODULE_NAME = "ebsynth_torch_jit"

# Find the directory where the C++/CUDA source files are located
# This assumes this loader file is in the same directory as the 'engines' folder
# or has a predictable relative path. Let's make it robust.
_ext_dir = Path(__file__).parent.parent / Path(__file__).parent / "ebsynth_extension"

# List all the source files for the extension
_source_files = [
    _ext_dir / "ext.cpp",
    _ext_dir / "dispatch.cu",
    _ext_dir / "kernels.cu",
]

# Convert Path objects to strings for the compiler
_source_files_str = [str(p) for p in _source_files]

# JIT compilation using torch.utils.cpp_extension.load()
# This will be executed only once, the first time this module is imported.
# PyTorch caches the compiled library in a build directory.
try:
    print(f"Attempting to JIT compile and load CUDA extension '{MODULE_NAME}'...")
    ebsynth_torch = torch.utils.cpp_extension.load(
        name=MODULE_NAME,
        sources=_source_files_str,
        # Use verbose=True to see the compiler commands and debug issues
        verbose=True,
    )
    print("CUDA extension loaded successfully via JIT compilation.")
except Exception as e:
    print("=" * 50)
    print(f"[ERROR] Failed to JIT compile the CUDA extension '{MODULE_NAME}'.")
    print("Please ensure you have a compatible C++ compiler (MSVC on Windows)")
    print("and the NVIDIA CUDA Toolkit installed.")
    print(f"Error details: {e}")
    print("=" * 50)
    ebsynth_torch = None
