import platform
import importlib
from pathlib import Path

import torch

from ezsynth.consts import FORCE_EBSYNTH_JIT_LOADER, FORCE_EBSYNTH_WHEEL
from ezsynth.utils.ebsynth_jit_platform import (
    check_python_dev_headers,
    ebsynth_jit_cflags,
    ebsynth_jit_cuda_cflags,
    ebsynth_jit_extra_include_dirs,
    ebsynth_jit_ldflags,
)

FORCE_JIT = FORCE_EBSYNTH_JIT_LOADER
FORCE_WHEEL = FORCE_EBSYNTH_WHEEL

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# State variables
ebsynth_torch = None
_module_source = None


def _log(msg):
    print(f"[ebsynth_loader] {msg}")

# 1. Try to load pre-compiled extension (wheel)
if not FORCE_JIT:
    try:
        ebsynth_torch = importlib.import_module("ebsynth_torch")

        _module_source = "pre-compiled"
        _log("Loaded pre-compiled extension.")
    except ImportError:
        if FORCE_WHEEL:
            raise ImportError(
                "ezsynth.consts.FORCE_EBSYNTH_WHEEL is True but failed to import ebsynth_torch. "
                "Please install the pre-compiled wheel."
            )
        pass

# If pre-compiled not available or FORCE_JIT is set, use JIT compilation
if ebsynth_torch is None and not FORCE_WHEEL:
    import torch.utils.cpp_extension

    MODULE_NAME = "ebsynth_torch_jit"
    _ext_dir = Path(__file__).parent / "ebsynth_extension"

    # CPU sources
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

    # CUDA sources
    if cuda_available:
        _source_files.extend([
            _ext_dir / "dispatch.cu",
            _ext_dir / "kernels.cu",
            _ext_dir / "integral_image.cu",
        ])

    _source_files_str = [str(p) for p in _source_files]

    try:
        backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
        _log(f"JIT compiling {backend_type} extension '{MODULE_NAME}'...")

        # Check for Python headers on Windows specifically
        if platform.system() == "Windows" and not check_python_dev_headers():
            _log(
                "WARNING: Python headers (Python.h) seem missing. JIT compilation will likely fail."
            )
            _log(
                "If you are using ComfyUI Portable, you MUST use a pre-compiled wheel."
            )

        extra_cflags = ["-DCPU_ONLY"] if not cuda_available else []
        extra_cflags.extend(ebsynth_jit_cflags())

        ebsynth_torch = torch.utils.cpp_extension.load(
            name=MODULE_NAME,
            sources=_source_files_str,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=ebsynth_jit_cuda_cflags(),
            extra_ldflags=ebsynth_jit_ldflags(),
            extra_include_paths=ebsynth_jit_extra_include_dirs(),
            verbose=True,
        )

        _module_source = "jit"
        print(f"{backend_type} extension loaded successfully via JIT compilation.")

    except Exception as e:
        print("=" * 50)
        backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
        print(
            f"[ERROR] Failed to JIT compile the {backend_type} extension '{MODULE_NAME}'."
        )

        system = platform.system()
        if system == "Darwin":
            print("Please ensure you have Xcode Command Line Tools installed:")
            print("  xcode-select --install")
            print("For OpenMP support, also install libomp:")
            print("  brew install libomp")
        elif system == "Windows":
            print("Please ensure you have a compatible C++ compiler (MSVC on Windows)")
            if cuda_available:
                print("and the NVIDIA CUDA Toolkit installed.")
        else:
            print("Please ensure you have a compatible C++ compiler (GCC/Clang)")
            print("and development headers installed.")

        print(f"Error details: {e}")
        print("=" * 50)
        ebsynth_torch = None


def is_cuda_available():
    """Check if the compiled extension has CUDA support."""
    return cuda_available


def get_backend_type():
    """Get the type of backend (CPU-only or CPU+CUDA)."""
    return "CPU+CUDA" if cuda_available else "CPU-only"


def get_module_source():
    """Get the source of the loaded module (pre-compiled or jit)."""
    return _module_source


# Backwards compatibility: expose the module at the package level
if ebsynth_torch is not None:
    # Re-export all attributes from the extension module
    for _attr_name in dir(ebsynth_torch):
        if not _attr_name.startswith("_"):
            globals()[_attr_name] = getattr(ebsynth_torch, _attr_name)
