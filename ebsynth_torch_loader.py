# ezsynth/ebsynth_torch_loader.py

import importlib.util
import os
import platform
import sys
from pathlib import Path

import torch

# Configuration
FORCE_JIT = os.environ.get("FORCE_EBSYNTH_JIT", "0") == "1"
FORCE_WHEEL = os.environ.get("FORCE_EBSYNTH_WHEEL", "0") == "1"

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# State variables
ebsynth_torch = None
_module_source = None


def _log(msg):
    print(f"[ebsynth_loader] {msg}")


def _check_python_headers():
    """Check if Python development headers are likely present."""
    # This is a heuristic. Python.h is usually in include directory.
    include_path = Path(sys.base_prefix) / "include"

    # On Windows, it's typically pythonXY/include/Python.h
    if platform.system() == "Windows":
        # Check standard include path
        if (include_path / "Python.h").exists():
            return True
        # Check if we are in a venv, headers might be in base prefix
        return (Path(sys.base_prefix) / "include" / "Python.h").exists()

    # On POSIX, it's often python3.x/Python.h
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if (include_path / python_version / "Python.h").exists():
        return True

    return False


# 1. Try to load pre-compiled extension (wheel)
if not FORCE_JIT:
    try:
        import ebsynth_torch

        _module_source = "pre-compiled"
        _log("Loaded pre-compiled extension.")
    except ImportError:
        if FORCE_WHEEL:
            raise ImportError(
                "FORCE_EBSYNTH_WHEEL=1 but failed to import ebsynth_torch. "
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

    def _get_platform_cflags():
        cflags = ["-O2"]
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin":
            cflags.append("-std=c++17")
            cflags.append("-ffast-math")
            if machine == "arm64":
                cflags.extend(["-arch", "arm64", "-mcpu=apple-m4"])
            elif machine == "x86_64":
                cflags.extend(["-arch", "x86_64"])
            cflags.extend(["-Xpreprocessor", "-fopenmp"])
        elif system == "Windows":
            cflags.extend(["/openmp", "/O2", "/fp:fast", "/MP", "/EHsc"])
            if machine in ["AMD64", "x86_64"]:
                cflags.extend(["/arch:AVX2", "/D__SSE4_2__", "/D__AVX2__"])
        else:
            cflags.append("-std=c++17")
            cflags.append("-fopenmp")
            if machine in ["x86_64", "AMD64"]:
                cflags.append("-march=native")
            elif machine == "arm64":
                cflags.append("-mcpu=native")
        return cflags

    def _get_platform_ldflags():
        ldflags = []
        if platform.system() == "Darwin":
            libomp_paths = [
                "/opt/homebrew/opt/libomp/lib",
                "/usr/local/opt/libomp/lib",
            ]
            for path in libomp_paths:
                if os.path.exists(path):
                    ldflags.extend([f"-L{path}", "-lomp", f"-Wl,-rpath,{path}"])
                    break
        return ldflags

    def _get_platform_include_dirs():
        include_dirs = []
        if platform.system() == "Darwin":
            libomp_includes = [
                "/opt/homebrew/opt/libomp/include",
                "/usr/local/opt/libomp/include",
            ]
            for path in libomp_includes:
                if os.path.exists(path):
                    include_dirs.append(path)
                    break
        return include_dirs

    try:
        backend_type = "CPU+CUDA" if cuda_available else "CPU-only"
        _log(f"JIT compiling {backend_type} extension '{MODULE_NAME}'...")

        # Check for Python headers on Windows specifically
        if platform.system() == "Windows" and not _check_python_headers():
            _log(
                "WARNING: Python headers (Python.h) seem missing. JIT compilation will likely fail."
            )
            _log(
                "If you are using ComfyUI Portable, you MUST use a pre-compiled wheel."
            )

        extra_cflags = ["-DCPU_ONLY"] if not cuda_available else []
        extra_cflags.extend(_get_platform_cflags())

        ebsynth_torch = torch.utils.cpp_extension.load(
            name=MODULE_NAME,
            sources=_source_files_str,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=[],
            extra_ldflags=_get_platform_ldflags(),
            extra_include_paths=_get_platform_include_dirs(),
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
