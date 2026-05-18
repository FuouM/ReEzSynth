"""Compiler flags and header checks for JIT-building ``ebsynth_torch``."""

import os
import platform
import sys
from pathlib import Path


def check_python_dev_headers() -> bool:
    """Heuristic: whether Python development headers (``Python.h``) are likely present."""
    include_path = Path(sys.base_prefix) / "include"

    if platform.system() == "Windows":
        if (include_path / "Python.h").exists():
            return True
        return (Path(sys.base_prefix) / "include" / "Python.h").exists()

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if (include_path / python_version / "Python.h").exists():
        return True

    return False


def ebsynth_jit_cflags() -> list[str]:
    """Host C++ compiler flags for the native extension."""
    cflags = ["-O2"]
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        cflags.append("-std=c++17")
        cflags.append("-ffast-math")
        if machine == "arm64":
            cflags.extend(["-arch", "arm64", "-mcpu=native"])
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


def ebsynth_jit_cuda_cflags() -> list[str]:
    """CUDA compiler flags for JIT builds."""
    return ["-O2", "--use_fast_math"]


def ebsynth_jit_ldflags() -> list[str]:
    """Extra linker flags, currently macOS Homebrew libomp."""
    ldflags: list[str] = []
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


def ebsynth_jit_extra_include_dirs() -> list[str]:
    """Extra include dirs, currently macOS Homebrew libomp headers."""
    include_dirs: list[str] = []
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
