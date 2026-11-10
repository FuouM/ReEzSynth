"""Compiler flags and header checks for JIT-building ``ebsynth_torch``.

Flags are portable by default so a build cache can be shared between machines:

- Host code targets the toolchain's baseline ISA. Set ``EZSYNTH_NATIVE_ARCH=1`` to
  tune for the build machine (``-march=native`` / ``-mcpu=native`` /
  ``/arch:AVX2``); such a binary may crash with an illegal instruction elsewhere.
- The C++ standard is set explicitly on every platform. Host code
  (``cxx_standard()``) uses C++17, or C++20 for PyTorch >= 2.12, whose headers
  require it; an explicit flag also overrides MSVC's C++14 default. CUDA code is
  always C++17 (``CUDA_CXX_STANDARD``): the ``*.cu`` sources include no PyTorch
  headers, so they build with older nvcc too (CUDA 12.4's cudafe++ crashes in
  C++20 mode with MSVC 14.41).
- CUDA architectures are left to PyTorch: ``ebsynth_jit_cuda_cflags()`` never
  contains an ``-arch``/``-gencode`` flag, so ``torch.utils.cpp_extension``
  builds for ``TORCH_CUDA_ARCH_LIST`` when it is set, and otherwise for the
  compute capability of every visible GPU (plus PTX for the newest one).
"""

import os
import platform
import sys
from pathlib import Path

import torch

NATIVE_ARCH_ENV = "EZSYNTH_NATIVE_ARCH"
CUDA_CXX_STANDARD = "c++17"


def cxx_standard() -> str:
    """C++ standard for host code: the minimum the installed PyTorch headers accept."""
    return "c++20" if torch.__version__ >= "2.12" else "c++17"


def native_arch_enabled() -> bool:
    """Whether ``EZSYNTH_NATIVE_ARCH`` opts in to machine-specific host tuning."""
    value = os.environ.get(NATIVE_ARCH_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_ninja_on_path() -> None:
    """Put the ``ninja`` dependency's executable on PATH for ``torch.utils.cpp_extension``.

    PyTorch's JIT loader runs ``ninja`` from PATH, which does not include the virtualenv's
    scripts directory when the interpreter is started directly (``.venv/Scripts/python``).
    """
    import ninja

    path = os.environ.get("PATH", "")
    if ninja.BIN_DIR not in path.split(os.pathsep):
        os.environ["PATH"] = ninja.BIN_DIR + os.pathsep + path


def check_python_dev_headers() -> bool:
    """Heuristic: whether Python development headers (``Python.h``) are likely present."""
    include_path = Path(sys.base_prefix) / "include"

    if platform.system() == "Windows":
        return (include_path / "Python.h").exists()

    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return bool((include_path / python_version / "Python.h").exists())


def ebsynth_jit_cflags() -> list[str]:
    """Host C++ compiler flags for the native extension."""
    system = platform.system()
    machine = platform.machine()
    native = native_arch_enabled()

    if system == "Windows":
        std = cxx_standard()
        cflags = [f"/std:{std}", "/openmp", "/O2", "/fp:fast", "/MP", "/EHsc"]
        if native and machine in ("AMD64", "x86_64"):
            cflags.append("/arch:AVX2")
        return cflags

    cflags = ["-O2", f"-std={cxx_standard()}"]
    if system == "Darwin":
        cflags.append("-ffast-math")
        if machine == "arm64":
            cflags.extend(["-arch", "arm64"])
        elif machine == "x86_64":
            cflags.extend(["-arch", "x86_64"])
        cflags.extend(["-Xpreprocessor", "-fopenmp"])
    else:
        cflags.append("-fopenmp")
    if native:
        if machine in ("x86_64", "AMD64"):
            cflags.append("-march=native")
        elif machine in ("arm64", "aarch64"):
            cflags.append("-mcpu=native")
    return cflags


def ebsynth_jit_cuda_cflags() -> list[str]:
    """CUDA compiler flags for JIT builds (no arch flags; see module docstring).

    PyTorch passes its own ``-std=c++20`` to nvcc first; nvcc keeps the last
    ``-std`` value, so this one wins.
    """
    return ["-O2", "--use_fast_math", f"-std={CUDA_CXX_STANDARD}"]


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
