"""Loader for the native ``ebsynth_torch`` C++/CUDA extension.

This is the only implementation of "import the pre-compiled wheel, else JIT-build
``ebsynth_extension/``". Loading is lazy: nothing happens until
:func:`ensure_extension_loaded` is called (the ``cuda`` synthesis backend does so),
so the PyTorch / Taichi backends never trigger a compile.

Controls (environment, read via :mod:`ezsynth.consts`):

- ``FORCE_EBSYNTH_JIT_LOADER=1``: skip the wheel import and always JIT-build.
- ``FORCE_EBSYNTH_WHEEL=1``: only the wheel is allowed; a missing wheel raises
  :class:`ImportError` instead of compiling.
- Compiler flags come from :mod:`ezsynth.utils.ebsynth_jit_platform`
  (``EZSYNTH_NATIVE_ARCH``, ``TORCH_CUDA_ARCH_LIST``).
"""

import importlib
import logging
import platform
from pathlib import Path
from types import ModuleType

import torch

from . import consts
from .utils.ebsynth_jit_platform import (
    check_python_dev_headers,
    ebsynth_jit_cflags,
    ebsynth_jit_cuda_cflags,
    ebsynth_jit_extra_include_dirs,
    ebsynth_jit_ldflags,
    ensure_ninja_on_path,
)

logger = logging.getLogger(__name__)

WHEEL_MODULE_NAME = "ebsynth_torch"
JIT_MODULE_NAME = "ebsynth_torch_jit"

EXTENSION_DIR = Path(__file__).resolve().parent / "ebsynth_extension"
_CPU_SOURCES = (
    "ext.cpp",
    "dispatch.cpp",
    "cpu/cost_functions_cpu.cpp",
    "cpu/omega_ops_cpu.cpp",
    "cpu/voting_cpu.cpp",
    "cpu/mask_ops_cpu.cpp",
    "cpu/integral_image_cpu.cpp",
    "cpu/patchmatch_cpu.cpp",
    "cpu/dispatch_cpu.cpp",
)
# CUDA builds only: torch-free kernels (*.cu) plus the PyTorch-facing glue.
_CUDA_SOURCES = ("cuda_dispatch.cpp", "dispatch.cu", "kernels.cu", "integral_image.cu")

# Loader state; read through the accessors below or ``ezsynth.consts``.
module: ModuleType | None = None
module_source: str | None = None  # "pre-compiled" or "jit"
cuda_available = False
_load_attempted = False


def _log(msg: str) -> None:
    logger.info(f"[ebsynth_loader] {msg}")


def ensure_extension_loaded() -> bool:
    """Load the native extension once; return whether it is available."""
    global _load_attempted, module, module_source, cuda_available
    if module is not None:
        return True
    if _load_attempted:
        return False
    _load_attempted = True

    loaded: ModuleType | None = None
    source: str | None = None
    if consts.FORCE_EBSYNTH_JIT_LOADER:
        if consts.JIT_VERBOSE:
            _log("FORCE_EBSYNTH_JIT_LOADER set; skipping the pre-compiled wheel.")
    else:
        loaded = _import_wheel()
        source = "pre-compiled" if loaded is not None else None
    if loaded is None and not consts.FORCE_EBSYNTH_WHEEL:
        loaded = _jit_build(with_cuda=torch.cuda.is_available())
        source = "jit" if loaded is not None else None

    if loaded is None:
        logger.warning("\n[WARNING] ebsynth_torch extension not available.")
        logger.warning("Only PyTorch / Taichi backends can be used without it.")
        logger.warning(
            "To enable the C++ extension, ensure a C++ compiler is installed."
        )
        logger.warning(
            "For CUDA support, also ensure the NVIDIA CUDA Toolkit is installed.\n"
        )
        return False

    module = loaded
    module_source = source
    cuda_available = _module_has_cuda(loaded) and torch.cuda.is_available()
    if consts.JIT_VERBOSE:
        _log(f"Extension ready ({source}, {get_backend_type()}).")
    return True


def get_backend_type() -> str:
    """``"CPU+CUDA"`` when the loaded extension can run CUDA tensors, else ``"CPU-only"``."""
    return "CPU+CUDA" if cuda_available else "CPU-only"


def _module_has_cuda(mod: ModuleType) -> bool:
    """Whether ``mod`` was compiled with its CUDA kernels.

    Builds of this repository export ``compiled_with_cuda()``. Older wheels that
    lack it cannot be verified, so they are reported as CPU-only.
    """
    probe = getattr(mod, "compiled_with_cuda", None)
    return bool(probe()) if callable(probe) else False


def _import_wheel() -> ModuleType | None:
    try:
        mod = importlib.import_module(WHEEL_MODULE_NAME)
    except ImportError as err:
        if consts.FORCE_EBSYNTH_WHEEL:
            raise ImportError(
                "FORCE_EBSYNTH_WHEEL is set but importing the pre-compiled "
                f"'{WHEEL_MODULE_NAME}' wheel failed. Install the wheel or unset "
                "FORCE_EBSYNTH_WHEEL to allow JIT compilation."
            ) from err
        return None
    _log("Loaded pre-compiled extension.")
    return mod


def _jit_build(with_cuda: bool) -> ModuleType | None:
    import torch.utils.cpp_extension

    sources = [str(EXTENSION_DIR / s) for s in _CPU_SOURCES]
    if with_cuda:
        sources.extend(str(EXTENSION_DIR / s) for s in _CUDA_SOURCES)
    backend_type = "CPU+CUDA" if with_cuda else "CPU-only"
    system = platform.system()

    _log(f"JIT compiling {backend_type} extension '{JIT_MODULE_NAME}'...")
    if system == "Windows" and not check_python_dev_headers():
        _log(
            "WARNING: Python headers (Python.h) seem missing. "
            "JIT compilation will likely fail."
        )
        _log("If you are using ComfyUI Portable, you MUST use a pre-compiled wheel.")

    extra_cflags = [] if with_cuda else ["-DCPU_ONLY"]
    extra_cflags.extend(ebsynth_jit_cflags())
    ensure_ninja_on_path()
    try:
        mod = torch.utils.cpp_extension.load(
            name=JIT_MODULE_NAME,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=ebsynth_jit_cuda_cflags(),
            extra_ldflags=ebsynth_jit_ldflags(),
            extra_include_paths=ebsynth_jit_extra_include_dirs(),
            verbose=True,
        )
    except Exception as e:
        logger.error("=" * 50)
        logger.error(
            f"[ERROR] Failed to JIT compile the {backend_type} extension "
            f"'{JIT_MODULE_NAME}'."
        )
        if system == "Darwin":
            logger.error("Please ensure you have Xcode Command Line Tools installed:")
            logger.error("  xcode-select --install")
            logger.error("For OpenMP support, also install libomp:")
            logger.error("  brew install libomp")
        elif system == "Windows":
            logger.error(
                "Please ensure you have a compatible C++ compiler (MSVC on Windows)"
            )
            if with_cuda:
                logger.error("and the NVIDIA CUDA Toolkit installed.")
        else:
            logger.error("Please ensure you have a compatible C++ compiler (GCC/Clang)")
            logger.error("and development headers installed.")
        logger.error(f"Error details: {e}")
        logger.error("=" * 50)
        return None

    logger.info(f"{backend_type} extension loaded successfully via JIT compilation.")
    return mod
