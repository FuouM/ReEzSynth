"""Backward-compatible entry point for the native ``ebsynth_torch`` extension.

Importing this module loads the extension through :mod:`ezsynth.native` (the
single wheel-or-JIT implementation) and re-exports its functions at module level.
"""

from ezsynth import native

native.ensure_extension_loaded()
ebsynth_torch = native.module


def is_cuda_available():
    """Check if the loaded extension can run CUDA tensors."""
    return native.cuda_available


def get_backend_type():
    """Get the type of backend (CPU-only or CPU+CUDA)."""
    return native.get_backend_type()


def get_module_source():
    """Get the source of the loaded module (pre-compiled or jit)."""
    return native.module_source


if ebsynth_torch is not None:
    for _attr_name in dir(ebsynth_torch):
        if not _attr_name.startswith("_"):
            globals()[_attr_name] = getattr(ebsynth_torch, _attr_name)
