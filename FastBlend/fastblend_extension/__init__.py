import torch
from typing import Any, Optional

_extension_module: Optional[Any] = None
_extension_load_attempted = False


def _get_extension():
    global _extension_module, _extension_load_attempted
    if _extension_load_attempted:
        return _extension_module
    _extension_load_attempted = True
    from ..src.fastblend_extension_loader import load_fastblend_extension

    _extension_module = load_fastblend_extension()
    return _extension_module


def is_available():
    """True only if a CUDA device exists and the extension JIT load succeeded."""
    if not torch.cuda.is_available():
        return False
    if _extension_load_attempted:
        return _extension_module is not None
    return _get_extension() is not None


def remap(source_style, nnf, patch_size, pad_size):
    ext = _get_extension()
    if ext is None:
        raise RuntimeError("FastBlend CUDA extension not available")
    return ext.remap(source_style, nnf, patch_size, pad_size)


def patch_error(source, nnf, target, patch_size, pad_size):
    ext = _get_extension()
    if ext is None:
        raise RuntimeError("FastBlend CUDA extension not available")
    return ext.patch_error(source, nnf, target, patch_size, pad_size)


def pairwise_patch_error(source_a, nnf_a, source_b, nnf_b, patch_size, pad_size):
    ext = _get_extension()
    if ext is None:
        raise RuntimeError("FastBlend CUDA extension not available")
    return ext.pairwise_patch_error(
        source_a, nnf_a, source_b, nnf_b, patch_size, pad_size
    )
