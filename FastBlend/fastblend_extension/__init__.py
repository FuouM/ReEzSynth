from typing import Any

import torch

_extension_module: Any | None = None
_extension_error: Exception | None = None
_extension_load_attempted = False


def _get_extension():
    """Load the JIT extension once; cache the module or the build error."""
    global _extension_module, _extension_error, _extension_load_attempted
    if not _extension_load_attempted:
        _extension_load_attempted = True
        from ..src.fastblend_extension_loader import load_fastblend_extension

        try:
            _extension_module = load_fastblend_extension()
        except RuntimeError as e:
            _extension_error = e
    return _extension_module


def _require_extension():
    ext = _get_extension()
    if ext is None:
        raise RuntimeError(
            "FastBlend CUDA extension not available"
        ) from _extension_error
    return ext


def is_available():
    """True only if a CUDA device exists and the extension JIT load succeeded.

    Never raises: a failed build is cached and reported as ``False``.
    """
    if not torch.cuda.is_available():
        return False
    return _get_extension() is not None


def remap(source_style, nnf, patch_size, pad_size):
    return _require_extension().remap(source_style, nnf, patch_size, pad_size)


def patch_error(source, nnf, target, patch_size, pad_size):
    return _require_extension().patch_error(source, nnf, target, patch_size, pad_size)


def pairwise_patch_error(source_a, nnf_a, source_b, nnf_b, patch_size, pad_size):
    return _require_extension().pairwise_patch_error(
        source_a, nnf_a, source_b, nnf_b, patch_size, pad_size
    )
