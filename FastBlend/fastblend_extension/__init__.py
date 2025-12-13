import os
import sys

import torch

# Import the JIT-compiled extension
try:
    from ..fastblend_extension_loader import fastblend_extension

    _extension_available = fastblend_extension is not None
except ImportError:
    fastblend_extension = None
    _extension_available = False


def is_available():
    """Check if the CUDA extension is available."""
    return _extension_available and torch.cuda.is_available()


def remap(source_style, nnf, patch_size, pad_size):
    """Apply NNF remapping for FastBlend."""
    if not is_available():
        raise RuntimeError("FastBlend CUDA extension not available")
    return fastblend_extension.remap(source_style, nnf, patch_size, pad_size)


def patch_error(source, nnf, target, patch_size, pad_size):
    """Compute patch error for FastBlend."""
    if not is_available():
        raise RuntimeError("FastBlend CUDA extension not available")
    return fastblend_extension.patch_error(source, nnf, target, patch_size, pad_size)


def pairwise_patch_error(source_a, nnf_a, source_b, nnf_b, patch_size, pad_size):
    """Compute pairwise patch error for FastBlend."""
    if not is_available():
        raise RuntimeError("FastBlend CUDA extension not available")
    return fastblend_extension.pairwise_patch_error(
        source_a, nnf_a, source_b, nnf_b, patch_size, pad_size
    )
