# ezsynth/engines/backends/__init__.py
from .base import BaseSynthesisBackend
from .pytorch_backend import PyTorchBackend

# CUDA backend disabled - using JIT compilation instead via ebsynth_torch_loader
# To use CUDA backend, the extension will be compiled at runtime on first use
# Import CudaBackend only if CUDA extension is available
from ...consts import CUDA_EXTENSION_AVAILABLE

if CUDA_EXTENSION_AVAILABLE:
    from .cuda_backend import CudaBackend
else:
    # CudaBackend not available without extension, PyTorchBackend handles CUDA via torch_ops
    CudaBackend = None

__all__ = ["BaseSynthesisBackend", "CudaBackend", "PyTorchBackend"]
