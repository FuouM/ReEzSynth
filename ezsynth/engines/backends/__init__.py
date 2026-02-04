# ezsynth/engines/backends/__init__.py
# CUDA backend disabled - using JIT compilation instead via ebsynth_torch_loader
# To use CUDA backend, the extension will be compiled at runtime on first use
# Import CudaBackend only if CUDA extension is available
from ...consts import CUDA_EXTENSION_AVAILABLE
from .base import BaseSynthesisBackend
from .pytorch_backend import PyTorchBackend

if CUDA_EXTENSION_AVAILABLE:
    from .cuda_backend import CudaBackend
else:
    # CudaBackend not available without extension, PyTorchBackend handles CUDA via torch_ops
    CudaBackend = None

# Import TaichiBackend (always available if taichi is installed)
try:
    from .taichi_backend import TaichiBackend
except ImportError:
    TaichiBackend = None

__all__ = ["BaseSynthesisBackend", "CudaBackend", "PyTorchBackend", "TaichiBackend"]
