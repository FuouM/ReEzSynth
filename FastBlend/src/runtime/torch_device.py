"""PyTorch device selection for balanced-mode staging."""

import platform

import torch


def balanced_mode_torch_device(resolved_backend: str, gpu_id: int) -> torch.device:
    """
    Device for ``FrameCache`` / ``GPUAccumulator`` tensors before ``estimate_nnf``.

    CUDA and CuPy paths use CUDA tensors. Taichi matches ``PyramidPatchMatcherTaichi``.
    """
    if resolved_backend == "cuda":
        return torch.device("cuda", gpu_id)
    if resolved_backend == "cupy":
        return torch.device("cuda", gpu_id)
    if resolved_backend == "taichi":
        if torch.cuda.is_available():
            return torch.device("cuda", gpu_id)
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")
