"""Taichi backend for FastBlend — init, launch helpers, thin matcher classes.

Kernels live in :mod:`taichi_kernels` (same split as ``ezsynth`` ``taichi_kernels`` /
``taichi_backend``).
"""

from __future__ import annotations

import os
import platform

import taichi as ti
import torch

from . import taichi_kernels as tk
from .patch_matcher_core import FaceBlitPatchMatcherCore, PyramidPatchMatcherCore


def get_ti_arch():
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        return ti.metal
    if torch.cuda.is_available():
        return ti.cuda
    return ti.cpu


_ti_initialized = False


def ensure_ti_init():
    global _ti_initialized
    if not _ti_initialized:
        arch = get_ti_arch()
        ti.init(arch=arch, log_level=ti.WARN, random_seed=42)
        if os.environ.get("FASTBLEND_TAICHI_INIT_VERBOSE", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            print(f"[Taichi FastBlend] Initialized with arch: {arch}")
        _ti_initialized = True


class FastBlendTaichiBackend:
    """Backward-compat: ``.remap_kernel`` etc. point at :mod:`taichi_kernels`."""

    remap_kernel = tk.remap_kernel
    patch_error_kernel = tk.patch_error_kernel
    pairwise_patch_error_kernel = tk.pairwise_patch_error_kernel

    def __init__(self):
        ensure_ti_init()


def _launch_remap(
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
    nnf: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    ensure_ti_init()
    batch_size = source.shape[0]
    target = torch.zeros(
        (batch_size, height + pad_size * 2, width + pad_size * 2, channel),
        dtype=torch.float32,
        device=source.device,
    )
    tk.remap_kernel(
        source,
        nnf,
        target,
        height,
        width,
        channel,
        patch_size,
        pad_size,
    )
    return target


def _launch_patch_error(
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
    source: torch.Tensor,
    nnf: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    ensure_ti_init()
    batch_size = source.shape[0]
    error = torch.zeros(
        (batch_size, height, width), dtype=torch.float32, device=source.device
    )
    tk.patch_error_kernel(
        source,
        nnf,
        target,
        error,
        height,
        width,
        channel,
        patch_size,
        pad_size,
    )
    return error


def _launch_pairwise_patch_error(
    height: int,
    width: int,
    channel: int,
    patch_size: int,
    pad_size: int,
    source: torch.Tensor,
    nnf: torch.Tensor,
) -> torch.Tensor:
    ensure_ti_init()
    source_a, nnf_a = source[0::2].contiguous(), nnf[0::2].contiguous()
    source_b, nnf_b = source[1::2].contiguous(), nnf[1::2].contiguous()
    batch_size = source_a.shape[0]
    error = torch.zeros(
        (batch_size, height, width), dtype=torch.float32, device=source.device
    )
    tk.pairwise_patch_error_kernel(
        source_a,
        nnf_a,
        source_b,
        nnf_b,
        error,
        height,
        width,
        channel,
        patch_size,
        pad_size,
    )
    return error.repeat_interleave(2, dim=0)


def _pyramid_device_string(gpu_id: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{gpu_id}"
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return "mps"
    return "cpu"


class PatchMatcherTaichi(FaceBlitPatchMatcherCore):
    def __init__(self, *args, **kwargs):
        ensure_ti_init()
        super().__init__(*args, **kwargs)

    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return _launch_remap(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            nnf,
            source,
        )

    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return _launch_patch_error(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            source,
            nnf,
            target,
        )

    def _kernel_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        return _launch_pairwise_patch_error(
            self.height,
            self.width,
            self.channel,
            self.patch_size,
            self.pad_size,
            source,
            nnf,
        )


class PyramidPatchMatcherTaichi(PyramidPatchMatcherCore):
    def __init__(
        self,
        image_height,
        image_width,
        channel,
        minimum_patch_size,
        threads_per_block=16,
        num_iter=5,
        gpu_id=0,
        guide_weight=10.0,
        use_mean_target_style=False,
        use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity",
    ):
        self.device = _pyramid_device_string(gpu_id)
        super().__init__(
            image_height,
            image_width,
            channel,
            minimum_patch_size,
            threads_per_block=threads_per_block,
            num_iter=num_iter,
            gpu_id=gpu_id,
            guide_weight=guide_weight,
            use_mean_target_style=use_mean_target_style,
            use_pairwise_patch_error=use_pairwise_patch_error,
            tracking_window_size=tracking_window_size,
            initialize=initialize,
        )

    @property
    def _pyramid_torch_device(self) -> torch.device:
        return torch.device(self.device)

    def _new_level_matcher(self, height: int, width: int, channel: int):
        return PatchMatcherTaichi(
            height,
            width,
            channel,
            minimum_patch_size=self.minimum_patch_size,
            threads_per_block=self.threads_per_block,
            num_iter=self.num_iter,
            gpu_id=self.gpu_id,
            guide_weight=self.guide_weight,
            use_mean_target_style=self.use_mean_target_style,
            use_pairwise_patch_error=self.use_pairwise_patch_error,
            tracking_window_size=self.tracking_window_size,
        )

    def apply_nnf_to_image(
        self, nnf: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        pm = self.patch_matchers[-1]
        return pm.unpad_image(pm.apply_nnf_to_image(nnf, pm.pad_image(image)))
