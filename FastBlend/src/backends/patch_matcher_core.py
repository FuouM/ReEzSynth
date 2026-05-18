"""Shared FaceBlit-style patch matching (Python loop + backend-specific kernels).

Comparison to ``ebsynth_extension/patchmatch.cu`` (and ``ezsynth/torch_ops/patchmatch_ops``):

**Not the same algorithm.** EBSynth-style PatchMatch uses per-pixel ``try_patch`` with an
omega (uniformity) map, weighted SSD or NCC, optional bilateral/modulation, search
pruning, and GPU kernels that alternate forward/backward grid sweeps. FastBlend follows
a FaceBlit-style flow: batched Torch propagation (four shifted neighbors in random
order), a fixed count of full-field random jitter steps, optional temporal ``track``,
and a simple ``guide_weight * SSD_guide + SSD_style`` cost on float ``HWC`` tensors.

**Propagation:** EBSynth reads a spatial neighbor's match and applies a direction-dependent
offset (odd/even passes, ``torch.roll``-equivalent logic in PyTorch). FastBlend uses
``neighboor_step`` (concatenate shifted NNF + integer bump on one channel). Those rules
are not equivalent.

**Random search:** EBSynth halves a radius around the *current* match per pixel
(``while (r >= 1) { ... r /= 2; }``) with CURAND. FastBlend draws i.i.d. offsets in
``[-random_search_range, +range]`` for all pixels, repeated ``random_search_steps``
times.

**NNF layout:** In this codebase, FastBlend stores ``nnf[..., 0]`` = row (y),
``nnf[..., 1]`` = column (x). The native extension uses ``nnf[..., 0]`` = source x,
``nnf[..., 1]`` = source y (see ``patchmatch.cu`` / ``patchmatch_ops``). Conventions are
internally consistent per stack but are **not** interchangeable without transposing.

Because of cost model, omega, sweep order, random search, and NNF channel order, aligning
FastBlend with EBSynth would be a substantive reimplementation, not a small sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class PatchMatchTensors:
    """Padded guide + style tensors for one pyramid level or NNF iteration."""

    source_guide: torch.Tensor
    target_guide: torch.Tensor
    source_style: torch.Tensor
    target_style: torch.Tensor


class FaceBlitPatchMatcherCore(ABC):
    """Propagation / random search / tracking in PyTorch; subclasses implement remap + SSD kernels."""

    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        minimum_patch_size: int,
        threads_per_block: int = 16,
        num_iter: int = 5,
        gpu_id: int = 0,
        guide_weight: float = 10.0,
        random_search_steps: int = 3,
        random_search_range: int = 4,
        use_mean_target_style: bool = False,
        use_pairwise_patch_error: bool = False,
        tracking_window_size: int = 0,
    ):
        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.random_search_steps = random_search_steps
        self.random_search_range = random_search_range
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        self.patch_size_list = [minimum_patch_size + i * 2 for i in range(num_iter)][
            ::-1
        ]
        self.pad_size = self.patch_size_list[0] // 2
        self.patch_size = self.patch_size_list[0]

        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block,
        )
        self.block = (threads_per_block, threads_per_block)

    @abstractmethod
    def _kernel_remap(self, nnf: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Return padded target (B, H+2p, W+2p, C)."""

    @abstractmethod
    def _kernel_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return per-pixel patch SSD map (B, H, W)."""

    @abstractmethod
    def _kernel_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        """Interleaved batch pairwise error; length matches ``source`` batch."""

    def _level_execution_context(self) -> ContextManager[None]:
        return nullcontext()

    def pad_image(self, image: torch.Tensor) -> torch.Tensor:
        pad_size = self.pad_size
        return (
            torch.nn.functional.pad(
                image.permute(0, 3, 1, 2),
                (pad_size, pad_size, pad_size, pad_size),
                mode="reflect",
            )
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def unpad_image(self, image: torch.Tensor) -> torch.Tensor:
        pad_size = self.pad_size
        return image[:, pad_size:-pad_size, pad_size:-pad_size, :].contiguous()

    def apply_nnf_to_image(
        self, nnf: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        return self._kernel_remap(nnf, source)

    def get_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self._kernel_patch_error(source, nnf, target)

    def get_pairwise_patch_error(
        self, source: torch.Tensor, nnf: torch.Tensor
    ) -> torch.Tensor:
        return self._kernel_pairwise_patch_error(source, nnf)

    def get_error(self, tensors: PatchMatchTensors, nnf: torch.Tensor) -> torch.Tensor:
        error_guide = self.get_patch_error(
            tensors.source_guide, nnf, tensors.target_guide
        )
        if self.use_mean_target_style:
            target_style = self.apply_nnf_to_image(nnf, tensors.source_style)
            target_style = target_style.mean(dim=0, keepdim=True)
            target_style = target_style.repeat(tensors.source_guide.shape[0], 1, 1, 1)
        else:
            target_style = tensors.target_style
        if self.use_pairwise_patch_error:
            error_style = self.get_pairwise_patch_error(tensors.source_style, nnf)
        else:
            error_style = self.get_patch_error(tensors.source_style, nnf, target_style)
        return error_guide * self.guide_weight + error_style

    def clamp_bound(self, nnf: torch.Tensor) -> torch.Tensor:
        nnf[..., 0] = torch.clamp(nnf[..., 0], 0, self.height - 1)
        nnf[..., 1] = torch.clamp(nnf[..., 1], 0, self.width - 1)
        return nnf

    def random_step(self, nnf: torch.Tensor, r: int) -> torch.Tensor:
        batch_size = nnf.shape[0]
        step = torch.randint(
            -r,
            r + 1,
            (batch_size, self.height, self.width, 2),
            dtype=torch.int32,
            device=nnf.device,
        )
        return self.clamp_bound(nnf + step)

    def neighboor_step(self, nnf: torch.Tensor, d: int) -> torch.Tensor:
        if d == 0:
            upd_nnf = torch.cat([nnf[:, :1, :, :], nnf[:, :-1, :, :]], dim=1)
            upd_nnf[:, :, :, 0] += 1
        elif d == 1:
            upd_nnf = torch.cat([nnf[:, :, :1, :], nnf[:, :, :-1, :]], dim=2)
            upd_nnf[:, :, :, 1] += 1
        elif d == 2:
            upd_nnf = torch.cat([nnf[:, 1:, :, :], nnf[:, -1:, :, :]], dim=1)
            upd_nnf[:, :, :, 0] -= 1
        elif d == 3:
            upd_nnf = torch.cat([nnf[:, :, 1:, :], nnf[:, :, -1:, :]], dim=2)
            upd_nnf[:, :, :, 1] -= 1
        else:
            raise ValueError(f"invalid direction {d}")
        return self.clamp_bound(upd_nnf)

    def shift_nnf(self, nnf: torch.Tensor, d: int) -> torch.Tensor:
        if d > 0:
            d = min(nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[d:], nnf[-1:].repeat(d, 1, 1, 1)], dim=0)
        else:
            d = max(-nnf.shape[0], d)
            upd_nnf = torch.cat([nnf[:1].repeat(-d, 1, 1, 1), nnf[:-d]], dim=0)
        return upd_nnf

    def track_step(self, nnf: torch.Tensor, d: int) -> torch.Tensor:
        if self.use_pairwise_patch_error:
            upd_nnf = torch.zeros_like(nnf)
            upd_nnf[0::2] = self.shift_nnf(nnf[0::2], d)
            upd_nnf[1::2] = self.shift_nnf(nnf[1::2], d)
        else:
            upd_nnf = self.shift_nnf(nnf, d)
        return upd_nnf

    def update(
        self,
        tensors: PatchMatchTensors,
        nnf: torch.Tensor,
        err: torch.Tensor,
        upd_nnf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        upd_err = self.get_error(tensors, upd_nnf)
        upd_idx = upd_err < err
        # Boolean masked assignment can mis-count indices on MPS; where is stable.
        return (
            torch.where(upd_idx.unsqueeze(-1), upd_nnf, nnf),
            torch.where(upd_idx, upd_err, err),
        )

    def propagation(
        self,
        tensors: PatchMatchTensors,
        nnf: torch.Tensor,
        err: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        directions = torch.randperm(4, device=nnf.device)
        for d in directions:
            upd_nnf = self.neighboor_step(nnf, int(d))
            nnf, err = self.update(tensors, nnf, err, upd_nnf)
        return nnf, err

    def random_search(
        self,
        tensors: PatchMatchTensors,
        nnf: torch.Tensor,
        err: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.random_search_steps):
            upd_nnf = self.random_step(nnf, self.random_search_range)
            nnf, err = self.update(tensors, nnf, err, upd_nnf)
        return nnf, err

    def track(
        self,
        tensors: PatchMatchTensors,
        nnf: torch.Tensor,
        err: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for d in range(1, self.tracking_window_size + 1):
            upd_nnf = self.track_step(nnf, d)
            nnf, err = self.update(tensors, nnf, err, upd_nnf)
            upd_nnf = self.track_step(nnf, -d)
            nnf, err = self.update(tensors, nnf, err, upd_nnf)
        return nnf, err

    def iteration(
        self,
        tensors: PatchMatchTensors,
        nnf: torch.Tensor,
        err: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nnf, err = self.propagation(tensors, nnf, err)
        nnf, err = self.random_search(tensors, nnf, err)
        nnf, err = self.track(tensors, nnf, err)
        return nnf, err

    def estimate_nnf(
        self,
        source_guide: torch.Tensor,
        target_guide: torch.Tensor,
        source_style: torch.Tensor,
        nnf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._level_execution_context():
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)

            for it in range(self.num_iter):
                self.patch_size = self.patch_size_list[it]
                target_style = self.apply_nnf_to_image(nnf, source_style)
                tensors = PatchMatchTensors(
                    source_guide, target_guide, source_style, target_style
                )
                err = self.get_error(tensors, nnf)
                nnf, err = self.iteration(tensors, nnf, err)

            target_style = self.unpad_image(self.apply_nnf_to_image(nnf, source_style))
        return nnf, target_style


class PyramidPatchMatcherCore(ABC):
    """Image pyramid wiring shared by CUDA, CuPy, and Taichi backends."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channel: int,
        minimum_patch_size: int,
        threads_per_block: int = 16,
        num_iter: int = 5,
        gpu_id: int = 0,
        guide_weight: float = 10.0,
        use_mean_target_style: bool = False,
        use_pairwise_patch_error: bool = False,
        tracking_window_size: int = 0,
        initialize: str = "identity",
    ):
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size
        self.initialize = initialize

        maximum_patch_size = minimum_patch_size + (num_iter - 1) * 2
        self.pyramid_level = int(
            np.log2(min(image_height, image_width) / maximum_patch_size)
        )
        self.pyramid_heights: list[int] = []
        self.pyramid_widths: list[int] = []
        self.patch_matchers: list[FaceBlitPatchMatcherCore] = []

        for level in range(self.pyramid_level):
            height = image_height // (2 ** (self.pyramid_level - 1 - level))
            width = image_width // (2 ** (self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(self._new_level_matcher(height, width, channel))

    @abstractmethod
    def _new_level_matcher(
        self, height: int, width: int, channel: int
    ) -> FaceBlitPatchMatcherCore: ...

    @property
    @abstractmethod
    def _pyramid_torch_device(self) -> torch.device: ...

    def _pyramid_run_context(self) -> ContextManager[None]:
        return nullcontext()

    def _coerce_input(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return (
                torch.from_numpy(x).float().to(self._pyramid_torch_device).contiguous()
            )
        return x

    def resample_image(self, images: torch.Tensor, level: int) -> torch.Tensor:
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        images = images.permute(0, 3, 1, 2)
        # MPS implements ``mode="area"`` via adaptive_avg_pool2d, which requires
        # input dimensions divisible by output sizes (often false for pyramids).
        dev = images.device
        if dev.type == "mps":
            images_resample = torch.nn.functional.interpolate(
                images.cpu(), size=(height, width), mode="area"
            ).to(dev)
        else:
            images_resample = torch.nn.functional.interpolate(
                images, size=(height, width), mode="area"
            )
        return images_resample.permute(0, 2, 3, 1).contiguous()

    def initialize_nnf(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        device = self._pyramid_torch_device
        if self.initialize == "random":
            nnf = torch.stack(
                [
                    torch.randint(
                        0,
                        height,
                        (batch_size, height, width),
                        device=device,
                        dtype=torch.int32,
                    ),
                    torch.randint(
                        0,
                        width,
                        (batch_size, height, width),
                        device=device,
                        dtype=torch.int32,
                    ),
                ],
                dim=3,
            )
        elif self.initialize == "identity":
            y_coords = (
                torch.arange(height, device=device, dtype=torch.int32)
                .view(height, 1)
                .repeat(1, width)
            )
            x_coords = (
                torch.arange(width, device=device, dtype=torch.int32)
                .view(1, width)
                .repeat(height, 1)
            )
            nnf = torch.stack([y_coords, x_coords], dim=2)
            nnf = nnf.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        else:
            raise NotImplementedError(self.initialize)
        return nnf.contiguous()

    def update_nnf(self, nnf: torch.Tensor, level: int) -> torch.Tensor:
        nnf = nnf.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) * 2
        nnf[:, :, 1::2, 0] += 1
        nnf[:, 1::2, :, 1] += 1

        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[1] or width != nnf.shape[2]:
            nnf = nnf.permute(0, 3, 1, 2).float()
            nnf = torch.nn.functional.interpolate(
                nnf, size=(height, width), mode="bilinear", align_corners=False
            )
            nnf = nnf.permute(0, 2, 3, 1).int()
            # Match ``clamp_bound``: channel 0 = row (y) ∈ [0, height-1], channel 1 = col (x) ∈ [0, width-1].
            nnf[..., 0] = torch.clamp(nnf[..., 0], 0, height - 1)
            nnf[..., 1] = torch.clamp(nnf[..., 1], 0, width - 1)

        return nnf.contiguous()

    @abstractmethod
    def apply_nnf_to_image(
        self, nnf: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        """Backend-specific (padded vs unpadded) final remap."""
        ...

    def estimate_nnf(
        self,
        source_guide: Union[np.ndarray, torch.Tensor],
        target_guide: Union[np.ndarray, torch.Tensor],
        source_style: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._pyramid_run_context():
            source_guide = self._coerce_input(source_guide)
            target_guide = self._coerce_input(target_guide)
            source_style = self._coerce_input(source_style)

            nnf: Optional[torch.Tensor] = None
            for level in range(self.pyramid_level):
                if level == 0:
                    nnf = self.initialize_nnf(
                        source_guide.shape[0],
                        self.pyramid_heights[0],
                        self.pyramid_widths[0],
                    )
                else:
                    nnf = self.update_nnf(nnf, level)

                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)

                assert nnf is not None
                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )

        assert nnf is not None
        return nnf, target_style
