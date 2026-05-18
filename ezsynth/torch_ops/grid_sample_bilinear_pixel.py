"""Bilinear ``grid_sample`` with (x, y) in pixel coordinates (``align_corners=True``)."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F


def grid_sample_bilinear_pixel_coords(
    img: torch.Tensor,
    coords: torch.Tensor,
    *,
    disable_cudnn_on_cuda: bool = False,
    return_oob_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample ``img`` at locations given by ``coords[..., :2]`` as x, y pixel indices.

    Coordinates follow the same convention as ``torch.nn.functional.grid_sample``
    with ``align_corners=True`` after mapping from pixel space to ``[-1, 1]``.
    """
    h, w = img.shape[-2:]
    x_grid, y_grid = coords.split([1, 1], dim=-1)
    x_norm = 2 * x_grid / (w - 1) - 1
    y_norm = 2 * y_grid / (h - 1) - 1
    grid = torch.cat([x_norm, y_norm], dim=-1)

    def _sample() -> torch.Tensor:
        return F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    if img.device.type == "cuda" and disable_cudnn_on_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            sampled = _sample()
    else:
        sampled = _sample()

    if not return_oob_mask:
        return sampled
    mask = (x_norm > -1) & (y_norm > -1) & (x_norm < 1) & (y_norm < 1)
    return sampled, mask.float()
