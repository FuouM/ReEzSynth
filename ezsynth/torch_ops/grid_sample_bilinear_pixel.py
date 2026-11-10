"""Bilinear ``grid_sample`` with (x, y) in pixel coordinates (``align_corners=True``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_sample_bilinear_pixel_coords(
    img: torch.Tensor,
    coords: torch.Tensor,
    *,
    disable_cudnn_on_cuda: bool = False,
    return_oob_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ``img`` at locations given by ``coords[..., :2]`` as x, y pixel indices.

    Coordinates follow the same convention as ``torch.nn.functional.grid_sample``
    with ``align_corners=True`` after mapping from pixel space to ``[-1, 1]``.

    A size-1 axis has no ``[-1, 1]`` mapping (``size - 1 == 0``); it is sampled as a
    one-pixel image with zero padding: the axis gets one zero pixel appended, so
    coordinate ``t`` yields ``value * (1 - |t|)`` for ``|t| < 1`` and 0 beyond. The
    OOB mask is 0 along such an axis (its open interior ``(0, size - 1)`` is empty).
    """
    h, w = img.shape[-2:]
    degenerate_x, degenerate_y = w == 1, h == 1
    if degenerate_x or degenerate_y:
        img = F.pad(img, (0, int(degenerate_x), 0, int(degenerate_y)))
    h_eff, w_eff = img.shape[-2:]
    x_grid, y_grid = coords.split([1, 1], dim=-1)
    x_norm = 2 * x_grid / (w_eff - 1) - 1
    y_norm = 2 * y_grid / (h_eff - 1) - 1
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
    if degenerate_x or degenerate_y:
        mask = torch.zeros_like(mask)
    return sampled, mask.float()
