# ezsynth/torch_ops/jfa_ops.py
"""
Jump Flooding (JFA) correspondence search for the PyTorch backend.

Replaces PatchMatch's 4-neighbour propagation + random search: for step sizes
``max(H_t, W_t) // 2, ..., 2, 1`` every pixel ``p`` tries the matches of its eight
neighbours ``p + d * step`` shifted back by the same offset (``nnf[p + d] - d``,
the coherent continuation of the neighbour's match). Long steps carry good
matches across the image in ``O(log N)`` passes; a final random search with a
small radius refines detail.

Reference: Rong & Tan, "Jump Flooding in GPU with Applications to Voronoi Diagram
and Distance Transform", I3D 2006.
"""

from __future__ import annotations

import torch

from .microprofile import region as _mp_region
from .omega_ops import (
    omega_mean_map_buffer,
    omega_patch_mean_map_into,
    update_omega_map,
)
from .patchmatch_ops import random_search_step, try_patch_batch

# Eight neighbour directions (dx, dy).
JFA_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)

# (H, W, device_str) -> (ys (H, 1), xs (1, W)) long index vectors
_JFA_GRID_CACHE: dict = {}


def _grid_vectors(height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device))
    entry = _JFA_GRID_CACHE.get(key)
    if entry is None:
        entry = (
            torch.arange(height, device=device).view(height, 1),
            torch.arange(width, device=device).view(1, width),
        )
        _JFA_GRID_CACHE[key] = entry
    return entry


def jfa_step_sizes(height: int, width: int) -> list[int]:
    """Halving step sizes ``max(height, width) // 2 .. 1`` (empty for 1x1)."""
    steps = []
    step = max(height, width) // 2
    while step >= 1:
        steps.append(step)
        step //= 2
    return steps


def jfa_fine_radius(source_h: int, source_w: int) -> int:
    """Initial radius of the random search that follows the flood passes."""
    return max(source_h, source_w) // 16


def jfa_step(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches,
    target_style_patches,
    source_guide_patches,
    target_guide_patches,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    step_size: int,
    mask: torch.Tensor,
    cost_function_mode: int,
    omega_best: float,
    target_modulation_patches=None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> None:
    """One flood pass at ``step_size``; only active (``mask == 255``) pixels change."""
    device = nnf.device
    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_style_patches.shape[:2]
    active = mask == 255
    ys, xs = _grid_vectors(H_t, W_t, device)

    omega_mean_map = None
    if uniformity_weight != 0.0:
        omega_mean_map = omega_mean_map_buffer(device, H_s, W_s)
        omega_patch_mean_map_into(omega_mean_map, omega_map, patch_size)

    old_parts = []
    new_parts = []
    for dx, dy in JFA_DIRECTIONS:
        with _mp_region("jfa:candidates", device):
            ny = ys + dy * step_size
            nx = xs + dx * step_size
            inside = (ny >= 0) & (ny < H_t) & (nx >= 0) & (nx < W_t)
            if not bool(inside.any()):
                continue
            neighbor = nnf[ny.clamp(0, H_t - 1), nx.clamp(0, W_t - 1)]
            neighbor[..., 0] -= dx * step_size
            neighbor[..., 1] -= dy * step_size
            # Pixels whose neighbour is outside the frame propose their own match.
            candidate = torch.where((inside & active).unsqueeze(-1), neighbor, nnf)

        _, new_errors, update = try_patch_batch(
            candidate,
            nnf,
            error_map,
            omega_map,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            cost_function_mode,
            omega_best,
            target_modulation_patches=target_modulation_patches,
            use_bilateral=use_bilateral,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            n_size_step=n_size_step,
            omega_mean_map=omega_mean_map,
        )
        with _mp_region("jfa:apply", device):
            uy, ux = (update & active).nonzero(as_tuple=True)
            if uy.numel() == 0:
                continue
            new_coords = candidate[uy, ux]
            old_parts.append(nnf[uy, ux].clone())
            new_parts.append(new_coords)
            nnf[uy, ux] = new_coords
            error_map[uy, ux] = new_errors[uy, ux]

    if old_parts:
        with _mp_region("jfa:omega_update", device):
            update_omega_map(
                omega_map,
                torch.cat(old_parts, dim=0),
                torch.cat(new_parts, dim=0),
                patch_size,
                clear_device_cache=False,
            )


def jfa_search(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches,
    target_style_patches,
    source_guide_patches,
    target_guide_patches,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    mask: torch.Tensor,
    cost_function_mode: int,
    omega_best: float,
    search_pruning_threshold: float,
    generator: torch.Generator | None = None,
    target_modulation_patches=None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> None:
    """All flood passes (halving steps), then one fine random search."""
    cost_kw = {
        "target_modulation_patches": target_modulation_patches,
        "use_bilateral": use_bilateral,
        "sigma_spatial": sigma_spatial,
        "sigma_color": sigma_color,
        "n_size_step": n_size_step,
    }
    H_t, W_t = nnf.shape[:2]
    for step_size in jfa_step_sizes(H_t, W_t):
        jfa_step(
            nnf,
            error_map,
            omega_map,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            step_size,
            mask,
            cost_function_mode,
            omega_best,
            **cost_kw,
        )

    H_s, W_s = source_style_patches.shape[:2]
    fine_radius = jfa_fine_radius(H_s, W_s)
    if fine_radius >= 1:
        random_search_step(
            nnf,
            error_map,
            omega_map,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            uniformity_weight,
            patch_size,
            fine_radius,
            mask,
            search_pruning_threshold,
            cost_function_mode,
            omega_best,
            generator=generator,
            **cost_kw,
        )
