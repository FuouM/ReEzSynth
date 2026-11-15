# ezsynth/torch_ops/multicand_ops.py
"""
Multi-candidate (k-NN) PatchMatch for the PyTorch backend.

Each target pixel keeps its usual best match (``nnf`` / ``error_map``, ranked by
patch cost plus the uniformity term) and ``k - 1`` *alternative* matches ranked by
patch cost only (:class:`CandidateSet`). Alternatives are distinct from each other
and from the best match. They:

- seed propagation: every neighbour alternative is tried as a candidate, not only
  the neighbour's best match;
- seed random search: every radius also samples around each alternative;
- are promoted when they beat the best match after the target image changes;
- blend into weighted voting with the usual ``1 / (1 + error)`` weights.

Displaced best matches and rejected candidates are offered to the alternative set,
so the k matches per pixel are the best distinct candidates seen so far.
"""

from __future__ import annotations

import torch

from .microprofile import region as _mp_region
from .omega_ops import (
    omega_mean_map_buffer,
    omega_patch_mean_map_into,
    update_omega_map,
)
from .patchmatch_ops import (
    _get_offset_tensor,
    _prop_neighbor_nnf_buf,
    grid_patch_errors,
    try_patch_batch,
)


class CandidateSet:
    """
    ``slots`` alternative matches per target pixel.

    ``nnf``: (H, W, slots, 2) int32 source coordinates; ``errors``: (H, W, slots)
    float32 patch costs against the current target, ``inf`` marks an empty slot.
    """

    def __init__(self, height: int, width: int, slots: int, nnf: torch.Tensor):
        if slots < 1:
            raise ValueError("CandidateSet needs at least one alternative slot")
        self.slots = slots
        self.nnf = nnf.unsqueeze(2).repeat(1, 1, slots, 1).contiguous()
        self.errors = torch.full(
            (height, width, slots), float("inf"), dtype=torch.float32, device=nnf.device
        )

    def slot(self, index: int) -> torch.Tensor:
        """(H, W, 2) contiguous copy of one slot's coordinates."""
        return self.nnf[:, :, index].contiguous()

    def offer(
        self,
        ys: torch.Tensor,
        xs: torch.Tensor,
        coords: torch.Tensor,
        errors: torch.Tensor,
        primary_nnf: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Offer one candidate per listed pixel (``ys``/``xs`` must be unique).

        A candidate replaces the pixel's worst alternative when its error is lower
        and it differs from the stored alternatives and the best match.
        """
        with _mp_region("mc:offer", coords.device):
            slots = self.nnf[ys, xs]
            slot_errors = self.errors[ys, xs]
            coords = coords.to(slots.dtype)
            duplicate = (slots == coords.unsqueeze(1)).all(dim=-1).any(dim=-1)
            duplicate |= (primary_nnf[ys, xs] == coords).all(dim=-1)
            worst_errors, worst_slots = slot_errors.max(dim=1)
            take = (errors < worst_errors) & ~duplicate & mask
            picked = take.nonzero(as_tuple=True)[0]
            if picked.numel() == 0:
                return
            py, px, ps = ys[picked], xs[picked], worst_slots[picked]
            self.nnf[py, px, ps] = coords[picked]
            self.errors[py, px, ps] = errors[picked].to(self.errors.dtype)

    def offer_grid(
        self,
        coords: torch.Tensor,
        errors: torch.Tensor,
        mask: torch.Tensor,
        primary_nnf: torch.Tensor,
    ) -> None:
        """:meth:`offer` with an (H, W, 2) candidate grid restricted to ``mask``."""
        ys, xs = mask.nonzero(as_tuple=True)
        if ys.numel() == 0:
            return
        self.offer(
            ys,
            xs,
            coords[ys, xs],
            errors[ys, xs],
            primary_nnf,
            torch.ones_like(ys, dtype=torch.bool),
        )

    def refresh(self, errors_of) -> None:
        """Re-score stored alternatives with ``errors_of(coords_grid)`` (new target)."""
        for index in range(self.slots):
            present = torch.isfinite(self.errors[:, :, index])
            fresh = errors_of(self.slot(index))
            self.errors[:, :, index] = torch.where(
                present, fresh, self.errors[:, :, index]
            )


def _cost_kwargs(
    use_bilateral: bool, sigma_spatial: float, sigma_color: float, n_size_step: int
) -> dict:
    return {
        "use_bilateral": use_bilateral,
        "sigma_spatial": sigma_spatial,
        "sigma_color": sigma_color,
        "n_size_step": n_size_step,
    }


def refresh_and_promote(
    alternatives: CandidateSet,
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
    target_modulation_patches=None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> None:
    """
    Re-score alternatives against the current target, then swap an alternative
    with the best match wherever it now has the lower total (patch + uniformity)
    cost. ``error_map`` must already hold the best matches' current patch costs.
    """
    device = nnf.device
    cost_kw = _cost_kwargs(use_bilateral, sigma_spatial, sigma_color, n_size_step)

    def errors_of(coords: torch.Tensor) -> torch.Tensor:
        return grid_patch_errors(
            coords,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            patch_size,
            cost_function_mode,
            target_modulation_patches,
            **cost_kw,
            region="mc:refresh",
        )

    with _mp_region("mc:refresh", device):
        alternatives.refresh(errors_of)

    active = mask == 255
    H_s, W_s = source_style_patches.shape[:2]
    omega_mean_map = None
    if uniformity_weight != 0.0:
        omega_mean_map = omega_mean_map_buffer(device, H_s, W_s)
        omega_patch_mean_map_into(omega_mean_map, omega_map, patch_size)

    old_parts = []
    new_parts = []
    for index in range(alternatives.slots):
        candidate = alternatives.slot(index)
        candidate_errors = alternatives.errors[:, :, index].contiguous()
        _, _, update = try_patch_batch(
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
            omega_mean_map=omega_mean_map,
            candidate_patch_errors=candidate_errors,
            **cost_kw,
        )
        ys, xs = (update & active).nonzero(as_tuple=True)
        if ys.numel() == 0:
            continue
        old_coords = nnf[ys, xs].clone()
        old_errors = error_map[ys, xs].clone()
        new_coords = candidate[ys, xs]
        nnf[ys, xs] = new_coords
        error_map[ys, xs] = candidate_errors[ys, xs]
        alternatives.nnf[ys, xs, index] = old_coords
        alternatives.errors[ys, xs, index] = old_errors
        old_parts.append(old_coords)
        new_parts.append(new_coords)

    if old_parts:
        update_omega_map(
            omega_map,
            torch.cat(old_parts, dim=0),
            torch.cat(new_parts, dim=0),
            patch_size,
            clear_device_cache=False,
        )


def _shift_neighbor(source: torch.Tensor, buf: torch.Tensor, axis: int, shift: int):
    """``buf[y, x] = source[y, x - shift]`` along ``axis`` with edge replication."""
    if axis == 1:
        if shift == 1:
            torch.cat((source[:, :1], source[:, :-1]), dim=1, out=buf)
        else:
            torch.cat((source[:, 1:], source[:, -1:]), dim=1, out=buf)
    elif shift == 1:
        torch.cat((source[:1], source[:-1]), dim=0, out=buf)
    else:
        torch.cat((source[1:], source[-1:]), dim=0, out=buf)
    return buf


def propagation_step_multicand(
    alternatives: CandidateSet,
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
    is_odd: bool,
    mask: torch.Tensor,
    cost_function_mode: int,
    omega_best: float,
    target_modulation_patches=None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> None:
    """
    Propagation over the best match and every alternative of the left/top
    (``is_odd``) or right/bottom neighbour, same scan convention as
    :func:`~ezsynth.torch_ops.patchmatch_ops.propagation_step`.
    """
    device = nnf.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2
    active = mask == 255
    cost_kw = _cost_kwargs(use_bilateral, sigma_spatial, sigma_color, n_size_step)
    shift = 1 if is_odd else -1

    omega_mean_map = None
    if uniformity_weight != 0.0:
        omega_mean_map = omega_mean_map_buffer(device, H_s, W_s)
        omega_patch_mean_map_into(omega_mean_map, omega_map, patch_size)

    neighbor = _prop_neighbor_nnf_buf(nnf)
    present_buf = torch.empty(nnf.shape[:2], dtype=torch.bool, device=device)
    old_parts = []
    new_parts = []
    for axis, offset in ((1, (shift, 0)), (0, (0, shift))):
        offset_t = _get_offset_tensor(offset, device)
        for index in range(-1, alternatives.slots):
            with _mp_region("mc:prop_candidates", device):
                if index < 0:
                    candidate = _shift_neighbor(nnf, neighbor, axis, shift) + offset_t
                else:
                    candidate = (
                        _shift_neighbor(alternatives.slot(index), neighbor, axis, shift)
                        + offset_t
                    )
                    present = _shift_neighbor(
                        torch.isfinite(alternatives.errors[:, :, index]),
                        present_buf,
                        axis,
                        shift,
                    )
                    # Empty neighbour slots propose the current match (a no-op).
                    candidate = torch.where(present.unsqueeze(-1), candidate, nnf)
                candidate = candidate.to(nnf.dtype)
                valid = (
                    (candidate[..., 0] >= r)
                    & (candidate[..., 0] < W_s - r)
                    & (candidate[..., 1] >= r)
                    & (candidate[..., 1] < H_s - r)
                )
            candidate_errors = grid_patch_errors(
                candidate,
                source_style_patches,
                target_style_patches,
                source_guide_patches,
                target_guide_patches,
                style_weights,
                guide_weights,
                patch_size,
                cost_function_mode,
                target_modulation_patches,
                **cost_kw,
                region="mc:prop",
            )
            _, _, update = try_patch_batch(
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
                omega_mean_map=omega_mean_map,
                candidate_patch_errors=candidate_errors,
                **cost_kw,
            )
            with _mp_region("mc:prop_apply", device):
                update &= active
                offer_coords = torch.where(update.unsqueeze(-1), nnf, candidate)
                offer_errors = torch.where(update, error_map, candidate_errors)
                ys, xs = update.nonzero(as_tuple=True)
                if ys.numel() > 0:
                    new_coords = candidate[ys, xs]
                    old_parts.append(nnf[ys, xs].clone())
                    new_parts.append(new_coords)
                    nnf[ys, xs] = new_coords
                    error_map[ys, xs] = candidate_errors[ys, xs]
                alternatives.offer_grid(
                    offer_coords, offer_errors, active & (update | valid), nnf
                )

    if old_parts:
        update_omega_map(
            omega_map,
            torch.cat(old_parts, dim=0),
            torch.cat(new_parts, dim=0),
            patch_size,
            clear_device_cache=False,
        )


def initial_candidate_set(nnf: torch.Tensor, multicand_k: int) -> CandidateSet:
    """Empty alternative set for ``multicand_k`` matches per pixel (k - 1 slots)."""
    height, width = nnf.shape[:2]
    return CandidateSet(height, width, multicand_k - 1, nnf)
