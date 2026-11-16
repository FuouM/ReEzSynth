"""
PatchMatch algorithm operations for EBSynth.

This module implements the core PatchMatch operations:
- try_patch_batch: Evaluate candidate patches and update NNF
- propagation_step: Spatial coherence via neighbor propagation
- random_search_step: Exploration via random offsets
"""

from typing import TYPE_CHECKING

import torch

from ezsynth.consts import COST_FUNCTION_NCC, COST_FUNCTION_SAD

from .microprofile import region as _mp_region
from .omega_ops import (
    gather_omega_scores_from_patches,
    gather_omega_scores_pair,
    omega_mean_map_buffer,
    omega_patch_mean_map_into,
    update_omega_map,
)
from .patch_ops import (
    ImagePatches,
    compute_patch_cost_sampled,
    compute_patch_ncc_vectorized,
    compute_patch_ssd_style_guide_fused,
    compute_patch_ssd_vectorized,
)

if TYPE_CHECKING:
    from .multicand_ops import CandidateSet

# OPTIMIZATION: Cache for small offset tensors to avoid repeated allocations
_OFFSET_TENSOR_CACHE = {}


# Reused random-search buffers keyed by device / NNF dtype. The active set is
# usually full-frame, so this removes repeated temporary allocations per radius.
_RANDOM_SEARCH_BUFFER_CACHE: dict = {}

# Reused neighbor shifted NNF for propagation (``torch.cat(..., out=buf)``).
_PROP_NEIGHBOR_NNF_BUF: dict = {}

# (H_t, W_t, device_str) -> (ys, xs) long row/column of every flattened target pixel.
_TARGET_GRID_CACHE: dict = {}

# device_str -> (N, 2) long ``(x=i, y=0)`` coordinates used to address chunk-local grids.
_LOCAL_NNF_CACHE: dict = {}


def _prop_neighbor_nnf_buf(nnf: torch.Tensor) -> torch.Tensor:
    """Cached buffer for shifted neighbor ``nnf`` (same layout as ``nnf``)."""
    key = (nnf.shape, str(nnf.device), nnf.dtype)
    buf = _PROP_NEIGHBOR_NNF_BUF.get(key)
    if buf is None or buf.shape != nnf.shape:
        buf = torch.empty_like(nnf)
        _PROP_NEIGHBOR_NNF_BUF[key] = buf
    return buf


def _get_offset_tensor(offset_tuple, device):
    """Get or create cached offset tensor."""
    key = (offset_tuple, str(device))
    if key not in _OFFSET_TENSOR_CACHE:
        _OFFSET_TENSOR_CACHE[key] = torch.tensor(list(offset_tuple), device=device)
    return _OFFSET_TENSOR_CACHE[key]


def _random_search_buffers(
    num_active: int, nnf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (str(nnf.device), nnf.dtype)
    entry = _RANDOM_SEARCH_BUFFER_CACHE.get(key)
    if entry is None or entry[0].shape[0] < num_active:
        rand_x = torch.empty((num_active,), dtype=nnf.dtype, device=nnf.device)
        rand_y = torch.empty((num_active,), dtype=nnf.dtype, device=nnf.device)
        candidate = torch.empty((num_active, 2), dtype=nnf.dtype, device=nnf.device)
        entry = (rand_x, rand_y, candidate)
        _RANDOM_SEARCH_BUFFER_CACHE[key] = entry
    rand_x, rand_y, candidate = entry
    return rand_x[:num_active], rand_y[:num_active], candidate[:num_active]


def _target_grid_coords(
    height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device))
    entry = _TARGET_GRID_CACHE.get(key)
    if entry is None:
        flat = torch.arange(height * width, device=device)
        entry = (flat // width, flat % width)
        _TARGET_GRID_CACHE[key] = entry
    return entry


def _local_nnf(count: int, device: torch.device) -> torch.Tensor:
    key = str(device)
    buf = _LOCAL_NNF_CACHE.get(key)
    if buf is None or buf.shape[0] < count:
        buf = torch.zeros((count, 2), dtype=torch.long, device=device)
        buf[:, 0] = torch.arange(count, device=device)
        _LOCAL_NNF_CACHE[key] = buf
    return buf[:count]


def _uses_image_patches(*grids) -> bool:
    return any(isinstance(grid, ImagePatches) for grid in grids)


def _patch_errors(
    source_style_patches,
    target_style_patches: torch.Tensor,
    source_guide_patches,
    target_guide_patches: torch.Tensor,
    candidate_coords: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    patch_size: int,
    cost_function_mode: int,
    target_modulation_patches: torch.Tensor | None,
    ssd_kw: dict,
    *,
    region: str,
) -> torch.Tensor:
    """SSD/NCC/SAD patch cost of ``candidate_coords`` against materialized grids."""
    dev = candidate_coords.device
    if ssd_kw["use_bilateral"] or ssd_kw["n_size_step"] > 1:
        with _mp_region(f"{region}:sampled", dev):
            return compute_patch_cost_sampled(
                source_style_patches,
                target_style_patches,
                source_guide_patches,
                target_guide_patches,
                candidate_coords,
                style_weights,
                guide_weights,
                target_modulation_patches,
                cost_function_mode=cost_function_mode,
                **ssd_kw,
            )
    if cost_function_mode == COST_FUNCTION_NCC:
        with _mp_region(f"{region}:ncc", dev):
            return compute_patch_ncc_vectorized(
                source_style_patches,
                target_style_patches,
                source_guide_patches,
                target_guide_patches,
                candidate_coords,
                patch_size,
                style_weights,
                guide_weights,
                target_modulation_patches,
            )
    has_guide = source_guide_patches.numel() > 0
    l1 = cost_function_mode == COST_FUNCTION_SAD
    if has_guide:
        with _mp_region(f"{region}:ssd_fused", dev):
            return compute_patch_ssd_style_guide_fused(
                source_style_patches,
                target_style_patches,
                source_guide_patches,
                target_guide_patches,
                candidate_coords,
                style_weights,
                guide_weights,
                target_modulation_patches,
                l1=l1,
            )
    with _mp_region(f"{region}:ssd_style", dev):
        return compute_patch_ssd_vectorized(
            source_style_patches,
            target_style_patches,
            candidate_coords,
            style_weights,
            None,
            l1=l1,
        )


def _gather_chunk(grid, ys: torch.Tensor, xs: torch.Tensor, indices: dict, role: str):
    """Patches of ``grid`` centered at (ys, xs); pixel indices are shared per grid size."""
    if isinstance(grid, ImagePatches):
        key = (role, grid.shape[0], grid.shape[1])
        index = indices.get(key)
        if index is None:
            index = grid.pixel_index(ys, xs)
            indices[key] = index
        return grid.gather_index(index)
    return grid[ys, xs]


def _dense_pixel_cost(
    source: ImagePatches,
    target: ImagePatches,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    weights: torch.Tensor,
    modulation: ImagePatches | None,
    l1: bool,
) -> torch.Tensor:
    """Weighted dense SSD/SAD straight from (n, ps*ps, C) pixel gathers.

    Same terms as :func:`compute_patch_ssd_style_guide_fused` on unfolded patches,
    without building the channel-major patch rows or re-indexing them.
    """
    diff = source.pixels[source_index].float()
    diff.sub_(target.pixels[target_index])
    if l1:
        diff.abs_()
    else:
        diff.square_()
    if modulation is None:
        return (diff.sum(dim=1) * weights.view(1, -1)).sum(dim=1)
    # Same products and channel-major reduction as the materialized kernel, so both
    # strategies give identical errors.
    diff.mul_(modulation.pixels[target_index].float().div_(255.0))
    diff.mul_(weights.view(1, 1, -1))
    return diff.transpose(1, 2).contiguous().sum(dim=(1, 2))


def _dense_ncc_style_cost(
    source: ImagePatches,
    target: ImagePatches,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    style_weights: torch.Tensor,
) -> torch.Tensor:
    """Style term of :func:`compute_patch_ncc_vectorized` from pixel gathers."""
    epsilon = 1e-6
    s_vals = source.pixels[source_index].float().mean(dim=2)
    t_vals = target.pixels[target_index].float().mean(dim=2)
    mean_s = s_vals.mean(dim=1, keepdim=True)
    mean_t = t_vals.mean(dim=1, keepdim=True)
    std_s = s_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon
    std_t = t_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon
    cov = ((s_vals - mean_s) * (t_vals - mean_t)).mean(dim=1)
    ncc = (cov / (std_s.squeeze(1) * std_t.squeeze(1))).clamp(-1.0, 1.0)
    return (1.0 - ncc) * style_weights[0] * float(source_index.shape[1])


def _dense_bounded_errors(
    sy: torch.Tensor,
    sx: torch.Tensor,
    ty: torch.Tensor,
    tx: torch.Tensor,
    grids: tuple,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    cost_function_mode: int,
    has_guide: bool,
) -> torch.Tensor:
    """Dense (no bilateral, ``n_size_step == 1``) patch cost of one chunk."""
    source_style, target_style, source_guide, target_guide, modulation = grids
    source_index = source_style.pixel_index(sy, sx)
    target_index = target_style.pixel_index(ty, tx)
    l1 = cost_function_mode == COST_FUNCTION_SAD
    if cost_function_mode == COST_FUNCTION_NCC:
        error = _dense_ncc_style_cost(
            source_style, target_style, source_index, target_index, style_weights
        )
    else:
        error = _dense_pixel_cost(
            source_style,
            target_style,
            source_index,
            target_index,
            style_weights,
            None,
            l1,
        )
    if has_guide:
        error += _dense_pixel_cost(
            source_guide,
            target_guide,
            source_index,
            target_index,
            guide_weights,
            modulation,
            l1,
        )
    return error


def _bounded_patch_errors(
    candidates: torch.Tensor,
    target_ys: torch.Tensor,
    target_xs: torch.Tensor,
    source_style_patches,
    target_style_patches,
    source_guide_patches,
    target_guide_patches,
    target_modulation_patches,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    patch_size: int,
    cost_function_mode: int,
    ssd_kw: dict,
    *,
    region: str,
) -> torch.Tensor:
    """
    Patch cost of ``candidates`` (N, 2) for target pixels (target_ys, target_xs),
    gathering at most ``chunk_pixels`` patches per grid at a time.

    Dense costs (no bilateral, ``n_size_step == 1``) are computed straight from
    (n, ps*ps, C) pixel gathers. Otherwise each chunk's matched source patches become
    a (1, n, D) grid addressed by ``(x=i, y=0)`` and the materialized cost kernels
    run on it. Both give the materialized path's values (up to float reduction order).
    """
    device = candidates.device
    H_s, W_s = source_style_patches.shape[:2]
    grids = (
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        target_modulation_patches,
    )
    chunk = min(g.chunk_pixels for g in grids if isinstance(g, ImagePatches))
    has_guide = source_guide_patches.numel() > 0
    used = [source_style_patches, target_style_patches, target_modulation_patches]
    if has_guide:
        used += [source_guide_patches, target_guide_patches]
    dense = (
        not ssd_kw["use_bilateral"]
        and ssd_kw["n_size_step"] == 1
        and all(isinstance(g, ImagePatches) for g in used if g is not None)
    )
    total = candidates.shape[0]
    errors = torch.empty(total, dtype=torch.float32, device=device)
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        count = end - start
        cand = candidates[start:end]
        sx = cand[:, 0].clamp(0, W_s - 1)
        sy = cand[:, 1].clamp(0, H_s - 1)
        ty = target_ys[start:end]
        tx = target_xs[start:end]
        if dense:
            with _mp_region(f"{region}:dense", device):
                errors[start:end] = _dense_bounded_errors(
                    sy,
                    sx,
                    ty,
                    tx,
                    grids,
                    style_weights,
                    guide_weights,
                    cost_function_mode,
                    has_guide,
                )
            continue
        indices: dict = {}
        ss = _gather_chunk(source_style_patches, sy, sx, indices, "source")
        ts = _gather_chunk(target_style_patches, ty, tx, indices, "target")
        if has_guide:
            sg = _gather_chunk(source_guide_patches, sy, sx, indices, "source")
            tg = _gather_chunk(target_guide_patches, ty, tx, indices, "target")
        else:
            sg = torch.empty((count, 0), dtype=torch.float32, device=device)
            tg = sg
        mod = (
            _gather_chunk(target_modulation_patches, ty, tx, indices, "target")
            if target_modulation_patches is not None
            else None
        )
        indices.clear()
        errors[start:end] = _patch_errors(
            ss.unsqueeze(0),
            ts,
            sg.unsqueeze(0),
            tg,
            _local_nnf(count, device),
            style_weights,
            guide_weights,
            patch_size,
            cost_function_mode,
            mod,
            ssd_kw,
            region=region,
        )
    return errors


def grid_patch_errors(
    candidate_coords: torch.Tensor,
    source_style_patches,
    target_style_patches,
    source_guide_patches,
    target_guide_patches,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    patch_size: int,
    cost_function_mode: int,
    target_modulation_patches=None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
    *,
    region: str = "try_patch",
) -> torch.Tensor:
    """
    Patch cost of an (H_t, W_t, 2) candidate grid, one candidate per target pixel.

    Works on materialized unfold grids and on :class:`ImagePatches` views (then the
    cost is gathered in chunks of ``chunk_pixels`` target pixels).
    """
    ssd_kw = {
        "use_bilateral": use_bilateral,
        "sigma_spatial": sigma_spatial,
        "sigma_color": sigma_color,
        "n_size_step": n_size_step,
    }
    if _uses_image_patches(
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        target_modulation_patches,
    ):
        if candidate_coords.dim() != 3:
            raise ValueError("on-demand patch grids need an (H, W, 2) candidate grid")
        H_t, W_t = candidate_coords.shape[:2]
        target_ys, target_xs = _target_grid_coords(H_t, W_t, candidate_coords.device)
        return _bounded_patch_errors(
            candidate_coords.reshape(-1, 2),
            target_ys,
            target_xs,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            target_modulation_patches,
            style_weights,
            guide_weights,
            patch_size,
            cost_function_mode,
            ssd_kw,
            region=region,
        ).view(H_t, W_t)
    return _patch_errors(
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        candidate_coords,
        style_weights,
        guide_weights,
        patch_size,
        cost_function_mode,
        target_modulation_patches,
        ssd_kw,
        region=region,
    )


class ActivePatchCost:
    """
    Patch cost of (N, 2) candidates for a fixed list of target pixels.

    Materialized grids gather the target-side patches once (they are invariant
    while the active set is fixed); :class:`ImagePatches` views gather them per
    chunk instead.
    """

    def __init__(
        self,
        target_ys: torch.Tensor,
        target_xs: torch.Tensor,
        source_style_patches,
        target_style_patches,
        source_guide_patches,
        target_guide_patches,
        style_weights: torch.Tensor,
        guide_weights: torch.Tensor,
        patch_size: int,
        cost_function_mode: int,
        target_modulation_patches=None,
        use_bilateral: bool = False,
        sigma_spatial: float = 4.0,
        sigma_color: float = 10.0,
        n_size_step: int = 1,
    ):
        self.target_ys = target_ys
        self.target_xs = target_xs
        self.source_style_patches = source_style_patches
        self.target_style_patches = target_style_patches
        self.source_guide_patches = source_guide_patches
        self.target_guide_patches = target_guide_patches
        self.target_modulation_patches = target_modulation_patches
        self.style_weights = style_weights
        self.guide_weights = guide_weights
        self.patch_size = patch_size
        self.cost_function_mode = cost_function_mode
        self.ssd_kw = {
            "use_bilateral": use_bilateral,
            "sigma_spatial": sigma_spatial,
            "sigma_color": sigma_color,
            "n_size_step": n_size_step,
        }
        self.bounded = _uses_image_patches(
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            target_modulation_patches,
        )
        self.target_style_v = self.target_guide_v = None
        self.target_mod_v = None
        if not self.bounded:
            self.target_style_v = target_style_patches[target_ys, target_xs]
            self.target_guide_v = (
                target_guide_patches[target_ys, target_xs]
                if target_guide_patches.numel() > 0
                else target_guide_patches
            )
            self.target_mod_v = (
                target_modulation_patches[target_ys, target_xs]
                if target_modulation_patches is not None
                else None
            )

    def __call__(self, candidates: torch.Tensor) -> torch.Tensor:
        if self.bounded:
            return _bounded_patch_errors(
                candidates,
                self.target_ys,
                self.target_xs,
                self.source_style_patches,
                self.target_style_patches,
                self.source_guide_patches,
                self.target_guide_patches,
                self.target_modulation_patches,
                self.style_weights,
                self.guide_weights,
                self.patch_size,
                self.cost_function_mode,
                self.ssd_kw,
                region="rs",
            )
        return _patch_errors(
            self.source_style_patches,
            self.target_style_v,
            self.source_guide_patches,
            self.target_guide_v,
            candidates,
            self.style_weights,
            self.guide_weights,
            self.patch_size,
            self.cost_function_mode,
            self.target_mod_v,
            self.ssd_kw,
            region="rs",
        )


def try_patch_batch(
    candidate_coords: torch.Tensor,
    current_nnf: torch.Tensor,
    current_errors: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    cost_function_mode: int,
    omega_best: float,
    target_modulation_patches: torch.Tensor | None = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
    *,
    omega_mean_map: torch.Tensor | None = None,
    candidate_patch_errors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch errors of ``candidate_coords`` and the mask of pixels where they beat the
    current match: ``(errors, update_mask)``, ``errors`` taking the candidate's patch
    error where it wins and ``current_errors`` elsewhere. Writes nothing.

    ``candidate_patch_errors`` skips the cost evaluation when the caller already
    computed :func:`grid_patch_errors` for ``candidate_coords``.
    """
    _dev = candidate_coords.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2
    valid_candidates = (
        (candidate_coords[..., 0] >= r)
        & (candidate_coords[..., 0] < W_s - r)
        & (candidate_coords[..., 1] >= r)
        & (candidate_coords[..., 1] < H_s - r)
    )

    if candidate_patch_errors is None:
        candidate_patch_errors = grid_patch_errors(
            candidate_coords,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            patch_size,
            cost_function_mode,
            target_modulation_patches,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
        )

    if uniformity_weight == 0.0:
        with _mp_region("try_patch:omega_gather_and_cost", _dev):
            update_mask = (candidate_patch_errors < current_errors) & valid_candidates
    else:
        if omega_mean_map is None:
            with _mp_region("try_patch:omega_mean_build", _dev):
                omega_mean_buf = omega_mean_map_buffer(_dev, H_s, W_s)
                omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
                omega_mean_map = omega_mean_buf
        with _mp_region("try_patch:omega_gather_and_cost", _dev):
            if candidate_coords.shape == current_nnf.shape:
                candidate_omega_scores, current_omega_scores = gather_omega_scores_pair(
                    omega_mean_map, candidate_coords, current_nnf
                )
            else:
                candidate_omega_scores = gather_omega_scores_from_patches(
                    omega_mean_map, candidate_coords
                )
                current_omega_scores = gather_omega_scores_from_patches(
                    omega_mean_map, current_nnf
                )
            patch_pixel_count = patch_size * patch_size
            inv_norm = 1.0 / (patch_pixel_count * omega_best)
            uf = uniformity_weight * inv_norm
            candidate_total_error = candidate_patch_errors + uf * candidate_omega_scores
            current_total_error = current_errors + uf * current_omega_scores
            update_mask = (
                candidate_total_error < current_total_error
            ) & valid_candidates
    with _mp_region("try_patch:select_write", _dev):
        updated_errors = torch.where(
            update_mask, candidate_patch_errors, current_errors
        )
    return updated_errors, update_mask


def propagation_step(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    is_odd: bool,
    mask: torch.Tensor,
    cost_function_mode: int,
    omega_best: float,
    target_modulation_patches: torch.Tensor | None = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
):
    """
    Propagation step: try neighbors' matches for spatial coherence, with one omega
    map update for both directions.

    Same scan as CUDA:
    - is_odd=True  => Forward Pass (Iter 1, 3...): Look Left/Top, Propagate +1
    - is_odd=False => Backward Pass (Iter 0, 2...): Look Right/Bottom, Propagate -1
    """
    H_s, W_s = source_style_patches.shape[:2]
    device = nnf.device
    active_mask = mask == 255
    nnf_before = nnf.clone()

    omega_mean_map = None
    if uniformity_weight != 0.0:
        with _mp_region("prop:omega_mean_map", device):
            omega_mean_buf = omega_mean_map_buffer(device, H_s, W_s)
            omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
            omega_mean_map = omega_mean_buf

    # Forward pass reads the left/top neighbour (shift right/down) and propagates +1;
    # backward pass reads the right/bottom neighbour and propagates -1.
    shift = 1 if is_odd else -1

    # --- Horizontal Propagation ---
    nb = _prop_neighbor_nnf_buf(nnf)
    if shift == 1:
        torch.cat((nnf[:, :1, :], nnf[:, :-1, :]), dim=1, out=nb)
    else:
        torch.cat((nnf[:, 1:, :], nnf[:, -1:, :]), dim=1, out=nb)

    horiz_candidates = nb + _get_offset_tensor((shift, 0), device)

    new_errors, horiz_updates = try_patch_batch(
        horiz_candidates,
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
        target_modulation_patches,
        use_bilateral,
        sigma_spatial,
        sigma_color,
        n_size_step,
        omega_mean_map=omega_mean_map,
    )
    horiz = horiz_updates & active_mask
    _accept(nnf, error_map, horiz_candidates, new_errors, horiz)

    # --- Vertical Propagation ---
    if shift == 1:
        torch.cat((nnf[:1, :, :], nnf[:-1, :, :]), dim=0, out=nb)
    else:
        torch.cat((nnf[1:, :, :], nnf[-1:, :, :]), dim=0, out=nb)

    vert_candidates = nb + _get_offset_tensor((0, shift), device)

    new_errors, vert_updates = try_patch_batch(
        vert_candidates,
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
        target_modulation_patches,
        use_bilateral,
        sigma_spatial,
        sigma_color,
        n_size_step,
        omega_mean_map=omega_mean_map,
    )
    vert = vert_updates & active_mask
    _accept(nnf, error_map, vert_candidates, new_errors, vert)

    # One omega update for both directions: from the NNF before the step to the
    # final one, for the pixels either direction moved.
    with _mp_region("prop:omega_updates", device):
        moved = (horiz | vert).nonzero(as_tuple=True)
        update_omega_map(
            omega_map,
            nnf_before[moved],
            nnf[moved],
            patch_size,
        )


def _accept(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    candidates: torch.Tensor,
    candidate_errors: torch.Tensor,
    update: torch.Tensor,
) -> None:
    """Write the ``update`` pixels' candidates and errors (masked, no host sync)."""
    nnf.copy_(torch.where(update.unsqueeze(-1), candidates.to(nnf.dtype), nnf))
    error_map.copy_(torch.where(update, candidate_errors, error_map))


def random_search_step(
    nnf: torch.Tensor,
    error_map: torch.Tensor,
    omega_map: torch.Tensor,
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    uniformity_weight: float,
    patch_size: int,
    initial_radius: int,
    mask: torch.Tensor,
    search_pruning_threshold: float,
    cost_function_mode: int,
    omega_best: float,
    generator: torch.Generator = None,
    target_modulation_patches: torch.Tensor | None = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
    alternatives: "CandidateSet | None" = None,
):
    """
    Random search with pruning.

    With ``alternatives`` (multicand mode) every radius also samples around each
    stored alternative match, and displaced or rejected candidates are offered to
    the alternative set.
    """
    device = nnf.device
    H_s, W_s = source_style_patches.shape[:2]
    r = patch_size // 2

    with _mp_region("rs:prep", device):
        # --- 1. Pruning: Select only active pixels ---
        active_mask = mask == 255
        if search_pruning_threshold > 0:
            active_mask &= error_map >= search_pruning_threshold

        active_indices = active_mask.nonzero(as_tuple=True)
        if active_indices[0].numel() == 0:
            return  # Exit early if no pixels to process

        y_coords, x_coords = active_indices
        num_active = y_coords.numel()

        rand_buf_x, rand_buf_y, candidate_buf = _random_search_buffers(num_active, nnf)

        # Constants of the loop below.
        patch_pixel_count = patch_size * patch_size
        omega_normalization = patch_pixel_count * omega_best
        uniformity_factor = uniformity_weight / omega_normalization

        # Current NNF and errors of the active pixels.
        current_nnf_active = nnf[y_coords, x_coords]
        current_errors_active = error_map[y_coords, x_coords]

        cost = ActivePatchCost(
            y_coords,
            x_coords,
            source_style_patches,
            target_style_patches,
            source_guide_patches,
            target_guide_patches,
            style_weights,
            guide_weights,
            patch_size,
            cost_function_mode,
            target_modulation_patches,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
        )
        alt_active = alt_present = None
        if alternatives is not None:
            alt_active = alternatives.nnf[y_coords, x_coords]
            alt_present = torch.isfinite(alternatives.errors[y_coords, x_coords])

        use_omega_cost = uniformity_weight != 0.0
        omega_mean_buf = None
        omega_mean_map = None
        curr_omega_active = None
        if use_omega_cost:
            # Reuse a single mean-map buffer across radius levels (avoids new tensor each rebuild).
            omega_mean_buf = omega_mean_map_buffer(device, H_s, W_s)
            omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)
            omega_mean_map = omega_mean_buf
            curr_omega_active = gather_omega_scores_from_patches(
                omega_mean_map, current_nnf_active
            )

    def _draw_offsets(radius: int) -> None:
        torch.randint(
            -radius,
            radius + 1,
            (num_active,),
            device=device,
            generator=generator,
            out=rand_buf_x,
        )
        torch.randint(
            -radius,
            radius + 1,
            (num_active,),
            device=device,
            generator=generator,
            out=rand_buf_y,
        )

    num_bases = 1 if alternatives is None else 1 + alternatives.slots
    radius = initial_radius
    while radius >= 1:
        with _mp_region("rs:radius_iter", device):
            # Base 0 is the current match; multicand mode also samples around
            # every stored alternative.
            for base_index in range(num_bases):
                # --- 2. Generate candidates only for active pixels ---
                _draw_offsets(radius)
                base = (
                    current_nnf_active
                    if base_index == 0
                    else alt_active[:, base_index - 1]
                )
                candidate_buf[:, 0].copy_(base[:, 0])
                candidate_buf[:, 1].copy_(base[:, 1])
                candidate_buf[:, 0].add_(rand_buf_x)
                candidate_buf[:, 1].add_(rand_buf_y)
                cand_v = candidate_buf

                # --- 3. Validate active candidates ---
                valid_mask_active = (
                    (cand_v[..., 0] >= r)
                    & (cand_v[..., 0] < W_s - r)
                    & (cand_v[..., 1] >= r)
                    & (cand_v[..., 1] < H_s - r)
                )
                if base_index > 0:
                    valid_mask_active &= alt_present[:, base_index - 1]

                # --- 4. Compute errors for active candidates ---
                # Keeping the full active set avoids a costly MPS nonzero/compaction
                # per radius; invalid candidates are masked out before any write.
                cand_errors_v = cost(cand_v)

                # --- 5. Compare total error and update ---
                if use_omega_cost:
                    cand_omega_v = gather_omega_scores_from_patches(
                        omega_mean_map, cand_v
                    )
                    cand_total_error = cand_errors_v + uniformity_factor * cand_omega_v
                    curr_total_error = (
                        current_errors_active + uniformity_factor * curr_omega_active
                    )
                else:
                    cand_total_error = cand_errors_v
                    curr_total_error = current_errors_active

                update_mask_v = (
                    cand_total_error < curr_total_error
                ) & valid_mask_active

                if alternatives is not None:
                    # Snapshot before the primary update: displaced matches become
                    # alternatives, rejected valid candidates are offered as well.
                    offer_coords = torch.where(
                        update_mask_v.unsqueeze(-1), current_nnf_active, cand_v
                    )
                    offer_errors = torch.where(
                        update_mask_v, current_errors_active, cand_errors_v
                    )

                # Omega moves only the updated pixels (compacted: a dense masked
                # scatter over every active pixel costs more than the host sync).
                moved = update_mask_v.nonzero(as_tuple=True)[0]
                update_omega_map(
                    omega_map,
                    current_nnf_active[moved],
                    cand_v[moved],
                    patch_size,
                )
                current_nnf_active.copy_(
                    torch.where(
                        update_mask_v.unsqueeze(-1),
                        cand_v.to(current_nnf_active.dtype),
                        current_nnf_active,
                    )
                )
                current_errors_active.copy_(
                    torch.where(update_mask_v, cand_errors_v, current_errors_active)
                )
                nnf[y_coords, x_coords] = current_nnf_active.to(nnf.dtype)
                error_map[y_coords, x_coords] = current_errors_active
                if use_omega_cost:
                    curr_omega_active.copy_(
                        torch.where(update_mask_v, cand_omega_v, curr_omega_active)
                    )
                    omega_patch_mean_map_into(omega_mean_buf, omega_map, patch_size)

                if alternatives is not None:
                    alternatives.offer(
                        y_coords,
                        x_coords,
                        offer_coords,
                        offer_errors,
                        nnf,
                        valid_mask_active,
                    )

        radius //= 2
