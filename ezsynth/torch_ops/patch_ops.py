# ezsynth/torch_ops/patch_ops.py
"""
Patch extraction and distance computation operations for EBSynth.

This module contains vectorized PyTorch implementations of patch-based
operations that form the core of the PatchMatch algorithm.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..consts import (
    TORCH_COMPILE_FUSED_SSD,
    TORCH_COMPILE_PATCH_SSD,
)
from .device_cache import clear_torch_device_cache
from .microprofile import region as _mp_region


def extract_patches(
    image: torch.Tensor,
    patch_size: int,
    *,
    as_float: bool = False,
    clear_device_cache: bool = True,
) -> torch.Tensor:
    """
    Extract all patches from an image using unfold.

    Args:
        image: (H, W, C) tensor
        patch_size: Size of square patches (typically 7)

    Returns:
        patches: (H, W, patch_size*patch_size*C) tensor where each position
                contains the flattened patch centered at that pixel
    """
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")

    H, W, C = image.shape
    padding = patch_size // 2

    with _mp_region("extract_patches:unfold", image.device):
        image_nchw = image.permute(2, 0, 1).unsqueeze(0)

        is_uint8 = image_nchw.dtype == torch.uint8
        if is_uint8:
            image_nchw = image_nchw.float()

        # Use replicate padding to match CUDA implementation
        image_padded = F.pad(
            image_nchw, (padding, padding, padding, padding), mode="replicate"
        )
        patches = F.unfold(image_padded, kernel_size=patch_size, padding=0)

        if is_uint8 and not as_float:
            patches = patches.round().clamp(0, 255).to(torch.uint8)

        patches = patches.view(C * patch_size * patch_size, H * W)
        patches = patches.permute(1, 0).view(H, W, -1)

        result = patches.contiguous()

    # Clear device-specific cache to reduce memory usage
    if clear_device_cache:
        clear_torch_device_cache(image.device)

    return result


def extract_patches_from_coords(
    image: torch.Tensor, coords: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """
    Extract patches from specific coordinates using advanced indexing.
    More memory efficient than unfolding the whole image when we only need a subset.

    Args:
        image: (H, W, C) tensor
        coords: (N, 2) tensor of (x, y) top-left coordinates
        patch_size: Size of patches

    Returns:
        patches: (N, C*ps*ps) flattened patches
    """
    H, W, C = image.shape
    N = coords.shape[0]

    # Ensure coords are long for indexing
    coords = coords.long()

    # Create grid of offsets
    # (ps, ps)
    ys = torch.arange(patch_size, device=image.device)
    xs = torch.arange(patch_size, device=image.device)

    # (ps*ps)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.flatten()
    grid_x = grid_x.flatten()

    # Compute all sampling coordinates
    # coords: (N, 2) -> (N, 1)
    # grid: (ps*ps) -> (1, ps*ps)
    # sample_y: (N, ps*ps)
    sample_y = coords[:, 1:2] + grid_y.unsqueeze(0)
    sample_x = coords[:, 0:1] + grid_x.unsqueeze(0)

    # Clamp to be safe (though input coords should be valid for top-left)
    # We need to clamp the *bottom-right* of the patch too
    # But assuming valid top-left coords that allow for a full patch:
    # sample_y will range from y to y+ps-1.
    # If y < H-ps+1, then y+ps-1 < H.
    sample_y = sample_y.clamp(0, H - 1)
    sample_x = sample_x.clamp(0, W - 1)

    # Gather pixels
    # image: (H, W, C)
    # We want (N, ps*ps, C)

    # Advanced indexing with broadcasting
    # sample_y, sample_x are (N, ps*ps)
    # We need to expand them to (N, ps*ps, C)? No, we can index directly if we handle C carefully.
    # Actually, image[sample_y, sample_x] will give (N, ps*ps, C)

    patches = image[sample_y, sample_x]  # (N, ps*ps, C)

    # Flatten to (N, C*ps*ps) to match expected format for some ops,
    # OR keep as (N, C, ps, ps) depending on usage.
    # The existing extract_patches returns (H, W, patch_size*patch_size*C) -> flattened patches.
    # Let's match that "flattened patch" structure: (N, C*ps*ps)
    # But wait, existing extract_patches returns (H, W, -1) where -1 is C*ps*ps.
    # Here we have N patches.

    # patches is (N, ps*ps, C).
    # We need to permute to (N, C, ps*ps) then flatten?
    # Let's check how unfold does it.
    # Unfold (1, C, H, W) -> (1, C*ps*ps, L).
    # So the channel dimension comes first in the flattened vector.
    # i.e. [R0, R1... G0, G1... B0, B1...]

    patches = patches.permute(0, 2, 1)  # (N, C, ps*ps)
    patches = patches.reshape(N, -1)  # (N, C*ps*ps)

    return patches


# (patch_size, n_size_step, device_id) -> (k_idx long[K], spatial_sq float[K]) on that device
_SSD_SPARSE_PATCH_CACHE: dict = {}


def _ssd_sparse_patch_layout(
    patch_size: int, n_size_step: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear patch indices and spatial_sq for the n_size_step sampling grid."""
    dev_key = str(device)
    key = (patch_size, n_size_step, dev_key)
    if key not in _SSD_SPARSE_PATCH_CACHE:
        ps = patch_size
        r = ps // 2
        idx_list = []
        sp_list = []
        for py in range(-r, r + 1, n_size_step):
            for px in range(-r, r + 1, n_size_step):
                ky = py + r
                kx = px + r
                k_sp = ky * ps + kx
                idx_list.append(k_sp)
                sp_list.append(float(py * py + px * px))
        _SSD_SPARSE_PATCH_CACHE[key] = (
            torch.tensor(idx_list, dtype=torch.long, device=device),
            torch.tensor(sp_list, dtype=torch.float32, device=device),
        )
    return _SSD_SPARSE_PATCH_CACHE[key]


def _ssd_bilateral_position_weights(
    matched_style_nc_ps: torch.Tensor,
    patch_size: int,
    use_bilateral: bool,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
) -> torch.Tensor:
    """
    Per-patch spatial weights (length ps²), zero outside the n_size_step grid.
    Matches cost_functions_cpu.cpp: weight 1 or exp(-spatial/(2σ_s²) - color/(2σ_c²))
    with color distance in **source style** vs patch center.
    """
    N, _, ps_sq = matched_style_nc_ps.shape
    ps = patch_size
    r = ps // 2
    device = matched_style_nc_ps.device
    dtype = torch.float32
    k0 = r * ps + r
    center = matched_style_nc_ps[:, :, k0].float()
    k_idx, spatial_sq = _ssd_sparse_patch_layout(patch_size, n_size_step, device)
    cur = matched_style_nc_ps[:, :, k_idx].float()
    color_sq = ((cur - center.unsqueeze(-1)) ** 2).sum(dim=1)
    if use_bilateral:
        inv_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
        inv_sc = 1.0 / (2.0 * sigma_color * sigma_color)
        spatial_b = spatial_sq.to(dtype=dtype).view(1, -1)
        W_sparse = torch.exp(-spatial_b * inv_ss - color_sq * inv_sc)
    else:
        W_sparse = torch.ones((N, k_idx.shape[0]), device=device, dtype=dtype)
    W = torch.zeros(N, ps_sq, device=device, dtype=dtype)
    W[:, k_idx] = W_sparse
    return W


def compute_patch_ssd_vectorized(
    source_patches: torch.Tensor,
    target_patches: torch.Tensor,
    nnf: torch.Tensor,
    weights: torch.Tensor,
    target_modulation_patches: Optional[torch.Tensor] = None,
    *,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
    source_style_patches_for_bilateral: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes SSD. Handles both 2D grid and 1D list of target patches.

    When ``target_modulation_patches`` is set (same layout as ``target_patches``),
    scales each per-pixel guide squared difference by ``mod/255`` to match the
    CPU extension / Taichi cost (modulation applies to guide channels only).

    Bilateral weighting (``use_bilateral``) follows ``compute_patch_ssd_split_cpu``:
    Gaussian on spatial offset and summed squared **style** channel differences
    from the source patch center. ``source_style_patches_for_bilateral`` must be
    the style patch grid when ``source_patches`` are guide patches; if omitted,
    ``source_patches`` are used (correct for the style term).
    """
    H_s, W_s = source_patches.shape[:2]

    # Store original shape and flatten if necessary
    original_shape = target_patches.shape[:-1]
    if target_patches.dim() > 2:
        target_patches = target_patches.flatten(0, -2)
        nnf = nnf.flatten(0, -2)
        if target_modulation_patches is not None:
            target_modulation_patches = target_modulation_patches.flatten(0, -2)

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)
    matched_source_patches = source_patches[source_y, source_x]

    diff = matched_source_patches.float()
    diff.sub_(target_patches)
    diff.square_()
    C = len(weights)
    ps_sq = source_patches.shape[2] // C
    patch_size = int(round(ps_sq**0.5))
    if patch_size * patch_size != ps_sq:
        raise ValueError("patch flatten size must be a perfect square")

    # Dense unweighted SSD: W is all-ones on the full patch (n_size_step==1).
    if not use_bilateral and n_size_step == 1:
        diff_reshaped = diff.view(-1, C, ps_sq)
        if target_modulation_patches is None:
            error = (diff_reshaped.sum(dim=2) * weights.view(1, -1)).sum(dim=1)
        else:
            mod = target_modulation_patches.float().view(-1, C, ps_sq) / 255.0
            error = (diff_reshaped * mod * weights.view(1, -1, 1)).sum(dim=(1, 2))
    else:
        diff_reshaped = diff.view(-1, C, ps_sq)
        weighted_diff = diff_reshaped * weights.view(1, -1, 1)
        if target_modulation_patches is not None:
            mod = target_modulation_patches.float().view(-1, C, ps_sq) / 255.0
            weighted_diff = weighted_diff * mod
        style_grid = (
            source_style_patches_for_bilateral
            if source_style_patches_for_bilateral is not None
            else source_patches
        )
        matched_style = style_grid[source_y, source_x].view(
            -1, style_grid.shape[-1] // ps_sq, ps_sq
        )
        W = _ssd_bilateral_position_weights(
            matched_style.float(),
            patch_size,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
        )
        err_sum = (weighted_diff * W.unsqueeze(1)).sum(dim=(1, 2))
        w_sum = W.sum(dim=1).clamp(min=1e-6)
        error = err_sum / w_sum * float(ps_sq)

    # Reshape back to original if it was a grid
    if len(original_shape) > 1:
        return error.view(original_shape)
    return error


def compute_patch_ssd_style_guide_fused(
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    nnf: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    target_modulation_patches: Optional[torch.Tensor],
    *,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> torch.Tensor:
    """
    Style + guide SSD in one pass: one NNF flatten/clamp, one bilateral ``W`` when
    used, and two gathers from the same source indices (less Python than two
    separate ``compute_patch_ssd_vectorized`` calls).

    **Parity:** match the legacy two-call path only when ``not use_bilateral`` and
    ``n_size_step == 1`` (dense SSD). Call sites must gate on that; for bilateral
    or sparse ``n_size_step > 1``, keep two ``compute_patch_ssd_vectorized`` calls.
    """
    H_s, W_s = source_style_patches.shape[:2]
    Cs = int(style_weights.shape[0])
    Cg = int(guide_weights.shape[0])
    ps_sq_style = source_style_patches.shape[2] // Cs
    ps_sq_guide = source_guide_patches.shape[2] // Cg
    patch_size = int(round(ps_sq_style**0.5))
    if patch_size * patch_size != ps_sq_style or ps_sq_style != ps_sq_guide:
        raise ValueError("style/guide patch layouts must match for fused SSD")

    original_shape = target_style_patches.shape[:-1]
    ts = target_style_patches
    tg = target_guide_patches
    nnf_work = nnf
    mod = target_modulation_patches
    if ts.dim() > 2:
        ts = ts.flatten(0, -2)
        tg = tg.flatten(0, -2)
        nnf_work = nnf_work.flatten(0, -2)
        if mod is not None:
            mod = mod.flatten(0, -2)

    source_x = nnf_work[..., 0].clamp(0, W_s - 1)
    source_y = nnf_work[..., 1].clamp(0, H_s - 1)
    ms = source_style_patches[source_y, source_x]
    mg = source_guide_patches[source_y, source_x]

    diff_s = ms.float()
    diff_s.sub_(ts)
    diff_s.square_()
    diff_g = mg.float()
    diff_g.sub_(tg)
    diff_g.square_()
    if not use_bilateral and n_size_step == 1:
        ds = diff_s.view(-1, Cs, ps_sq_style)
        dg = diff_g.view(-1, Cg, ps_sq_style)
        style_error = (ds.sum(dim=2) * style_weights.view(1, -1)).sum(dim=1)
        if mod is None:
            guide_error = (dg.sum(dim=2) * guide_weights.view(1, -1)).sum(dim=1)
        else:
            guide_error = (
                dg
                * (mod.float().view(-1, Cg, ps_sq_style) / 255.0)
                * guide_weights.view(1, -1, 1)
            ).sum(dim=(1, 2))
        error = style_error + guide_error
    else:
        wd_s = diff_s.view(-1, Cs, ps_sq_style) * style_weights.view(1, -1, 1)
        wd_g = diff_g.view(-1, Cg, ps_sq_style) * guide_weights.view(1, -1, 1)
        if mod is not None:
            wd_g = wd_g * (mod.float().view(-1, Cg, ps_sq_style) / 255.0)
        matched_style_nc_ps = ms.float().view(-1, Cs, ps_sq_style)
        W = _ssd_bilateral_position_weights(
            matched_style_nc_ps,
            patch_size,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
        )
        err_sum = (wd_s * W.unsqueeze(1)).sum(dim=(1, 2)) + (wd_g * W.unsqueeze(1)).sum(
            dim=(1, 2)
        )
        w_sum = W.sum(dim=1).clamp(min=1e-6)
        error = err_sum / w_sum * float(ps_sq_style)

    if len(original_shape) > 1:
        return error.view(original_shape)
    return error


def compute_patch_stats(
    patches: torch.Tensor, patch_size: int, num_channels: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute statistics for NCC: vals (grayscale), mean, std.

    Args:
        patches: (..., C*ps*ps) flattened patches
        patch_size: int
        num_channels: int

    Returns:
        vals: (..., ps*ps) - channel-averaged pixel values
        mean: (..., 1) - mean of vals
        std: (..., 1) - std of vals
    """
    ps_sq = patch_size * patch_size
    epsilon = 1e-6

    # Reshape to (..., C, ps*ps)
    patches_reshaped = patches.view(*patches.shape[:-1], num_channels, ps_sq)

    # Average over channels to get "grayscale" equivalent for NCC
    vals = patches_reshaped.float().mean(dim=-2)  # (..., ps_sq)

    mean = vals.mean(dim=-1, keepdim=True)  # (..., 1)
    std = vals.std(dim=-1, keepdim=True, unbiased=False) + epsilon  # (..., 1)

    return vals, mean, std


def compute_patch_ncc_vectorized(
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    nnf: torch.Tensor,
    patch_size: int,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    source_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    target_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    target_modulation_patches: Optional[torch.Tensor] = None,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
    n_size_step: int = 1,
) -> torch.Tensor:
    """Computes NCC/SSD. Handles both 2D grid and 1D list of target patches."""
    H_s, W_s = source_style_patches.shape[:2]
    C_style = style_weights.shape[0]
    ps_sq = patch_size * patch_size
    epsilon = 1e-6

    # Store original shape and flatten if necessary
    original_shape = target_style_patches.shape[:-1]
    if target_style_patches.dim() > 2:
        target_style_patches = target_style_patches.flatten(0, -2)
        if target_guide_patches.numel() > 0:
            target_guide_patches = target_guide_patches.flatten(0, -2)
        if target_modulation_patches is not None:
            target_modulation_patches = target_modulation_patches.flatten(0, -2)
        nnf = nnf.flatten(0, -2)

    source_x = nnf[..., 0].clamp(0, W_s - 1)
    source_y = nnf[..., 1].clamp(0, H_s - 1)
    matched_source_patches = source_style_patches[source_y, source_x]

    # --- NCC Computation (on flattened data) ---
    num_active = matched_source_patches.shape[0]
    matched_source = matched_source_patches.view(num_active, C_style, ps_sq)
    target = target_style_patches.view(num_active, C_style, ps_sq)

    s_vals = matched_source.float().mean(dim=1)
    t_vals = target.float().mean(dim=1)

    mean_s = s_vals.mean(dim=1, keepdim=True)
    mean_t = t_vals.mean(dim=1, keepdim=True)

    std_s = s_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon
    std_t = t_vals.std(dim=1, keepdim=True, unbiased=False) + epsilon

    cov = ((s_vals - mean_s) * (t_vals - mean_t)).mean(dim=1)
    ncc = cov / (std_s.squeeze(1) * std_t.squeeze(1))
    ncc = ncc.clamp(-1.0, 1.0)
    style_error = (1.0 - ncc) * style_weights[0] * float(ps_sq)

    # --- Guide SSD ---
    if source_guide_patches.numel() > 0:
        guide_error = compute_patch_ssd_vectorized(
            source_guide_patches,
            target_guide_patches,
            nnf,
            guide_weights,
            target_modulation_patches,
            use_bilateral=use_bilateral,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            n_size_step=n_size_step,
            source_style_patches_for_bilateral=source_style_patches,
        )
        total_error = style_error + guide_error
    else:
        total_error = style_error

    # Reshape back to original if it was a grid
    if len(original_shape) > 1:
        return total_error.view(original_shape)
    return total_error


if TORCH_COMPILE_PATCH_SSD:
    compute_patch_ssd_vectorized = torch.compile(
        compute_patch_ssd_vectorized, dynamic=True
    )

if TORCH_COMPILE_FUSED_SSD:
    compute_patch_ssd_style_guide_fused = torch.compile(
        compute_patch_ssd_style_guide_fused, dynamic=True
    )
