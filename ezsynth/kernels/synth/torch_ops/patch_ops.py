# ezsynth/torch_ops/patch_ops.py
"""
Patch extraction and distance computation operations for EBSynth.

This module contains vectorized PyTorch implementations of patch-based
operations that form the core of the PatchMatch algorithm.
"""

import torch
import torch.nn.functional as F

from ....consts import (
    COST_FUNCTION_NCC,
    COST_FUNCTION_SAD,
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
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Extract all patches from an image using unfold.

    Args:
        image: (H, W, C) tensor
        patch_size: Size of square patches (typically 7)
        as_float: keep uint8 input as floating point patches instead of rounding
            back to uint8.
        dtype: floating dtype of ``as_float`` patches; None keeps float32 for uint8
            input and the input dtype for floating input.

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
        if as_float and dtype is not None:
            image_nchw = image_nchw.to(dtype)
        elif is_uint8:
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
    H, W, _C = image.shape
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


# (patch_size, device_str) -> (dy[K], dx[K]) long offsets in unfold order
_PATCH_OFFSET_CACHE: dict = {}


def _patch_offsets(
    patch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (patch_size, str(device))
    entry = _PATCH_OFFSET_CACHE.get(key)
    if entry is None:
        r = patch_size // 2
        k = torch.arange(patch_size * patch_size, device=device, dtype=torch.long)
        entry = (k // patch_size - r, k % patch_size - r)
        _PATCH_OFFSET_CACHE[key] = entry
    return entry


class ImagePatches:
    """
    On-demand view of ``extract_patches(image, patch_size, as_float=True, dtype=dtype)``.

    Holds only the (H, W, C) image and gathers flattened patches for requested
    centers, so memory scales with the number of requested patches instead of
    ``H * W * C * patch_size**2``. Values are bit-identical to the unfold grid:
    replicate padding equals clamping each sample coordinate to the image, and the
    flattened layout is channel-major then kernel row-major, like ``F.unfold``.

    ``chunk_pixels`` is the number of patches a consumer may gather at once.
    """

    def __init__(
        self,
        image: torch.Tensor,
        patch_size: int,
        chunk_pixels: int,
        dtype: torch.dtype | None = None,
    ):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        height, width, channels = image.shape
        if dtype is not None:
            values = image.to(dtype)
        else:
            values = image.float() if image.dtype == torch.uint8 else image
        self.pixels = values.reshape(height * width, channels)
        self.patch_size = patch_size
        self.chunk_pixels = max(1, int(chunk_pixels))
        self.shape = torch.Size((height, width, channels * patch_size * patch_size))
        self.device = image.device
        self.dtype = values.dtype

    def numel(self) -> int:
        return self.shape.numel()

    def pixel_index(self, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """(N, patch_size**2) linear pixel indices of the patches centered at (ys, xs)."""
        height, width = self.shape[0], self.shape[1]
        dy, dx = _patch_offsets(self.patch_size, self.device)
        rows = (ys.long().unsqueeze(1) + dy).clamp_(0, height - 1)
        cols = (xs.long().unsqueeze(1) + dx).clamp_(0, width - 1)
        return rows.mul_(width).add_(cols)

    def gather_index(self, index: torch.Tensor) -> torch.Tensor:
        """(N, C * patch_size**2) patches for indices from :meth:`pixel_index`."""
        values = self.pixels[index]  # (N, ps*ps, C)
        return values.transpose(1, 2).reshape(index.shape[0], -1)

    def gather(self, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        return self.gather_index(self.pixel_index(ys, xs))


def _gather_rows(
    grid: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor
) -> torch.Tensor:
    """``grid[ys, xs]`` for an (H, W, D) patch grid as one ``index_select`` of rows
    (faster than advanced indexing on CUDA)."""
    width = grid.shape[1]
    return grid.reshape(-1, grid.shape[-1]).index_select(0, (ys * width + xs).view(-1))


# (patch_size, n_size_step, device) -> (k_idx long[K], spatial_sq float[K]) on that device
_SAMPLED_PATCH_LAYOUT: dict = {}


def _sampled_patch_layout(
    patch_size: int, n_size_step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear patch indices and squared offsets of the ``n_size_step`` sampling grid."""
    key = (patch_size, n_size_step, str(device))
    if key not in _SAMPLED_PATCH_LAYOUT:
        r = patch_size // 2
        offsets = [
            (py, px)
            for py in range(-r, r + 1, n_size_step)
            for px in range(-r, r + 1, n_size_step)
        ]
        _SAMPLED_PATCH_LAYOUT[key] = (
            torch.tensor(
                [(py + r) * patch_size + px + r for py, px in offsets],
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(
                [float(py * py + px * px) for py, px in offsets],
                dtype=torch.float32,
                device=device,
            ),
        )
    return _SAMPLED_PATCH_LAYOUT[key]


def compute_patch_ssd_vectorized(
    source_patches: torch.Tensor,
    target_patches: torch.Tensor,
    nnf: torch.Tensor,
    weights: torch.Tensor,
    target_modulation_patches: torch.Tensor | None = None,
    *,
    l1: bool = False,
) -> torch.Tensor:
    """Dense SSD (or SAD with ``l1=True``) over the whole patch. Handles both a 2D
    grid and a 1D list of target patches.

    When ``target_modulation_patches`` is set (same layout as ``target_patches``),
    scales each per-pixel guide squared difference by ``mod/255`` to match the
    CPU extension / Taichi cost (modulation applies to guide channels only).
    Bilateral or ``n_size_step > 1`` costs: :func:`compute_patch_cost_sampled`.
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
    matched_source_patches = _gather_rows(source_patches, source_y, source_x)

    diff = matched_source_patches.float()
    diff.sub_(target_patches)
    if l1:
        diff.abs_()
    else:
        diff.square_()
    C = len(weights)
    ps_sq = source_patches.shape[2] // C
    diff_reshaped = diff.view(-1, C, ps_sq)
    if target_modulation_patches is None:
        error = (diff_reshaped.sum(dim=2) * weights.view(1, -1)).sum(dim=1)
    else:
        mod = target_modulation_patches.float().view(-1, C, ps_sq) / 255.0
        error = (diff_reshaped * mod * weights.view(1, -1, 1)).sum(dim=(1, 2))

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
    target_modulation_patches: torch.Tensor | None,
    *,
    l1: bool = False,
) -> torch.Tensor:
    """
    Dense style + guide SSD (or SAD with ``l1=True``) in one pass: one NNF
    flatten/clamp and two gathers from the same source indices (less Python than two
    separate ``compute_patch_ssd_vectorized`` calls).
    """
    H_s, W_s = source_style_patches.shape[:2]
    Cs = int(style_weights.shape[0])
    Cg = int(guide_weights.shape[0])
    ps_sq_style = source_style_patches.shape[2] // Cs
    ps_sq_guide = source_guide_patches.shape[2] // Cg
    patch_size = round(ps_sq_style**0.5)
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
    ms = _gather_rows(source_style_patches, source_y, source_x)
    mg = _gather_rows(source_guide_patches, source_y, source_x)

    diff_s = ms.float()
    diff_s.sub_(ts)
    if l1:
        diff_s.abs_()
    else:
        diff_s.square_()
    diff_g = mg.float()
    diff_g.sub_(tg)
    if l1:
        diff_g.abs_()
    else:
        diff_g.square_()
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

    if len(original_shape) > 1:
        return error.view(original_shape)
    return error


def compute_patch_stats(
    patches: torch.Tensor, patch_size: int, num_channels: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    source_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    target_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    target_modulation_patches: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense NCC style cost plus guide SSD. Handles both 2D grid and 1D list of
    target patches. Bilateral or ``n_size_step > 1``: :func:`compute_patch_cost_sampled`.
    """
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
    matched_source_patches = _gather_rows(source_style_patches, source_y, source_x)

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
        )
        total_error = style_error + guide_error
    else:
        total_error = style_error

    # Reshape back to original if it was a grid
    if len(original_shape) > 1:
        return total_error.view(original_shape)
    return total_error


def compute_patch_cost_sampled(
    source_style_patches: torch.Tensor,
    target_style_patches: torch.Tensor,
    source_guide_patches: torch.Tensor,
    target_guide_patches: torch.Tensor,
    nnf: torch.Tensor,
    style_weights: torch.Tensor,
    guide_weights: torch.Tensor,
    target_modulation_patches: torch.Tensor | None,
    *,
    cost_function_mode: int,
    use_bilateral: bool,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
) -> torch.Tensor:
    """Patch cost over the ``n_size_step`` sampling grid with optional bilateral
    weights, term for term as Taichi ``compute_patch_ssd`` / ``compute_patch_ncc``
    (and WGSL ``patch_cost``). Values are the weighted sums, not normalized by the
    sample count or weight total.

    SSD/SAD weight of offset ``p``: ``exp(-|p|^2 / (2 sigma_s^2) - ((s_p - s_0)^2 +
    (t_p - t_0)^2) / (2 sigma_c^2))`` with ``s``/``t`` the channel-mean source/target
    style and ``0`` the patch centre. NCC weight: the same spatial term and the
    centre difference ``(s_0 - t_0)^2``; NCC statistics use the weighted moments and
    the style term is ``(1 - ncc) * style_weights[0] * sum(weights)``. Guide terms are
    squared for NCC and follow ``cost_function_mode`` otherwise.
    """
    H_s, W_s = source_style_patches.shape[:2]
    original_shape = target_style_patches.shape[:-1]
    ts = target_style_patches.flatten(0, -2)
    nnf = nnf.flatten(0, -2)
    sx = nnf[:, 0].clamp(0, W_s - 1)
    sy = nnf[:, 1].clamp(0, H_s - 1)
    Cs = int(style_weights.shape[0])
    ps_sq = ts.shape[-1] // Cs
    patch_size = round(ps_sq**0.5)
    centre = (patch_size // 2) * patch_size + patch_size // 2
    k_idx, spatial_sq = _sampled_patch_layout(patch_size, n_size_step, ts.device)

    source_style = (
        _gather_rows(source_style_patches, sy, sx).view(-1, Cs, ps_sq).float()
    )
    target_style = ts.view(-1, Cs, ps_sq).float()
    s_mean = source_style.mean(dim=1)  # (N, ps_sq)
    t_mean = target_style.mean(dim=1)
    s_centre = s_mean[:, centre : centre + 1]
    t_centre = t_mean[:, centre : centre + 1]
    s_k, t_k = s_mean[:, k_idx], t_mean[:, k_idx]  # (N, K)
    ncc = cost_function_mode == COST_FUNCTION_NCC

    if use_bilateral:
        inv_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
        inv_sc = 1.0 / (2.0 * sigma_color * sigma_color)
        if ncc:
            colour_sq = (s_centre - t_centre).square()
        else:
            colour_sq = (s_k - s_centre).square() + (t_k - t_centre).square()
        weight = torch.exp(-spatial_sq.view(1, -1) * inv_ss - colour_sq * inv_sc)
    else:
        weight = torch.ones_like(s_k)

    per_offset = torch.zeros_like(s_k)
    if source_guide_patches.numel() > 0:
        Cg = int(guide_weights.shape[0])
        source_guide = _gather_rows(source_guide_patches, sy, sx).view(-1, Cg, ps_sq)
        source_guide = source_guide[:, :, k_idx]
        target_guide = target_guide_patches.flatten(0, -2).view(-1, Cg, ps_sq)
        diff = source_guide.float() - target_guide[:, :, k_idx].float()
        diff = diff.abs() if cost_function_mode == COST_FUNCTION_SAD else diff.square()
        diff = diff * guide_weights.view(1, -1, 1)
        if target_modulation_patches is not None:
            mod = target_modulation_patches.flatten(0, -2).view(-1, Cg, ps_sq)
            diff = diff * (mod[:, :, k_idx].float() / 255.0)
        per_offset = per_offset + diff.sum(dim=1)

    if ncc:
        epsilon = 1e-6
        guide_error = (weight * per_offset).sum(dim=1)
        total = weight.sum(dim=1)
        safe = total.clamp(min=epsilon)
        mean_s = (weight * s_k).sum(dim=1) / safe
        mean_t = (weight * t_k).sum(dim=1) / safe
        var_s = (weight * s_k.square()).sum(dim=1) / safe - mean_s.square()
        var_t = (weight * t_k.square()).sum(dim=1) / safe - mean_t.square()
        std_s = var_s.clamp(min=0.0).sqrt()
        std_t = var_t.clamp(min=0.0).sqrt()
        cov = (weight * s_k * t_k).sum(dim=1) / safe - mean_s * mean_t
        defined = (std_s > epsilon) & (std_t > epsilon)
        corr = torch.where(defined, cov / (std_s * std_t), torch.zeros_like(cov))
        style_error = (1.0 - corr.clamp(-1.0, 1.0)) * style_weights[0] * total
        error = torch.where(total > epsilon, style_error + guide_error, guide_error)
    else:
        diff = source_style[:, :, k_idx] - target_style[:, :, k_idx]
        diff = diff.abs() if cost_function_mode == COST_FUNCTION_SAD else diff.square()
        per_offset = per_offset + (diff * style_weights.view(1, -1, 1)).sum(dim=1)
        error = (weight * per_offset).sum(dim=1)

    if len(original_shape) > 1:
        return error.view(original_shape)
    return error


if TORCH_COMPILE_PATCH_SSD:
    compute_patch_ssd_vectorized = torch.compile(
        compute_patch_ssd_vectorized, dynamic=True
    )

if TORCH_COMPILE_FUSED_SSD:
    compute_patch_ssd_style_guide_fused = torch.compile(
        compute_patch_ssd_style_guide_fused, dynamic=True
    )
