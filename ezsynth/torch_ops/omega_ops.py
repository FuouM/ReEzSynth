# ezsynth/torch_ops/omega_ops.py
"""
Omega map operations for tracking patch usage in EBSynth.

The omega map tracks how many times each source pixel is used in the current
NNF, enabling a uniformity penalty that discourages overuse of popular patches.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from .device_cache import clear_torch_device_cache

# conv2d kernels for box-filter omega (device, patch_size, dtype) -> cached tensor
_OMEGA_BOX_KERNEL: dict = {}
# patch coverage offsets (device, patch_size) -> (ps, ps, 2) tensor
_OMEGA_PATCH_OFFSETS: dict = {}
# scatter delta buffers (device, dtype) -> (minus_one, plus_one)
_OMEGA_DELTA_CACHE: dict = {}
# Reusable float32 mean-map buffers for hot paths (device_str, H, W)
_OMEGA_MEAN_BUF: dict = {}
# Float conversion scratch for omega conv input (same keys as mean buf)
_OMEGA_FLOAT_SCRATCH: dict = {}
# Padded (1,1,H+2r,W+2r) conv inputs — avoids allocating ``F.pad`` output each rebuild
_OMEGA_PADDED_CONV: dict = {}


def _omega_box_kernel(
    patch_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (patch_size, str(device), dtype)
    if key not in _OMEGA_BOX_KERNEL:
        _OMEGA_BOX_KERNEL[key] = torch.ones(
            1, 1, patch_size, patch_size, device=device, dtype=dtype
        )
    return _OMEGA_BOX_KERNEL[key]


def _omega_patch_offsets(patch_size: int, device: torch.device) -> torch.Tensor:
    key = (patch_size, str(device))
    if key not in _OMEGA_PATCH_OFFSETS:
        r = patch_size // 2
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-r, r + 1, device=device),
            torch.arange(-r, r + 1, device=device),
            indexing="ij",
        )
        _OMEGA_PATCH_OFFSETS[key] = torch.stack([offsets_x, offsets_y], dim=-1)
    return _OMEGA_PATCH_OFFSETS[key]


def _omega_scatter_deltas(
    n: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), dtype)
    entry = _OMEGA_DELTA_CACHE.get(key)
    if entry is None or entry[0].shape[0] < n:
        minus_one = torch.full((n,), -1, dtype=dtype, device=device)
        plus_one = torch.ones((n,), dtype=dtype, device=device)
        entry = (minus_one, plus_one)
        _OMEGA_DELTA_CACHE[key] = entry
    minus_one, plus_one = entry
    return minus_one[:n], plus_one[:n]


def populate_omega_map(
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    source_shape: Tuple[int, int],  # (H_s, W_s)
    patch_size: int,
) -> torch.Tensor:
    """
    Count how many target patches reference each source pixel.

    ACCELERATED VERSION: Uses Metal kernels on Apple Silicon for maximum performance.
    Falls back to optimized PyTorch implementation.

    For each NNF entry pointing to a source patch center, increment counts
    for all pixels covered by that patch (patch_size x patch_size region).

    Args:
        nnf: Nearest neighbor field mapping target pixels to source coordinates
        source_shape: (H_s, W_s) dimensions of source image
        patch_size: Size of patches (odd number)

    Returns:
        omega_map: (H_s, W_s) int32 tensor with usage counts

    Implementation Strategy:
    - For each NNF entry, expand to cover patch_size x patch_size region
    - Use scatter_add to efficiently accumulate counts
    - Clamp coordinates to valid range
    """

    H_t, W_t = nnf.shape[:2]
    H_s, W_s = source_shape
    device = nnf.device

    omega_map = torch.zeros((H_s, W_s), dtype=torch.int32, device=device)

    offsets = _omega_patch_offsets(patch_size, device)

    # Expand NNF to all patch pixels
    # nnf: (H_t, W_t, 2) -> (H_t, W_t, 1, 1, 2) + (1, 1, ps, ps, 2) = (H_t, W_t, ps, ps, 2)
    expanded_coords = nnf.unsqueeze(2).unsqueeze(3) + offsets.unsqueeze(0).unsqueeze(0)

    # Flatten to list of all coordinates: (H_t * W_t * ps * ps, 2)
    all_coords = expanded_coords.reshape(-1, 2)

    # Clamp to valid range
    all_coords[:, 0].clamp_(0, W_s - 1)
    all_coords[:, 1].clamp_(0, H_s - 1)

    # Convert to linear indices for scatter_add
    linear_indices = all_coords[:, 1] * W_s + all_coords[:, 0]

    # Count occurrences using bincount, then reshape to (H_s, W_s)
    counts = torch.bincount(linear_indices, minlength=H_s * W_s)
    omega_map = counts.view(H_s, W_s).to(torch.int32)

    return omega_map


def _omega_float_scratch(device: torch.device, H: int, W: int) -> torch.Tensor:
    """Cached float32 ``(H, W)`` copy of ``omega_map`` for conv2d input (no per-call alloc)."""
    key = (str(device), H, W)
    buf = _OMEGA_FLOAT_SCRATCH.get(key)
    if buf is None or buf.shape != (H, W):
        buf = torch.empty((H, W), dtype=torch.float32, device=device)
        _OMEGA_FLOAT_SCRATCH[key] = buf
    return buf


def _omega_padded_conv_input(
    device: torch.device, H_s: int, W_s: int, patch_size: int
) -> torch.Tensor:
    r = patch_size // 2
    ph, pw = H_s + 2 * r, W_s + 2 * r
    key = (str(device), H_s, W_s, patch_size)
    buf = _OMEGA_PADDED_CONV.get(key)
    if buf is None or buf.shape[-2] != ph or buf.shape[-1] != pw:
        buf = torch.zeros((1, 1, ph, pw), dtype=torch.float32, device=device)
        _OMEGA_PADDED_CONV[key] = buf
    return buf


def omega_patch_mean_map_into(
    out: torch.Tensor,
    omega_map: torch.Tensor,
    patch_size: int,
) -> None:
    """
    Box-filter mean of ``omega_map`` into ``out`` (float32, shape ``H_s×W_s``).

    Same numerics as :func:`omega_patch_mean_map`; avoids allocating a new tensor
    when callers reuse a scratch buffer (e.g. random-search radius iterations).
    """
    device = omega_map.device
    H_s, W_s = omega_map.shape[-2], omega_map.shape[-1]
    r = patch_size // 2
    ps = patch_size
    dtype = torch.float32
    fs = _omega_float_scratch(device, H_s, W_s)
    fs.copy_(omega_map)
    padded = _omega_padded_conv_input(device, H_s, W_s, patch_size)
    padded[0, 0, r : r + H_s, r : r + W_s].copy_(fs)
    kernel = _omega_box_kernel(ps, device, dtype)
    sums = F.conv2d(padded, kernel)
    inv_ps = 1.0 / float(ps * ps)
    out.copy_(sums[0, 0])
    out.mul_(inv_ps)


def omega_mean_map_buffer(device: torch.device, H: int, W: int) -> torch.Tensor:
    """Return a cached float32 ``(H, W)`` buffer for :func:`omega_patch_mean_map_into`."""
    key = (str(device), H, W)
    buf = _OMEGA_MEAN_BUF.get(key)
    if buf is None or buf.shape != (H, W):
        buf = torch.empty((H, W), dtype=torch.float32, device=device)
        _OMEGA_MEAN_BUF[key] = buf
    return buf


def omega_patch_mean_map(omega_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Per-source-pixel mean omega over a ``patch_size``×``patch_size`` patch with
    implicit **zero** edge padding (same as ``F.unfold(..., padding=r)`` + mean).

    Implemented as a box ``conv2d``, which is much faster than ``unfold`` at
    large ``H_s×W_s`` (e.g. megapixel Stylit).
    """
    out = torch.empty(
        omega_map.shape[-2:],
        dtype=torch.float32,
        device=omega_map.device,
    )
    omega_patch_mean_map_into(out, omega_map, patch_size)
    return out


def gather_omega_scores_from_patches(
    omega_mean_map: torch.Tensor, nnf: torch.Tensor
) -> torch.Tensor:
    """
    Look up precomputed per-source-cell mean patch omega at each NNF center.

    Args:
        omega_mean_map: (H_s, W_s) float32 from :func:`omega_patch_mean_map`
        nnf: (..., 2) int32 last dimension (sx, sy)

    Returns:
        omega_scores: ``nnf.shape[:-1]`` float32
    """
    H_s, W_s = omega_mean_map.shape
    original_shape = nnf.shape[:-1]
    flat = nnf.reshape(-1, 2)
    source_x = flat[:, 0].clamp(0, W_s - 1).long()
    source_y = flat[:, 1].clamp(0, H_s - 1).long()
    return omega_mean_map[source_y, source_x].view(original_shape)


def gather_omega_scores_pair(
    omega_mean_map: torch.Tensor,
    nnf_a: torch.Tensor,
    nnf_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two omega lookups for ``nnf_a`` / ``nnf_b`` (same leading shape as in
    ``try_patch_batch``). Uses one 1-D ``gather`` on a flattened map so the device
    schedules a single read pass instead of two advanced-index kernels.
    """
    H_s, W_s = omega_mean_map.shape
    if nnf_a.shape != nnf_b.shape:
        raise ValueError("gather_omega_scores_pair requires matching nnf shapes")
    original_shape = nnf_a.shape[:-1]
    flat_a = nnf_a.reshape(-1, 2)
    flat_b = nnf_b.reshape(-1, 2)
    sx_a = flat_a[:, 0].clamp(0, W_s - 1).long()
    sy_a = flat_a[:, 1].clamp(0, H_s - 1).long()
    sx_b = flat_b[:, 0].clamp(0, W_s - 1).long()
    sy_b = flat_b[:, 1].clamp(0, H_s - 1).long()
    omega_flat = omega_mean_map.reshape(-1)
    lin_a = sy_a * W_s + sx_a
    lin_b = sy_b * W_s + sx_b
    pair_idx = torch.stack((lin_a, lin_b), dim=1).reshape(-1)
    both = omega_flat.gather(0, pair_idx).reshape(-1, 2)
    return both[:, 0].view(original_shape), both[:, 1].view(original_shape)


def unfold_omega_patches(omega_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Deprecated alias for :func:`omega_patch_mean_map` (2D map, not unfold layout)."""
    return omega_patch_mean_map(omega_map, patch_size)


def compute_omega_scores(
    omega_map: torch.Tensor,  # (H_s, W_s) int32
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
) -> torch.Tensor:
    """
    Compute average omega value for each patch in the NNF.

    Args:
        omega_map: Usage counts for each source pixel
        nnf: Current nearest neighbor field
        patch_size: Size of patches

    Returns:
        omega_scores: ``nnf.shape[:-1]`` float32 average omega values per patch
    """
    op = omega_patch_mean_map(omega_map, patch_size)
    scores = gather_omega_scores_from_patches(op, nnf)
    clear_torch_device_cache(scores.device)
    return scores


def update_omega_map(
    omega_map: torch.Tensor,  # (H_s, W_s) int32, modified in-place
    old_coords: torch.Tensor,  # (N, 2) int32 - coordinates to decrement
    new_coords: torch.Tensor,  # (N, 2) int32 - coordinates to increment
    patch_size: int,
    *,
    clear_device_cache: bool = True,
):
    """
    Update omega map when NNF entries change.

    When a target pixel changes from pointing to one source patch to another,
    we need to decrement the old patch region and increment the new one.

    Args:
        omega_map: Current usage counts, modified in-place
        old_coords: Source coordinates to decrement (N, 2)
        new_coords: Source coordinates to increment (N, 2)
        patch_size: Size of patches
        clear_device_cache: If True, clear MPS/CUDA cache after scatter when enabled in consts.
    - Expand coordinates to cover patch regions
    - Use scatter_add to efficiently update counts
    """
    H_s, W_s = omega_map.shape
    device = omega_map.device
    # N = old_coords.shape[0]

    offsets = _omega_patch_offsets(patch_size, device)

    # Expand coordinates to patch coverage
    old_expanded = old_coords.unsqueeze(1).unsqueeze(2) + offsets  # (N, ps, ps, 2)
    new_expanded = new_coords.unsqueeze(1).unsqueeze(2) + offsets

    old_flat = old_expanded.reshape(-1, 2)  # (N*ps*ps, 2)
    new_flat = new_expanded.reshape(-1, 2)

    # Clamp coordinates
    old_flat[:, 0].clamp_(0, W_s - 1)
    old_flat[:, 1].clamp_(0, H_s - 1)
    new_flat[:, 0].clamp_(0, W_s - 1)
    new_flat[:, 1].clamp_(0, H_s - 1)

    # Convert to linear indices
    old_linear = old_flat[:, 1] * W_s + old_flat[:, 0]
    new_linear = new_flat[:, 1] * W_s + new_flat[:, 0]

    # Update counts: decrement old, increment new
    omega_flat = omega_map.view(-1)
    dec, inc = _omega_scatter_deltas(old_linear.numel(), device, omega_flat.dtype)
    omega_flat.scatter_add_(0, old_linear, dec)
    omega_flat.scatter_add_(0, new_linear, inc)

    if clear_device_cache:
        clear_torch_device_cache(device)
