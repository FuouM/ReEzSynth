# ezsynth/torch_ops/voting_ops.py
"""
Voting operations for image reconstruction in EBSynth.

These functions reconstruct the target image by averaging overlapping source patches
that contribute to each target pixel according to the current NNF.

Offsets are applied in a single vectorized pass over ``K = patch_size²`` (or in
chunks when memory is tight) to avoid Python double-loop dispatch on large grids.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..consts import vote_chunk_budget_bytes
from .device_cache import clear_torch_device_cache
from .microprofile import region as _mp_region

# (patch_size, device_str) -> (Py[K], Px[K]) long tensors
_VOTE_OFFSET_CACHE: dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
# (H_t, W_t, device_str) -> (Ty, Tx) meshgrid long tensors
_VOTE_TYTX_CACHE: dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _vote_target_grid(
    h_t: int, w_t: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (h_t, w_t, str(device))
    if key not in _VOTE_TYTX_CACHE:
        Ty, Tx = torch.meshgrid(
            torch.arange(h_t, device=device, dtype=torch.long),
            torch.arange(w_t, device=device, dtype=torch.long),
            indexing="ij",
        )
        _VOTE_TYTX_CACHE[key] = (Ty, Tx)
    return _VOTE_TYTX_CACHE[key]


def _vote_flat_offsets(
    patch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(Py, Px)`` length ``K`` matching nested ``for py / for px`` order (``ij`` meshgrid)."""
    r = patch_size // 2
    key = (patch_size, str(device))
    if key not in _VOTE_OFFSET_CACHE:
        py_ = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        px_ = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        Py, Px = torch.meshgrid(py_, px_, indexing="ij")
        _VOTE_OFFSET_CACHE[key] = (Py.reshape(-1), Px.reshape(-1))
    return _VOTE_OFFSET_CACHE[key]


def _vote_chunk_k(h: int, w: int, c: int, k_total: int) -> int:
    """How many offsets to stack at once (float32 ``vals`` is ``k*h*w*c*4`` bytes)."""
    budget_bytes = vote_chunk_budget_bytes()
    per = h * w * max(c, 1) * 4
    if per <= 0:
        return max(1, k_total)
    n = budget_bytes // max(per, 1)
    return max(1, min(k_total, max(1, n)))


def vote_plain(
    source_style: torch.Tensor,  # (H_s, W_s, C) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    patch_size: int,
    *,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
) -> torch.Tensor:
    """
    Reconstruct target image by averaging overlapping patches.

    Optional bilateral weights (same convention as patch SSD bilateral in the
    CPU extension): Gaussian in offset ``(px, py)`` and in summed squared
    per-channel differences between the gathered source sample and the source
    at the **receiving** pixel's NNF center ``nnf[ty, tx]``.
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    device = source_style.device
    dtype = source_style.dtype
    nnf_index = nnf if nnf.dtype == torch.long else nnf.to(torch.long)

    Ty, Tx = _vote_target_grid(H_t, W_t, device)

    with _mp_region("vote:plain", device):
        Py, Px = _vote_flat_offsets(patch_size, device)
        K = Py.numel()
        chunk = _vote_chunk_k(H_t, W_t, C, K)

        accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

        inv_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
        inv_sc = 1.0 / (2.0 * sigma_color * sigma_color)

        if use_bilateral:
            rcx = nnf_index[Ty, Tx, 0].clamp(0, W_s - 1)
            rcy = nnf_index[Ty, Tx, 1].clamp(0, H_s - 1)
            center = source_style[rcy, rcx].float()
            spatial_sq_offsets = Py.float().square() + Px.float().square()

        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            Pyb, Pxb = Py[k0:k1], Px[k0:k1]
            kb = k1 - k0

            Nty = Ty.unsqueeze(0) - Pyb.view(kb, 1, 1)
            Ntx = Tx.unsqueeze(0) - Pxb.view(kb, 1, 1)
            valid = (Ntx >= 0) & (Ntx < W_t) & (Nty >= 0) & (Nty < H_t)
            nny = Nty.clamp(0, H_t - 1)
            nnx = Ntx.clamp(0, W_t - 1)
            scx = nnf_index[nny, nnx, 0]
            scy = nnf_index[nny, nnx, 1]
            sx = (scx + Pxb.view(kb, 1, 1)).clamp(0, W_s - 1)
            sy = (scy + Pyb.view(kb, 1, 1)).clamp(0, H_s - 1)
            vals = source_style[sy, sx].float()

            if use_bilateral:
                spatial_sq = spatial_sq_offsets[k0:k1].view(kb, 1, 1)
                color_sq = ((vals - center.unsqueeze(0)) ** 2).sum(dim=-1)
                w = torch.exp(-spatial_sq * inv_ss - color_sq * inv_sc)
                w = w * valid.to(torch.float32)
                accumulator += (vals * w.unsqueeze(-1)).sum(dim=0)
                weight_sum += w.sum(dim=0)
            else:
                vf = valid.to(torch.float32)
                accumulator += (vals * vf.unsqueeze(-1)).sum(dim=0)
                weight_sum += vf.sum(dim=0)

        target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)
        target_style = target_style.clamp(0, 255).to(dtype)

    clear_torch_device_cache(device)

    return target_style


def vote_weighted(
    source_style: torch.Tensor,  # (H_s, W_s, C) uint8
    nnf: torch.Tensor,  # (H_t, W_t, 2) int32
    error_map: torch.Tensor,  # (H_t, W_t) float32
    patch_size: int,
    *,
    use_bilateral: bool = False,
    sigma_spatial: float = 4.0,
    sigma_color: float = 10.0,
) -> torch.Tensor:
    """
    Weighted voting using patch errors, optionally multiplied by bilateral
    weights (see :func:`vote_plain`).
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    device = source_style.device
    dtype = source_style.dtype
    nnf_index = nnf if nnf.dtype == torch.long else nnf.to(torch.long)

    Ty, Tx = _vote_target_grid(H_t, W_t, device)

    with _mp_region("vote:weighted", device):
        Py, Px = _vote_flat_offsets(patch_size, device)
        K = Py.numel()
        chunk = _vote_chunk_k(H_t, W_t, C, K)

        accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

        inv_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
        inv_sc = 1.0 / (2.0 * sigma_color * sigma_color)

        if use_bilateral:
            rcx = nnf_index[Ty, Tx, 0].clamp(0, W_s - 1)
            rcy = nnf_index[Ty, Tx, 1].clamp(0, H_s - 1)
            center = source_style[rcy, rcx].float()
            spatial_sq_offsets = Py.float().square() + Px.float().square()

        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            Pyb, Pxb = Py[k0:k1], Px[k0:k1]
            kb = k1 - k0

            Nty = Ty.unsqueeze(0) - Pyb.view(kb, 1, 1)
            Ntx = Tx.unsqueeze(0) - Pxb.view(kb, 1, 1)
            valid = (Ntx >= 0) & (Ntx < W_t) & (Nty >= 0) & (Nty < H_t)
            nny = Nty.clamp(0, H_t - 1)
            nnx = Ntx.clamp(0, W_t - 1)
            scx = nnf_index[nny, nnx, 0]
            scy = nnf_index[nny, nnx, 1]
            sx = (scx + Pxb.view(kb, 1, 1)).clamp(0, W_s - 1)
            sy = (scy + Pyb.view(kb, 1, 1)).clamp(0, H_s - 1)
            vals = source_style[sy, sx].float()
            err_n = error_map[nny, nnx]
            w = 1.0 / (1.0 + err_n)
            vf = valid.to(torch.float32)
            w = w * vf
            if use_bilateral:
                spatial_sq = spatial_sq_offsets[k0:k1].view(kb, 1, 1)
                color_sq = ((vals - center.unsqueeze(0)) ** 2).sum(dim=-1)
                bw = torch.exp(-spatial_sq * inv_ss - color_sq * inv_sc)
                w = w * bw
            accumulator += (vals * w.unsqueeze(-1)).sum(dim=0)
            weight_sum += w.sum(dim=0)

        target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)
        target_style = target_style.clamp(0, 255).to(dtype)

    clear_torch_device_cache(device)

    return target_style
