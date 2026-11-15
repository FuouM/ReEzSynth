# ezsynth/torch_ops/voting_ops.py
"""
Voting operations for image reconstruction in EBSynth.

These functions reconstruct the target image by averaging overlapping source patches
that contribute to each target pixel according to the current NNF.

Offsets are applied in a single vectorized pass over ``K = patch_size²`` (or in
chunks when memory is tight) to avoid Python double-loop dispatch on large grids.
"""

from __future__ import annotations

import torch

from ....consts import vote_chunk_budget_bytes
from .microprofile import region as _mp_region

# (patch_size, device_str) -> (Py[K], Px[K]) long tensors
_VOTE_OFFSET_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
# (H_t, W_t, device_str) -> (Ty, Tx) meshgrid long tensors
_VOTE_TYTX_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _vote_target_grid(
    h_t: int, w_t: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _vote_bilateral(
    source_style: torch.Tensor,
    nnf: torch.Tensor,
    error_map: torch.Tensor | None,
    patch_size: int,
    sigma_spatial: float,
    sigma_color: float,
) -> torch.Tensor:
    """Bilateral vote, term for term as Taichi ``voting_kernel`` (and WGSL
    ``vote_bilateral``).

    Each neighbour ``n = t - p`` inside the target votes with source pixel
    ``s = nnf[n] + p`` when ``s`` lies inside the source (no clamping), weighted by
    ``1 / (1 + error[n])`` (``error_map`` given) times ``exp(-|p|^2 / (2 sigma_s^2) -
    (m(s) - m(nnf[t]))^2 / (2 sigma_c^2))`` with ``m`` the channel mean. A pixel whose
    weights sum to at most 1e-4 takes ``source[nnf[t]]``, as in the reference kernel.
    """
    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    device = source_style.device
    nnf_index = nnf if nnf.dtype == torch.long else nnf.to(torch.long)
    Ty, Tx = _vote_target_grid(H_t, W_t, device)
    Py, Px = _vote_flat_offsets(patch_size, device)
    K = Py.numel()
    chunk = _vote_chunk_k(H_t, W_t, C, K)
    inv_ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_sc = 1.0 / (2.0 * sigma_color * sigma_color)
    spatial_sq = (Py.square() + Px.square()).to(torch.float32)

    own = source_style[nnf_index[..., 1], nnf_index[..., 0]]  # (H_t, W_t, C)
    own_mean = own.float().mean(dim=-1)
    accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk)
        kb = k1 - k0
        Pyb, Pxb = Py[k0:k1].view(kb, 1, 1), Px[k0:k1].view(kb, 1, 1)
        Nty, Ntx = Ty.unsqueeze(0) - Pyb, Tx.unsqueeze(0) - Pxb
        inside = (Ntx >= 0) & (Ntx < W_t) & (Nty >= 0) & (Nty < H_t)
        nny, nnx = Nty.clamp(0, H_t - 1), Ntx.clamp(0, W_t - 1)
        sx = nnf_index[nny, nnx, 0] + Pxb
        sy = nnf_index[nny, nnx, 1] + Pyb
        inside &= (sx >= 0) & (sx < W_s) & (sy >= 0) & (sy < H_s)
        vals = source_style[sy.clamp(0, H_s - 1), sx.clamp(0, W_s - 1)].float()
        colour_sq = (vals.mean(dim=-1) - own_mean.unsqueeze(0)).square()
        w = torch.exp(-spatial_sq[k0:k1].view(kb, 1, 1) * inv_ss - colour_sq * inv_sc)
        if error_map is not None:
            w = (1.0 / (1.0 + error_map[nny, nnx])) * w
        w = w * inside.to(torch.float32)
        accumulator += (vals * w.unsqueeze(-1)).sum(dim=0)
        weight_sum += w.sum(dim=0)

    voted = (
        (accumulator / weight_sum.unsqueeze(-1)).clamp(0, 255).to(source_style.dtype)
    )
    return torch.where((weight_sum > 1e-4).unsqueeze(-1), voted, own)


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
    Reconstruct target image by averaging overlapping patches (source coordinates
    clamped to the source). ``use_bilateral``: :func:`_vote_bilateral`.
    """
    device = source_style.device
    if use_bilateral:
        with _mp_region("vote:plain_bilateral", device):
            return _vote_bilateral(
                source_style, nnf, None, patch_size, sigma_spatial, sigma_color
            )

    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    nnf_index = nnf if nnf.dtype == torch.long else nnf.to(torch.long)
    Ty, Tx = _vote_target_grid(H_t, W_t, device)

    with _mp_region("vote:plain", device):
        Py, Px = _vote_flat_offsets(patch_size, device)
        K = Py.numel()
        chunk = _vote_chunk_k(H_t, W_t, C, K)
        accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)
        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            kb = k1 - k0
            Pyb, Pxb = Py[k0:k1].view(kb, 1, 1), Px[k0:k1].view(kb, 1, 1)
            Nty, Ntx = Ty.unsqueeze(0) - Pyb, Tx.unsqueeze(0) - Pxb
            valid = (Ntx >= 0) & (Ntx < W_t) & (Nty >= 0) & (Nty < H_t)
            nny, nnx = Nty.clamp(0, H_t - 1), Ntx.clamp(0, W_t - 1)
            sx = (nnf_index[nny, nnx, 0] + Pxb).clamp(0, W_s - 1)
            sy = (nnf_index[nny, nnx, 1] + Pyb).clamp(0, H_s - 1)
            vf = valid.to(torch.float32)
            accumulator += (source_style[sy, sx].float() * vf.unsqueeze(-1)).sum(dim=0)
            weight_sum += vf.sum(dim=0)
        target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)
        target_style = target_style.clamp(0, 255).to(source_style.dtype)

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
    alternatives: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Voting weighted by ``1 / (1 + error)`` of each voting patch (source coordinates
    clamped). ``use_bilateral``: :func:`_vote_bilateral` with the best matches only.

    ``alternatives`` = ``(nnf_k (H_t, W_t, S, 2), errors_k (H_t, W_t, S))`` adds the
    multicand alternative matches as extra voters with the same ``1 / (1 + error)``
    weights (empty slots carry ``inf`` errors and get zero weight).
    """
    device = source_style.device
    if use_bilateral:
        with _mp_region("vote:weighted_bilateral", device):
            return _vote_bilateral(
                source_style, nnf, error_map, patch_size, sigma_spatial, sigma_color
            )

    H_t, W_t = nnf.shape[:2]
    H_s, W_s, C = source_style.shape
    nnf_index = nnf if nnf.dtype == torch.long else nnf.to(torch.long)
    Ty, Tx = _vote_target_grid(H_t, W_t, device)

    with _mp_region("vote:weighted", device):
        Py, Px = _vote_flat_offsets(patch_size, device)
        K = Py.numel()
        chunk = _vote_chunk_k(H_t, W_t, C, K)
        accumulator = torch.zeros((H_t, W_t, C), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((H_t, W_t), dtype=torch.float32, device=device)

        layers = [(nnf_index, error_map)]
        if alternatives is not None:
            alt_nnf, alt_errors = alternatives
            layers.extend(
                (alt_nnf[:, :, s], alt_errors[:, :, s]) for s in range(alt_nnf.shape[2])
            )

        for layer_coords, layer_errors in layers:
            layer_nnf = (
                layer_coords
                if layer_coords.dtype == torch.long
                else layer_coords.to(torch.long)
            )
            for k0 in range(0, K, chunk):
                k1 = min(K, k0 + chunk)
                kb = k1 - k0
                Pyb, Pxb = Py[k0:k1].view(kb, 1, 1), Px[k0:k1].view(kb, 1, 1)
                Nty, Ntx = Ty.unsqueeze(0) - Pyb, Tx.unsqueeze(0) - Pxb
                valid = (Ntx >= 0) & (Ntx < W_t) & (Nty >= 0) & (Nty < H_t)
                nny, nnx = Nty.clamp(0, H_t - 1), Ntx.clamp(0, W_t - 1)
                sx = (layer_nnf[nny, nnx, 0] + Pxb).clamp(0, W_s - 1)
                sy = (layer_nnf[nny, nnx, 1] + Pyb).clamp(0, H_s - 1)
                w = valid.to(torch.float32) / (1.0 + layer_errors[nny, nnx])
                accumulator += (source_style[sy, sx].float() * w.unsqueeze(-1)).sum(
                    dim=0
                )
                weight_sum += w.sum(dim=0)

        target_style = accumulator / weight_sum.unsqueeze(2).clamp(min=1e-6)
        target_style = target_style.clamp(0, 255).to(source_style.dtype)

    return target_style
