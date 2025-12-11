import math

import torch
import torch.nn.functional as F

__all__ = [
    "_calc_mls_delta",
    "warp_mls_similarity",
]


def _calc_mls_delta(
    src_points: torch.Tensor,
    dst_points: torch.Tensor,
    out_h: int,
    out_w: int,
    grid_points_x: list[int] | None = None,
    grid_points_y: list[int] | None = None,
    grid_size: int = 10,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute displacement field at grid nodes (only nodes are filled)."""
    device = device or src_points.device
    # Follow C++ ImgWarp_MLS: oldDotL = dst (target), newDotL = src (style)
    new_pts = src_points.to(device, dtype=torch.float32)  # style/source
    old_pts = dst_points.to(device, dtype=torch.float32)  # target/destination

    if grid_points_x is not None:
        xs = grid_points_x
    else:
        xs = list(range(0, out_w, grid_size))
        if xs[-1] != out_w - 1:
            xs.append(out_w - 1)

    if grid_points_y is not None:
        ys = grid_points_y
    else:
        ys = list(range(0, out_h, grid_size))
        if ys[-1] != out_h - 1:
            ys.append(out_h - 1)

    # Output grid shape matches the provided grid points
    grid_h = len(ys)
    grid_w = len(xs)

    # Build grid as a single tensor so computations can be vectorized
    xs_t = torch.tensor(xs, device=device, dtype=torch.float32)
    ys_t = torch.tensor(ys, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys_t, xs_t, indexing="ij")
    grid_pts = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (G, 2) in (x, y)

    # Pairwise differences to destination landmarks
    diff = grid_pts[:, None, :] - old_pts[None, :, :]  # (G, N, 2)
    dist2 = (diff * diff).sum(dim=-1)  # (G, N)

    # Handle exact matches (grid point coincides with control point)
    has_anchor = dist2 == 0
    anchor_hit = has_anchor.any(dim=1)
    anchor_idx = dist2.argmin(dim=1)

    rdx_flat = torch.zeros(grid_pts.shape[0], device=device, dtype=torch.float32)
    rdy_flat = torch.zeros_like(rdx_flat)

    if anchor_hit.any():
        anchor_targets = new_pts[anchor_idx[anchor_hit]]
        anchor_pts = grid_pts[anchor_hit]
        rdx_flat[anchor_hit] = anchor_targets[:, 0] - anchor_pts[:, 0]
        rdy_flat[anchor_hit] = anchor_targets[:, 1] - anchor_pts[:, 1]

    # Remaining grid nodes: full MLS computation (vectorized)
    work_mask = ~anchor_hit
    if work_mask.any():
        gp = grid_pts[work_mask]  # (M, 2)
        dist2_work = dist2[work_mask]  # (M, N)
        w = 1.0 / dist2_work

        sw = w.sum(dim=1, keepdim=True)  # (M, 1)
        pstar = (w @ old_pts) / sw  # target mean (M, 2)
        qstar = (w @ new_pts) / sw  # source mean (M, 2)

        pi = old_pts.unsqueeze(0) - pstar.unsqueeze(1)  # (M, N, 2)
        pij = torch.stack([-pi[..., 1], pi[..., 0]], dim=-1)  # (M, N, 2)

        miu_s = (w * (pi * pi).sum(dim=-1)).sum(dim=1)  # (M,)
        miu_s = miu_s.clamp_min(1e-8)  # avoid divide-by-zero in degenerate cases

        cur_v = gp - pstar  # (M, 2)
        cur_vj = torch.stack([-cur_v[:, 1], cur_v[:, 0]], dim=-1)  # (M, 2)

        # Dot products for the MLS similarity transform
        dot_pi_cv = (pi * cur_v.unsqueeze(1)).sum(dim=-1)  # (M, N)
        dot_pij_cv = (pij * cur_v.unsqueeze(1)).sum(dim=-1)  # (M, N)
        dot_pi_cvj = (pi * cur_vj.unsqueeze(1)).sum(dim=-1)  # (M, N)
        dot_pij_cvj = (pij * cur_vj.unsqueeze(1)).sum(dim=-1)  # (M, N)

        new_x = new_pts[:, 0]
        new_y = new_pts[:, 1]

        tmp_x = (dot_pi_cv * new_x - dot_pij_cv * new_y) * w / miu_s.unsqueeze(1)
        tmp_y = (-dot_pi_cvj * new_x + dot_pij_cvj * new_y) * w / miu_s.unsqueeze(1)

        new_p = torch.stack(
            [tmp_x.sum(dim=1) + qstar[:, 0], tmp_y.sum(dim=1) + qstar[:, 1]], dim=1
        )  # (M, 2)

        rdx_flat[work_mask] = new_p[:, 0] - gp[:, 0]
        rdy_flat[work_mask] = new_p[:, 1] - gp[:, 1]

    rdx = rdx_flat.view(grid_h, grid_w)
    rdy = rdy_flat.view(grid_h, grid_w)
    return rdx, rdy


def warp_mls_similarity(
    image: torch.Tensor,
    src_points: torch.Tensor,
    dst_points: torch.Tensor,
    *,
    grid_size: int = 10,
) -> torch.Tensor:
    """Warp image using Moving Least Squares (similarity)."""
    if image.ndim == 3:
        img = image.unsqueeze(0)
    else:
        img = image
    _, c, h, w = img.shape
    device = img.device
    # src_points/dst_points are expected as (x, y)
    device = img.device
    # src_points/dst_points are expected as (x, y)

    # Fix: Use strictly uniform grid extended beyond boundaries to avoid F.interpolate distortion
    # We want grid points at 0, 10, 20... ensuring we cover the whole image.
    max_x = int(math.ceil((w - 1) / grid_size) * grid_size)
    max_y = int(math.ceil((h - 1) / grid_size) * grid_size)

    xs = list(range(0, max_x + 1, grid_size))
    ys = list(range(0, max_y + 1, grid_size))

    rdx, rdy = _calc_mls_delta(
        src_points,
        dst_points,
        h,
        w,
        grid_points_x=xs,
        grid_points_y=ys,
        grid_size=grid_size,
        device=device,
    )

    # rdx/rdy are now shape (len(ys), len(xs)) - the coarse grid

    # Interpolate delta field to dense resolution of the EXTENDED size
    # shape (1, 2, grid_h, grid_w)
    coarse = torch.stack([rdx, rdy], dim=0).unsqueeze(0)

    # Target size is (max_y + 1, max_x + 1) to cover all pixels from 0 to max coordinate
    # with 1:1 mapping at integer coords because we use align_corners=True
    extended_h = max_y + 1
    extended_w = max_x + 1

    dense_delta = F.interpolate(
        coarse, size=(extended_h, extended_w), mode="bilinear", align_corners=True
    )

    # Crop to actual image size
    dense_delta = dense_delta[:, :, :h, :w]

    dense_dx = dense_delta[:, 0].squeeze(0)
    dense_dy = dense_delta[:, 1].squeeze(0)

    base_x = torch.linspace(0, w - 1, w, device=device)
    base_y = torch.linspace(0, h - 1, h, device=device)
    base_grid_y, base_grid_x = torch.meshgrid(base_y, base_x, indexing="ij")

    sample_x = base_grid_x + dense_dx
    sample_y = base_grid_y + dense_dy
    sample_x = sample_x.clamp(0, w - 1)
    sample_y = sample_y.clamp(0, h - 1)

    norm_x = (sample_x / (w - 1)) * 2 - 1
    norm_y = (sample_y / (h - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (x, y) order

    # C++ interpolation is often closer to align_corners=False for images
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return warped
