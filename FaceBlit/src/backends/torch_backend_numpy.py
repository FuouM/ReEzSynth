"""NumPy / SciPy helpers for FaceBlit (no PyTorch, no Numba).

Used by :mod:`torch_backend` for CPU paths, LUT reference implementation, I/O, and geometry.
Numba-accelerated CPU paths live in :mod:`torch_backend_numba` and are wired from ``torch_backend``.
"""

from __future__ import annotations

import heapq
from collections import deque
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from src.utils_io import ensure_grayscale, to_uint8

PathLike = Union[str, Path]


def compute_guided_error(
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_guide: np.ndarray | None,
    target_pos: Tuple[int, int],
    style_pos: Tuple[int, int],
    lambda_pos: int,
    lambda_app: int,
) -> int:
    """Compute guided error between target and style pixels."""
    ty, tx = target_pos
    sy, sx = style_pos

    # Position error (L1 distance in position guide space).
    # Guides are uint8; subtract in int32 to avoid unsigned wrap / overflow warnings.
    tr = int(target_pos_guide[ty, tx, 2])
    tg = int(target_pos_guide[ty, tx, 1])
    sr = int(style_pos_guide[sy, sx, 2])
    sg = int(style_pos_guide[sy, sx, 1])
    pos_error = abs(tr - sr) + abs(tg - sg)

    # Appearance error (optionally disabled)
    app_error = 0
    if target_app_guide is not None and style_app_guide is not None:
        ta = int(target_app_guide[ty, tx])
        sa = int(style_app_guide[sy, sx])
        app_error = abs(ta - sa)

    return int(lambda_pos * int(pos_error) + lambda_app * int(app_error))


def dfs_seed_grow(
    target_seed_point: Tuple[int, int],
    style_seed_point: Tuple[int, int],
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_guide: np.ndarray | None,
    result_img: np.ndarray,
    style_img: np.ndarray,
    covered_pixels: np.ndarray,
    chunk_number: int,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
) -> None:
    """Perform DFS seed growing from seed point."""
    h, w = target_pos_guide.shape[:2]

    # Priority queue: (error, y, x)
    pq = []
    heapq.heappush(pq, (0, target_seed_point[0], target_seed_point[1]))

    # Track visited pixels
    visited = np.zeros((h, w), dtype=bool)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

    while pq:
        error, ty, tx = heapq.heappop(pq)

        if visited[ty, tx] or covered_pixels[ty, tx] != 0:
            continue

        visited[ty, tx] = True

        # Mark as covered and assign style color
        covered_pixels[ty, tx] = chunk_number
        sy, sx = style_seed_point
        if 0 <= sy < style_img.shape[0] and 0 <= sx < style_img.shape[1]:
            result_img[ty, tx] = style_img[sy, sx]

        # Explore neighbors
        for dy, dx in directions:
            ny, nx = ty + dy, tx + dx
            if (
                0 <= ny < h
                and 0 <= nx < w
                and not visited[ny, nx]
                and covered_pixels[ny, nx] == 0
            ):
                # Compute error for this neighbor
                neighbor_error = compute_guided_error(
                    style_pos_guide,
                    target_pos_guide,
                    style_app_guide,
                    target_app_guide,
                    (ny, nx),
                    style_seed_point,
                    lambda_pos,
                    lambda_app,
                )

                if neighbor_error <= threshold:
                    heapq.heappush(pq, (neighbor_error, ny, nx))


def _pixel_out_of_range(
    style_pixel: Tuple[int, int], target_pixel: Tuple[int, int], shape: Tuple[int, int]
) -> bool:
    """Mirror C++ pixelOutOfImageRange: both coords are checked against target dims."""
    h, w = shape
    sy, sx = style_pixel
    ty, tx = target_pixel
    return (
        sx < 0
        or tx < 0
        or sy < 0
        or ty < 0
        or sx > w - 1
        or tx > w - 1
        or sy > h - 1
        or ty > h - 1
    )


def _compute_style_seed_point(
    target_pos_guide: np.ndarray,
    target_app_guide: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    row_t: int,
    col_t: int,
    style_shape: Tuple[int, int],
    lambda_app: int,
) -> tuple[int, int]:
    """Match C++ computeStyleSeedPoint (pos-only fallback when appearance disabled)."""
    style_h, style_w = style_shape
    pos_vals = target_pos_guide[row_t, col_t]
    pos_r = int(pos_vals[2])
    pos_g = int(pos_vals[1])

    if lambda_app == 0 or target_app_guide is None or look_up_cube is None:
        x_norm = style_w / 256.0
        y_norm = style_h / 256.0
        sx = int(pos_r * x_norm)
        sy = int(pos_g * y_norm)
    else:
        app_val = int(ensure_grayscale(target_app_guide)[row_t, col_t])
        coords = look_up_cube[pos_r, pos_g, app_val]
        sx = int(coords[0])
        sy = int(coords[1])

    # Clamp to style dimensions to avoid OOB access
    sx = int(np.clip(sx, 0, style_w - 1))
    sy = int(np.clip(sy, 0, style_h - 1))
    return sy, sx  # return as (y, x)


def dfs_seed_grow_voting_numpy(
    target_seed_point: Tuple[int, int],
    style_seed_point: Tuple[int, int],
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_guide: np.ndarray | None,
    nnf: np.ndarray,
    covered_pixels: np.ndarray,
    chunk_number: int,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
) -> None:
    """Replica of C++ DFSSeedGrow_voting using a FIFO queue (pure Python / NumPy)."""
    # print("DFS PYTHON")
    h, w = target_pos_guide.shape[:2]
    q: deque[tuple[int, int]] = deque()
    q.append((0, 0))

    while q:
        dy, dx = q.popleft()
        ty = target_seed_point[0] + dy
        tx = target_seed_point[1] + dx
        sy = style_seed_point[0] + dy
        sx = style_seed_point[1] + dx

        if _pixel_out_of_range((sy, sx), (ty, tx), (h, w)):
            continue
        if covered_pixels[ty, tx] != 0:
            continue

        error = compute_guided_error(
            style_pos_guide,
            target_pos_guide,
            style_app_guide,
            target_app_guide,
            (ty, tx),
            (sy, sx),
            lambda_pos,
            lambda_app,
        )

        if error < threshold or (dy == 0 and dx == 0):
            nnf[ty, tx, 0] = sy
            nnf[ty, tx, 1] = sx
            covered_pixels[ty, tx] = chunk_number

            q.append((dy - 1, dx))
            q.append((dy + 1, dx))
            q.append((dy, dx - 1))
            q.append((dy, dx + 1))


def initialize_nnf_vectorized(
    target_pos_guide: np.ndarray,
    target_app_guide: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    style_shape: Tuple[int, int],
    stylization_rect: Tuple[int, int, int, int],
    # kept for call-site compatibility; unused
) -> np.ndarray:
    """LUT-based NNF for every target pixel (full frame; head-only init left borders black under voting)."""
    h_t, w_t = target_pos_guide.shape[:2]
    nnf = np.zeros((h_t, w_t, 2), dtype=np.int32)

    coords_x, coords_y = _lookup_coords_from_guides(
        target_pos_guide, target_app_guide, look_up_cube, style_shape
    )

    nnf[..., 0] = coords_y
    nnf[..., 1] = coords_x

    return nnf


def prepare_dfs_voting_arrays(
    target_pos_guide: np.ndarray,
    target_app_gray: np.ndarray | None,
    h_t: int,
    w_t: int,
    x: int,
    y: int,
    h: int,
    w: int,
    look_up_cube: np.ndarray | None,
    style_h: int,
    style_w: int,
    lambda_app: int,
):
    covered_pixels = np.zeros((h_t, w_t), dtype=np.int32)

    seeds: list[tuple[int, int, tuple[int, int]]] = []
    for row_t in range(y, y + h):
        for col_t in range(x, x + w):
            if covered_pixels[row_t, col_t] != 0:
                continue
            style_seed_point = _compute_style_seed_point(
                target_pos_guide,
                target_app_gray,
                look_up_cube,
                row_t,
                col_t,
                (style_h, style_w),
                lambda_app,
            )
            seeds.append((row_t, col_t, style_seed_point))

    return covered_pixels, seeds


def denoise_nnf_python(nnf: np.ndarray, patch_size: int) -> np.ndarray:
    """Mode filter on the nearest-neighbour field (matches C++ denoiseNNF)."""
    h, w, _ = nnf.shape
    half = patch_size // 2
    result = nnf.copy()

    for row in range(h):
        for col in range(w):
            counts: dict[tuple[int, int], int] = {}
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    ry = row + dy
                    rx = col + dx
                    if ry < 0 or ry >= h or rx < 0 or rx >= w:
                        continue
                    nearest = nnf[ry, rx]
                    offset = (int(nearest[0]) - ry, int(nearest[1]) - rx)
                    counts[offset] = counts.get(offset, 0) + 1

            if counts:
                best_offset = max(counts.items(), key=lambda kv: kv[1])[0]
                result[row, col, 0] = best_offset[0] + row
                result[row, col, 1] = best_offset[1] + col

    return result


def build_nnf_dfs_voting_numpy(
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_gray: np.ndarray | None,
    h_t: int,
    w_t: int,
    threshold: int | None,
    x: int,
    y: int,
    h: int,
    w: int,
    look_up_cube: np.ndarray | None,
    style_h: int,
    style_w: int,
    lambda_app: int,
    lambda_pos: int,
    patch_size: int,
) -> np.ndarray:
    """DFS-based region growing NNF, then :func:`denoise_nnf` (pure Python / NumPy)."""
    (
        style_pos_int,
        target_pos_int,
        style_app_int,
        target_app_int,
        nnf,
        covered_pixels,
        seeds,
        thresh,
        _use_app,
    ) = prepare_dfs_voting_arrays(
        style_pos_guide,
        target_pos_guide,
        style_app_guide,
        target_app_gray,
        h_t,
        w_t,
        threshold,
        x,
        y,
        h,
        w,
        look_up_cube,
        style_h,
        style_w,
        lambda_app,
    )

    chunk_number = 1
    for row_t, col_t, style_seed_point in seeds:
        if covered_pixels[row_t, col_t] != 0:
            continue
        target_seed_point = (row_t, col_t)
        dfs_seed_grow_voting_numpy(
            target_seed_point,
            style_seed_point,
            style_pos_int,
            target_pos_int,
            style_app_int,
            target_app_int,
            nnf,
            covered_pixels,
            chunk_number,
            thresh,
            lambda_pos,
            lambda_app,
        )
        chunk_number += 1

    return denoise_nnf_python(nnf, patch_size)


def voting_on_rgb(
    style_image: np.ndarray, nnf: np.ndarray, patch_size: int
) -> np.ndarray:
    """Patch voting as in C++ votingOnRGB, optimized vectorized version."""
    h, w, _ = nnf.shape
    style_h, style_w = style_image.shape[:2]
    half = patch_size // 2

    # Accumulators use int32 to avoid overflow (faster than int64 for small patch sizes)
    acc = np.zeros((h, w, 3), dtype=np.int32)
    counts = np.zeros((h, w), dtype=np.int32)

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            # Compute valid neighbor region
            ry_min = max(0, -dy)
            ry_max = min(h, h - dy)
            rx_min = max(0, -dx)
            rx_max = min(w, w - dx)

            if ry_min >= ry_max or rx_min >= rx_max:
                continue

            # Slice the valid region
            neighbor_slice = (
                slice(ry_min + dy, ry_max + dy),
                slice(rx_min + dx, rx_max + dx),
            )

            # Get NNF values for neighbors
            nearest = nnf[neighbor_slice]
            row_style = nearest[..., 0] - dy
            col_style = nearest[..., 1] - dx

            # Validate style coordinates
            valid_mask = (
                (row_style >= 0)
                & (row_style < style_h)
                & (col_style >= 0)
                & (col_style < style_w)
            )

            if not np.any(valid_mask):
                continue

            # Sample style image
            row_style_valid = row_style[valid_mask]
            col_style_valid = col_style[valid_mask]
            vals = style_image[row_style_valid, col_style_valid].astype(np.int32)

            # Create target indices for accumulation
            target_rows, target_cols = np.where(valid_mask)
            target_rows += ry_min
            target_cols += rx_min

            # Accumulate
            np.add.at(acc[..., 0], (target_rows, target_cols), vals[:, 0])
            np.add.at(acc[..., 1], (target_rows, target_cols), vals[:, 1])
            np.add.at(acc[..., 2], (target_rows, target_cols), vals[:, 2])
            np.add.at(counts, (target_rows, target_cols), 1)

    output = np.zeros((h, w, 3), dtype=np.uint8)
    mask = counts > 0
    if np.any(mask):
        counts_safe = counts.clip(min=1)
        output[mask] = (acc[mask] // counts_safe[mask, None]).astype(np.uint8)
    return output


def _lookup_coords_from_guides(
    target_pos: np.ndarray,
    target_app: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    style_shape: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h_s, w_s = style_shape
    pos_r = to_uint8(target_pos[..., 2])
    pos_g = to_uint8(target_pos[..., 1])

    if look_up_cube is not None and target_app is not None:
        app = ensure_grayscale(target_app)
        coords = look_up_cube[pos_r, pos_g, app]
        x = coords[..., 0].astype(np.int32)
        y = coords[..., 1].astype(np.int32)
    else:
        x = np.round((pos_r.astype(np.float32) / 255.0) * (w_s - 1)).astype(np.int32)
        y = np.round((pos_g.astype(np.float32) / 255.0) * (h_s - 1)).astype(np.int32)

    x_clipped = np.clip(x, 0, w_s - 1)
    y_clipped = np.clip(y, 0, h_s - 1)

    return x_clipped, y_clipped


def _apply_rect_mask(
    rect: Tuple[int, int, int, int], shape: Tuple[int, int]
) -> np.ndarray:
    x, y, w, h = rect
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


def style_blit(
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_guide: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    style_image: np.ndarray,
    stylization_rect: Tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    h_t, w_t = target_pos_guide.shape[:2]
    style_h, style_w = style_image.shape[:2]

    coords_x, coords_y = _lookup_coords_from_guides(
        target_pos_guide, target_app_guide, look_up_cube, (style_h, style_w)
    )

    result = np.zeros((h_t, w_t, 3), dtype=np.uint8)
    sample = style_image[coords_y, coords_x]
    if stylization_rect is not None:
        mask = _apply_rect_mask(stylization_rect, (h_t, w_t))
        result[mask] = sample[mask]
    else:
        result = sample

    return result


def compute_look_up_cube(
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    lambda_pos: int = 10,
    lambda_app: int = 2,
    search_radius: int = 30,
) -> np.ndarray:
    """Match C++ getLookUpCube: local search + app smoothing."""
    pos = to_uint8(style_pos_guide)
    app = ensure_grayscale(style_app_guide)
    h, w = app.shape
    pos_r = pos[..., 2].astype(np.int32)
    pos_g = pos[..., 1].astype(np.int32)

    look_up = np.zeros((256, 256, 256, 2), dtype=np.uint16)
    x_norm = w / 256.0
    y_norm = h / 256.0
    use_app = lambda_app != 0 and style_app_guide is not None
    k_inf = np.iinfo(np.int32).max // 4

    for x in range(256):  # pos red
        seed_col = int(x * x_norm)
        col_start = max(0, seed_col - search_radius)
        col_end = min(w, seed_col + search_radius)
        for y in range(256):  # pos green
            seed_row = int(y * y_norm)
            row_start = max(0, seed_row - search_radius)
            row_end = min(h, seed_row + search_radius)

            seed_coord = np.array([seed_col, seed_row], dtype=np.uint16)

            if not use_app:
                look_up[x, y, :, 0] = seed_coord[0]
                look_up[x, y, :, 1] = seed_coord[1]
                continue

            best_base = np.full(256, k_inf, dtype=np.int32)
            best_coord = np.tile(seed_coord[None, :], (256, 1)).astype(np.int32)

            pos_r_patch = pos_r[row_start:row_end, col_start:col_end]
            pos_g_patch = pos_g[row_start:row_end, col_start:col_end]
            app_patch = app[row_start:row_end, col_start:col_end]

            base_err = (np.abs(pos_g_patch - y) + np.abs(pos_r_patch - x)) * lambda_pos

            flat_err = base_err.reshape(-1)
            flat_app = app_patch.reshape(-1)
            flat_rows, flat_cols = np.meshgrid(
                np.arange(row_start, row_end),
                np.arange(col_start, col_end),
                indexing="ij",
            )
            flat_rows = flat_rows.reshape(-1)
            flat_cols = flat_cols.reshape(-1)

            unique_apps = np.unique(flat_app)
            for a in unique_apps:
                mask = flat_app == a
                errs = flat_err[mask]
                if errs.size == 0:
                    continue
                idx = int(np.argmin(errs))
                if errs[idx] < best_base[a]:
                    best_base[a] = int(errs[idx])
                    best_coord[a] = np.array(
                        [flat_cols[mask][idx], flat_rows[mask][idx]], dtype=np.int32
                    )

            forward_cost = best_base.copy()
            backward_cost = best_base.copy()
            forward_idx = np.arange(256, dtype=np.int32)
            backward_idx = np.arange(256, dtype=np.int32)

            for i in range(1, 256):
                cand = min(int(forward_cost[i - 1]) + lambda_app, k_inf)
                if cand < forward_cost[i]:
                    forward_cost[i] = cand
                    forward_idx[i] = forward_idx[i - 1]

            for i in range(254, -1, -1):
                cand = min(int(backward_cost[i + 1]) + lambda_app, k_inf)
                if cand < backward_cost[i]:
                    backward_cost[i] = cand
                    backward_idx[i] = backward_idx[i + 1]

            for z in range(256):
                best_cost = forward_cost[z]
                best_idx = forward_idx[z]
                if backward_cost[z] < best_cost:
                    best_cost = backward_cost[z]
                    best_idx = backward_idx[z]

                if best_cost >= k_inf:
                    look_up[x, y, z] = seed_coord
                else:
                    look_up[x, y, z, 0] = np.uint16(best_coord[best_idx][0])
                    look_up[x, y, z, 1] = np.uint16(best_coord[best_idx][1])

    return look_up
