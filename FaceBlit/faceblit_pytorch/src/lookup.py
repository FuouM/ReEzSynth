from collections import deque
from pathlib import Path

import numpy as np
import torch

from .image_utils import _ensure_grayscale, _to_uint8

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional

    def tqdm(x, **kwargs):
        return x


try:  # pragma: no cover - used only when available
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None

__all__ = [
    "_build_seed_maps",
    "_distance_transform_with_indices",
    "compute_look_up_cube",
    "compute_look_up_cube_optimized",
    "save_look_up_cube",
    "load_look_up_cube",
]


def _build_seed_maps(
    pos_r: np.ndarray, pos_g: np.ndarray, app: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-app seed maps in bin space."""
    flat_bins = (pos_g.astype(np.int32).ravel() * 256) + pos_r.astype(np.int32).ravel()
    flat_app = app.astype(np.uint8).ravel()
    seeds = {}
    for z in np.unique(flat_app):
        mask_z = flat_app == z
        bins_z = flat_bins[mask_z]
        if bins_z.size == 0:
            continue
        # First occurrence per bin
        unique_bins, first_idx = np.unique(bins_z, return_index=True)
        src_idx = np.nonzero(mask_z)[0][first_idx]
        rows = src_idx // pos_r.shape[1]
        cols = src_idx % pos_r.shape[1]
        y_bins = (unique_bins // 256).astype(np.int32)
        x_bins = (unique_bins % 256).astype(np.int32)
        seeds[int(z)] = (
            y_bins,
            x_bins,
            np.stack([cols.astype(np.int32), rows.astype(np.int32)], axis=-1),
        )
    return seeds


def _distance_transform_with_indices(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distance, nearest_y, nearest_x) for a binary mask of seeds."""
    if mask.shape != (256, 256):
        raise ValueError("mask must be 256x256 in bin space")
    if not mask.any():
        return (
            np.full_like(mask, np.inf, dtype=np.float32),
            np.full_like(mask, -1, dtype=np.int16),
            np.full_like(mask, -1, dtype=np.int16),
        )

    if distance_transform_edt is not None:
        dist, indices = distance_transform_edt(~mask, return_indices=True)
        nearest_y = indices[0].astype(np.int16)
        nearest_x = indices[1].astype(np.int16)
        return dist.astype(np.float32), nearest_y, nearest_x

    # Fallback BFS (Manhattan distance)
    dist = np.full(mask.shape, np.inf, dtype=np.float32)
    nearest_y = np.full(mask.shape, -1, dtype=np.int16)
    nearest_x = np.full(mask.shape, -1, dtype=np.int16)
    q: deque[tuple[int, int]] = deque()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        dist[y, x] = 0.0
        nearest_y[y, x] = y
        nearest_x[y, x] = x
        q.append((y, x))

    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 256 and 0 <= nx < 256:
                if dist[ny, nx] > dist[y, x] + 1:
                    dist[ny, nx] = dist[y, x] + 1
                    nearest_y[ny, nx] = nearest_y[y, x]
                    nearest_x[ny, nx] = nearest_x[y, x]
                    q.append((ny, nx))
    return dist, nearest_y, nearest_x


def compute_look_up_cube(
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    lambda_pos: int = 10,
    lambda_app: int = 2,
    search_radius: int = 30,
) -> np.ndarray:
    """Match C++ getLookUpCube: local search + app smoothing."""
    pos = _to_uint8(style_pos_guide)
    app = _ensure_grayscale(style_app_guide)
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

            def safe_add(v: int) -> int:
                if v >= k_inf - lambda_app:
                    return k_inf
                return v + lambda_app

            for i in range(1, 256):
                cand = safe_add(int(forward_cost[i - 1]))
                if cand < forward_cost[i]:
                    forward_cost[i] = cand
                    forward_idx[i] = forward_idx[i - 1]

            for i in range(254, -1, -1):
                cand = safe_add(int(backward_cost[i + 1]))
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


def compute_look_up_cube_optimized(
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    lambda_pos: int = 10,
    lambda_app: int = 2,
    search_radius: int = 30,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Vectorized version of compute_look_up_cube using PyTorch.
    Support GPU acceleration (CUDA/MPS).
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)

    # Prepare inputs
    if style_pos_guide.ndim == 3 and style_pos_guide.shape[2] == 3:
        # BGR to RGB? No, guide generation already handles order locally,
        # but here we just need channels. original uses BGR indices [2] and [1].
        # pos[..., 2] is Red (x index), pos[..., 1] is Green (y index).
        pos_r = torch.from_numpy(style_pos_guide[..., 2].astype(np.int32)).to(device)
        pos_g = torch.from_numpy(style_pos_guide[..., 1].astype(np.int32)).to(device)
    else:
        raise ValueError("style_pos_guide must be HxWx3")

    app = torch.from_numpy(style_app_guide.astype(np.int32)).to(device)
    h, w = app.shape

    # Output buffer: (256, 256, 256, 2)
    # Dimensions: [target_r(x), target_g(y), target_app(z), coords(2)]
    # Use int32 for intermediate storage
    output = torch.zeros((256, 256, 256, 2), dtype=torch.int16, device=device)

    # Precompute norms
    x_norm = w / 256.0
    y_norm = h / 256.0

    # We iterate over target_x (0..255)
    # And process all target_y (0..255) in parallel vectors

    # Optimization: To avoid OOM, processing one X at a time is safe.
    # Total memory for 256 Ys * patch_size is small.

    k_inf = 2**30

    # Grid for y iteration
    y_indices = torch.arange(256, device=device, dtype=torch.int32)  # [0..255]

    # Prepare constants
    l_pos = torch.tensor(lambda_pos, device=device, dtype=torch.int32)
    l_app = torch.tensor(lambda_app, device=device, dtype=torch.int32)

    for x in tqdm(range(256), desc="Computing LUT"):
        seed_col = int(x * x_norm)
        c_start = max(0, seed_col - search_radius)
        c_end = min(w, seed_col + search_radius)

        # Crop strips based on x
        strip_pos_r = pos_r[:, c_start:c_end]  # (H, W_strip)
        strip_pos_g = pos_g[:, c_start:c_end]
        strip_app = app[:, c_start:c_end]

        # Calculate row bounds for all Ys
        seed_rows = (y_indices.float() * y_norm).long()
        r_starts = (seed_rows - search_radius).clamp(min=0)
        r_ends = (seed_rows + search_radius).clamp(max=h)

        # We need to process each 'y' (0..255).
        # Since r_start/r_end varies per y, we can't easily do a rectangular batch
        # unless we pad or just extract the max union.
        # But 'search_radius' is small (30). The window height is ~60.
        # The strips are full height H.

        # Vectorized gather approach:
        # Create a unified buffer of all patches?
        # A simple loop over Y inside X is actually decently fast on GPU if operations are batched?
        # No, 256x256 = 65k iterations is too slow if loop is in Python.
        # We MUST batch Y.

        # Strategy:
        # Each 'y' needs a window [r_start[y]:r_end[y], :].
        # These windows overlap heavily.
        # We can construct a mask or gather indices.
        # Let's create a tensor of indices.

        max_h_window = search_radius * 2 + 1
        window_width = c_end - c_start

        # Construct gather indices (Batch=256, max_h, width)
        # Offsets relative to r_start
        # If r_end - r_start < max_h, we need masking.

        # Base rows: (256, 1) + (1, max_h)
        rel_rows = torch.arange(max_h_window, device=device).unsqueeze(0)  # (1, max_h)
        abs_rows = r_starts.unsqueeze(1) + rel_rows  # (256, max_h)

        # Mask for valid rows
        valid_mask = abs_rows < r_ends.unsqueeze(1)

        # Clamping for safety (masked values won't be used)
        abs_rows_clamped = abs_rows.clamp(max=h - 1)

        # Extract patches: (256, max_h, W_strip)
        # Using fancy indexing
        # Expand indices to match strip width
        # abs_rows_exp: (256, max_h, 1)

        # We need (256, max_h, W_strip)
        # strip is (H, W_strip)
        # We select rows from strip.

        # Gather rows
        # Flattening H for gather?
        # patches_r = strip_pos_r[abs_rows_clamped] # shape (256, max_h, W_strip)
        patches_r = strip_pos_r[abs_rows_clamped]
        patches_g = strip_pos_g[abs_rows_clamped]
        patches_app = strip_app[abs_rows_clamped]

        # Compute Error
        # base_err = (abs(pg - y) + abs(pr - x)) * lambda_pos
        # y is varying per batch (0..255), x is constant scalar

        # y needs to be broadcast: (256, 1, 1)
        y_view = y_indices.view(256, 1, 1)

        base_err = (torch.abs(patches_g - y_view) + torch.abs(patches_r - x)) * l_pos

        # Mask out invalid pixels (out of r_start/r_end bounds)
        # valid_mask is (256, max_h). Expand to width.
        full_mask = valid_mask.unsqueeze(2).expand_as(base_err)

        # Set error to infinity for invalid pixels
        base_err = torch.where(full_mask, base_err, torch.tensor(k_inf, device=device))

        # Now Reduction: For each batch 'b' (y), and each unique app value 'z', find min base_err.
        # We need the argmin (coordinates).
        # Grid X coords for patches: range(c_start, c_end) -> (1, 1, W_strip)
        # Grid Y coords for patches: abs_rows_clamped -> (256, max_h, 1)

        grid_cols = torch.arange(c_start, c_end, device=device, dtype=torch.int32).view(
            1, 1, -1
        )
        grid_rows = abs_rows_clamped  # (256, max_h)

        # Flatten everything per batch
        # (256, N_pixels)
        flat_err = base_err.view(256, -1)  # (256, K)
        flat_app = patches_app.view(256, -1)  # (256, K)
        flat_cols = grid_cols.expand(256, max_h_window, -1).reshape(256, -1)
        flat_rows = (
            grid_rows.unsqueeze(2)
            .expand(256, max_h_window, window_width)
            .reshape(256, -1)
        )

        # We need independent reduction per batch.
        # Offset app keys by batch index: batch_keys = app + batch_idx * 256
        batch_offset = (y_indices * 256).unsqueeze(1)  # (256, 1)
        flat_keys = (
            flat_app.long() + batch_offset.long()
        )  # (256, K) -> flatten to (256*K)

        # Pack (Error, Index) into int64 for argmin
        # Error is ~255*20 = 5000. Fits in 16 bits easily.
        # Index is just an identifier to retrieve row/col.
        # Actually we need to store (row, col).
        # Let's pack row/col into one int32: (row << 16) | col
        # Combine error: (error << 32) | (packed_coords)

        packed_coords = (flat_rows << 16) | flat_cols
        scores = (flat_err.long() << 32) | packed_coords.long()

        # Global flatten
        all_keys = flat_keys.view(-1)
        all_scores = scores.view(-1)

        # Scatter Reduce Min
        # Output buffer size: 256 * 256 (all possible Y * App combinations)
        # Initialize with infinity-like score
        init_val = torch.tensor(k_inf, device=device).long() << 32

        # torch.scatter_reduce_ is available in 1.12+, 'amin' mode
        # or scatter_reduce in 2.0
        # If unavailable, use custom scatter min?
        # Assuming torch 2.0+ based on user context

        min_scores = torch.full(
            (256 * 256,), init_val.item(), device=device, dtype=torch.int64
        )

        if hasattr(torch.Tensor, "scatter_reduce_"):
            if device.type == "mps" and all_scores.dtype == torch.int64:
                # MPS workaround for int64 atomic min
                min_scores_cpu = min_scores.cpu()
                min_scores_cpu.scatter_reduce_(
                    0,
                    all_keys.cpu(),
                    all_scores.cpu(),
                    reduce="amin",
                    include_self=True,
                )
                min_scores = min_scores_cpu.to(device)
            else:
                min_scores.scatter_reduce_(
                    0, all_keys, all_scores, reduce="amin", include_self=True
                )
        else:
            # Fallback for older torch versions if needed
            min_scores.scatter_reduce_(
                0, all_keys, all_scores, reduce="amin", include_self=True
            )

        # Unpack results: (256, 256) -> (Y, App)
        min_scores = min_scores.view(256, 256)

        # If best_err >= k_inf, it means no pixel found for that app bin.
        # In original code, we retain seed_coord if cost >= k_inf after smoothing.
        # The smoothing step handles this propagation.

        # Smoothing (Forward/Backward) without Python loops.
        # Reformulate DP as prefix/suffix minima with linear offset.
        def _smooth_scores(
            min_scores_local: torch.Tensor, work_device: torch.device
        ) -> tuple[torch.Tensor, torch.Tensor]:
            idx_range = torch.arange(256, device=work_device, dtype=torch.int64)
            l_app_long = l_app.to(device=work_device, dtype=torch.int64)

            best_costs = (min_scores_local >> 32).to(torch.int64)  # (256, 256)
            best_coords = (min_scores_local & 0xFFFFFFFF).to(torch.int32)

            # Forward prefix minima: min_{k<=z} cost[k] + l_app * (z-k)
            base_forward = best_costs - l_app_long * idx_range  # subtract offset
            f_vals, f_idx = torch.cummin(base_forward, dim=1)
            forward_costs = f_vals + l_app_long * idx_range
            forward_coords = best_coords.gather(1, f_idx)

            # Backward suffix minima via reversed arrays
            best_costs_rev = torch.flip(best_costs, dims=[1])
            best_coords_rev = torch.flip(best_coords, dims=[1])
            base_backward = best_costs_rev - l_app_long * idx_range
            b_vals_rev, b_idx_rev = torch.cummin(base_backward, dim=1)
            backward_costs_rev = b_vals_rev + l_app_long * idx_range
            backward_coords_rev = best_coords_rev.gather(1, b_idx_rev)

            backward_costs = torch.flip(backward_costs_rev, dims=[1])
            backward_coords = torch.flip(backward_coords_rev, dims=[1])

            use_forward = forward_costs <= backward_costs
            final_costs_local = torch.where(
                use_forward, forward_costs, backward_costs
            ).int()
            final_coords_local = torch.where(
                use_forward, forward_coords, backward_coords
            )
            return final_costs_local, final_coords_local

        if device.type == "mps":
            # cummin missing on MPS; run small DP on CPU then move back.
            final_costs_cpu, final_coords_cpu = _smooth_scores(
                min_scores.cpu(), torch.device("cpu")
            )
            final_costs = final_costs_cpu.to(device)
            final_coords = final_coords_cpu.to(device)
        else:
            final_costs, final_coords = _smooth_scores(min_scores, device)

        # Unpack final
        final_rows = (final_coords >> 16).short()
        final_cols = (final_coords & 0xFFFF).short()

        # Fill holes: if cost >= k_inf, use seed coord
        # Seed coord depends on Y (and X).
        # seed_rows: (256,)
        # seed_col: scalar

        seed_r = seed_rows.short().unsqueeze(1).expand(256, 256)
        seed_c = torch.full((256, 256), seed_col, device=device, dtype=torch.int16)

        invalid_mask = final_costs >= k_inf
        final_cols = torch.where(invalid_mask, seed_c, final_cols)
        final_rows = torch.where(invalid_mask, seed_r, final_rows)

        # Store in output: [x, y, z, 2]
        # output[x, :, :, 0] = final_cols
        # output[x, :, :, 1] = final_rows

        output[x, :, :, 0] = final_cols
        output[x, :, :, 1] = final_rows

    return output.cpu().numpy().astype(np.uint16)


def save_look_up_cube(lut: np.ndarray, path: Path | str) -> Path:
    arr = np.asarray(lut, dtype=np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return Path(path)


def load_look_up_cube(path: Path | str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = 256 * 256 * 256 * 2
    if data.size != expected:
        raise ValueError(f"Unexpected LUT size {data.size}, expected {expected}")
    return data.reshape((256, 256, 256, 2))
