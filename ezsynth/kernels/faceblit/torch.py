"""PyTorch FaceBlit kernels: LUT build, NNF denoising and voting on any torch device."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from FaceBlit.src.utils_io import ensure_grayscale, get_torch_device


def _denoise_nnf_torch(
    nnf_y: torch.Tensor, nnf_x: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mode filter on NNF offsets using torch for GPU acceleration."""
    device = nnf_y.device
    h, w = nnf_y.shape
    half = patch_size // 2

    grid_y = torch.arange(h, device=device).view(h, 1)
    grid_x = torch.arange(w, device=device).view(1, w)

    # Offsets are typically small; clamp to keep packing float-safe.
    offset_y = (nnf_y - grid_y).clamp(-1023, 1023)
    offset_x = (nnf_x - grid_x).clamp(-1023, 1023)

    bias = 1024
    base = 2048
    packed = (offset_y + bias) * base + (offset_x + bias)
    packed = packed.unsqueeze(0).unsqueeze(0).float()

    patches = F.unfold(packed, kernel_size=patch_size, padding=half)
    mode = torch.mode(patches, dim=1).values.view(h, w)

    mode_int = mode.long()
    out_offset_y = (mode_int // base) - bias
    out_offset_x = (mode_int % base) - bias

    return out_offset_y + grid_y, out_offset_x + grid_x


def _voting_on_rgb_torch(
    style_image: torch.Tensor, nnf_y: torch.Tensor, nnf_x: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """Torch implementation of voting-based synthesis."""
    device = nnf_y.device
    h_t, w_t = nnf_y.shape
    style_h, style_w = style_image.shape[:2]

    half = patch_size // 2
    k = patch_size * patch_size

    # Prepare style flattened
    # (3, style_h * style_w)
    style_flat = style_image.to(torch.int32).permute(2, 0, 1).reshape(3, -1)

    # Unfold NNF to get neighbor mappings
    nnf_y_f = nnf_y.unsqueeze(0).unsqueeze(0).float()
    nnf_x_f = nnf_x.unsqueeze(0).unsqueeze(0).float()

    patches_y = F.unfold(nnf_y_f, kernel_size=patch_size, padding=half)  # (1, k, H*W)
    patches_x = F.unfold(nnf_x_f, kernel_size=patch_size, padding=half)

    # Offsets per position in the patch (row-major)
    offsets = [
        (dy, dx) for dy in range(-half, half + 1) for dx in range(-half, half + 1)
    ]
    dy_vec = torch.tensor([d[0] for d in offsets], device=device).view(k, 1)
    dx_vec = torch.tensor([d[1] for d in offsets], device=device).view(k, 1)

    # Adjust style coords for voting alignment
    adj_y = patches_y - dy_vec
    adj_x = patches_x - dx_vec

    valid = (adj_y >= 0) & (adj_y < style_h) & (adj_x >= 0) & (adj_x < style_w)

    adj_y = adj_y.clamp(0, style_h - 1).long()
    adj_x = adj_x.clamp(0, style_w - 1).long()

    lin = adj_y * style_w + adj_x  # (k, H*W)
    lin_flat = lin.view(-1)

    vals_flat = torch.gather(
        style_flat,
        1,
        lin_flat.unsqueeze(0).expand(3, -1),
    )
    vals = vals_flat.view(3, k, -1)  # (3, k, H*W)

    valid_f = valid.view(k, -1).float()
    acc = (vals * valid_f.unsqueeze(0)).sum(dim=1)  # (3, H*W)
    counts = valid_f.sum(dim=0).clamp(min=1.0)

    output_flat = (acc / counts).round().clamp(0, 255).to(torch.uint8)
    output = output_flat.view(3, h_t, w_t).permute(1, 2, 0)
    return output


def style_blit_voting_torch(
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    target_app_guide: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    style_image: np.ndarray,
    stylization_rect: tuple[int, int, int, int] | None,
    patch_size: int,
    lambda_pos: int,
    lambda_app: int,
    device: torch.device,
) -> np.ndarray:
    """GPU-accelerated voting stylization using torch."""
    h_t, w_t = target_pos_guide.shape[:2]
    style_h, style_w = style_image.shape[:2]

    target_pos_t = torch.as_tensor(target_pos_guide, device=device, dtype=torch.int32)
    style_img_t = torch.as_tensor(style_image, device=device, dtype=torch.int32)

    pos_r = target_pos_t[..., 2].clamp(0, 255).long()
    pos_g = target_pos_t[..., 1].clamp(0, 255).long()

    if look_up_cube is not None and target_app_guide is not None and lambda_app != 0:
        lut_t = torch.as_tensor(look_up_cube, device=device, dtype=torch.int32)
        app = torch.as_tensor(
            ensure_grayscale(target_app_guide), device=device, dtype=torch.int32
        ).clamp(0, 255)
        coords = lut_t[pos_r, pos_g, app.long()]
        coord_x = coords[..., 0].long()
        coord_y = coords[..., 1].long()
    else:
        coord_x = torch.round(pos_r.float() / 255.0 * (style_w - 1)).long()
        coord_y = torch.round(pos_g.float() / 255.0 * (style_h - 1)).long()

    coord_x = coord_x.clamp(0, style_w - 1)
    coord_y = coord_y.clamp(0, style_h - 1)

    nnf_y, nnf_x = coord_y, coord_x

    if patch_size > 1:
        nnf_y, nnf_x = _denoise_nnf_torch(nnf_y, nnf_x, patch_size)

    output_t = _voting_on_rgb_torch(style_img_t, nnf_y, nnf_x, patch_size)

    if stylization_rect is not None:
        x, y, w, h = stylization_rect
        mask = torch.zeros((h_t, w_t), device=device, dtype=torch.bool)
        mask[y : y + h, x : x + w] = True
        output_t = torch.where(mask[..., None], output_t, torch.zeros_like(output_t))

    return output_t.cpu().numpy()


def compute_look_up_cube_optimized(
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    lambda_pos: int = 10,
    lambda_app: int = 2,
    search_radius: int = 30,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Vectorized ``compute_look_up_cube`` (``numpy.py``) on any torch device."""
    device = get_torch_device(device)

    # BGR guide: red (index 2) is the x coordinate, green (index 1) the y coordinate.
    if style_pos_guide.ndim == 3 and style_pos_guide.shape[2] == 3:
        pos_r = torch.from_numpy(style_pos_guide[..., 2].astype(np.int32)).to(device)
        pos_g = torch.from_numpy(style_pos_guide[..., 1].astype(np.int32)).to(device)
    else:
        raise ValueError("style_pos_guide must be HxWx3")

    app = torch.from_numpy(style_app_guide.astype(np.int32)).to(device)
    h, w = app.shape

    # Output buffer: [pos_r(x), pos_g(y), app(z), coords(2)]
    output = torch.zeros((256, 256, 256, 2), dtype=torch.int16, device=device)

    x_norm = w / 256.0
    y_norm = h / 256.0

    # One x at a time keeps memory small; all 256 y values are batched.
    k_inf = 2**30

    y_indices = torch.arange(256, device=device, dtype=torch.int32)

    l_pos = torch.tensor(lambda_pos, device=device, dtype=torch.int32)
    l_app = torch.tensor(lambda_app, device=device, dtype=torch.int32)

    for x in tqdm(range(256), desc="Computing LUT"):
        seed_col = int(x * x_norm)
        c_start = max(0, seed_col - search_radius)
        c_end = min(w, seed_col + search_radius)

        # Column strip around the seed column for this x
        strip_pos_r = pos_r[:, c_start:c_end]  # (H, W_strip)
        strip_pos_g = pos_g[:, c_start:c_end]
        strip_app = app[:, c_start:c_end]

        # Calculate row bounds for all Ys
        seed_rows = (y_indices.float() * y_norm).long()
        r_starts = (seed_rows - search_radius).clamp(min=0)
        r_ends = (seed_rows + search_radius).clamp(max=h)

        # Each y searches rows [r_start[y], r_end[y]); gather a fixed-height
        # window per y and mask the rows past r_end.
        max_h_window = search_radius * 2 + 1
        window_width = c_end - c_start

        rel_rows = torch.arange(max_h_window, device=device).unsqueeze(0)  # (1, max_h)
        abs_rows = r_starts.unsqueeze(1) + rel_rows  # (256, max_h)

        valid_mask = abs_rows < r_ends.unsqueeze(1)

        # Masked rows are clamped for safe indexing and never win.
        abs_rows_clamped = abs_rows.clamp(max=h - 1)

        # (256, max_h, W_strip)
        patches_r = strip_pos_r[abs_rows_clamped]
        patches_g = strip_pos_g[abs_rows_clamped]
        patches_app = strip_app[abs_rows_clamped]

        # base_err = (|pg - y| + |pr - x|) * lambda_pos, y broadcast per batch
        y_view = y_indices.view(256, 1, 1)
        base_err = (torch.abs(patches_g - y_view) + torch.abs(patches_r - x)) * l_pos

        # Invalid rows (outside r_start/r_end) get infinite error.
        full_mask = valid_mask.unsqueeze(2).expand_as(base_err)
        base_err = torch.where(full_mask, base_err, torch.tensor(k_inf, device=device))

        # Per (y, app) bin: min error and its (row, col).
        grid_cols = torch.arange(c_start, c_end, device=device, dtype=torch.int32).view(
            1, 1, -1
        )
        grid_rows = abs_rows_clamped  # (256, max_h)

        flat_err = base_err.view(256, -1)  # (256, K)
        flat_app = patches_app.view(256, -1)  # (256, K)
        flat_cols = grid_cols.expand(256, max_h_window, -1).reshape(256, -1)
        flat_rows = (
            grid_rows.unsqueeze(2)
            .expand(256, max_h_window, window_width)
            .reshape(256, -1)
        )

        # Independent reduction per y: offset app keys by y * 256.
        batch_offset = (y_indices * 256).unsqueeze(1)  # (256, 1)
        flat_keys = flat_app.long() + batch_offset.long()  # (256, K)

        # Pack (error << 32) | (row << 16) | col so one amin gives error and coords.
        packed_coords = (flat_rows << 16) | flat_cols
        scores = (flat_err.long() << 32) | packed_coords.long()

        # Global flatten
        all_keys = flat_keys.view(-1)
        all_scores = scores.view(-1)

        # Scatter-min into all 256 * 256 (y, app) bins.
        init_val = torch.tensor(k_inf, device=device).long() << 32
        min_scores = torch.full(
            (256 * 256,), init_val.item(), device=device, dtype=torch.int64
        )
        if device.type == "mps":
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

        min_scores = min_scores.view(256, 256)  # (y, app)

        # Bins with no pixel (cost >= k_inf) are filled by the app smoothing below,
        # else fall back to the seed coord. The forward/backward DP is computed as
        # prefix/suffix minima with a linear offset.
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

        # Fill holes: if cost >= k_inf, use the seed coord.
        seed_r = seed_rows.short().unsqueeze(1).expand(256, 256)
        seed_c = torch.full((256, 256), seed_col, device=device, dtype=torch.int16)

        invalid_mask = final_costs >= k_inf
        final_cols = torch.where(invalid_mask, seed_c, final_cols)
        final_rows = torch.where(invalid_mask, seed_r, final_rows)

        # Store in output: [x, y, z, 2]
        output[x, :, :, 0] = final_cols
        output[x, :, :, 1] = final_rows

    return output.cpu().numpy().astype(np.uint16)
