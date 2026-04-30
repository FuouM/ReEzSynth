"""Taichi kernels for packed LUT build and NNF voting (device tensors)."""

import taichi as ti
import torch

from src.utils_io import get_taichi_arch

_ti_initialized = False
_ti_arch = None


def _normalize_ti_device_key(device) -> str | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    return str(device).lower().split(":", maxsplit=1)[0]


def ensure_ti_init(device=None):
    """Initialize Taichi; respects PyTorch-style ``device`` (cpu / cuda / mps).

    Calls :func:`ti.reset` if the requested arch changes so CPU vs GPU can be switched.
    """
    global _ti_initialized, _ti_arch
    arch = get_taichi_arch(_normalize_ti_device_key(device))
    if _ti_initialized and _ti_arch is not None and _ti_arch != arch:
        ti.reset()
        _ti_initialized = False
    if not _ti_initialized:
        ti.init(arch=arch, log_level=ti.INFO, random_seed=42)
        print(f"[Taichi FaceBlit] Initialized with arch: {arch}")
        _ti_initialized = True
        _ti_arch = arch


def compute_lut_taichi(
    style_pos_guide,
    style_app_guide,
    lambda_pos=10,
    lambda_app=2,
    search_radius=30,
    device=None,
):
    ensure_ti_init(device)
    h_s, w_s = style_app_guide.shape[:2]
    # Using 31-bit packing: (cost << 20) | (sx << 10) | sy
    # Init with large cost (2047 << 20) to fit in signed int32
    k_inf_packed = 2047 << 20
    packed_min = torch.full(
        (256, 256, 256), k_inf_packed, dtype=torch.int32, device=device
    )

    style_pos_r = style_pos_guide[..., 2].to(device).contiguous()
    style_pos_g = style_pos_guide[..., 1].to(device).contiguous()
    style_app = style_app_guide.to(device).contiguous()

    x_norm = (w_s - 1) / 255.0
    y_norm = (h_s - 1) / 255.0

    compute_lut_search_packed_kernel(
        style_pos_r,
        style_pos_g,
        style_app,
        packed_min,
        lambda_pos,
        search_radius,
        w_s,
        h_s,
        x_norm,
        y_norm,
    )

    seed_x = (torch.arange(256).float() * x_norm).short().to(device)
    seed_y = (torch.arange(256).float() * y_norm).short().to(device)
    compute_lut_smooth_kernel(packed_min, lambda_app, seed_x, seed_y)

    return packed_min


@ti.kernel
def compute_lut_search_packed_kernel(
    style_pos_r: ti.types.ndarray(),
    style_pos_g: ti.types.ndarray(),
    style_app: ti.types.ndarray(),
    packed_min: ti.types.ndarray(),
    lambda_pos: int,
    search_radius: int,
    w_s: int,
    h_s: int,
    x_norm: float,
    y_norm: float,
):
    # Parallelize over target_x and target_y
    for tr, tg in ti.ndrange(256, 256):
        # tr is target Red (X guide), tg is target Green (Y guide)
        seed_col = int(tr * x_norm)
        seed_row = int(tg * y_norm)

        c_start = ti.max(0, seed_col - search_radius)
        c_end = ti.min(w_s, seed_col + search_radius)
        r_start = ti.max(0, seed_row - search_radius)
        r_end = ti.min(h_s, seed_row + search_radius)

        for sx in range(c_start, c_end):
            for sy in range(r_start, r_end):
                app_val = style_app[sy, sx]
                err = (
                    ti.abs(int(style_pos_r[sy, sx]) - tr)
                    + ti.abs(int(style_pos_g[sy, sx]) - tg)
                ) * lambda_pos
                # Use 11-bit cost (2047) to stay within signed int32 range.
                # 2047 is plenty for matches within search radius (~600).
                clamped_err = ti.min(err, 2047)
                new_val = (ti.i32(clamped_err) << 20) | (ti.i32(sx) << 10) | ti.i32(sy)
                ti.atomic_min(packed_min[tr, tg, app_val], new_val)


@ti.kernel
def compute_lut_smooth_kernel(
    packed_min: ti.types.ndarray(),
    lambda_app: int,
    seed_x: ti.types.ndarray(),
    seed_y: ti.types.ndarray(),
):
    # Linear smoothing over app_val (z) for each (tx, ty)
    k_inf_cost = ti.i32(2047)
    l_app_scaled = ti.max(1, ti.i32(lambda_app))

    for tr, tg in ti.ndrange(256, 256):
        s_x = ti.i32(seed_x[tr])
        s_y = ti.i32(seed_y[tg])
        seed_coords_packed = (s_x << 10) | s_y

        # Forward pass
        prev_cost = k_inf_cost
        prev_coords_packed = seed_coords_packed
        for z in range(256):
            curr_packed = packed_min[tr, tg, z]
            curr_cost = curr_packed >> 20

            prop_cost = prev_cost + l_app_scaled
            if prop_cost < curr_cost:
                packed_min[tr, tg, z] = (
                    ti.min(prop_cost, k_inf_cost) << 20
                ) | prev_coords_packed
                prev_cost = prop_cost
                # prev_coords_packed already holds the propagated source
            elif curr_cost < k_inf_cost:
                # Valid cell wins over propagation — anchor to it
                prev_cost = curr_cost
                prev_coords_packed = curr_packed & 0xFFFFF  # 20 bits for coords
            # else: empty cell, leave prev_cost / prev_coords unchanged so
            # subsequent steps can still propagate from the last real cell

        # Backward pass
        prev_cost = k_inf_cost
        prev_coords_packed = seed_coords_packed
        for z_inv in range(256):
            z = 255 - z_inv
            curr_packed = packed_min[tr, tg, z]
            curr_cost = curr_packed >> 20

            prop_cost = prev_cost + l_app_scaled
            if prop_cost < curr_cost:
                packed_min[tr, tg, z] = (
                    ti.min(prop_cost, k_inf_cost) << 20
                ) | prev_coords_packed
                prev_cost = prop_cost
            elif curr_cost < k_inf_cost:
                # Valid cell wins — anchor
                prev_cost = curr_cost
                prev_coords_packed = curr_packed & 0xFFFFF
            # else: empty cell, preserve propagation state

        # Final hole filling
        for z in range(256):
            if (packed_min[tr, tg, z] >> 20) >= k_inf_cost:
                packed_min[tr, tg, z] = (k_inf_cost << 20) | seed_coords_packed


def stylize_blit_taichi(
    style_image,
    style_pos_guide,
    style_app_guide,
    target_pos_guide,
    target_app_guide,
    look_up_cube_packed,
    device,
    stylization_rect=None,
    patch_size=3,
    lambda_pos=10,
    lambda_app=2,
    threshold=50,
    use_vectorized=True,
    denoise_iters=3,
):
    ensure_ti_init(device)
    if threshold is None:
        threshold = 50
    h_t, w_t = target_app_guide.shape[:2]
    nnf = torch.zeros((h_t, w_t, 2), dtype=torch.int32, device=device)
    output = torch.zeros((h_t, w_t, 3), dtype=torch.uint8, device=device)

    tp_r = target_pos_guide[..., 2].to(device).contiguous()
    tp_g = target_pos_guide[..., 1].to(device).contiguous()
    ta = target_app_guide.to(device).contiguous()
    style_image_ti = style_image.to(device).contiguous()
    sp_r = style_pos_guide[..., 2].to(device).contiguous()
    sp_g = style_pos_guide[..., 1].to(device).contiguous()
    sa = style_app_guide.to(device).contiguous()

    lut_t = torch.as_tensor(
        look_up_cube_packed, dtype=torch.int32, device=device
    ).contiguous()

    if use_vectorized:
        lookup_nnf_kernel(tp_r, tp_g, ta, lut_t, nnf)
    else:
        if stylization_rect is None:
            rect_x, rect_y, rect_w, rect_h = 0, 0, w_t, h_t
        else:
            rect_x, rect_y, rect_w, rect_h = stylization_rect
        covered_pixels = torch.zeros((h_t, w_t), dtype=torch.int32, device=device)
        q_y = torch.empty(h_t * w_t, dtype=torch.int32, device=device)
        q_x = torch.empty(h_t * w_t, dtype=torch.int32, device=device)
        initialize_nnf_dfs_kernel(
            tp_r,
            tp_g,
            ta,
            sp_r,
            sp_g,
            sa,
            lut_t,
            nnf,
            covered_pixels,
            q_y,
            q_x,
            rect_x,
            rect_y,
            rect_w,
            rect_h,
            threshold,
            lambda_pos,
            lambda_app,
        )

    sh_s, sw_s = style_image_ti.shape[0], style_image_ti.shape[1]
    if patch_size > 1:
        # Apply denoise multiple times to increase offset coherence
        # across the image, matching the quality of the reference DFS.
        for _ in range(denoise_iters):
            denoised_nnf = torch.zeros_like(nnf)
            denoise_nnf_kernel(nnf, denoised_nnf, patch_size, sh_s, sw_s)
            nnf = denoised_nnf

    voting_kernel(output, style_image_ti, nnf, patch_size)
    return output.detach().cpu().numpy()


@ti.kernel
def lookup_nnf_kernel(
    target_pos_r: ti.types.ndarray(),
    target_pos_g: ti.types.ndarray(),
    target_app: ti.types.ndarray(),
    look_up_cube_packed: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
):
    for ty, tx in ti.ndrange(target_app.shape[0], target_app.shape[1]):
        tr = target_pos_r[ty, tx]
        tg = target_pos_g[ty, tx]
        ta = target_app[ty, tx]

        # Using Red channel as X and Green channel as Y to index the LUT
        packed = look_up_cube_packed[tr, tg, ta]
        nnf[ty, tx, 1] = int((packed >> 10) & 0x3FF)  # sx (10 bits)
        nnf[ty, tx, 0] = int(packed & 0x3FF)  # sy (10 bits)


@ti.kernel
def denoise_nnf_kernel(
    nnf: ti.types.ndarray(),
    denoised_nnf: ti.types.ndarray(),
    patch_size: int,
    sh: int,
    sw: int,
):
    h, w = nnf.shape[0], nnf.shape[1]
    r = patch_size // 2
    for ty, tx in ti.ndrange(h, w):
        best_y_off = 0
        best_x_off = 0
        max_count = 0
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            ny, nx = (
                ti.max(0, ti.min(h - 1, ty + py)),
                ti.max(0, ti.min(w - 1, tx + px)),
            )
            curr_y_off = nnf[ny, nx, 0] - ny
            curr_x_off = nnf[ny, nx, 1] - nx

            count = 0
            for py2, px2 in ti.ndrange((-r, r + 1), (-r, r + 1)):
                ny2, nx2 = (
                    ti.max(0, ti.min(h - 1, ty + py2)),
                    ti.max(0, ti.min(w - 1, tx + px2)),
                )
                if (nnf[ny2, nx2, 0] - ny2 == curr_y_off) and (
                    nnf[ny2, nx2, 1] - nx2 == curr_x_off
                ):
                    count += 1

            if count > max_count:
                max_count = count
                best_y_off = curr_y_off
                best_x_off = curr_x_off

        denoised_nnf[ty, tx, 0] = ti.max(0, ti.min(sh - 1, best_y_off + ty))
        denoised_nnf[ty, tx, 1] = ti.max(0, ti.min(sw - 1, best_x_off + tx))


@ti.kernel
def initialize_nnf_dfs_kernel(
    target_pos_r: ti.types.ndarray(),
    target_pos_g: ti.types.ndarray(),
    target_app: ti.types.ndarray(),
    style_pos_r: ti.types.ndarray(),
    style_pos_g: ti.types.ndarray(),
    style_app: ti.types.ndarray(),
    look_up_cube_packed: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
    covered_pixels: ti.types.ndarray(),
    q_y: ti.types.ndarray(),
    q_x: ti.types.ndarray(),
    rect_x: int,
    rect_y: int,
    rect_w: int,
    rect_h: int,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
):
    h_t, w_t = target_app.shape[0], target_app.shape[1]
    h_s, w_s = style_app.shape[0], style_app.shape[1]
    max_q = h_t * w_t
    chunk_number = 1

    ti.loop_config(serialize=True)
    for row_t in range(rect_y, rect_y + rect_h):
        for col_t in range(rect_x, rect_x + rect_w):
            if covered_pixels[row_t, col_t] == 0:
                tr_seed = target_pos_r[row_t, col_t]
                tg_seed = target_pos_g[row_t, col_t]
                seed_sx = ti.i32(
                    ti.cast(tr_seed, ti.f32) * ti.cast(w_s, ti.f32) / 256.0
                )
                seed_sy = ti.i32(
                    ti.cast(tg_seed, ti.f32) * ti.cast(h_s, ti.f32) / 256.0
                )
                if lambda_app != 0:
                    ta_seed = target_app[row_t, col_t]
                    packed = look_up_cube_packed[tr_seed, tg_seed, ta_seed]
                    seed_sx = ti.i32((packed >> 10) & 0x3FF)
                    seed_sy = ti.i32(packed & 0x3FF)
                seed_sx = ti.max(0, ti.min(w_s - 1, seed_sx))
                seed_sy = ti.max(0, ti.min(h_s - 1, seed_sy))

                head = 0
                tail = 1
                q_y[0] = row_t
                q_x[0] = col_t

                while head < tail:
                    ty = q_y[head]
                    tx = q_x[head]
                    head += 1

                    dy = ty - row_t
                    dx = tx - col_t
                    sy = seed_sy + dy
                    sx = seed_sx + dx

                    if (
                        0 <= ty < h_t
                        and 0 <= tx < w_t
                        and 0 <= sy < h_s
                        and 0 <= sx < w_s
                    ):
                        if covered_pixels[ty, tx] == 0:
                            pos_err = ti.abs(
                                ti.i32(target_pos_r[ty, tx])
                                - ti.i32(style_pos_r[sy, sx])
                            ) + ti.abs(
                                ti.i32(target_pos_g[ty, tx])
                                - ti.i32(style_pos_g[sy, sx])
                            )
                            app_err = ti.abs(
                                ti.i32(target_app[ty, tx]) - ti.i32(style_app[sy, sx])
                            )
                            error = lambda_pos * pos_err + lambda_app * app_err

                            if error < threshold or (dy == 0 and dx == 0):
                                nnf[ty, tx, 0] = sy
                                nnf[ty, tx, 1] = sx
                                covered_pixels[ty, tx] = chunk_number

                                if ty > 0 and tail < max_q:
                                    q_y[tail] = ty - 1
                                    q_x[tail] = tx
                                    tail += 1
                                if ty + 1 < h_t and tail < max_q:
                                    q_y[tail] = ty + 1
                                    q_x[tail] = tx
                                    tail += 1
                                if tx > 0 and tail < max_q:
                                    q_y[tail] = ty
                                    q_x[tail] = tx - 1
                                    tail += 1
                                if tx + 1 < w_t and tail < max_q:
                                    q_y[tail] = ty
                                    q_x[tail] = tx + 1
                                    tail += 1

                chunk_number += 1

    for ty, tx in ti.ndrange(h_t, w_t):
        if covered_pixels[ty, tx] == 0:
            tr = target_pos_r[ty, tx]
            tg = target_pos_g[ty, tx]
            ta = target_app[ty, tx]
            packed = look_up_cube_packed[tr, tg, ta]
            nnf[ty, tx, 1] = ti.max(0, ti.min(w_s - 1, ti.i32((packed >> 10) & 0x3FF)))
            nnf[ty, tx, 0] = ti.max(0, ti.min(h_s - 1, ti.i32(packed & 0x3FF)))


@ti.kernel
def voting_kernel(
    output_image: ti.types.ndarray(),
    source_style: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
    patch_size: int,
):
    th, tw = output_image.shape[0], output_image.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    r = patch_size // 2
    for ty, tx in ti.ndrange(th, tw):
        sum_b = 0.0
        sum_g = 0.0
        sum_r = 0.0
        count = 0.0
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            ny, nx = ty + py, tx + px
            if 0 <= ny < th and 0 <= nx < tw:
                # Clamp source coordinates to style image boundaries.
                # This matches Torch's behavior and improves edge quality.
                sy = ti.max(0, ti.min(sh - 1, nnf[ny, nx, 0] - py))
                sx = ti.max(0, ti.min(sw - 1, nnf[ny, nx, 1] - px))
                sum_b += float(source_style[sy, sx, 0])
                sum_g += float(source_style[sy, sx, 1])
                sum_r += float(source_style[sy, sx, 2])
                count += 1.0
        if count > 0:
            output_image[ty, tx, 0] = ti.u8(
                ti.max(0, ti.min(255, ti.round(sum_b / count)))
            )
            output_image[ty, tx, 1] = ti.u8(
                ti.max(0, ti.min(255, ti.round(sum_g / count)))
            )
            output_image[ty, tx, 2] = ti.u8(
                ti.max(0, ti.min(255, ti.round(sum_r / count)))
            )
        else:
            # Fallback to nearest neighbor if voting failed
            sy = ti.max(0, ti.min(sh - 1, nnf[ty, tx, 0]))
            sx = ti.max(0, ti.min(sw - 1, nnf[ty, tx, 1]))
            output_image[ty, tx, 0] = source_style[sy, sx, 0]
            output_image[ty, tx, 1] = source_style[sy, sx, 1]
            output_image[ty, tx, 2] = source_style[sy, sx, 2]
