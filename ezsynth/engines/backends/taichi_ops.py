"""Taichi-accelerated image kernels and helpers (module-level)."""

import numpy as np
import taichi as ti


# --- Color Conversion Kernels ---

@ti.func
def bgr_to_lab_func(bgr):
    # Normalize to [0, 1]
    b = bgr[0] / 255.0
    g = bgr[1] / 255.0
    r = bgr[2] / 255.0

    # Gamma correction
    r = ti.pow((r + 0.055) / 1.055, 2.4) if r > 0.04045 else r / 12.92
    g = ti.pow((g + 0.055) / 1.055, 2.4) if g > 0.04045 else g / 12.92
    b = ti.pow((b + 0.055) / 1.055, 2.4) if b > 0.04045 else b / 12.92

    # RGB to XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # XYZ to Lab
    # D65 white point
    xw, yw, zw = 0.95047, 1.0, 1.08883

    x /= xw
    y /= yw
    z /= zw

    fx = ti.pow(x, 1.0 / 3.0) if x > 0.008856 else (7.787 * x) + (16.0 / 116.0)
    fy = ti.pow(y, 1.0 / 3.0) if y > 0.008856 else (7.787 * y) + (16.0 / 116.0)
    fz = ti.pow(z, 1.0 / 3.0) if z > 0.008856 else (7.787 * z) + (16.0 / 116.0)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)

    return ti.Vector([L, a, b_])

@ti.func
def lab_to_bgr_func(lab):
    L, a, b_ = lab[0], lab[1], lab[2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_ / 200.0

    x = fx**3 if fx**3 > 0.008856 else (fx - 16.0 / 116.0) / 7.787
    y = fy**3 if fy**3 > 0.008856 else (fy - 16.0 / 116.0) / 7.787
    z = fz**3 if fz**3 > 0.008856 else (fz - 16.0 / 116.0) / 7.787

    # D65 white point
    xw, yw, zw = 0.95047, 1.0, 1.08883
    x *= xw
    y *= yw
    z *= zw

    # XYZ to RGB
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Gamma correction
    r = 1.055 * ti.pow(r, 1 / 2.4) - 0.055 if r > 0.0031308 else 12.92 * r
    g = 1.055 * ti.pow(g, 1 / 2.4) - 0.055 if g > 0.0031308 else 12.92 * g
    b = 1.055 * ti.pow(b, 1 / 2.4) - 0.055 if b > 0.0031308 else 12.92 * b

    return ti.Vector([
        ti.max(0, ti.min(255, b * 255.0)),
        ti.max(0, ti.min(255, g * 255.0)),
        ti.max(0, ti.min(255, r * 255.0)),
    ])

# --- Histogram Blending Kernels ---

@ti.kernel
def compute_stats_kernel(
    img: ti.types.ndarray(), mean: ti.types.ndarray(), std: ti.types.ndarray()
):
    h, w = img.shape[0], img.shape[1]
    sum_val = ti.Vector([0.0, 0.0, 0.0])
    sum_sq = ti.Vector([0.0, 0.0, 0.0])

    # Parallel reduction
    ti.loop_config(block_dim=256)
    for i, j in ti.ndrange(h, w):
        bgr = ti.Vector([img[i, j, 0], img[i, j, 1], img[i, j, 2]])
        lab = bgr_to_lab_func(bgr)
        sum_val += lab
        sum_sq += lab * lab

    N = float(h * w)
    m = sum_val / N
    s = ti.sqrt(ti.max(0.0, sum_sq / N - m * m))

    for c in ti.static(range(3)):
        mean[c] = m[c]
        std[c] = s[c]

@ti.kernel
def hist_blend_apply_kernel(
    a: ti.types.ndarray(),
    b: ti.types.ndarray(),
    mask: ti.types.ndarray(),
    a_mean: ti.types.ndarray(),
    a_std: ti.types.ndarray(),
    b_mean: ti.types.ndarray(),
    b_std: ti.types.ndarray(),
    min_e_mean: ti.types.ndarray(),
    min_e_std: ti.types.ndarray(),
    weight1: float,
    weight2: float,
    out: ti.types.ndarray(),
):
    h, w = a.shape[0], a.shape[1]

    t_mean = ti.Vector([0.5 * 256, 0.5 * 256, 0.5 * 256])
    t_std = ti.Vector([(1.0 / 36.0) * 256, (1.0 / 36.0) * 256, (1.0 / 36.0) * 256])

    m_a = ti.Vector([a_mean[0], a_mean[1], a_mean[2]])
    s_a = ti.Vector([a_std[0], a_std[1], a_std[2]])
    m_b = ti.Vector([b_mean[0], b_mean[1], b_mean[2]])
    s_b = ti.Vector([b_std[0], b_std[1], b_std[2]])

    for i, j in ti.ndrange(h, w):
        bgr_a = ti.Vector([a[i, j, 0], a[i, j, 1], a[i, j, 2]])
        bgr_b = ti.Vector([b[i, j, 0], b[i, j, 1], b[i, j, 2]])
        lab_a = bgr_to_lab_func(bgr_a)
        lab_b = bgr_to_lab_func(bgr_b)

        # Normalize
        a_norm = (lab_a - m_a) * t_std / (s_a + 1e-6) + t_mean
        b_norm = (lab_b - m_b) * t_std / (s_b + 1e-6) + t_mean

        # Blend
        ab_lab = (a_norm * weight1 + b_norm * weight2 - 128.0) / 0.5 + 128.0

        out[i, j, 0] = ab_lab[0]
        out[i, j, 1] = ab_lab[1]
        out[i, j, 2] = ab_lab[2]

@ti.kernel
def hist_blend_final_pass_kernel(
    ab_lab_temp: ti.types.ndarray(),
    ab_mean: ti.types.ndarray(),
    ab_std: ti.types.ndarray(),
    min_e_mean: ti.types.ndarray(),
    min_e_std: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    h, w = ab_lab_temp.shape[0], ab_lab_temp.shape[1]
    m_ab = ti.Vector([ab_mean[0], ab_mean[1], ab_mean[2]])
    s_ab = ti.Vector([ab_std[0], ab_std[1], ab_std[2]])
    m_me = ti.Vector([min_e_mean[0], min_e_mean[1], min_e_mean[2]])
    s_me = ti.Vector([min_e_std[0], min_e_std[1], min_e_std[2]])

    for i, j in ti.ndrange(h, w):
        lab = ti.Vector([
            ab_lab_temp[i, j, 0],
            ab_lab_temp[i, j, 1],
            ab_lab_temp[i, j, 2],
        ])
        lab_final = (lab - m_ab) * s_me / (s_ab + 1e-6) + m_me
        res = lab_to_bgr_func(lab_final)
        out[i, j, 0] = res[0]
        out[i, j, 1] = res[1]
        out[i, j, 2] = res[2]

# --- Warping Kernels ---

@ti.func
def get_val_reflect(
    src: ti.template(), y: int, x: int, c: int, h: int, w: int
):
    ry = y
    rx = x
    if ry < 0:
        ry = -ry
    if ry >= h:
        ry = 2 * h - 2 - ry
    if rx < 0:
        rx = -rx
    if rx >= w:
        rx = 2 * w - 2 - rx
    ry = ti.max(0, ti.min(h - 1, ry))
    rx = ti.max(0, ti.min(w - 1, rx))

    val = 0.0
    if ti.static(len(src.shape) > 2):
        val = src[ry, rx, c]
    else:
        val = src[ry, rx]
    return float(val)

@ti.kernel
def bilinear_warp_kernel(
    src: ti.types.ndarray(), flow: ti.types.ndarray(), dst: ti.types.ndarray()
):
    h, w = src.shape[0], src.shape[1]

    for i, j in ti.ndrange(h, w):
        fx = flow[i, j, 0]
        fy = flow[i, j, 1]

        x = float(j) + fx
        y = float(i) + fy

        # Bilinear interpolation
        x0 = int(ti.floor(x))
        y0 = int(ti.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = x - float(x0)
        wy1 = y - float(y0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        if ti.static(len(src.shape) > 2):
            for ch in range(src.shape[2]):
                v00 = get_val_reflect(src, y0, x0, ch, h, w)
                v01 = get_val_reflect(src, y0, x1, ch, h, w)
                v10 = get_val_reflect(src, y1, x0, ch, h, w)
                v11 = get_val_reflect(src, y1, x1, ch, h, w)
                dst[i, j, ch] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (
                    wx0 * v10 + wx1 * v11
                )
        else:
            v00 = get_val_reflect(src, y0, x0, 0, h, w)
            v01 = get_val_reflect(src, y0, x1, 0, h, w)
            v10 = get_val_reflect(src, y1, x0, 0, h, w)
            v11 = get_val_reflect(src, y1, x1, 0, h, w)
            dst[i, j] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (
                wx0 * v10 + wx1 * v11
            )

@ti.func
def finite_or_zero(value: float) -> float:
    out = value
    if value != value or value > 1.0e20 or value < -1.0e20:
        out = 0.0
    return out

@ti.kernel
def soft_splat_kernel(
    src: ti.types.ndarray(),
    flow: ti.types.ndarray(),
    dst_color: ti.types.ndarray(),
    dst_weight: ti.types.ndarray(),
    src_guide: ti.types.ndarray(),
    tgt_guide: ti.types.ndarray(),
    use_bilateral: ti.template(),
):
    h, w = src.shape[0], src.shape[1]
    for i, j in ti.ndrange(h, w):
        fx = finite_or_zero(float(flow[i, j, 0]))
        fy = finite_or_zero(float(flow[i, j, 1]))

        tx = float(j) + fx
        ty = float(i) + fy

        ix = int(ti.floor(tx))
        iy = int(ti.floor(ty))

        flow_mag = ti.sqrt(fx * fx + fy * fy)
        sigma = 0.35 + 0.15 * ti.exp(-flow_mag / 10.0)

        src_g = ti.Vector([0.0, 0.0, 0.0])
        if ti.static(use_bilateral):
            for c in ti.static(range(3)):
                src_g[c] = finite_or_zero(float(src_guide[i, j, c]))

        for dy, dx in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            target_x = ix + dx
            target_y = iy + dy
            if 0 <= target_x < w and 0 <= target_y < h:
                dist_sq = (float(target_x) - tx) ** 2 + (float(target_y) - ty) ** 2
                weight = ti.exp(-dist_sq / (2 * sigma * sigma))

                if ti.static(use_bilateral):
                    color_dist_sq = 0.0
                    for c in ti.static(range(3)):
                        tgt_val = finite_or_zero(
                            float(tgt_guide[target_y, target_x, c])
                        )
                        color_dist_sq += (src_g[c] - tgt_val) ** 2
                    tau = 30.0
                    weight *= ti.exp(-color_dist_sq / (2 * tau * tau))

                if weight == weight and weight > 1e-4 and weight < 1.0e20:
                    ti.atomic_add(dst_weight[target_y, target_x], weight)
                    if ti.static(len(src.shape) > 2):
                        for c in range(src.shape[2]):
                            src_val = finite_or_zero(float(src[i, j, c]))
                            ti.atomic_add(
                                dst_color[target_y, target_x, c],
                                weight * src_val,
                            )
                    else:
                        src_val = finite_or_zero(float(src[i, j]))
                        ti.atomic_add(
                            dst_color[target_y, target_x],
                            weight * src_val,
                        )

@ti.kernel
def normalize_splat_kernel(
    dst_color: ti.types.ndarray(),
    dst_weight: ti.types.ndarray(),
    out: ti.types.ndarray(),
    is_uint8: ti.template(),
):
    h, w = dst_color.shape[0], dst_color.shape[1]
    for i, j in ti.ndrange(h, w):
        weight = finite_or_zero(float(dst_weight[i, j]))
        if weight > 1e-4 and weight < 1.0e20:
            if ti.static(len(out.shape) > 2):
                for c in range(out.shape[2]):
                    val = finite_or_zero(float(dst_color[i, j, c]) / weight)
                    if ti.static(is_uint8):
                        out[i, j, c] = ti.u8(ti.max(0.0, ti.min(255.0, val)))
                    else:
                        out[i, j, c] = val
            else:
                val = finite_or_zero(float(dst_color[i, j]) / weight)
                if ti.static(is_uint8):
                    out[i, j] = ti.u8(ti.max(0.0, ti.min(255.0, val)))
                else:
                    out[i, j] = val
        else:
            if ti.static(len(out.shape) > 2):
                for c in range(out.shape[2]):
                    if ti.static(is_uint8):
                        out[i, j, c] = ti.u8(0)
                    else:
                        out[i, j, c] = 0.0
            else:
                if ti.static(is_uint8):
                    out[i, j] = ti.u8(0)
                else:
                    out[i, j] = 0.0

@ti.kernel
def pull_kernel(
    src_color: ti.types.ndarray(),
    src_weight: ti.types.ndarray(),
    dst_color: ti.types.ndarray(),
    dst_weight: ti.types.ndarray(),
):
    h_dst, w_dst = dst_color.shape[0], dst_color.shape[1]
    h_src, w_src = src_color.shape[0], src_color.shape[1]
    for i, j in ti.ndrange(h_dst, w_dst):
        sum_w = 0.0
        if ti.static(len(src_color.shape) > 2):
            for c in range(src_color.shape[2]):
                sum_c = 0.0
                for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
                    si, sj = i * 2 + di, j * 2 + dj
                    if 0 <= si < h_src and 0 <= sj < w_src:
                        dist_sq = di * di + dj * dj
                        gw = ti.exp(-dist_sq / 2.0)
                        sum_c += finite_or_zero(
                            float(src_color[si, sj, c])
                        ) * gw
                        if c == 0:
                            sum_w += finite_or_zero(
                                float(src_weight[si, sj])
                            ) * gw
                dst_color[i, j, c] = sum_c
        else:
            sum_c = 0.0
            for di, dj in ti.static(ti.ndrange((-2, 3), (-2, 3))):
                si, sj = i * 2 + di, j * 2 + dj
                if 0 <= si < h_src and 0 <= sj < w_src:
                    dist_sq = di * di + dj * dj
                    gw = ti.exp(-dist_sq / 2.0)
                    sum_c += finite_or_zero(float(src_color[si, sj])) * gw
                    sum_w += finite_or_zero(float(src_weight[si, sj])) * gw
            dst_color[i, j] = sum_c
        dst_weight[i, j] = finite_or_zero(sum_w)

@ti.kernel
def push_kernel(
    src_color: ti.types.ndarray(),
    src_weight: ti.types.ndarray(),
    dst_color: ti.types.ndarray(),
    dst_weight: ti.types.ndarray(),
):
    h_dst, w_dst = dst_color.shape[0], dst_color.shape[1]
    for i, j in ti.ndrange(h_dst, w_dst):
        dw = ti.max(0.0, ti.min(1.0, finite_or_zero(float(dst_weight[i, j]))))
        if dw < 0.95:
            si, sj = i // 2, j // 2
            sw = finite_or_zero(float(src_weight[si, sj]))
            if sw > 1e-4:
                alpha = ti.max(0.0, ti.min(1.0, 1.0 - dw))
                if ti.static(len(src_color.shape) > 2):
                    for c in range(src_color.shape[2]):
                        coarse = finite_or_zero(
                            float(src_color[si, sj, c]) / sw
                        )
                        current = finite_or_zero(
                            float(dst_color[i, j, c]) / ti.max(dw, 1e-6)
                        )
                        dst_color[i, j, c] = dw * (
                            current
                        ) + alpha * coarse
                else:
                    coarse = finite_or_zero(float(src_color[si, sj]) / sw)
                    current = finite_or_zero(
                        float(dst_color[i, j]) / ti.max(dw, 1e-6)
                    )
                    dst_color[i, j] = dw * current + alpha * coarse
                dst_weight[i, j] = 1.0

# --- Poisson CG Solver ---

@ti.kernel
def compute_laplacian_kernel(
    x: ti.types.ndarray(), lap: ti.types.ndarray(), weight_sq: float
):
    h, w = x.shape[0], x.shape[1]
    for i, j in ti.ndrange(h, w):
        center = x[i, j]
        gx_sq = 0.0
        gy_sq = 0.0

        if j > 0:
            gx_sq += center - x[i, j - 1]
        if j < w - 1:
            gx_sq += center - x[i, j + 1]
        if i > 0:
            gy_sq += center - x[i - 1, j]
        if i < h - 1:
            gy_sq += center - x[i + 1, j]

        lap[i, j] = weight_sq * (gx_sq + gy_sq) + center

@ti.kernel
def compute_rhs_kernel(
    gx: ti.types.ndarray(),
    gy: ti.types.ndarray(),
    target: ti.types.ndarray(),
    rhs: ti.types.ndarray(),
    weight_sq: float,
):
    h, w = gx.shape[0], gx.shape[1]
    for i, j in ti.ndrange(h, w):
        val_gx = gx[i, j]
        if j > 0:
            val_gx -= gx[i, j - 1]

        val_gy = gy[i, j]
        if i > 0:
            val_gy -= gy[i - 1, j]

        rhs[i, j] = weight_sq * (val_gx + val_gy) + target[i, j]

def poisson_solver_cg(gx, gy, target, weight, max_iter=100, tol=1e-5):
    h, w = target.shape
    x = np.zeros_like(target)
    weight_sq = weight * weight
    rhs = np.zeros_like(target)
    compute_rhs_kernel(gx, gy, target, rhs, weight_sq)

    r = np.zeros_like(target)
    Ap = np.zeros_like(target)
    compute_laplacian_kernel(x, Ap, weight_sq)
    r = rhs - Ap
    p = r.copy()
    rsold = np.sum(r * r)

    if rsold < tol:
        return x

    for i in range(max_iter):
        compute_laplacian_kernel(p, Ap, weight_sq)
        alpha = rsold / (np.sum(p * Ap) + 1e-10)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x
