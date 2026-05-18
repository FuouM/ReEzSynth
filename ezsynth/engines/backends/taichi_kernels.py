"""Taichi kernels for :mod:`taichi_backend` (module-level ``@ti.kernel`` / ``@ti.func``)."""

import taichi as ti

from ...consts import COST_FUNCTION_NCC, EBSYNTH_VOTEMODE_WEIGHTED


@ti.func
def get_omega(
    omega_map: ti.template(),
    x: int,
    y: int,
    patch_size: int,
    sw: int,
    sh: int,
):
    r = patch_size // 2
    sum_val = 0
    for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
        cur_x = ti.max(0, ti.min(x + px, sw - 1))
        cur_y = ti.max(0, ti.min(y + py, sh - 1))
        sum_val += omega_map[cur_y, cur_x]
    return float(sum_val)


@ti.func
def patch_omega_atomic_add(
    omega_map: ti.template(),
    cx: int,
    cy: int,
    patch_size: int,
    sw: int,
    sh: int,
    delta: int,
):
    r = patch_size // 2
    for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
        cur_x = ti.max(0, ti.min(cx + px, sw - 1))
        cur_y = ti.max(0, ti.min(cy + py, sh - 1))
        ti.atomic_add(omega_map[cur_y, cur_x], delta)


@ti.func
def query_sat(sat: ti.template(), x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = ti.max(x1, 0)
    y1 = ti.max(y1, 0)
    x2 = ti.min(x2, w - 1)
    y2 = ti.min(y2, h - 1)
    br = sat[y2, x2]
    bl = sat[y2, x1 - 1] if x1 > 0 else 0.0
    tr = sat[y1 - 1, x2] if y1 > 0 else 0.0
    tl = sat[y1 - 1, x1 - 1] if (x1 > 0 and y1 > 0) else 0.0
    return br - bl - tr + tl


@ti.func
def compute_patch_ncc(
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    patch_size: int,
    source_style: ti.template(),
    target_style: ti.template(),
    source_guide: ti.template(),
    target_guide: ti.template(),
    modulation_guide: ti.template(),
    use_modulation: int,
    style_weights: ti.template(),
    guide_weights: ti.template(),
    source_style_sat: ti.template(),
    source_style_sq_sat: ti.template(),
    target_style_sat: ti.template(),
    target_style_sq_sat: ti.template(),
    sw: int,
    sh: int,
    tw: int,
    th: int,
    NSC: int,
    NGC: int,
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    r = patch_size // 2
    epsilon = 1e-6
    final_error = 0.0

    if use_bilateral == 0 and n_size_step == 1:
        # Optimized non-bilateral path using SAT
        N = float(patch_size * patch_size)
        sum_s = query_sat(source_style_sat, sx - r, sy - r, sx + r, sy + r, sw, sh)
        sum_sq_s = query_sat(
            source_style_sq_sat, sx - r, sy - r, sx + r, sy + r, sw, sh
        )
        sum_t = query_sat(target_style_sat, tx - r, ty - r, tx + r, ty + r, tw, th)
        sum_sq_t = query_sat(
            target_style_sq_sat, tx - r, ty - r, tx + r, ty + r, tw, th
        )

        mean_s = sum_s / N
        mean_t = sum_t / N
        std_s = ti.sqrt(ti.max(0.0, sum_sq_s / N - mean_s * mean_s))
        std_t = ti.sqrt(ti.max(0.0, sum_sq_t / N - mean_t * mean_t))

        sum_st = 0.0
        guide_error = 0.0
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            idx_sx, idx_sy = (
                ti.max(0, ti.min(sx + px, sw - 1)),
                ti.max(0, ti.min(sy + py, sh - 1)),
            )
            idx_tx, idx_ty = (
                ti.max(0, ti.min(tx + px, tw - 1)),
                ti.max(0, ti.min(ty + py, th - 1)),
            )

            s_val_g, t_val_g = 0.0, 0.0
            for c in range(NSC):
                s_val_g += float(source_style[idx_sy, idx_sx, c])
                t_val_g += float(target_style[idx_ty, idx_tx, c])
            sum_st += (s_val_g / NSC) * (t_val_g / NSC)

            for c in range(NGC):
                diff = float(source_guide[idx_sy, idx_sx, c]) - float(
                    target_guide[idx_ty, idx_tx, c]
                )
                mod = (
                    float(modulation_guide[idx_ty, idx_tx, c]) / 255.0
                    if use_modulation
                    else 1.0
                )
                guide_error += guide_weights[c] * mod * diff * diff

        cov = sum_st / N - mean_s * mean_t
        ncc = cov / (std_s * std_t) if (std_s > epsilon and std_t > epsilon) else 0.0
        ncc = ti.min(1.0, ti.max(-1.0, ncc))
        final_error = (1.0 - ncc) * style_weights[0] * N + guide_error

    else:
        # Bilateral or multi-scale path (no SAT optimization)
        inv_2_sigma_spatial_sq = -1.0 / (2.0 * sigma_spatial * sigma_spatial)
        inv_2_sigma_color_sq = -1.0 / (2.0 * sigma_color * sigma_color)

        sum_s = 0.0
        sum_sq_s = 0.0
        sum_t = 0.0
        sum_sq_t = 0.0
        sum_st = 0.0
        sum_weight = 0.0
        guide_error = 0.0

        # Step manually because Ti.range doesn't support 3 args
        steps = (2 * r) // n_size_step + 1
        for i, j in ti.ndrange(steps, steps):
            px = -r + i * n_size_step
            py = -r + j * n_size_step
            if px <= r and py <= r:
                idx_sx, idx_sy = (
                    ti.max(0, ti.min(sx + px, sw - 1)),
                    ti.max(0, ti.min(sy + py, sh - 1)),
                )
                idx_tx, idx_ty = (
                    ti.max(0, ti.min(tx + px, tw - 1)),
                    ti.max(0, ti.min(ty + py, th - 1)),
                )

                weight = 1.0
                if use_bilateral:
                    s_val_c, t_val_c = 0.0, 0.0
                    for c in range(NSC):
                        s_val_c += float(source_style[sy, sx, c])
                        t_val_c += float(target_style[ty, tx, c])
                    s_val_c /= NSC
                    t_val_c /= NSC

                    dist_sq = float(px * px + py * py)
                    color_diff = s_val_c - t_val_c
                    color_diff_sq = color_diff * color_diff
                    weight = ti.exp(
                        dist_sq * inv_2_sigma_spatial_sq
                        + color_diff_sq * inv_2_sigma_color_sq
                    )

                s_val, t_val = 0.0, 0.0
                for c in range(NSC):
                    sv = float(source_style[idx_sy, idx_sx, c])
                    tv = float(target_style[idx_ty, idx_tx, c])
                    s_val += sv
                    t_val += tv
                s_val /= NSC
                t_val /= NSC

                sum_s += s_val * weight
                sum_sq_s += s_val * s_val * weight
                sum_t += t_val * weight
                sum_sq_t += t_val * t_val * weight
                sum_st += s_val * t_val * weight
                sum_weight += weight

                for c in range(NGC):
                    diff = float(source_guide[idx_sy, idx_sx, c]) - float(
                        target_guide[idx_ty, idx_tx, c]
                    )
                    mod = (
                        float(modulation_guide[idx_ty, idx_tx, c]) / 255.0
                        if use_modulation
                        else 1.0
                    )
                    guide_error += weight * guide_weights[c] * mod * diff * diff

        if sum_weight > epsilon:
            mean_s = sum_s / sum_weight
            mean_t = sum_t / sum_weight
            std_s = ti.sqrt(ti.max(0.0, sum_sq_s / sum_weight - mean_s * mean_s))
            std_t = ti.sqrt(ti.max(0.0, sum_sq_t / sum_weight - mean_t * mean_t))

            cov = sum_st / sum_weight - mean_s * mean_t
            ncc = (
                cov / (std_s * std_t) if (std_s > epsilon and std_t > epsilon) else 0.0
            )
            ncc = ti.min(1.0, ti.max(-1.0, ncc))
            final_error = (1.0 - ncc) * style_weights[0] * sum_weight + guide_error
        else:
            final_error = guide_error

    return float(final_error)


@ti.func
def compute_patch_ssd(
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    patch_size: int,
    source_style: ti.template(),
    target_style: ti.template(),
    source_guide: ti.template(),
    target_guide: ti.template(),
    modulation_guide: ti.template(),
    use_modulation: int,
    style_weights: ti.template(),
    guide_weights: ti.template(),
    ebest: float,
    sw: int,
    sh: int,
    tw: int,
    th: int,
    NSC: int,
    NGC: int,
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    r = patch_size // 2
    error = 0.0

    inv_2_sigma_spatial_sq = -1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2_sigma_color_sq = -1.0 / (2.0 * sigma_color * sigma_color)

    # Center colors for bilateral
    s_val_center, t_val_center = 0.0, 0.0
    if use_bilateral:
        for c in range(NSC):
            s_val_center += float(source_style[sy, sx, c])
            t_val_center += float(target_style[ty, tx, c])
        s_val_center /= NSC
        t_val_center /= NSC

    steps = (2 * r) // n_size_step + 1
    for i, j in ti.ndrange(steps, steps):
        px = -r + i * n_size_step
        py = -r + j * n_size_step
        if px <= r and py <= r:
            idx_sx, idx_sy = (
                ti.max(0, ti.min(sx + px, sw - 1)),
                ti.max(0, ti.min(sy + py, sh - 1)),
            )
            idx_tx, idx_ty = (
                ti.max(0, ti.min(tx + px, tw - 1)),
                ti.max(0, ti.min(ty + py, th - 1)),
            )

            weight = 1.0
            if use_bilateral:
                s_val_c, t_val_c = 0.0, 0.0
                for c in range(NSC):
                    s_val_c += float(source_style[idx_sy, idx_sx, c])
                    t_val_c += float(target_style[idx_ty, idx_tx, c])
                s_val_c /= NSC
                t_val_c /= NSC

                dist_sq = float(px * px + py * py)
                color_diff_sq = (s_val_c - s_val_center) ** 2 + (
                    t_val_c - t_val_center
                ) ** 2
                weight = ti.exp(
                    dist_sq * inv_2_sigma_spatial_sq
                    + color_diff_sq * inv_2_sigma_color_sq
                )

            inner_break = 0
            for c in range(NSC):
                diff = float(source_style[idx_sy, idx_sx, c]) - float(
                    target_style[idx_ty, idx_tx, c]
                )
                error += weight * style_weights[c] * diff * diff
                if error > ebest:
                    inner_break = 1
                    break
            if inner_break:
                break

            for c in range(NGC):
                diff = float(source_guide[idx_sy, idx_sx, c]) - float(
                    target_guide[idx_ty, idx_tx, c]
                )
                mod = (
                    float(modulation_guide[idx_ty, idx_tx, c]) / 255.0
                    if use_modulation
                    else 1.0
                )
                error += weight * guide_weights[c] * mod * diff * diff
                if error > ebest:
                    inner_break = 1
                    break
            if inner_break:
                break
        if error > ebest:
            break
    return error


@ti.kernel
def compute_integral_image(src: ti.types.ndarray(), dst: ti.types.ndarray(), sqr: int):
    h, w = src.shape[0], src.shape[1]
    NSC = src.shape[2]
    # Rows
    for y in range(h):
        s = 0.0
        for x in range(w):
            val = 0.0
            for c in range(NSC):
                val += float(src[y, x, c])
            val /= NSC
            if sqr:
                val = val * val
            s += val
            dst[y, x] = s
    # Columns
    for x in range(w):
        s = 0.0
        for y in range(h):
            s += dst[y, x]
            dst[y, x] = s


@ti.kernel
def populate_omega(
    nnf: ti.types.ndarray(), omega_map: ti.types.ndarray(), patch_size: int
):
    sh, sw = omega_map.shape[0], omega_map.shape[1]
    for ty, tx in ti.ndrange(nnf.shape[0], nnf.shape[1]):
        patch_omega_atomic_add(
            omega_map, nnf[ty, tx, 0], nnf[ty, tx, 1], patch_size, sw, sh, 1
        )


@ti.kernel
def compute_error_map_kernel(
    nnf: ti.types.ndarray(),
    error_map: ti.types.ndarray(),
    source_style: ti.types.ndarray(),
    target_style: ti.types.ndarray(),
    source_guide: ti.types.ndarray(),
    target_guide: ti.types.ndarray(),
    modulation_guide: ti.types.ndarray(),
    use_modulation: int,
    patch_size: int,
    style_weights: ti.types.ndarray(),
    guide_weights: ti.types.ndarray(),
    cost_mode: int,
    s_sat: ti.types.ndarray(),
    s_sq_sat: ti.types.ndarray(),
    t_sat: ti.types.ndarray(),
    t_sq_sat: ti.types.ndarray(),
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    th, tw = error_map.shape[0], error_map.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    NSC, NGC = source_style.shape[2], source_guide.shape[2]
    for ty, tx in ti.ndrange(th, tw):
        sx, sy = nnf[ty, tx, 0], nnf[ty, tx, 1]
        if cost_mode == COST_FUNCTION_NCC:
            error_map[ty, tx] = compute_patch_ncc(
                sx,
                sy,
                tx,
                ty,
                patch_size,
                source_style,
                target_style,
                source_guide,
                target_guide,
                modulation_guide,
                use_modulation,
                style_weights,
                guide_weights,
                s_sat,
                s_sq_sat,
                t_sat,
                t_sq_sat,
                sw,
                sh,
                tw,
                th,
                NSC,
                NGC,
                use_bilateral,
                sigma_spatial,
                sigma_color,
                n_size_step,
            )
        else:
            error_map[ty, tx] = compute_patch_ssd(
                sx,
                sy,
                tx,
                ty,
                patch_size,
                source_style,
                target_style,
                source_guide,
                target_guide,
                modulation_guide,
                use_modulation,
                style_weights,
                guide_weights,
                1e20,
                sw,
                sh,
                tw,
                th,
                NSC,
                NGC,
                use_bilateral,
                sigma_spatial,
                sigma_color,
                n_size_step,
            )


@ti.kernel
def patchmatch_step_kernel(
    nnf: ti.types.ndarray(),
    error_map: ti.types.ndarray(),
    omega_map: ti.types.ndarray(),
    source_style: ti.types.ndarray(),
    target_style: ti.types.ndarray(),
    source_guide: ti.types.ndarray(),
    target_guide: ti.types.ndarray(),
    modulation_guide: ti.types.ndarray(),
    use_modulation: int,
    style_weights: ti.types.ndarray(),
    guide_weights: ti.types.ndarray(),
    patch_size: int,
    is_odd: int,
    uniformity_weight: float,
    mask: ti.types.ndarray(),
    cost_mode: int,
    omega_best: float,
    s_sat: ti.types.ndarray(),
    s_sq_sat: ti.types.ndarray(),
    t_sat: ti.types.ndarray(),
    t_sq_sat: ti.types.ndarray(),
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    th, tw = error_map.shape[0], error_map.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    NSC, NGC = source_style.shape[2], source_guide.shape[2]
    pixel_count = float(patch_size * patch_size)
    step = 1 if is_odd == 0 else -1

    for y_it, x_it in ti.ndrange(th, tw):
        tx, ty = (tw - 1 - x_it, th - 1 - y_it) if is_odd == 0 else (x_it, y_it)
        if mask[ty, tx] == 0:
            continue

        best_sx, best_sy = nnf[ty, tx, 0], nnf[ty, tx, 1]
        best_total_err = error_map[ty, tx] + uniformity_weight * (
            get_omega(omega_map, best_sx, best_sy, patch_size, sw, sh)
            / pixel_count
            / omega_best
        )

        for i in range(2):
            nx, ny = (tx + step, ty) if i == 0 else (tx, ty + step)
            if 0 <= nx < tw and 0 <= ny < th:
                cand_sx, cand_sy = (
                    nnf[ny, nx, 0] - (step if i == 0 else 0),
                    nnf[ny, nx, 1] - (0 if i == 0 else step),
                )
                r = patch_size // 2
                if r <= cand_sx < sw - r and r <= cand_sy < sh - r:
                    new_err = 0.0
                    if cost_mode == COST_FUNCTION_NCC:
                        new_err = compute_patch_ncc(
                            cand_sx,
                            cand_sy,
                            tx,
                            ty,
                            patch_size,
                            source_style,
                            target_style,
                            source_guide,
                            target_guide,
                            modulation_guide,
                            use_modulation,
                            style_weights,
                            guide_weights,
                            s_sat,
                            s_sq_sat,
                            t_sat,
                            t_sq_sat,
                            sw,
                            sh,
                            tw,
                            th,
                            NSC,
                            NGC,
                            use_bilateral,
                            sigma_spatial,
                            sigma_color,
                            n_size_step,
                        )
                    else:
                        new_err = compute_patch_ssd(
                            cand_sx,
                            cand_sy,
                            tx,
                            ty,
                            patch_size,
                            source_style,
                            target_style,
                            source_guide,
                            target_guide,
                            modulation_guide,
                            use_modulation,
                            style_weights,
                            guide_weights,
                            best_total_err,
                            sw,
                            sh,
                            tw,
                            th,
                            NSC,
                            NGC,
                            use_bilateral,
                            sigma_spatial,
                            sigma_color,
                            n_size_step,
                        )

                    new_total_err = new_err + uniformity_weight * (
                        get_omega(omega_map, cand_sx, cand_sy, patch_size, sw, sh)
                        / pixel_count
                        / omega_best
                    )
                    if new_total_err < best_total_err:
                        patch_omega_atomic_add(
                            omega_map, best_sx, best_sy, patch_size, sw, sh, -1
                        )
                        patch_omega_atomic_add(
                            omega_map, cand_sx, cand_sy, patch_size, sw, sh, 1
                        )
                        best_sx, best_sy, error_map[ty, tx], best_total_err = (
                            cand_sx,
                            cand_sy,
                            new_err,
                            new_total_err,
                        )
        nnf[ty, tx, 0], nnf[ty, tx, 1] = best_sx, best_sy


@ti.kernel
def random_search_kernel(
    nnf: ti.types.ndarray(),
    error_map: ti.types.ndarray(),
    omega_map: ti.types.ndarray(),
    source_style: ti.types.ndarray(),
    target_style: ti.types.ndarray(),
    source_guide: ti.types.ndarray(),
    target_guide: ti.types.ndarray(),
    modulation_guide: ti.types.ndarray(),
    use_modulation: int,
    style_weights: ti.types.ndarray(),
    guide_weights: ti.types.ndarray(),
    patch_size: int,
    radius: int,
    uniformity_weight: float,
    mask: ti.types.ndarray(),
    pruning_threshold: float,
    cost_mode: int,
    omega_best: float,
    s_sat: ti.types.ndarray(),
    s_sq_sat: ti.types.ndarray(),
    t_sat: ti.types.ndarray(),
    t_sq_sat: ti.types.ndarray(),
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    th, tw = error_map.shape[0], error_map.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    NSC, NGC = source_style.shape[2], source_guide.shape[2]
    pixel_count = float(patch_size * patch_size)

    for ty, tx in ti.ndrange(th, tw):
        if mask[ty, tx] == 0 or (
            pruning_threshold > 0 and error_map[ty, tx] < pruning_threshold
        ):
            continue

        best_sx, best_sy = nnf[ty, tx, 0], nnf[ty, tx, 1]
        best_total_err = error_map[ty, tx] + uniformity_weight * (
            get_omega(omega_map, best_sx, best_sy, patch_size, sw, sh)
            / pixel_count
            / omega_best
        )

        r = radius
        while r >= 1:
            off_x = int(ti.floor(ti.random() * (2 * r + 1))) - r
            off_y = int(ti.floor(ti.random() * (2 * r + 1))) - r
            cand_sx, cand_sy = best_sx + off_x, best_sy + off_y

            pr = patch_size // 2
            if pr <= cand_sx < sw - pr and pr <= cand_sy < sh - pr:
                new_err = 0.0
                if cost_mode == COST_FUNCTION_NCC:
                    new_err = compute_patch_ncc(
                        cand_sx,
                        cand_sy,
                        tx,
                        ty,
                        patch_size,
                        source_style,
                        target_style,
                        source_guide,
                        target_guide,
                        modulation_guide,
                        use_modulation,
                        style_weights,
                        guide_weights,
                        s_sat,
                        s_sq_sat,
                        t_sat,
                        t_sq_sat,
                        sw,
                        sh,
                        tw,
                        th,
                        NSC,
                        NGC,
                        use_bilateral,
                        sigma_spatial,
                        sigma_color,
                        n_size_step,
                    )
                else:
                    new_err = compute_patch_ssd(
                        cand_sx,
                        cand_sy,
                        tx,
                        ty,
                        patch_size,
                        source_style,
                        target_style,
                        source_guide,
                        target_guide,
                        modulation_guide,
                        use_modulation,
                        style_weights,
                        guide_weights,
                        best_total_err,
                        sw,
                        sh,
                        tw,
                        th,
                        NSC,
                        NGC,
                        use_bilateral,
                        sigma_spatial,
                        sigma_color,
                        n_size_step,
                    )

                new_total_err = new_err + uniformity_weight * (
                    get_omega(omega_map, cand_sx, cand_sy, patch_size, sw, sh)
                    / pixel_count
                    / omega_best
                )
                if new_total_err < best_total_err:
                    patch_omega_atomic_add(
                        omega_map, best_sx, best_sy, patch_size, sw, sh, -1
                    )
                    patch_omega_atomic_add(
                        omega_map, cand_sx, cand_sy, patch_size, sw, sh, 1
                    )
                    best_sx, best_sy, error_map[ty, tx], best_total_err = (
                        cand_sx,
                        cand_sy,
                        new_err,
                        new_total_err,
                    )
            r //= 2
        nnf[ty, tx, 0], nnf[ty, tx, 1] = best_sx, best_sy


@ti.kernel
def voting_kernel(
    output_image: ti.types.ndarray(),
    source_style: ti.types.ndarray(),
    target_style: ti.types.ndarray(),
    nnf: ti.types.ndarray(),
    error_map: ti.types.ndarray(),
    patch_size: int,
    mode: int,
    use_bilateral: int,
    sigma_spatial: float,
    sigma_color: float,
    n_size_step: int,
):
    th, tw, NSC = (
        output_image.shape[0],
        output_image.shape[1],
        source_style.shape[2],
    )
    sh, sw = source_style.shape[0], source_style.shape[1]
    r = patch_size // 2

    inv_2_sigma_spatial_sq = -1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2_sigma_color_sq = -1.0 / (2.0 * sigma_color * sigma_color)

    for ty, tx in ti.ndrange(th, tw):
        sum_color = ti.Vector([0.0, 0.0, 0.0, 0.0])
        sum_weight = 0.0

        # Bilateral center colors
        s_val_center, t_val_center = 0.0, 0.0
        if use_bilateral:
            sx_c, sy_c = nnf[ty, tx, 0], nnf[ty, tx, 1]
            for c in range(NSC):
                s_val_center += float(source_style[sy_c, sx_c, c])
                t_val_center += float(target_style[ty, tx, c])
            s_val_center /= NSC
            t_val_center /= NSC

        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            oy, ox = ty + py, tx + px
            if 0 <= ox < tw and 0 <= oy < th:
                sx, sy = nnf[oy, ox, 0] - px, nnf[oy, ox, 1] - py
                if 0 <= sx < sw and 0 <= sy < sh:
                    weight = (
                        1.0 / (1.0 + error_map[oy, ox])
                        if mode == EBSYNTH_VOTEMODE_WEIGHTED
                        else 1.0
                    )

                    if use_bilateral:
                        s_val_c, t_val_c = 0.0, 0.0
                        for c in range(NSC):
                            s_val_c += float(source_style[sy, sx, c])
                            t_val_c += float(target_style[ty, tx, c])
                        s_val_c /= NSC
                        t_val_c /= NSC

                        dist_sq = float(px * px + py * py)
                        color_diff_sq = (s_val_c - s_val_center) ** 2 + (
                            t_val_c - t_val_center
                        ) ** 2
                        weight *= ti.exp(
                            dist_sq * inv_2_sigma_spatial_sq
                            + color_diff_sq * inv_2_sigma_color_sq
                        )

                    for c in range(NSC):
                        if c < 4:
                            sum_color[c] += float(source_style[sy, sx, c]) * weight
                    sum_weight += weight
        if sum_weight > 0.0001:
            for c in range(NSC):
                if c < 4:
                    output_image[ty, tx, c] = ti.u8(
                        ti.max(0, ti.min(255, sum_color[c] / sum_weight))
                    )
        else:
            sx, sy = nnf[ty, tx, 0], nnf[ty, tx, 1]
            for c in range(NSC):
                if c < 4:
                    output_image[ty, tx, c] = source_style[sy, sx, c]


@ti.kernel
def vote_plain_kernel(
    source_style: ti.types.ndarray(dtype=ti.u8, ndim=3),
    nnf: ti.types.ndarray(dtype=ti.i32, ndim=3),
    acc: ti.types.ndarray(dtype=ti.f32, ndim=3),
    wsum: ti.types.ndarray(dtype=ti.f32, ndim=2),
    output_image: ti.types.ndarray(dtype=ti.u8, ndim=3),
    patch_size: ti.i32,
):
    th, tw = output_image.shape[0], output_image.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    NSC = source_style.shape[2]
    r = patch_size // 2
    for ty, tx in ti.ndrange(th, tw):
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            oy = ty - py
            ox = tx - px
            if 0 <= ox < tw and 0 <= oy < th:
                sx = ti.max(0, ti.min(nnf[oy, ox, 0] + px, sw - 1))
                sy = ti.max(0, ti.min(nnf[oy, ox, 1] + py, sh - 1))
                wsum[ty, tx] += 1.0
                for c in range(NSC):
                    acc[ty, tx, c] += ti.cast(source_style[sy, sx, c], ti.f32)
        weight = ti.max(wsum[ty, tx], 1e-6)
        for c in range(NSC):
            val = ti.max(0.0, ti.min(255.0, acc[ty, tx, c] / weight))
            output_image[ty, tx, c] = ti.u8(ti.cast(val, ti.i32))


@ti.kernel
def vote_weighted_kernel(
    source_style: ti.types.ndarray(dtype=ti.u8, ndim=3),
    nnf: ti.types.ndarray(dtype=ti.i32, ndim=3),
    error_map: ti.types.ndarray(dtype=ti.f32, ndim=2),
    acc: ti.types.ndarray(dtype=ti.f32, ndim=3),
    wsum: ti.types.ndarray(dtype=ti.f32, ndim=2),
    output_image: ti.types.ndarray(dtype=ti.u8, ndim=3),
    patch_size: ti.i32,
):
    th, tw = output_image.shape[0], output_image.shape[1]
    sh, sw = source_style.shape[0], source_style.shape[1]
    NSC = source_style.shape[2]
    r = patch_size // 2
    for ty, tx in ti.ndrange(th, tw):
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            oy = ty - py
            ox = tx - px
            if 0 <= ox < tw and 0 <= oy < th:
                sx = ti.max(0, ti.min(nnf[oy, ox, 0] + px, sw - 1))
                sy = ti.max(0, ti.min(nnf[oy, ox, 1] + py, sh - 1))
                weight = 1.0 / (1.0 + error_map[oy, ox])
                wsum[ty, tx] += weight
                for c in range(NSC):
                    acc[ty, tx, c] += weight * ti.cast(source_style[sy, sx, c], ti.f32)
        weight_sum = ti.max(wsum[ty, tx], 1e-6)
        for c in range(NSC):
            val = ti.max(0.0, ti.min(255.0, acc[ty, tx, c] / weight_sum))
            output_image[ty, tx, c] = ti.u8(ti.cast(val, ti.i32))


def run_vote_dispatch(
    output_image,
    source_style,
    target_style,
    nnf,
    error_map,
    patch_size,
    mode,
    use_bilateral,
    sigma_spatial,
    sigma_color,
    n_size_step,
    acc,
    wsum,
):
    if use_bilateral:
        voting_kernel(
            output_image,
            source_style,
            target_style,
            nnf,
            error_map,
            patch_size,
            mode,
            use_bilateral,
            sigma_spatial,
            sigma_color,
            n_size_step,
        )
    elif mode == EBSYNTH_VOTEMODE_WEIGHTED:
        vote_weighted_kernel(
            source_style,
            nnf,
            error_map,
            acc,
            wsum,
            output_image,
            int(patch_size),
        )
    else:
        vote_plain_kernel(
            source_style,
            nnf,
            acc,
            wsum,
            output_image,
            int(patch_size),
        )


@ti.kernel
def eval_mask_kernel(
    mask: ti.types.ndarray(),
    current_img: ti.types.ndarray(),
    previous_img: ti.types.ndarray(),
    threshold: int,
):
    NSC = current_img.shape[2]
    for y, x in ti.ndrange(mask.shape[0], mask.shape[1]):
        max_diff = 0
        for c in range(NSC):
            d = ti.abs(int(current_img[y, x, c]) - int(previous_img[y, x, c]))
            if d > max_diff:
                max_diff = d
        mask[y, x] = ti.u8(255) if max_diff >= threshold else ti.u8(0)


@ti.kernel
def dilate_mask_kernel(
    dst: ti.types.ndarray(), src: ti.types.ndarray(), patch_size: int
):
    th, tw, r = dst.shape[0], dst.shape[1], patch_size // 2
    for y, x in ti.ndrange(th, tw):
        val = ti.u8(0)
        for py, px in ti.ndrange((-r, r + 1), (-r, r + 1)):
            ny, nx = y + py, x + px
            if 0 <= nx < tw and 0 <= ny < th and src[ny, nx] > 0:
                val = ti.u8(255)
                break
        dst[y, x] = val
