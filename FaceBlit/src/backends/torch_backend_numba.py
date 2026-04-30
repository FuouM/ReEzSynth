from __future__ import annotations

import numpy as np
from numba import njit

EMPTY_APP_GUIDE = np.zeros((1, 1), dtype=np.uint8)
EMPTY_LOOK_UP_CUBE = np.zeros((1, 1, 1, 2), dtype=np.uint16)


@njit(cache=True)
def dfs_seed_grow_voting_numba(
    tgt_seed_y: int,
    tgt_seed_x: int,
    sty_seed_y: int,
    sty_seed_x: int,
    style_pos_guide: np.ndarray,
    target_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    target_app_guide: np.ndarray,
    nnf: np.ndarray,
    covered_pixels: np.ndarray,
    chunk_number: int,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
    use_app: bool,
    q_y: np.ndarray,
    q_x: np.ndarray,
) -> None:
    # print("DFS NUMBA")
    h, w = target_pos_guide.shape[:2]
    max_q = h * w
    head = 0
    tail = 1
    q_y[0] = tgt_seed_y
    q_x[0] = tgt_seed_x

    sty_h, sty_w = style_pos_guide.shape[:2]

    while head < tail:
        ty = q_y[head]
        tx = q_x[head]
        head += 1

        dy = ty - tgt_seed_y
        dx = tx - tgt_seed_x
        sy = sty_seed_y + dy
        sx = sty_seed_x + dx

        if sy < 0 or sy >= sty_h or sx < 0 or sx >= sty_w:
            continue
        if ty < 0 or ty >= h or tx < 0 or tx >= w:
            continue
        if covered_pixels[ty, tx] != 0:
            continue

        pos_t = target_pos_guide[ty, tx]
        pos_s = style_pos_guide[sy, sx]
        pos_err = abs(int(pos_t[2]) - int(pos_s[2])) + abs(
            int(pos_t[1]) - int(pos_s[1])
        )

        app_err = 0
        if use_app:
            app_err = abs(int(target_app_guide[ty, tx]) - int(style_app_guide[sy, sx]))

        error = lambda_pos * pos_err + lambda_app * app_err

        if error < threshold or (dy == 0 and dx == 0):
            nnf[ty, tx, 0] = sy
            nnf[ty, tx, 1] = sx
            covered_pixels[ty, tx] = chunk_number

            if ty > 0 and tail < max_q:
                q_y[tail] = ty - 1
                q_x[tail] = tx
                tail += 1
            if ty + 1 < h and tail < max_q:
                q_y[tail] = ty + 1
                q_x[tail] = tx
                tail += 1
            if tx > 0 and tail < max_q:
                q_y[tail] = ty
                q_x[tail] = tx - 1
                tail += 1
            if tx + 1 < w and tail < max_q:
                q_y[tail] = ty
                q_x[tail] = tx + 1
                tail += 1


@njit(cache=True)
def denoise_nnf_numba(nnf: np.ndarray, patch_size: int) -> np.ndarray:
    h, w, _ = nnf.shape
    half = patch_size // 2
    result = np.empty_like(nnf)

    max_k = patch_size * patch_size
    offsets_y = np.empty(max_k, np.int32)
    offsets_x = np.empty(max_k, np.int32)

    for row in range(h):
        for col in range(w):
            k = 0
            for dy in range(-half, half + 1):
                ry = row + dy
                if ry < 0 or ry >= h:
                    continue
                for dx in range(-half, half + 1):
                    rx = col + dx
                    if rx < 0 or rx >= w:
                        continue
                    offsets_y[k] = int(nnf[ry, rx, 0]) - ry
                    offsets_x[k] = int(nnf[ry, rx, 1]) - rx
                    k += 1

            best_i = 0
            best_freq = 0
            for i in range(k):
                cy = offsets_y[i]
                cx = offsets_x[i]
                freq = 1
                for j in range(i + 1, k):
                    if offsets_y[j] == cy and offsets_x[j] == cx:
                        freq += 1
                if freq > best_freq:
                    best_freq = freq
                    best_i = i

            result[row, col, 0] = offsets_y[best_i] + row
            result[row, col, 1] = offsets_x[best_i] + col

    return result


@njit(cache=True)
def _build_nnf_dfs_voting_numba_core(
    target_pos_guide: np.ndarray,
    target_app_guide: np.ndarray,
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray,
    look_up_cube: np.ndarray,
    nnf: np.ndarray,
    covered_pixels: np.ndarray,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
    use_app: bool,
    use_lut: bool,
    x: int,
    y: int,
    h: int,
    w: int,
    style_h: int,
    style_w: int,
    q_y: np.ndarray,
    q_x: np.ndarray,
) -> None:
    chunk_number = 1
    x_norm = style_w / 256.0
    y_norm = style_h / 256.0

    for row_t in range(y, y + h):
        for col_t in range(x, x + w):
            if covered_pixels[row_t, col_t] != 0:
                continue

            pos_vals = target_pos_guide[row_t, col_t]
            pos_r = int(pos_vals[2])
            pos_g = int(pos_vals[1])
            if use_lut:
                app_val = int(target_app_guide[row_t, col_t])
                coords = look_up_cube[pos_r, pos_g, app_val]
                sty_seed_x = int(coords[0])
                sty_seed_y = int(coords[1])
            else:
                sty_seed_x = int(pos_r * x_norm)
                sty_seed_y = int(pos_g * y_norm)

            if sty_seed_x < 0:
                sty_seed_x = 0
            elif sty_seed_x >= style_w:
                sty_seed_x = style_w - 1
            if sty_seed_y < 0:
                sty_seed_y = 0
            elif sty_seed_y >= style_h:
                sty_seed_y = style_h - 1

            dfs_seed_grow_voting_numba(
                row_t,
                col_t,
                sty_seed_y,
                sty_seed_x,
                style_pos_guide,
                target_pos_guide,
                style_app_guide,
                target_app_guide,
                nnf,
                covered_pixels,
                chunk_number,
                threshold,
                lambda_pos,
                lambda_app,
                use_app,
                q_y,
                q_x,
            )
            chunk_number += 1


def initialize_nnf_dfs_voting_numba(
    target_pos_guide: np.ndarray,
    target_app_gray: np.ndarray | None,
    style_pos_guide: np.ndarray,
    style_app_guide: np.ndarray | None,
    look_up_cube: np.ndarray | None,
    threshold: int,
    lambda_pos: int,
    lambda_app: int,
    style_h: int,
    style_w: int,
    x: int,
    y: int,
    h: int,
    w: int,
    h_t: int,
    w_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the DFS NNF in one Numba call to avoid per-seed Python overhead."""
    target_pos_int = target_pos_guide.astype(np.int16, copy=False)
    style_pos_int = style_pos_guide.astype(np.int16, copy=False)
    target_app_int = (
        target_app_gray.astype(np.int16, copy=False)
        if target_app_gray is not None
        else EMPTY_APP_GUIDE
    )
    style_app_int = (
        style_app_guide.astype(np.int16, copy=False)
        if style_app_guide is not None
        else EMPTY_APP_GUIDE
    )
    lut = look_up_cube if look_up_cube is not None else EMPTY_LOOK_UP_CUBE
    use_lut = (
        lambda_app != 0 and target_app_gray is not None and look_up_cube is not None
    )
    use_app = (
        lambda_app != 0 and target_app_gray is not None and style_app_guide is not None
    )

    nnf = np.zeros((h_t, w_t, 2), dtype=np.int32)
    covered_pixels = np.zeros((h_t, w_t), dtype=np.int32)
    q_y = np.empty(h_t * w_t, dtype=np.int32)
    q_x = np.empty(h_t * w_t, dtype=np.int32)

    _build_nnf_dfs_voting_numba_core(
        target_pos_int,
        target_app_int,
        style_pos_int,
        style_app_int,
        lut,
        nnf,
        covered_pixels,
        threshold,
        lambda_pos,
        lambda_app,
        use_app,
        use_lut,
        x,
        y,
        h,
        w,
        style_h,
        style_w,
        q_y,
        q_x,
    )

    return nnf, covered_pixels


def build_nnf_dfs_voting_numba(
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
    """DFS voting NNF using :func:`dfs_seed_grow_voting_numba` and :func:`denoise_nnf_numba`."""
    thresh = 50 if threshold is None else threshold
    nnf, _covered_pixels = initialize_nnf_dfs_voting_numba(
        target_pos_guide,
        target_app_gray,
        style_pos_guide,
        style_app_guide,
        look_up_cube,
        thresh,
        lambda_pos,
        lambda_app,
        style_h,
        style_w,
        x,
        y,
        h,
        w,
        h_t,
        w_t,
    )

    return denoise_nnf_numba(nnf, patch_size)
