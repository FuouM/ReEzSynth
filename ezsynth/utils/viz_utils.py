# ezsynth/utils/viz_utils.py
import numpy as np


def make_colorwheel():
    """Generates a color wheel for optical flow visualization."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)
    col_idx = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY).astype(np.uint8)
    col_idx += RY
    # YG
    colorwheel[col_idx : col_idx + YG, 0] = 255 - np.floor(
        255 * np.arange(0, YG) / YG
    ).astype(np.uint8)
    colorwheel[col_idx : col_idx + YG, 1] = 255
    col_idx += YG
    # GC
    colorwheel[col_idx : col_idx + GC, 1] = 255
    colorwheel[col_idx : col_idx + GC, 2] = np.floor(
        255 * np.arange(0, GC) / GC
    ).astype(np.uint8)
    col_idx += GC
    # CB
    colorwheel[col_idx : col_idx + CB, 1] = 255 - np.floor(
        255 * np.arange(CB) / CB
    ).astype(np.uint8)
    colorwheel[col_idx : col_idx + CB, 2] = 255
    col_idx += CB
    # BM
    colorwheel[col_idx : col_idx + BM, 2] = 255
    colorwheel[col_idx : col_idx + BM, 0] = np.floor(
        255 * np.arange(0, BM) / BM
    ).astype(np.uint8)
    col_idx += BM
    # MR
    colorwheel[col_idx : col_idx + MR, 2] = 255 - np.floor(
        255 * np.arange(MR) / MR
    ).astype(np.uint8)
    colorwheel[col_idx : col_idx + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """Applies the flow color wheel to flow components u and v."""
    flow_image = np.zeros((*u.shape, 3), dtype=np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    flow_magnitude = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(-v, -u) / np.pi

    flow_k = (angle + 1) / 2 * (ncols - 1)

    k0 = np.floor(flow_k).astype(np.int32)
    k1 = (k0 + 1) % ncols

    # More descriptive variable name to avoid potential conflicts
    fraction = flow_k - k0

    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0

        color_val = (1 - fraction) * col0 + fraction * col1

        idx = flow_magnitude <= 1
        color_val[idx] = 1 - flow_magnitude[idx] * (1 - color_val[idx])
        color_val[~idx] *= 0.75

        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[..., ch_idx] = (255 * color_val).astype(np.uint8)

    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Converts a 2-channel flow field to a color image for visualization.
    """
    assert (
        flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    ), "Input flow must have shape [H,W,2]"

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u, v = flow_uv[..., 0], flow_uv[..., 1]

    flow_magnitude = np.sqrt(np.square(u) + np.square(v))
    # Defensive check for max on an empty array or all-zero flow
    if flow_magnitude.size > 0:
        magnitude_max = np.max(flow_magnitude)
    else:
        magnitude_max = 0.0

    epsilon = 1e-5
    # Defensive check to avoid division by zero if flow is all zeros
    if magnitude_max < epsilon:
        u_normalized = u
        v_normalized = v
    else:
        u_normalized = u / (magnitude_max + epsilon)
        v_normalized = v / (magnitude_max + epsilon)

    return flow_uv_to_colors(u_normalized, v_normalized, convert_to_bgr=convert_to_bgr)
