import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.utils_draw import draw_ellipse, fill_convex_poly
from src.utils_io import (
    bgr_to_yuv,
    ensure_grayscale,
    to_uint8,
)


def alpha_blend(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    sigma: float = 25.0,
) -> np.ndarray:
    """Alpha blend using PyTorch for blur operation."""
    fg = foreground.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    elif a.ndim == 3 and a.shape[2] == 1:
        pass
    elif a.ndim == 3 and a.shape[2] == 3:
        # already 3 channels
        pass
    else:
        a = a.reshape(alpha.shape[0], alpha.shape[1], -1)
        if a.shape[2] != 1 and a.shape[2] != 3:
            a = a[:, :, :1]

    # Use PyTorch for blur
    k = max(1, int(sigma))
    a_t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    a_t = F.avg_pool2d(a_t, kernel_size=k, stride=1, padding=k // 2)
    a = a_t.squeeze(0).permute(1, 2, 0).numpy()

    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = np.clip(a, 0.0, 1.0)
    out = fg * a + bg * (1.0 - a)
    return to_uint8(out * 255.0)


def get_app_guide(image_bgr: np.ndarray, stretch_hist: bool = True) -> np.ndarray:
    """
    Generate appearance guide (high-pass filter) using PyTorch.
    Matches C++: app = |img - pyrDown(img)|
    """
    gray = ensure_grayscale(image_bgr)

    # Use PyTorch for all operations
    t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    t_blurred = _gaussian_pyr_down(t, rounds=3)
    t_resized = F.interpolate(
        t_blurred, size=gray.shape[:2], mode="bilinear", align_corners=False
    )
    blur_float = (t_resized.squeeze(0).squeeze(0).detach().cpu().numpy()) * 255.0

    result = gray.astype(np.float32) - blur_float
    result = (result / 2.0) + 128.0
    result = np.clip(result, 0, 255).astype(np.uint8)

    if not stretch_hist:
        return result

    min_val = int(result.min())
    max_val = int(result.max())
    # Match C++: int margin = min(min, 255 - max);
    margin = min(min_val, 255 - max_val)
    min_val = margin
    max_val = 255 - margin

    # Avoid divide by zero
    diff = max_val - min_val
    if diff < 1:
        diff = 1

    stretched = ((result.astype(np.float32) - min_val) / float(diff)) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _gaussian_pyr_down(image: torch.Tensor, rounds: int = 3) -> torch.Tensor:
    """
    Simulate cv2.pyrDown using PyTorch: Gaussian blur + downsample.
    image: (1, 1, H, W) float tensor range [0, 1]
    """
    # Matched cv2.pyrDown Gaussian kernel: [1, 4, 6, 4, 1] / 16
    device = image.device
    kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device)
    kernel = kernel / 16.0
    # 2D separable kernel
    k_x = kernel.view(1, 1, 1, 5)
    k_y = kernel.view(1, 1, 5, 1)

    t = image
    for _ in range(rounds):
        # Padding for same size convolution: kernel 5 needs pad 2
        # mode='reflect' matches OpenCV border reflection reasonably well
        t = F.pad(t, (2, 2, 2, 2), mode="reflect")
        t = F.conv2d(t, k_x)
        t = F.conv2d(t, k_y)
        # Downsample
        t = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)
    return t


def gradient_guide(
    width: int,
    height: int,
    draw_grid: bool = False,
    *,
    device: torch.device | None = None,
    as_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    """Create a positional gradient guide equivalent to C++ getGradient."""
    xs = torch.linspace(0, 255, width, device=device)
    ys = torch.linspace(0, 255, height, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    b = torch.zeros_like(grid_x)
    g = grid_y
    r = grid_x
    guide = torch.stack([b, g, r], dim=-1).to(torch.uint8)
    if draw_grid:
        guide_np = guide.cpu().numpy()
        step = 10
        guide_np[step::step, :] = 255
        guide_np[:, step::step] = 255
        guide = torch.from_numpy(guide_np)

    return guide.cpu().numpy() if as_numpy else guide


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


def get_skin_mask(
    image_bgr: np.ndarray, landmarks: Sequence[Tuple[int, int]]
) -> np.ndarray:
    """Generate skin mask using PyTorch operations."""
    lm = np.asarray([(int(x), int(y)) for x, y in landmarks], dtype=np.int32)
    face_contour = lm[:17]
    face_width = face_contour[-1, 0] - face_contour[0, 0]
    forehead_roi = (
        face_contour[0, 0],
        max(face_contour[0, 1] - int(face_width * 0.75), 0),
        face_width,
        min(int(face_width * 0.75), face_contour[0, 1]),
    )
    x, y, w, h = forehead_roi
    forehead = image_bgr[y : y + h, x : x + w].copy()
    forehead_yuv = bgr_to_yuv(forehead)

    sample_points = [
        (int((w / 4) * 1), max(h - int(face_width / 4), 0)),
        (int((w / 4) * 2), max(h - int(face_width / 4), 0)),
        (int((w / 4) * 3), max(h - int(face_width / 4), 0)),
    ]
    samples = []
    for sx, sy in sample_points:
        sy = int(np.clip(sy, 0, h - 1))
        sx = int(np.clip(sx, 0, w - 1))
        samples.append(
            np.mean(
                forehead_yuv[max(0, sy - 5) : sy + 6, max(0, sx - 5) : sx + 6],
                axis=(0, 1),
            )
        )
    samples = np.array(samples)

    mask = np.zeros((h, w), dtype=np.float32)
    threshold = 50.0
    for row in range(h):
        for col in range(w):
            pix = forehead_yuv[row, col].astype(np.float32)
            errs = np.sum((samples - pix)[:, 1:] ** 2, axis=1)
            if np.min(errs) < threshold:
                mask[row, col] = 1.0

    full_mask = np.zeros(image_bgr.shape[:2], dtype=np.float32)
    full_mask[y : y + h, x : x + w] = mask
    full_mask = fill_convex_poly(full_mask, face_contour, 1.0)
    center = tuple(
        (face_contour[0] + (face_contour[-1] - face_contour[0]) // 2).tolist()
    )
    axes = (int(face_width / 2), int(face_width / 2.5))
    full_mask = draw_ellipse(full_mask, center, axes)
    return full_mask


def gray_hist_matching(input_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    src = ensure_grayscale(input_gray)
    ref = ensure_grayscale(ref_gray)

    # Use bincount instead of histogram(density=True) to avoid extra float work,
    # then build the mapping with a vectorized searchsorted instead of a loop.
    src_hist = np.bincount(src.ravel(), minlength=256).astype(np.float32)
    ref_hist = np.bincount(ref.ravel(), minlength=256).astype(np.float32)

    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    # Normalize to [0, 1]; guard against empty inputs.
    src_total = src_cdf[-1] if src_cdf[-1] > 0 else 1.0
    ref_total = ref_cdf[-1] if ref_cdf[-1] > 0 else 1.0
    src_cdf /= src_total
    ref_cdf /= ref_total

    mapping = np.searchsorted(ref_cdf, src_cdf, side="left")
    mapping = np.clip(mapping, 0, 255).astype(np.uint8)
    return mapping[src]


def get_head_area_rect(
    landmarks: Sequence[Tuple[int, int]], img_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    pts = np.asarray(landmarks, dtype=np.int32)
    width = pts[16, 0] - pts[0, 0]
    higher_y = min(pts[0, 1], pts[16, 1])
    height = pts[8, 1] - (higher_y - width // 2)
    x = max(int(pts[0, 0] - width * 0.1), 0)
    y = max(int((higher_y - width / 2.0) - (height * 0.2)), 0)
    max_w = img_size[1] - x
    max_h = img_size[0] - y
    w = int(min(width * 1.2, max_w))
    h = int(min(height * 1.4, max_h))
    return (x, y, w, h)
