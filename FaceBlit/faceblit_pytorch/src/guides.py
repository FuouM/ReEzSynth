import numpy as np
import torch
import torch.nn.functional as F

from .image_utils import _ensure_grayscale

__all__ = [
    "gradient_guide",
    "_gaussian_pyr_down",
    "get_app_guide",
    "gray_hist_matching",
]


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


def get_app_guide(image_bgr: np.ndarray, stretch_hist: bool = True) -> np.ndarray:
    """
    Generate appearance guide (high-pass filter) using PyTorch.
    Matches C++: app = |img - pyrDown(img)|
    """
    gray = _ensure_grayscale(image_bgr)

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


def gray_hist_matching(input_gray: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    src = _ensure_grayscale(input_gray)
    ref = _ensure_grayscale(ref_gray)

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
