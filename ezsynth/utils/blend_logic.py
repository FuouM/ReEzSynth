# ezsynth/utils/blend_logic.py
import time
from typing import Optional

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from tqdm import tqdm

# --- Optional Dependency Handling ---
try:
    import pyamg

    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False


# --- Algorithmic Alternatives ---


def seamless_clone_blending(frame_fwd, frame_bwd, mask):
    """Blends using OpenCV's seamlessClone."""
    h, w, _ = frame_fwd.shape
    # seamlessClone requires a single-channel mask
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Find bounding box of the mask to get a center point
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame_fwd  # Return forward frame if mask is empty

    x, y, wc, hc = cv2.boundingRect(np.concatenate(contours))
    center = (x + wc // 2, y + hc // 2)

    # The 'source' is the image providing the masked content (frame_bwd)
    # The 'destination' is the background (frame_fwd)
    return cv2.seamlessClone(frame_bwd, frame_fwd, mask, center, cv2.NORMAL_CLONE)


# --- Histogram Blending (Preprocessor) ---
def hist_blender(
    a: np.ndarray, b: np.ndarray, error_mask: np.ndarray, weight1=0.5, weight2=0.5
) -> np.ndarray:
    if len(error_mask.shape) == 2:
        error_mask = np.repeat(error_mask[:, :, np.newaxis], 3, axis=2)
    a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2Lab)
    min_error_lab = np.where(error_mask == 0, a_lab, b_lab)
    a_mean, a_std = np.mean(a_lab, axis=(0, 1)), np.std(a_lab, axis=(0, 1))
    b_mean, b_std = np.mean(b_lab, axis=(0, 1)), np.std(b_lab, axis=(0, 1))
    min_error_mean, min_error_std = (
        np.mean(min_error_lab, axis=(0, 1)),
        np.std(min_error_lab, axis=(0, 1)),
    )
    t_mean = np.full(3, 0.5 * 256, dtype=np.float32)
    t_std = np.full(3, (1 / 36) * 256, dtype=np.float32)
    a_lab_norm = ((a_lab - a_mean) * t_std / a_std + t_mean).astype(np.float32)
    b_lab_norm = ((b_lab - b_mean) * t_std / b_std + t_mean).astype(np.float32)
    ab_lab = (a_lab_norm * weight1 + b_lab_norm * weight2 - 128) / 0.5 + 128
    ab_mean, ab_std = np.mean(ab_lab, axis=(0, 1)), np.std(ab_lab, axis=(0, 1))
    ab_lab_final = (ab_lab - ab_mean) * min_error_std / ab_std + min_error_mean
    ab_lab_final = np.clip(np.round(ab_lab_final), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ab_lab_final, cv2.COLOR_Lab2BGR)


# --- Poisson Solvers ---
def construct_A_cpu(h: int, w: int, grad_weight: list[float]):
    st = time.time()
    indgx_x = np.zeros(2 * (h - 1) * w, dtype=int)
    indgx_y = np.zeros(2 * (h - 1) * w, dtype=int)
    vdx = np.ones(2 * (h - 1) * w)
    indgy_x = np.zeros(2 * h * (w - 1), dtype=int)
    indgy_y = np.zeros(2 * h * (w - 1), dtype=int)
    vdy = np.ones(2 * h * (w - 1))
    indgx_x[::2] = np.arange((h - 1) * w)
    indgx_y[::2] = indgx_x[::2]
    indgx_x[1::2] = indgx_x[::2]
    indgx_y[1::2] = indgx_x[::2] + w
    indgy_x[::2] = np.arange(h * (w - 1))
    indgy_y[::2] = indgy_x[::2]
    indgy_x[1::2] = indgy_x[::2]
    indgy_y[1::2] = indgy_x[::2] + 1
    vdx[1::2] = -1
    vdy[1::2] = -1
    Ix = scipy.sparse.eye(h * w, format="csc")
    Gx = scipy.sparse.coo_matrix(
        (vdx, (indgx_x, indgx_y)), shape=(h * w, h * w)
    ).tocsc()
    Gy = scipy.sparse.coo_matrix(
        (vdy, (indgy_x, indgy_y)), shape=(h * w, h * w)
    ).tocsc()
    As = [scipy.sparse.vstack([Gx * weight, Gy * weight, Ix]) for weight in grad_weight]
    print(f"Constructing Poisson matrix 'A' took {time.time() - st:.4f} s")
    return As


def poisson_fusion_cpu(blendI, I1, I2, mask, cache, solver, maxiter, grad_weights):
    Iab = cv2.cvtColor(blendI, cv2.COLOR_BGR2LAB).astype(np.float32)
    Ia = cv2.cvtColor(I1, cv2.COLOR_BGR2LAB).astype(np.float32)
    Ib = cv2.cvtColor(I2, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = (mask > 0).astype(np.float32)[..., np.newaxis]
    h, w, c = Iab.shape
    gx = np.zeros_like(Ia)
    gy = np.zeros_like(Ia)
    gx[:-1] = (Ia[:-1] - Ia[1:]) * (1 - m[:-1]) + (Ib[:-1] - Ib[1:]) * m[:-1]
    gy[:, :-1] = (Ia[:, :-1] - Ia[:, 1:]) * (1 - m[:, :-1]) + (
        Ib[:, :-1] - Ib[:, 1:]
    ) * m[:, :-1]

    gx_reshaped = np.clip(gx.reshape(h * w, c), -100, 100)
    gy_reshaped = np.clip(gy.reshape(h * w, c), -100, 100)
    Iab_reshaped = Iab.reshape(h * w, c)
    Iab_mean = np.mean(Iab_reshaped, axis=0)
    Iab_centered = Iab_reshaped - Iab_mean
    out_all = np.zeros((h * w, c), dtype=np.float32)

    for ch in range(c):
        b = np.vstack(
            [
                gx_reshaped[:, ch : ch + 1] * grad_weights[ch],
                gy_reshaped[:, ch : ch + 1] * grad_weights[ch],
                Iab_centered[:, ch : ch + 1],
            ]
        )
        if solver == "lsqr":
            A = cache["As"][ch]
            out_all[:, ch] = scipy.sparse.linalg.lsqr(A, b, iter_lim=maxiter)[0]
        elif solver == "lsmr":
            A = cache["As"][ch]
            out_all[:, ch] = scipy.sparse.linalg.lsmr(A, b, maxiter=maxiter)[0]
        elif solver == "cg":
            AtA, Atb = cache["AtAs"][ch], cache["As"][ch].T @ b
            out_all[:, ch] = scipy.sparse.linalg.cg(AtA, Atb, maxiter=maxiter)[0]
        elif solver == "amg":
            A, ml = cache["As"][ch], cache["MLs"][ch]
            out_all[:, ch] = ml.solve(A.T @ b, tol=1e-6, maxiter=maxiter, accel="cg")
    final = (out_all + Iab_mean).reshape(h, w, c)

    return cv2.cvtColor(np.clip(final, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --- Main Reconstructor Class ---
class Reconstructor:
    def __init__(
        self,
        solver: str = "lsqr",
        poisson_maxiter: Optional[int] = None,
        grad_weights: list[float] = [2.5, 0.5, 0.5],
    ):
        self.solver = solver
        self.poisson_maxiter = poisson_maxiter
        self.grad_weights = grad_weights
        self._cache = {}

        if self.solver == "amg" and not PYAMG_AVAILABLE:
            raise ImportError(
                "PyAMG is not installed. Please install it with 'pip install pyamg' to use the 'amg' solver."
            )

    def _prepare_cache_for_size(self, h, w):
        hw_key = (h, w)
        if hw_key in self._cache:
            return self._cache[hw_key]

        print(f"Preparing solver cache for resolution {w}x{h}...")
        size_cache = {}
        # Only construct matrices if a solver that needs them is selected
        if self.solver in ["lsqr", "lsmr", "cg", "amg"]:
            As = construct_A_cpu(h, w, self.grad_weights)
            size_cache["As"] = As
            if self.solver == "cg" or self.solver == "amg":
                size_cache["AtAs"] = [A.T @ A for A in As]
            if self.solver == "amg":
                print("  - Building AMG solver...")
                size_cache["MLs"] = [
                    pyamg.ruge_stuben_solver(AtA) for AtA in size_cache["AtAs"]
                ]

        self._cache[hw_key] = size_cache
        return size_cache

    def run(self, hist_blends, style_fwd, style_bwd, err_masks):
        num_frames = len(hist_blends)
        if num_frames == 0 or self.solver == "disabled":
            return hist_blends

        h, w, _ = hist_blends[0].shape
        solver_cache = self._prepare_cache_for_size(h, w)
        reconstructed_frames = []

        desc = f"Reconstruction ({self.solver})"
        for i in tqdm(range(num_frames), desc=desc):
            if self.solver == "seamless":
                frame = seamless_clone_blending(
                    style_fwd[i], style_bwd[i], err_masks[i]
                )
            else:  # All other CPU-based Poisson solvers
                frame = poisson_fusion_cpu(
                    hist_blends[i],
                    style_fwd[i],
                    style_bwd[i],
                    err_masks[i],
                    solver_cache,
                    self.solver,
                    self.poisson_maxiter,
                    self.grad_weights,
                )
            reconstructed_frames.append(frame)
        return reconstructed_frames
