# ezsynth/utils/blend_logic.py
import time
from typing import Optional

import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.fft import fft2, ifft2
from tqdm import tqdm


# --- Logic from original histogram_blend.py ---
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


# --- Logic from original reconstruction.py ---
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


def fft_poisson_solver_channel(
    target_img_ch: np.ndarray, grad_x_ch: np.ndarray, grad_y_ch: np.ndarray
) -> np.ndarray:
    """Solves the Poisson equation for a single channel using FFT."""
    h, w = target_img_ch.shape

    # Compute divergence of the gradient field
    # The divergence of a vector field F = (P, Q) is dP/dx + dQ/dy.
    # In discrete terms, this is (P[i+1] - P[i]) + (Q[j+1] - Q[j]).
    # Since our gradient is computed as I[i] - I[i+1], we use a forward difference.
    div = np.zeros((h, w), dtype=np.float32)
    div[1:, :] = grad_x_ch[:-1, :]
    div[:-1, :] -= grad_x_ch[:-1, :]
    div[:, 1:] += grad_y_ch[:, :-1]
    div[:, :-1] -= grad_y_ch[:, :-1]

    # Prepare FFT denominator (Laplacian in Fourier domain)
    kx = np.fft.fftfreq(w)[np.newaxis, :]
    ky = np.fft.fftfreq(h)[:, np.newaxis]
    denom = 2 * np.cos(2 * np.pi * kx) + 2 * np.cos(2 * np.pi * ky) - 4
    denom[0, 0] = 1.0  # Avoid division by zero at DC component

    # Solve in frequency domain
    result_fft = fft2(div) / denom
    # Restore the DC component (average value) from the target image
    result_fft[0, 0] = fft2(target_img_ch)[0, 0]

    return ifft2(result_fft).real


def poisson_fusion_cpu_optimized(
    blendI,
    I1,
    I2,
    mask,
    As,
    solver: str,
    maxiter: Optional[int],
    grad_weights: list[float],
):
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

    if solver == "fft":
        out_all = np.zeros_like(Iab)
        for channel in range(c):
            out_all[..., channel] = fft_poisson_solver_channel(
                Iab[..., channel], gx[..., channel], gy[..., channel]
            )
        final = out_all
    else:  # lsqr, lsmr
        gx_reshaped = np.clip(gx.reshape(h * w, c), -100, 100)
        gy_reshaped = np.clip(gy.reshape(h * w, c), -100, 100)
        Iab_reshaped = Iab.reshape(h * w, c)
        Iab_mean = np.mean(Iab_reshaped, axis=0)
        Iab_centered = Iab_reshaped - Iab_mean
        out_all = np.zeros((h * w, c), dtype=np.float32)

        for channel in range(c):
            b = np.vstack(
                [
                    gx_reshaped[:, channel : channel + 1] * grad_weights[channel],
                    gy_reshaped[:, channel : channel + 1] * grad_weights[channel],
                    Iab_centered[:, channel : channel + 1],
                ]
            )
            A = As[channel]
            if solver == "lsqr":
                out_all[:, channel] = scipy.sparse.linalg.lsqr(A, b, iter_lim=maxiter)[
                    0
                ]
            else:  # 'lsmr'
                out_all[:, channel] = scipy.sparse.linalg.lsmr(A, b, maxiter=maxiter)[0]

        final = (out_all + Iab_mean).reshape(h, w, c)

    final = np.clip(final, 0, 255)
    return cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_LAB2BGR)


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
        self._A_matrix_cache = {}

    def run(self, hist_blends, style_fwd, style_bwd, err_masks):
        reconstructed_frames = []
        num_frames = len(hist_blends)
        if num_frames == 0 or self.solver == "disabled":
            return hist_blends  # Return histogram blends if disabled

        h, w, _ = hist_blends[0].shape

        # Cache the 'A' matrix since it only depends on dimensions
        if self.solver not in ["fft", "disabled"]:
            if (h, w) not in self._A_matrix_cache:
                self._A_matrix_cache[(h, w)] = construct_A_cpu(h, w, self.grad_weights)
            As = self._A_matrix_cache[(h, w)]
        else:
            As = None

        for i in tqdm(range(num_frames), desc=f"Poisson Recon ({self.solver})"):
            frame = poisson_fusion_cpu_optimized(
                hist_blends[i],
                style_fwd[i],
                style_bwd[i],
                err_masks[i],
                As,
                self.solver,
                self.poisson_maxiter,
                self.grad_weights,
            )
            reconstructed_frames.append(frame)
        return reconstructed_frames
