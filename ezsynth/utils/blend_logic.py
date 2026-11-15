# ezsynth/utils/blend_logic.py
from __future__ import annotations

import logging
import time
from importlib.util import find_spec

import cv2
import numpy as np
from tqdm import tqdm

from ..kernels.imageops import ImageOpsEngine, load_op
from .progress import report_progress

logger = logging.getLogger(__name__)

# --- Optional Dependency Handling ---
# Availability is probed without importing; scipy and pyamg load on first use.
PYAMG_AVAILABLE = find_spec("pyamg") is not None
SCIPY_AVAILABLE = find_spec("scipy") is not None
# Solvers built on scipy.sparse matrices; "lsqr" runs on NumPy (imageops.opencv).
SCIPY_SOLVERS = ("lsmr", "cg", "amg")


# --- Algorithmic Alternatives ---


def seamless_clone_blending(frame_fwd, frame_bwd, mask):
    """Blends using OpenCV's seamlessClone."""
    _h, _w, _ = frame_fwd.shape
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
    a: np.ndarray,
    b: np.ndarray,
    error_mask: np.ndarray,
    weight1=0.5,
    weight2=0.5,
    engine: ImageOpsEngine = "opencv",
) -> np.ndarray:
    """Blend BGR frames ``a`` and ``b`` in Lab space with the statistics of the
    lower-error frame (``error_mask`` 0 where ``a`` wins).

    ``wgpu`` follows the OpenCV path on the GPU (Lab conversion within two levels of
    OpenCV's); ``taichi`` is an approximate float-Lab variant.
    """
    return load_op(engine, "hist_blend")(a, b, error_mask, weight1, weight2)


# --- Poisson Solvers ---
def construct_A_cpu(h: int, w: int, grad_weight: list[float]):
    import scipy.sparse

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
    logger.debug(f"Constructing Poisson matrix 'A' took {time.time() - st:.4f} s")
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
            out_all[:, ch] = load_op("opencv", "poisson_lsq")(
                gx_reshaped[:, ch].reshape(h, w),
                gy_reshaped[:, ch].reshape(h, w),
                Iab_centered[:, ch].reshape(h, w),
                grad_weights[ch],
                max_iter=maxiter,
            ).reshape(-1)
        elif solver == "lsmr":
            import scipy.sparse.linalg

            A = cache["As"][ch]
            out_all[:, ch] = scipy.sparse.linalg.lsmr(A, b, maxiter=maxiter)[0]
        elif solver == "cg":
            import scipy.sparse.linalg

            AtA, Atb = cache["AtAs"][ch], cache["As"][ch].T @ b
            out_all[:, ch] = scipy.sparse.linalg.cg(AtA, Atb, maxiter=maxiter)[0]
        elif solver == "amg":
            A, ml = cache["As"][ch], cache["MLs"][ch]
            out_all[:, ch] = ml.solve(A.T @ b, tol=1e-6, maxiter=maxiter, accel="cg")
        elif solver == "taichi-cg":
            out_all[:, ch] = load_op("taichi", "poisson_cg")(
                gx[..., ch],
                gy[..., ch],
                Iab_centered[..., ch].reshape(h, w),
                grad_weights[ch],
                max_iter=maxiter or 100,
            ).reshape(-1)
    final = (out_all + Iab_mean).reshape(h, w, c)

    return cv2.cvtColor(np.clip(final, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --- Main Reconstructor Class ---
class Reconstructor:
    def __init__(
        self,
        solver: str = "lsqr",
        poisson_maxiter: int | None = None,
        grad_weights: list[float] | None = None,
    ):
        self.solver = solver
        self.poisson_maxiter = poisson_maxiter
        self.grad_weights = (
            list(grad_weights) if grad_weights is not None else [2.5, 0.5, 0.5]
        )
        self._cache = {}

        if self.solver in SCIPY_SOLVERS and not SCIPY_AVAILABLE:
            raise ImportError(
                f"The {self.solver!r} Poisson solver needs SciPy: pip install scipy "
                "(or uv sync --extra scipy). The default 'lsqr' solver needs no SciPy."
            )
        if self.solver == "amg" and not PYAMG_AVAILABLE:
            raise ImportError(
                "PyAMG is not installed. Please install it with 'pip install pyamg' to use the 'amg' solver."
            )

    def _prepare_cache_for_size(self, h, w):
        hw_key = (h, w)
        if hw_key in self._cache:
            return self._cache[hw_key]

        logger.info(f"Preparing solver cache for resolution {w}x{h}...")
        size_cache = {}
        # Only construct matrices if a solver that needs them is selected
        if self.solver in SCIPY_SOLVERS:
            As = construct_A_cpu(h, w, self.grad_weights)
            size_cache["As"] = As
            if self.solver == "cg" or self.solver == "amg":
                size_cache["AtAs"] = [A.T @ A for A in As]
            if self.solver == "amg":
                import pyamg

                logger.debug("  - Building AMG solver...")
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
            report_progress(f"reconstruction ({self.solver})", i, num_frames)
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
