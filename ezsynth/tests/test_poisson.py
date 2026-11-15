"""The NumPy ``lsqr`` Poisson solve against the exact solution of the SciPy system."""

import json
from pathlib import Path

import numpy as np

from ezsynth.kernels.imageops.opencv import poisson_lsq

FIXTURE = Path(__file__).resolve().parents[2] / "web/test/fixtures-pass-blend.json"


def test_poisson_lsq_matches_exact_least_squares_solution():
    # ``u`` was solved exactly (spsolve on the normal equations of construct_A_cpu).
    fx = json.loads(FIXTURE.read_text())["solver"]
    h, w = fx["h"], fx["w"]
    u = poisson_lsq(
        np.array(fx["gx"]).reshape(h, w),
        np.array(fx["gy"]).reshape(h, w),
        np.array(fx["c"]).reshape(h, w),
        fx["k"],
    )
    np.testing.assert_allclose(u.ravel(), fx["u"], atol=1e-5)
