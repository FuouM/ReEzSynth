"""Parity fixtures for web/test, computed by ezsynth's own code.

    uv run --no-project --with numpy --with opencv-python-headless \\
        --with scipy python web/scripts/gen-parity-fixtures.py . \\
        web/test/fixtures-ebsynth-video.json web/test/fixtures-pass-blend.json

The functions are extracted from the source files, so torch is not needed.
"""

import ast
import json
import pathlib
import sys

import cv2
import numpy as np
import scipy.sparse.linalg


def extract(root: pathlib.Path, path: str, names: set[str]) -> str:
    src = (root / path).read_text()
    nodes = [n for n in ast.parse(src).body if getattr(n, "name", None) in names]
    return "\n\n".join(ast.get_source_segment(src, n) or "" for n in nodes)


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    ns: dict = {"np": np, "cv2": cv2}
    exec(extract(root, "ezsynth/edge/ops.py", {"create_gaussian_kernel"}), ns)
    exec(extract(root, "ezsynth/edge/classic.py", {"compute_classic_edge"}), ns)
    warp_names = {"_is_identity_flow", "_as_float32_array", "Warp", "PositionalGuide"}
    exec(
        "from __future__ import annotations\n"
        + extract(root, "ezsynth/utils/warp_utils.py", warp_names),
        ns,
    )

    rng = np.random.default_rng(7)
    h, w = 9, 13
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    edge = ns["compute_classic_edge"](rgb[..., ::-1].copy())  # ezsynth frames are BGR
    flow = rng.uniform(-3, 3, (h, w, 2)).astype(np.float32)
    warped_rgb = ns["Warp"](h, w).run_warping(rgb, -flow)
    pos = ns["PositionalGuide"](h, w)
    fixture = {
        "width": w,
        "height": h,
        "rgb": rgb.ravel().tolist(),
        "edge": edge.ravel().tolist(),
        "flow": flow.ravel().tolist(),
        "warpedRgbNegFlow": warped_rgb.ravel().tolist(),
        "pristine": pos.get_pristine_guide_uint8().ravel().tolist(),
        "posFromFlow": pos.create_from_flow(flow).ravel().tolist(),
    }
    with open(sys.argv[2], "w") as f:
        json.dump(fixture, f)

    # Pass blending (blend_logic.hist_blender + poisson_fusion_cpu), on smooth images so
    # gradients are meaningful.
    ns2: dict = {"np": np, "cv2": cv2, "time": __import__("time"), "logger": _Log()}
    blend_names = {"hist_blender", "construct_A_cpu", "poisson_fusion_cpu"}
    exec(extract(root, "ezsynth/utils/blend_logic.py", blend_names), ns2)
    ns2["TAICHI_AVAILABLE"] = False
    bh, bw = 12, 16
    yy, xx = np.mgrid[0:bh, 0:bw]
    a = np.stack([xx * 12, yy * 15, (xx + yy) * 6], -1).clip(0, 255).astype(np.uint8)
    b = np.stack([200 - xx * 5, 60 + yy * 8, 255 - (xx * yy) % 200], -1).astype(
        np.uint8
    )
    a = (a.astype(int) + rng.integers(-8, 9, a.shape)).clip(0, 255).astype(np.uint8)
    mask = ((xx + yy) % 7 < 3).astype(np.uint8)
    bgr = lambda x: x[..., ::-1].copy()
    hist = ns2["hist_blender"](bgr(a), bgr(b), mask)
    cache = {"As": ns2["construct_A_cpu"](bh, bw, [2.5, 0.5, 0.5])}
    fused = ns2["poisson_fusion_cpu"](
        hist, bgr(a), bgr(b), mask, cache, "lsqr", None, [2.5, 0.5, 0.5]
    )
    # The least-squares solve alone, solved exactly (lsqr converges to the same answer).
    sh, sw, k = 7, 9, 2.5
    gx = rng.normal(0, 10, sh * sw).astype(np.float32)
    gy = rng.normal(0, 10, sh * sw).astype(np.float32)
    c = rng.normal(0, 30, sh * sw).astype(np.float32)
    big_a = ns2["construct_A_cpu"](sh, sw, [k])[0]
    rhs = big_a.T @ np.concatenate([gx * k, gy * k, c])
    u = scipy.sparse.linalg.spsolve((big_a.T @ big_a).tocsc(), rhs)
    # cv2's 8-bit Lab conversions on a ramp and random colours.
    ramp = np.stack([np.arange(256)] * 3, -1)[None].astype(np.uint8)
    rnd = rng.integers(0, 256, (1, 512, 3), dtype=np.uint8)
    lab_rgb = np.concatenate([ramp, rnd], axis=1)
    lab_in = rng.integers(0, 256, (1, 512, 3), dtype=np.uint8)
    blend = {
        "solver": {
            "h": sh,
            "w": sw,
            "k": k,
            "gx": gx.tolist(),
            "gy": gy.tolist(),
            "c": c.tolist(),
            "u": u.tolist(),
        },
        "labRgb": lab_rgb.ravel().tolist(),
        "lab": cv2.cvtColor(bgr(lab_rgb), cv2.COLOR_BGR2Lab).ravel().tolist(),
        "labIn": lab_in.ravel().tolist(),
        "labBack": cv2.cvtColor(lab_in, cv2.COLOR_Lab2BGR)[..., ::-1].ravel().tolist(),
        "width": bw,
        "height": bh,
        "a": a.ravel().tolist(),
        "b": b.ravel().tolist(),
        "mask": mask.ravel().tolist(),
        "hist": hist[..., ::-1].ravel().tolist(),
        "fused": fused[..., ::-1].ravel().tolist(),
    }
    with open(sys.argv[3], "w") as f:
        json.dump(blend, f)


class _Log:
    def debug(self, *args, **kwargs) -> None:
        pass


if __name__ == "__main__":
    main()
