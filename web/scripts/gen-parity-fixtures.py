"""Parity fixtures for web/test/fixtures-ebsynth-video.json, computed by ezsynth's own code.

    uv run --no-project --with numpy --with opencv-python-headless \\
        python web/scripts/gen-parity-fixtures.py . web/test/fixtures-ebsynth-video.json

The functions are extracted from the source files, so torch is not needed.
"""

import ast
import json
import pathlib
import sys

import cv2
import numpy as np


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


if __name__ == "__main__":
    main()
