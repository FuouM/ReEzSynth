# Parity fixtures for web/test/fixtures-ebsynth-video.json, computed by ezsynth's own code:
#   uv run --no-project --with numpy --with opencv-python-headless python web/scripts/gen-parity-fixtures.py . web/test/fixtures-ebsynth-video.json
# (functions are extracted from the source files so torch is not needed).
import ast, json, sys, pathlib
import numpy as np, cv2
root = pathlib.Path(sys.argv[1])
def extract(path, names):
    src = (root / path).read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if getattr(n, "name", None) in names]
    return "\n\n".join(ast.get_source_segment(src, n) for n in keep)
ns = {"np": np, "cv2": cv2}
exec(extract("ezsynth/edge/ops.py", {"create_gaussian_kernel"}), ns)
exec(extract("ezsynth/edge/classic.py", {"compute_classic_edge"}), ns)
exec("from __future__ import annotations\n" + extract("ezsynth/utils/warp_utils.py", {"_is_identity_flow", "_as_float32_array", "Warp", "PositionalGuide"}), ns)

rng = np.random.default_rng(7)
h, w = 9, 13
rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
bgr = rgb[..., ::-1].copy()
edge = ns["compute_classic_edge"](bgr)           # (h, w) uint8
flow = rng.uniform(-3, 3, (h, w, 2)).astype(np.float32)
warp = ns["Warp"](h, w)
warped_rgb = warp.run_warping(rgb, -flow)         # uint8 path, directed flow
pos = ns["PositionalGuide"](h, w)
pristine = pos.get_pristine_guide_uint8()
pos_from_flow = pos.create_from_flow(flow)
json.dump({
    "width": w, "height": h,
    "rgb": rgb.ravel().tolist(),
    "edge": edge.ravel().tolist(),
    "flow": flow.ravel().tolist(),
    "warpedRgbNegFlow": warped_rgb.ravel().tolist(),
    "pristine": pristine.ravel().tolist(),
    "posFromFlow": pos_from_flow.ravel().tolist(),
}, open(sys.argv[2], "w"))
print("ok", edge.shape, warped_rgb.shape, pos_from_flow.shape)
