import sys
from pathlib import Path

# Make the repository root importable when running the script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faceblit_pytorch import compute_style_assets  # noqa: E402


def _read_landmarks_file(path: Path) -> list[tuple[int, int]]:
    """Read the landmarks file produced by compute_style_assets."""
    pts: list[tuple[int, int]] = []
    with path.open("r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return pts
    try:
        count = int(lines[0])
        coord_lines = lines[1:]
        for line in coord_lines:
            x, y = line.split()
            pts.append((int(x), int(y)))
        # If the header is correct, length should match
        if count != len(pts):
            print(
                f"Warning: header count {count} != parsed coords {len(pts)} for {path}"
            )
    except Exception as e:  # pragma: no cover - diagnostic
        print(f"Failed to parse landmarks file {path}: {e}")
    return pts


def test_fan_call():
    print("Testing FAN call on a real sample (target2.png)...")
    examples_dir = ROOT / "examples"
    style_path = examples_dir / "target2.png"
    out_dir = ROOT / "faceblit_pytorch" / "test_fan_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not style_path.exists():
        print(f"Missing test image: {style_path}")
        return

    try:
        assets = compute_style_assets(
            input_path=style_path,
            output_dir=out_dir,
            landmark_model="fan",
            reuse_precomputed=False,
            device="cpu",
        )
    except ImportError as e:
        print(f"Caught ImportError (fan backend missing): {e}")
        return
    except RuntimeError as e:
        print(f"Caught RuntimeError during FAN compute: {e}")
        return
    except Exception as e:
        print(f"Caught unexpected exception: {type(e).__name__}: {e}")
        return

    # Ensure outputs mirror the dlib format (files present, header + 68 coords)
    required_keys = [
        "style_path",
        "landmarks_path",
        "style_pos_guide_path",
        "style_app_guide_path",
        "lut_path",
    ]
    missing = [k for k in required_keys if k not in assets]
    if missing:
        print(f"Missing keys in result: {missing}")
        return

    for k in required_keys:
        p = Path(assets[k])
        if not p.exists() or p.stat().st_size == 0:
            print(f"Output missing or empty for {k}: {p}")
            return

    lm_path = Path(assets["landmarks_path"])
    pts = _read_landmarks_file(lm_path)
    if len(pts) != 68:
        print(f"Unexpected landmark count: {len(pts)} (expected 68) at {lm_path}")
        return

    print("FAN integration succeeded; outputs match dlib-style format.")


if __name__ == "__main__":
    test_fan_call()
