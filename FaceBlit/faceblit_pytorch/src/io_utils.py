from pathlib import Path
from typing import Tuple

__all__ = ["read_landmarks_file"]


def read_landmarks_file(path: Path | str) -> list[Tuple[int, int]]:
    text = Path(path).read_text().strip().splitlines()
    pts: list[Tuple[int, int]] = []
    start_idx = 1 if text and text[0].strip() == "68" else 0
    for line in text[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            pts.append((int(float(parts[0])), int(float(parts[1]))))
    return pts
