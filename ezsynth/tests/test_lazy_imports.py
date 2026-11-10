"""``import ezsynth.api`` must not pull in heavy optional dependencies."""

import json
import subprocess
import sys
from pathlib import Path

HEAVY_MODULES = ("phycv", "torchvision", "scipy", "taichi")
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_api_skips_heavy_modules():
    code = (
        "import json, sys\n"
        "import ezsynth.api\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in heavy)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == []
