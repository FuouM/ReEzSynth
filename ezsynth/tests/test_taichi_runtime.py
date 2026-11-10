"""Shared Taichi runtime: one init per process, arch switch on explicit device."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("taichi")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a subprocess so the CPU runtime does not leak into other tests.
_SCRIPT = """
import taichi as ti
from ezsynth.utils import taichi_runtime as rt
from ezsynth.engines.backends import taichi_backend as ez
from FastBlend.src.backends import taichi_backend as fb

assert rt.get_ti_arch("cpu") == ti.cpu
assert rt.get_ti_arch("cuda:0") in (ti.cuda, ti.cpu)
assert ez.ensure_ti_init is rt.ensure_ti_init
assert fb.ensure_ti_init is rt.ensure_ti_init

assert rt.ensure_ti_init("cpu") == ti.cpu
assert rt.ensure_ti_init() == ti.cpu  # None keeps the active runtime
assert fb.ensure_ti_init() == ti.cpu
assert rt.get_taichi_torch_device() == "cpu"
print("ok")
"""


def test_shared_runtime_cpu():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
