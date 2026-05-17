from pathlib import Path

import numpy as np

from ezsynth.utils.pipeline_utils import has_exact_cache_files, save_flow_cache


def test_cache_validation_requires_exact_numbered_files(tmp_path: Path):
    cache_dir = tmp_path / "flow"
    save_flow_cache([np.zeros((2, 2, 2), dtype=np.float32) for _ in range(2)], cache_dir)

    assert has_exact_cache_files(cache_dir, "npy", 2)

    np.save(cache_dir / "extra.npy", np.zeros((1,), dtype=np.float32))
    assert not has_exact_cache_files(cache_dir, "npy", 2)


def test_save_flow_cache_clears_stale_files(tmp_path: Path):
    cache_dir = tmp_path / "flow"
    save_flow_cache([np.zeros((2, 2, 2), dtype=np.float32) for _ in range(3)], cache_dir)
    save_flow_cache([np.zeros((2, 2, 2), dtype=np.float32)], cache_dir)

    assert has_exact_cache_files(cache_dir, "npy", 1)
    assert sorted(path.name for path in cache_dir.glob("*.npy")) == ["00000.npy"]
