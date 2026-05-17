from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from .io_utils import write_image


def expected_cache_paths(cache_dir: Path, extension: str, num_expected: int) -> List[Path]:
    """Return the exact numbered cache files expected for a complete cache hit."""
    return [cache_dir / f"{i:05d}.{extension}" for i in range(num_expected)]


def has_exact_cache_files(cache_dir: Path, extension: str, num_expected: int) -> bool:
    """Return true only when cache files exactly match the expected numbered set."""
    if num_expected < 0 or not cache_dir.exists():
        return False

    expected = expected_cache_paths(cache_dir, extension, num_expected)
    actual = sorted(cache_dir.glob(f"*.{extension}"))
    return actual == expected and all(path.is_file() for path in expected)


def _clear_cache_files(cache_dir: Path, extensions: Iterable[str]) -> None:
    for extension in extensions:
        for path in cache_dir.glob(f"*.{extension}"):
            path.unlink()


def load_cached_flow(cache_dir: Path):
    print(f"Loading optical flow from cache: {cache_dir}")
    flow_paths = sorted(cache_dir.glob("*.npy"))
    fwd_flows = [np.load(p) for p in tqdm(flow_paths, desc="Loading Cached Flow")]
    return fwd_flows


def save_flow_cache(fwd_flows, cache_dir: Path):
    print(f"Saving {len(fwd_flows)} flow fields to cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _clear_cache_files(cache_dir, ["npy"])
    for i, flow in enumerate(tqdm(fwd_flows, desc="Saving Flow Cache")):
        np.save(cache_dir / f"{i:05d}.npy", flow)


def save_edge_map_cache(edge_maps, cache_dir: Path):
    print(f"Saving {len(edge_maps)} edge maps to cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _clear_cache_files(cache_dir, ["png"])
    for i, edge_map in enumerate(edge_maps):
        write_image(cache_dir / f"{i:05d}.png", edge_map)
