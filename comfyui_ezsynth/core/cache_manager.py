# comfyui_ezsynth/core/cache_manager.py
"""
Caching utilities for ComfyUI nodes.
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class CacheManager:
    """
    Manages caching of expensive computations for reuse across node executions.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Base directory for cache. If None, uses system temp dir.
        """
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="ezsynth_cache_")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different cache types
        self.flow_cache = self.cache_dir / "flow"
        self.edge_cache = self.cache_dir / "edge"
        self.sparse_cache = self.cache_dir / "sparse"
        self.nnf_cache = self.cache_dir / "nnf"
        self.faceblit_cache = self.cache_dir / "faceblit"

        for subdir in [
            self.flow_cache,
            self.edge_cache,
            self.sparse_cache,
            self.nnf_cache,
            self.faceblit_cache,
        ]:
            subdir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from input parameters.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hexadecimal cache key string
        """
        import json

        key_data = []

        for arg in args:
            if isinstance(arg, (list, tuple)):
                if len(arg) > 0 and isinstance(arg[0], (int, float, str)):
                    key_data.append(str(tuple(arg)))
                else:
                    key_data.append(str(len(arg)))
            elif isinstance(arg, dict):
                key_data.append(json.dumps(arg, sort_keys=True))
            else:
                key_data.append(str(arg))

        for k, v in sorted(kwargs.items()):
            key_data.append(f"{k}={v}")

        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_flow_path(self, frame_hash: str) -> Path:
        """Get cache path for optical flow."""
        return self.flow_cache / f"{frame_hash}.npy"

    def get_edge_path(self, frame_hash: str, method: str) -> Path:
        """Get cache path for edge maps."""
        return self.edge_cache / f"{frame_hash}_{method}.npy"

    def get_sparse_path(self, frame_hash: str) -> Path:
        """Get cache path for sparse features."""
        return self.sparse_cache / f"{frame_hash}.npy"

    def get_nnf_path(self, style_hash: str, frame_idx: str) -> Path:
        """Get cache path for NNF."""
        return self.nnf_cache / f"{style_hash}_{frame_idx}.npy"

    def get_faceblit_path(self, image_hash: str, asset_type: str) -> Path:
        """Get cache path for FaceBlit assets."""
        return self.faceblit_cache / f"{image_hash}_{asset_type}"

    def load_cached_flow(self, frame_hash: str) -> Optional[Any]:
        """Load cached optical flow."""
        path = self.get_flow_path(frame_hash)
        if path.exists():
            import numpy as np

            return np.load(str(path))
        return None

    def save_cached_flow(self, frame_hash: str, flow: Any) -> None:
        """Save optical flow to cache."""
        import numpy as np

        path = self.get_flow_path(frame_hash)
        np.save(str(path), flow)

    def load_cached_edge(self, frame_hash: str, method: str) -> Optional[Any]:
        """Load cached edge map."""
        path = self.get_edge_path(frame_hash, method)
        if path.exists():
            import numpy as np

            return np.load(str(path))
        return None

    def save_cached_edge(self, frame_hash: str, method: str, edge: Any) -> None:
        """Save edge map to cache."""
        import numpy as np

        path = self.get_edge_path(frame_hash, method)
        np.save(str(path), edge)

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache directories.

        Args:
            cache_type: Type of cache to clear (None for all)
        """
        if cache_type is None:
            for subdir in [
                self.flow_cache,
                self.edge_cache,
                self.sparse_cache,
                self.nnf_cache,
                self.faceblit_cache,
            ]:
                if subdir.exists():
                    for f in subdir.glob("*"):
                        if f.is_file():
                            f.unlink()
        elif cache_type == "flow" and self.flow_cache.exists():
            for f in self.flow_cache.glob("*.npy"):
                f.unlink()
        elif cache_type == "edge" and self.edge_cache.exists():
            for f in self.edge_cache.glob("*.npy"):
                f.unlink()
        elif cache_type == "sparse" and self.sparse_cache.exists():
            for f in self.sparse_cache.glob("*.npy"):
                f.unlink()
        elif cache_type == "nnf" and self.nnf_cache.exists():
            for f in self.nnf_cache.glob("*.npy"):
                f.unlink()
        elif cache_type == "faceblit" and self.faceblit_cache.exists():
            for f in self.faceblit_cache.glob("*"):
                if f.is_file():
                    f.unlink()

    def get_cache_size(self) -> Dict[str, int]:
        """
        Get sizes of cache directories.

        Returns:
            Dictionary mapping cache type to size in bytes
        """
        sizes = {}
        for name, path in [
            ("flow", self.flow_cache),
            ("edge", self.edge_cache),
            ("sparse", self.sparse_cache),
            ("nnf", self.nnf_cache),
            ("faceblit", self.faceblit_cache),
        ]:
            if path.exists():
                total = sum(f.stat().st_size for f in path.glob("*") if f.is_file())
                sizes[name] = total
            else:
                sizes[name] = 0
        return sizes


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: Optional[str] = None) -> CacheManager:
    """
    Get global cache manager instance.

    Args:
        cache_dir: Optional cache directory override

    Returns:
        CacheManager instance
    """
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(cache_dir)
    return _global_cache_manager


def clear_global_cache(cache_type: Optional[str] = None) -> None:
    """Clear global cache."""
    manager = get_cache_manager()
    manager.clear_cache(cache_type)
