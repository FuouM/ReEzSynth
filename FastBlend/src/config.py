"""
Standalone FastBlend configuration.

FastBlend uses its own CUDA extension (``ezsynth/kernels/fastblend/cuda_src/``) for
patch costs and remap when the ``cuda`` backend is selected. That is separate from the
``ezsynth/ebsynth_extension`` / ``ebsynth_torch`` stack used by ``ezsynth`` for
full synthesis (PatchMatch + voting). Naming here avoids ``ebsynth_*`` to prevent
confusion with ``ezsynth``'s synthesis settings (``EbsynthParamsConfig``).
"""

from dataclasses import dataclass, replace


@dataclass
class FastBlendConfig:
    """Configuration for FastBlend post-processing."""

    # Core settings
    enabled: bool = True
    accuracy: int = 2  # 1=Fast, 2=Balanced, 3=Accurate
    window_size: int = 15
    batch_size: int = 16

    # Patch matching settings
    minimum_patch_size: int = 5
    num_iter: int = 5
    guide_weight: float = 10.0

    # Engine settings
    backend: str = "auto"  # "auto", "cuda", "cupy", "taichi", "wgpu"
    gpu_id: int = 0

    # Advanced settings
    initialize: str = "identity"  # initialization method
    tracking_window_size: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.accuracy not in [1, 2, 3]:
            raise ValueError("accuracy must be 1, 2, or 3")

        if self.backend not in ["auto", "cuda", "cupy", "taichi", "wgpu"]:
            raise ValueError(
                "backend must be 'auto', 'cuda', 'cupy', 'taichi', or 'wgpu'"
            )

        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")

        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.minimum_patch_size < 1:
            raise ValueError("minimum_patch_size must be >= 1")

        if self.num_iter < 1:
            raise ValueError("num_iter must be >= 1")

        if self.guide_weight < 0:
            raise ValueError("guide_weight must be >= 0")

    def get_pyramid_patch_matcher_config(self) -> dict:
        """Keyword settings for ``ezsynth.kernels.fastblend.build_pyramid_matcher``."""
        return {
            "minimum_patch_size": self.minimum_patch_size,
            "threads_per_block": 8,
            "num_iter": self.num_iter,
            "gpu_id": self.gpu_id,
            "guide_weight": self.guide_weight,
            "initialize": self.initialize,
            "tracking_window_size": self.tracking_window_size,
            "backend": self.backend,
        }


# Default configurations for different accuracy modes
FAST_CONFIG = FastBlendConfig(
    accuracy=1,
    window_size=5,
    batch_size=32,
    minimum_patch_size=7,
    num_iter=3,
    guide_weight=5.0,
)

BALANCED_CONFIG = FastBlendConfig(
    accuracy=2,
    window_size=15,
    batch_size=16,
    minimum_patch_size=5,
    num_iter=5,
    guide_weight=10.0,
)

ACCURATE_CONFIG = FastBlendConfig(
    accuracy=3,
    window_size=25,
    batch_size=8,
    minimum_patch_size=3,
    num_iter=8,
    guide_weight=15.0,
)


def get_default_config(accuracy: int = 2) -> FastBlendConfig:
    """Get a default configuration for the specified accuracy level."""
    if accuracy == 1:
        return replace(FAST_CONFIG)
    elif accuracy == 2:
        return replace(BALANCED_CONFIG)
    elif accuracy == 3:
        return replace(ACCURATE_CONFIG)
    else:
        raise ValueError("accuracy must be 1, 2, or 3")
