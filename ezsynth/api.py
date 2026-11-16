import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    MainConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .data import ProjectData
from .engines.synthesis_engine import EbsynthEngine
from .guide import GuideObject, load_guide
from .pipeline import SynthesisPipeline

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """
    A configuration class similar to the one in the old Ezsynth,
    bundling common synthesis parameters for ease of use.
    """

    # Ebsynth gen params
    uniformity: float = 3500.0
    patch_size: int = 7
    pyramid_levels: int = 6
    search_vote_iters: int = 12
    patch_match_iters: int = 6
    backend: str = "wgpu"
    extra_pass_3x3: bool = False
    # Guide weights
    edge_weight: float = 1.0
    image_weight: float = 6.0
    pos_weight: float = 2.0
    warp_weight: float = 0.5
    sparse_anchor_weight: float = 50.0
    # Blending params
    poisson_solver: str = "lsqr"
    poisson_maxiter: int | None = None
    # Pipeline params
    alpha: float = 0.75
    use_temporal_nnf_propagation: bool = True
    use_sparse_feature_guide: bool = True
    use_residual_transfer: bool = True
    cost_function: str = "ssd"
    search_mode: str = "patchmatch"
    multicand_k: int = 3
    precision: str = "float32"
    device: str | None = None
    use_optimization: bool = True
    use_bilateral: bool = False
    sigma_spatial: float = 4.0
    sigma_color: float = 10.0
    n_size_step: int = 1
    use_taichi_ops: bool = False


class Ezsynth:
    """
    High-level path-based video synthesis: builds a ``MainConfig`` from simple
    directory arguments and runs ``SynthesisPipeline`` (same core as the CLI).
    """

    def __init__(
        self,
        content_dir: str,
        style_paths: list[str],
        style_indices: list[int],
        output_dir: str | None = None,
        cache_dir: str | None = None,
        mask_dir: str | None = None,
        modulation_dir: str | None = None,
        config: RunConfig | None = None,
        edge_method: str = "Classic",
        flow_engine: str = "RAFT",
        flow_model: str = "sintel",
    ):
        """
        Initializes the Ezsynth pipeline with all necessary data and configurations.

        Args:
            content_dir (str): Path to the directory with content frames.
            style_paths (List[str]): List of paths to the style images.
            style_indices (List[int]): List of frame indices to apply the styles to.
            output_dir (Optional[str]): Directory to save the final frames. If None, a temporary directory is used and frames are not saved.
            cache_dir (Optional[str]): Directory for caching computations. If None, a temporary directory is used.
            mask_dir (Optional[str]): Path to the directory with mask frames.
            modulation_dir (Optional[str]): Path to the directory with modulation frames.
            config (RunConfig): An object containing detailed synthesis parameters.
            edge_method (str): The edge detection algorithm to use ('Classic', 'PAGE', 'PST').
            flow_engine (str): The optical flow engine to use ('RAFT', 'NeuFlow').
            flow_model (str): The specific model for the chosen flow engine.
        """
        config = config if config is not None else RunConfig()
        self.output_dir_path = Path(output_dir) if output_dir else None
        # Temporary directories are removed when this object is garbage collected.
        self._temp_output_dir = None
        self._temp_cache_dir = None

        if self.output_dir_path is None:
            self._temp_output_dir = tempfile.TemporaryDirectory()
            output_dir = self._temp_output_dir.name

        if cache_dir is None:
            self._temp_cache_dir = tempfile.TemporaryDirectory()
            cache_dir = self._temp_cache_dir.name

        self.configs = MainConfig(
            project=ProjectConfig(
                name=Path(content_dir).name,
                content_dir=content_dir,
                style_path=style_paths,
                style_indices=style_indices,
                output_dir=output_dir,
                cache_dir=cache_dir,
                mask_dir=mask_dir,
                modulation_dir=modulation_dir,
            ),
            precomputation=PrecomputationConfig(
                flow_engine=flow_engine,
                flow_model=flow_model,
                edge_method=edge_method,
            ),
            pipeline=PipelineConfig(
                pyramid_levels=config.pyramid_levels,
                alpha=config.alpha,
                use_residual_transfer=config.use_residual_transfer,
                use_temporal_nnf_propagation=config.use_temporal_nnf_propagation,
                use_sparse_feature_guide=config.use_sparse_feature_guide,
            ),
            blending=BlendingConfig(
                poisson_solver=config.poisson_solver,
                poisson_maxiter=config.poisson_maxiter,
                use_taichi_ops=config.use_taichi_ops,
            ),
            ebsynth_params=_ebsynth_params(config),
            debug=DebugConfig(),
        )

        self.data = ProjectData(self.configs.project)
        self.pipeline = SynthesisPipeline(self.configs, self.data)

        logger.info("\nEzsynth API initialized successfully.")

    def run(self) -> list[np.ndarray]:
        """
        Executes the synthesis pipeline.

        Returns:
            List[np.ndarray]: A list of the final stylized frames as NumPy arrays.
        """
        logger.info("\n--- Starting Synthesis via API ---")
        final_frames = self.pipeline.run()

        if self.output_dir_path:
            logger.info(
                f"\nSaving output to specified directory: {self.output_dir_path}"
            )
            self.data.save_output_frames(final_frames)
        else:
            logger.warning("\nOutput directory not specified, skipping save.")

        logger.info("\n--- Synthesis via API Finished ---")
        return final_frames


class ImageSynth:
    """
    Single-image stylization using ``EbsynthEngine`` with ``GuideObject`` inputs.
    """

    def __init__(
        self,
        style_image: str | np.ndarray,
        config: RunConfig | None = None,
    ):
        """
        Initializes the ImageSynth engine.

        Args:
            style_image (Union[str, np.ndarray]): Path to the style image or the image as a NumPy array.
            config (RunConfig): An object containing detailed synthesis parameters.
        """
        config = config if config is not None else RunConfig()
        self.style = load_guide(style_image, style_image).keyframe

        pipeline_cfg = PipelineConfig(
            pyramid_levels=config.pyramid_levels,
            use_residual_transfer=config.use_residual_transfer,
            use_temporal_nnf_propagation=config.use_temporal_nnf_propagation,
            use_sparse_feature_guide=config.use_sparse_feature_guide,
        )

        self.engine = EbsynthEngine(
            ebsynth_config=_ebsynth_params(config), pipeline_config=pipeline_cfg
        )
        logger.info("\nImageSynth API initialized successfully.")

    def run(
        self,
        guides: list[GuideObject | tuple[Any, Any, float]],
        benchmark: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the synthesis for a single target image using the provided guides.

        Args:
            guides: ``GuideObject`` instances (see :func:`load_guide`) or
                ``(source_guide, target_guide, weight)`` tuples whose images are
                file paths or uint8 BGR NumPy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the stylized image
                                           and the final error map.
        """
        if not guides:
            raise ValueError("At least one guide must be provided to the run() method.")

        processed_guides = [_as_guide(guide) for guide in guides]

        return self.engine.run(self.style, guides=processed_guides, benchmark=benchmark)


def _as_guide(guide: GuideObject | tuple[Any, Any, float]) -> GuideObject:
    if isinstance(guide, GuideObject):
        return guide
    if isinstance(guide, tuple | list) and len(guide) == 3:
        return load_guide(*guide)
    raise TypeError(
        "Each guide must be a GuideObject or a (source, target, weight) tuple, "
        f"got {type(guide).__name__}."
    )


def _ebsynth_params(config: RunConfig) -> EbsynthParamsConfig:
    return EbsynthParamsConfig(
        uniformity=config.uniformity,
        patch_size=config.patch_size,
        search_vote_iters=config.search_vote_iters,
        patch_match_iters=config.patch_match_iters,
        backend=config.backend,
        device=config.device,
        extra_pass_3x3=config.extra_pass_3x3,
        cost_function=config.cost_function,
        search_mode=config.search_mode,
        multicand_k=config.multicand_k,
        precision=config.precision,
        use_optimization=config.use_optimization,
        use_bilateral=config.use_bilateral,
        sigma_spatial=config.sigma_spatial,
        sigma_color=config.sigma_color,
        n_size_step=config.n_size_step,
        edge_weight=config.edge_weight,
        image_weight=config.image_weight,
        pos_weight=config.pos_weight,
        warp_weight=config.warp_weight,
        sparse_anchor_weight=config.sparse_anchor_weight,
    )
