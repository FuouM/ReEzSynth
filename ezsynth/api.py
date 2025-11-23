import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    FinalPassConfig,
    MainConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .data import ProjectData
from .engines.synthesis_engine import EbsynthEngine
from .pipeline import SynthesisPipeline
from .utils import io_utils


class RunConfig:
    """
    A configuration class similar to the one in the old Ezsynth,
    bundling common synthesis parameters for ease of use.
    """

    def __init__(
        self,
        uniformity=3500.0,
        patch_size=7,
        pyramid_levels=6,
        search_vote_iters=12,
        patch_match_iters=6,
        backend="cuda",
        extra_pass_3x3=False,
        edge_weight=1.0,
        image_weight=6.0,
        pos_weight=2.0,
        warp_weight=0.5,
        sparse_anchor_weight=50.0,
        use_lsqr=True,
        poisson_maxiter: Optional[int] = None,
        alpha: float = 0.75,
        colorize: bool = True,
        use_temporal_nnf_propagation: bool = True,
        use_sparse_feature_guide: bool = True,
        final_pass_enabled: bool = False,
        final_pass_strength: float = 1.0,
    ):
        # Ebsynth gen params
        self.uniformity = uniformity
        self.patch_size = patch_size
        self.pyramid_levels = pyramid_levels
        self.search_vote_iters = search_vote_iters
        self.patch_match_iters = patch_match_iters
        self.backend = backend
        self.extra_pass_3x3 = extra_pass_3x3

        # Guide weights
        self.edge_weight = edge_weight
        self.image_weight = image_weight
        self.pos_weight = pos_weight
        self.warp_weight = warp_weight
        self.sparse_anchor_weight = sparse_anchor_weight

        # Blending params
        self.use_lsqr = use_lsqr
        self.poisson_maxiter = poisson_maxiter

        # Pipeline params
        self.alpha = alpha
        self.colorize = colorize
        self.use_temporal_nnf_propagation = use_temporal_nnf_propagation
        self.use_sparse_feature_guide = use_sparse_feature_guide
        self.final_pass_enabled = final_pass_enabled
        self.final_pass_strength = final_pass_strength


class Ezsynth:
    """
    A high-level API for ReEzSynth that mimics the interface of older versions,
    providing a simplified way to run the synthesis pipeline without managing config files.
    """

    def __init__(
        self,
        content_dir: str,
        style_paths: List[str],
        style_indices: List[int],
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        modulation_dir: Optional[str] = None,
        config: RunConfig = RunConfig(),
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
        self.output_dir_path = Path(output_dir) if output_dir else None
        self._temp_output_dir = None
        self._temp_cache_dir = None

        if self.output_dir_path is None:
            self._temp_output_dir = tempfile.TemporaryDirectory()
            output_dir = self._temp_output_dir.name

        if cache_dir is None:
            self._temp_cache_dir = tempfile.TemporaryDirectory()
            cache_dir = self._temp_cache_dir.name

        # --- 1. Translate simple args into the structured MainConfig ---
        project_cfg = ProjectConfig(
            name=Path(content_dir).name,
            content_dir=content_dir,
            style_path=style_paths,
            style_indices=style_indices,
            output_dir=output_dir,
            cache_dir=cache_dir,
            mask_dir=mask_dir,
            modulation_dir=modulation_dir,
        )

        precomputation_cfg = PrecomputationConfig(
            flow_engine=flow_engine,
            flow_model=flow_model,
            edge_method=edge_method,
        )

        pipeline_cfg = PipelineConfig(
            pyramid_levels=config.pyramid_levels,
            alpha=config.alpha,
            colorize=config.colorize,
            use_temporal_nnf_propagation=config.use_temporal_nnf_propagation,
            use_sparse_feature_guide=config.use_sparse_feature_guide,
            final_pass=FinalPassConfig(
                enabled=config.final_pass_enabled,
                strength=config.final_pass_strength,
            ),
        )

        blending_cfg = BlendingConfig(
            use_lsqr=config.use_lsqr, poisson_maxiter=config.poisson_maxiter
        )

        ebsynth_params_cfg = EbsynthParamsConfig(
            uniformity=config.uniformity,
            patch_size=config.patch_size,
            search_vote_iters=config.search_vote_iters,
            patch_match_iters=config.patch_match_iters,
            extra_pass_3x3=config.extra_pass_3x3,
            edge_weight=config.edge_weight,
            image_weight=config.image_weight,
            pos_weight=config.pos_weight,
            warp_weight=config.warp_weight,
            sparse_anchor_weight=config.sparse_anchor_weight,
        )

        self.main_config = MainConfig(
            project=project_cfg,
            precomputation=precomputation_cfg,
            pipeline=pipeline_cfg,
            blending=blending_cfg,
            ebsynth_params=ebsynth_params_cfg,
            debug=DebugConfig(),  # Use default debug settings
        )

        # --- 2. Initialize the core pipeline components ---
        self.data = ProjectData(self.main_config.project)
        self.pipeline = SynthesisPipeline(self.main_config, self.data)

        print("\nEzsynth API initialized successfully.")

    def run(self) -> List[np.ndarray]:
        """
        Executes the synthesis pipeline.

        Returns:
            List[np.ndarray]: A list of the final stylized frames as NumPy arrays.
        """
        print("\n--- Starting Synthesis via API ---")
        final_frames = self.pipeline.run()

        if self.output_dir_path:
            print(f"\nSaving output to specified directory: {self.output_dir_path}")
            self.data.save_output_frames(final_frames)
        else:
            print("\nOutput directory not specified, skipping save.")

        print("\n--- Synthesis via API Finished ---")
        return final_frames

    def __del__(self):
        # Clean up temporary directories when the object is garbage collected
        if self._temp_output_dir:
            self._temp_output_dir.cleanup()
        if self._temp_cache_dir:
            self._temp_cache_dir.cleanup()


# --- START OF NEW CONTENT ---


def load_guide(
    source: Union[str, np.ndarray],
    target: Union[str, np.ndarray],
    weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Helper function to load a guide pair from paths or use existing NumPy arrays.
    """
    src_img = io_utils.read_image(source) if isinstance(source, str) else source
    tgt_img = io_utils.read_image(target) if isinstance(target, str) else target
    return (src_img, tgt_img, weight)


class ImageSynth:
    """
    A high-level API for single-image synthesis, similar to the old Ezsynth.
    This class is a lightweight wrapper around the core EbsynthEngine.
    """

    def __init__(
        self,
        style_image: Union[str, np.ndarray],
        config: RunConfig = RunConfig(),
    ):
        """
        Initializes the ImageSynth engine.

        Args:
            style_image (Union[str, np.ndarray]): Path to the style image or the image as a NumPy array.
            config (RunConfig): An object containing detailed synthesis parameters.
        """
        self.style = load_guide(style_image, style_image)[0]
        # Use load_guide to handle path or array

        ebsynth_params_cfg = EbsynthParamsConfig(
            uniformity=config.uniformity,
            patch_size=config.patch_size,
            search_vote_iters=config.search_vote_iters,
            patch_match_iters=config.patch_match_iters,
            backend=config.backend,
            extra_pass_3x3=config.extra_pass_3x3,
            # Weights are now passed directly to run()
        )

        pipeline_cfg = PipelineConfig(pyramid_levels=config.pyramid_levels)

        self.engine = EbsynthEngine(
            ebsynth_config=ebsynth_params_cfg, pipeline_config=pipeline_cfg
        )
        print("\nImageSynth API initialized successfully.")

    def run(
        self, guides: List[Tuple[Any, Any, float]], benchmark: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the synthesis for a single target image using the provided guides.

        Args:
            guides (List[Tuple[Any, Any, float]]): A list of guide tuples.
                Each tuple should be (source_guide, target_guide, weight).
                The guides can be file paths or NumPy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the stylized image
                                           and the final error map.
        """
        if not guides:
            raise ValueError("At least one guide must be provided to the run() method.")

        processed_guides = [load_guide(src, tgt, weight) for src, tgt, weight in guides]

        # EbsynthEngine handles the synthesis directly
        stylized_image, error_map = self.engine.run(
            self.style, guides=processed_guides, benchmark=benchmark
        )

        return stylized_image, error_map


# --- END OF NEW CONTENT ---
