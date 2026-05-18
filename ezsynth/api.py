import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .config import (
    BlendingConfig,
    DebugConfig,
    EbsynthParamsConfig,
    PipelineConfig,
    PrecomputationConfig,
    ProjectConfig,
)
from .data import ProjectData
from .engines.synthesis_engine import EbsynthEngine
from .guide import GuideObject
from .output import OutputManager
from .pipeline import SynthesisPipeline
from .service import SynthesisConfigs


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
        use_temporal_nnf_propagation: bool = True,
        use_sparse_feature_guide: bool = True,
        use_residual_transfer: bool = True,
        cost_function: str = "ssd",
        device: str = None,
        use_optimization: bool = True,
        use_bilateral: bool = False,
        sigma_spatial: float = 4.0,
        sigma_color: float = 10.0,
        n_size_step: int = 1,
        use_taichi_ops: bool = False,
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
        self.use_temporal_nnf_propagation = use_temporal_nnf_propagation
        self.use_sparse_feature_guide = use_sparse_feature_guide
        self.use_residual_transfer = use_residual_transfer
        self.cost_function = cost_function
        self.device = device
        self.use_optimization = use_optimization
        self.use_bilateral = use_bilateral
        self.sigma_spatial = sigma_spatial
        self.sigma_color = sigma_color
        self.n_size_step = n_size_step
        self.use_taichi_ops = use_taichi_ops


class Ezsynth:
    """
    High-level path-based video synthesis: builds ``SynthesisConfigs`` from simple
    directory arguments and runs ``SynthesisPipeline`` (same core as the CLI).
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

        # --- 1. Translate simple args into structured config sections ---
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
            use_residual_transfer=config.use_residual_transfer,
            use_temporal_nnf_propagation=config.use_temporal_nnf_propagation,
            use_sparse_feature_guide=config.use_sparse_feature_guide,
        )

        blending_cfg = BlendingConfig(
            use_lsqr=config.use_lsqr,
            poisson_maxiter=config.poisson_maxiter,
            use_taichi_ops=config.use_taichi_ops,
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

        self.configs = SynthesisConfigs(
            project=project_cfg,
            precomputation=precomputation_cfg,
            pipeline=pipeline_cfg,
            blending=blending_cfg,
            ebsynth_params=ebsynth_params_cfg,
            debug=DebugConfig(),  # Use default debug settings
        )

        # --- 2. Initialize the core pipeline components ---
        self.data = ProjectData.from_config(self.configs.project)
        self.pipeline = SynthesisPipeline(
            ebsynth_params_cfg=self.configs.ebsynth_params,
            pipeline_cfg=self.configs.pipeline,
            project_cfg=self.configs.project,
            precomputation_cfg=self.configs.precomputation,
            blending_cfg=self.configs.blending,
            data=self.data,
            debug_cfg=self.configs.debug,
        )

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
            OutputManager(self.data).handle(
                final_frames,
                save=True,
                visible_output_dir=self.output_dir_path,
            )
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


class ImageSynth:
    """
    Single-image stylization using ``EbsynthEngine`` with ``GuideObject`` inputs.
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
        self.style = GuideObject.load_guide(style_image, style_image).keyframe

        ebsynth_params_cfg = EbsynthParamsConfig(
            uniformity=config.uniformity,
            patch_size=config.patch_size,
            search_vote_iters=config.search_vote_iters,
            patch_match_iters=config.patch_match_iters,
            backend=config.backend,
            extra_pass_3x3=config.extra_pass_3x3,
            cost_function=config.cost_function,
            device=config.device,
            use_optimization=config.use_optimization,
            use_bilateral=config.use_bilateral,
            sigma_spatial=config.sigma_spatial,
            sigma_color=config.sigma_color,
            n_size_step=config.n_size_step,
            # Weights are now passed directly to run()
        )

        pipeline_cfg = PipelineConfig(
            pyramid_levels=config.pyramid_levels,
            use_residual_transfer=config.use_residual_transfer,
            use_temporal_nnf_propagation=config.use_temporal_nnf_propagation,
            use_sparse_feature_guide=config.use_sparse_feature_guide,
        )

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

        processed_guides = [
            GuideObject.load_guide(src, tgt, weight) for src, tgt, weight in guides
        ]

        stylized_image, error_map = self.engine.run(
            self.style, guides=processed_guides, benchmark=benchmark
        )

        return stylized_image, error_map
