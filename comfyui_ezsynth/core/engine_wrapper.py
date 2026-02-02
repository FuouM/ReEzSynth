# comfyui_ezsynth/core/engine_wrapper.py
"""
Wrapper around EbsynthEngine that handles tensor conversions
and provides a clean interface for ComfyUI nodes.
"""

from typing import List, Optional, Tuple

import torch

from ..types import EbsynthParams


class EbsynthNodeEngine:
    """
    Wrapper around EbsynthEngine that handles tensor conversions
    and provides a clean interface for ComfyUI nodes.
    """

    def __init__(self, params: Optional[EbsynthParams] = None):
        """
        Initialize the engine wrapper.

        Args:
            params: Ebsynth parameters. If None, uses defaults.
        """
        self.params = params or EbsynthParams()
        self._engine = None
        self._config = None

    def _ensure_engine(self, style_shape: Tuple[int, int, int]):
        """
        Lazily initialize the EbsynthEngine.

        Args:
            style_shape: Shape of style image (H, W, C)
        """
        if self._engine is not None:
            return

        from ezsynth.config import (
            EbsynthParamsConfig,
            FinalPassConfig,
            PipelineConfig,
        )
        from ezsynth.engines.synthesis_engine import EbsynthEngine

        self._config = {
            "ebsynth": EbsynthParamsConfig(
                uniformity=self.params.uniformity,
                patch_size=self.params.patch_size,
                search_vote_iters=self.params.search_vote_iters,
                patch_match_iters=self.params.patch_match_iters,
                extra_pass_3x3=self.params.extra_pass_3x3,
                edge_weight=self.params.edge_weight,
                image_weight=self.params.image_weight,
                pos_weight=self.params.pos_weight,
                warp_weight=self.params.warp_weight,
                sparse_anchor_weight=self.params.sparse_anchor_weight,
                backend=self.params.backend,
                cost_function=self.params.cost_function,
            ),
            "pipeline": PipelineConfig(
                pyramid_levels=self.params.pyramid_levels,
                use_residual_transfer=True,
                alpha=0.75,
                colorize=True,
                use_temporal_nnf_propagation=False,
                use_sparse_feature_guide=False,
                final_pass=FinalPassConfig(
                    enabled=False,
                    strength=1.0,
                ),
            ),
        }

        self._engine = EbsynthEngine(
            ebsynth_config=self._config["ebsynth"],
            pipeline_config=self._config["pipeline"],
        )

    def synthesize(
        self,
        style: torch.Tensor,
        guides: List[Tuple[torch.Tensor, torch.Tensor, float]],
        initial_nnf: Optional[torch.Tensor] = None,
        output_nnf: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Run synthesis on a single frame.

        Args:
            style: Style image tensor (H, W, C)
            guides: List of (source_guide, target_guide, weight) tuples
            initial_nnf: Optional initial NNF for temporal propagation
            output_nnf: Whether to output the final NNF

        Returns:
            Tuple of (stylized_image, error_map, optional_nnf)
        """
        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        self._ensure_engine(style.shape)

        style_np = tensor_to_numpy(style)
        guides_np = [
            (tensor_to_numpy(src), tensor_to_numpy(tgt), weight)
            for src, tgt, weight in guides
        ]
        initial_nnf_np = (
            tensor_to_numpy(initial_nnf) if initial_nnf is not None else None
        )

        result = self._engine.run(
            style_img=style_np,
            guides=guides_np,
            initial_nnf=initial_nnf_np,
            output_nnf=output_nnf,
        )

        if output_nnf:
            stylized, error, nnf = result
            return (
                numpy_to_tensor(stylized),
                numpy_to_tensor(error),
                numpy_to_tensor(nnf),
            )
        else:
            stylized, error = result
            return (
                numpy_to_tensor(stylized),
                numpy_to_tensor(error),
                None,
            )

    def get_device(self) -> str:
        """Get the device being used by the engine."""
        if self._engine is not None:
            return str(self._engine.device)
        return "unknown"

    def cleanup(self) -> None:
        """Clean up engine resources."""
        if self._engine is not None:
            del self._engine
            self._engine = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageSynthNodeEngine:
    """
    Wrapper around ImageSynth for simple single-image synthesis.
    """

    def __init__(self, params: Optional[EbsynthParams] = None):
        """
        Initialize the ImageSynth wrapper.

        Args:
            params: Ebsynth parameters
        """
        self.params = params or EbsynthParams()
        self._synth = None

    def synthesize(
        self,
        style: torch.Tensor,
        guides: List[Tuple[torch.Tensor, torch.Tensor, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run simple synthesis (no NNF propagation).

        Args:
            style: Style image tensor
            guides: List of (source_guide, target_guide, weight) tuples

        Returns:
            Tuple of (stylized_image, error_map)
        """
        from ezsynth.api import ImageSynth, RunConfig

        from ..core.tensor_utils import numpy_to_tensor, tensor_to_numpy

        style_np = tensor_to_numpy(style)

        if self._synth is None:
            run_config = RunConfig(
                uniformity=self.params.uniformity,
                patch_size=self.params.patch_size,
                pyramid_levels=self.params.pyramid_levels,
                search_vote_iters=self.params.search_vote_iters,
                patch_match_iters=self.params.patch_match_iters,
                backend=self.params.backend,
                extra_pass_3x3=self.params.extra_pass_3x3,
                cost_function=self.params.cost_function,
            )
            self._synth = ImageSynth(style_image=style_np, config=run_config)

        guides_np = [
            (tensor_to_numpy(src), tensor_to_numpy(tgt), weight)
            for src, tgt, weight in guides
        ]

        stylized_np, error_np = self._synth.run(guides=guides_np)

        return numpy_to_tensor(stylized_np), numpy_to_tensor(error_np)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._synth = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
