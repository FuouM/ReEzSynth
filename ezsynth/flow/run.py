"""Dispatch optical flow by ``PrecomputationConfig``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List

import numpy as np
import torch

from ..config import PrecomputationConfig
from ..engines.flow_engine import NeuFlowEngine, RAFTFlowEngine, TorchVisionFlowEngine
from .opencv import compute_opencv_flow_sequence
from .types import FlowEngineName

FlowSequenceFn = Callable[[List[np.ndarray]], List[np.ndarray]]


@contextmanager
def optical_flow_engine(
    precomputation_cfg: PrecomputationConfig,
) -> Iterator[FlowSequenceFn]:
    """
    Load a flow engine once, then run many short frame sequences.

    This mirrors the refactor repo's session API while reusing the current
    repo's engine classes.
    """
    engine_name: FlowEngineName = precomputation_cfg.flow_engine
    print("Instantiating Flow Engine (session)...")
    engine = None
    try:
        if engine_name == "RAFT":
            engine = RAFTFlowEngine(
                model_name=precomputation_cfg.flow_model,
                arch=engine_name,
            )

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return engine.compute(frames)

            yield _compute
        elif engine_name == "NeuFlow":
            engine = NeuFlowEngine(model_name=precomputation_cfg.flow_model)

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return engine.compute(frames)

            yield _compute
        elif engine_name == "OpenCV":

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return compute_opencv_flow_sequence(
                    frames,
                    precomputation_cfg.opencv_flow_method,
                )

            yield _compute
        elif engine_name == "TorchVision":
            engine = TorchVisionFlowEngine(
                model_name=precomputation_cfg.torchvision_flow_model
            )

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return engine.compute(frames)

            yield _compute
        else:
            raise ValueError(f"Unknown flow engine: {engine_name!r}")
    finally:
        print("Optical flow session complete. Releasing model from memory...")
        if engine is not None:
            del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_optical_flow_sequence(
    frames: List[np.ndarray],
    precomputation_cfg: PrecomputationConfig,
) -> List[np.ndarray]:
    """Forward optical flow for a frame sequence; releases heavy models after use."""
    print("Instantiating Flow Engine...")
    engine_name: FlowEngineName = precomputation_cfg.flow_engine
    engine = None
    try:
        if engine_name == "RAFT":
            engine = RAFTFlowEngine(
                model_name=precomputation_cfg.flow_model,
                arch=engine_name,
            )
            return engine.compute(frames)
        if engine_name == "NeuFlow":
            engine = NeuFlowEngine(model_name=precomputation_cfg.flow_model)
            return engine.compute(frames)
        if engine_name == "OpenCV":
            return compute_opencv_flow_sequence(
                frames,
                precomputation_cfg.opencv_flow_method,
            )
        if engine_name == "TorchVision":
            engine = TorchVisionFlowEngine(
                model_name=precomputation_cfg.torchvision_flow_model
            )
            return engine.compute(frames)
        raise ValueError(f"Unknown flow engine: {engine_name!r}")
    finally:
        print("Optical flow computation complete. Releasing model from memory...")
        if engine is not None:
            del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_backward_optical_flow_sequence(
    frames: List[np.ndarray],
    precomputation_cfg: PrecomputationConfig,
) -> List[np.ndarray]:
    """Backward optical flow for each adjacent pair: frame ``i + 1`` -> frame ``i``."""
    if len(frames) < 2:
        return []
    reversed_flows = compute_optical_flow_sequence(
        list(reversed(frames)),
        precomputation_cfg,
    )
    return list(reversed(reversed_flows))


def compute_bidirectional_optical_flow_sequence(
    frames: List[np.ndarray],
    precomputation_cfg: PrecomputationConfig,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Forward and backward adjacent flow."""
    if len(frames) < 2:
        return [], []

    with optical_flow_engine(precomputation_cfg) as compute:
        fwd = compute(frames)
        bwd_reversed = compute(list(reversed(frames)))
    return fwd, list(reversed(bwd_reversed))
