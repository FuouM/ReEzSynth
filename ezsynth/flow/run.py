"""Dispatch optical flow by ``PrecomputationConfig``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List
from typing import cast

import numpy as np
import torch

from ..config import PrecomputationConfig
from .neuflow_sequence import (
    _compute_neuflow_sequence_with_model,
    _load_neuflow_model,
    compute_neuflow_bidirectional,
    compute_neuflow_sequence,
)
from .opencv import compute_opencv_flow_sequence
from .raft_custom import (
    _compute_custom_raft_sequence_with_model,
    _load_raft_model,
    compute_custom_raft_bidirectional,
    compute_custom_raft_sequence,
)
from .torchvision_flow import (
    _compute_torchvision_raft_sequence_with_model,
    _load_torchvision_raft,
    compute_torchvision_raft_sequence,
)
from .types import FlowEngineName, NeuFlowCheckpointName, RaftCheckpointName

FlowSequenceFn = Callable[[List[np.ndarray]], List[np.ndarray]]


@contextmanager
def optical_flow_engine(
    precomputation_cfg: PrecomputationConfig,
) -> Iterator[FlowSequenceFn]:
    """
    Load flow weights once, then run many short frame sequences.
    """
    engine_name: FlowEngineName = precomputation_cfg.flow_engine
    print("Instantiating Flow Engine (session)...")
    try:
        if engine_name == "RAFT":
            model, dev = _load_raft_model(
                cast(RaftCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return _compute_custom_raft_sequence_with_model(
                    frames,
                    model,
                    dev,
                    show_progress=False,
                )

            yield _compute
        elif engine_name == "NeuFlow":
            model, dev, dtype, use_amp = _load_neuflow_model(
                cast(NeuFlowCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return _compute_neuflow_sequence_with_model(
                    frames,
                    model,
                    dev,
                    dtype,
                    use_amp,
                    show_progress=False,
                )

            yield _compute
        elif engine_name == "OpenCV":

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return compute_opencv_flow_sequence(
                    frames,
                    precomputation_cfg.opencv_flow_method,
                )

            yield _compute
        elif engine_name == "TorchVision":
            model, dev, transforms = _load_torchvision_raft(
                precomputation_cfg.torchvision_flow_model,
                "auto",
            )

            def _compute(frames: List[np.ndarray]) -> List[np.ndarray]:
                return _compute_torchvision_raft_sequence_with_model(
                    frames,
                    model,
                    dev,
                    transforms,
                    show_progress=False,
                )

            yield _compute
        else:
            raise ValueError(f"Unknown flow engine: {engine_name!r}")
    finally:
        print("Optical flow session complete. Releasing model from memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_optical_flow_sequence(
    frames: List[np.ndarray],
    precomputation_cfg: PrecomputationConfig,
) -> List[np.ndarray]:
    """Forward optical flow for a frame sequence; releases heavy models after use."""
    print("Instantiating Flow Engine...")
    engine_name: FlowEngineName = precomputation_cfg.flow_engine
    try:
        if engine_name == "RAFT":
            return compute_custom_raft_sequence(
                frames,
                cast(RaftCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )
        if engine_name == "NeuFlow":
            return compute_neuflow_sequence(
                frames,
                cast(NeuFlowCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )
        if engine_name == "OpenCV":
            return compute_opencv_flow_sequence(
                frames,
                precomputation_cfg.opencv_flow_method,
            )
        if engine_name == "TorchVision":
            return compute_torchvision_raft_sequence(
                frames,
                precomputation_cfg.torchvision_flow_model,
                "auto",
            )
        raise ValueError(f"Unknown flow engine: {engine_name!r}")
    finally:
        print("Optical flow computation complete. Releasing model from memory...")
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

    engine_name: FlowEngineName = precomputation_cfg.flow_engine
    try:
        if engine_name == "RAFT":
            return compute_custom_raft_bidirectional(
                frames,
                cast(RaftCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )
        if engine_name == "NeuFlow":
            return compute_neuflow_bidirectional(
                frames,
                cast(NeuFlowCheckpointName, precomputation_cfg.flow_model),
                "auto",
            )
        with optical_flow_engine(precomputation_cfg) as compute:
            fwd = compute(frames)
            bwd_reversed = compute(list(reversed(frames)))
        return fwd, list(reversed(bwd_reversed))
    finally:
        print("Bidirectional optical flow computation complete. Releasing model from memory...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
