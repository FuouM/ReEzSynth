import logging
from typing import cast

import numpy as np

from ..flow.neuflow_sequence import compute_neuflow_sequence
from ..flow.opencv import compute_opencv_flow_sequence
from ..flow.raft_custom import compute_custom_raft_sequence
from ..flow.torchvision_flow import compute_torchvision_raft_sequence
from ..flow.types import (
    NeuFlowCheckpointName,
    OpenCvFlowMethod,
    RaftCheckpointName,
    TorchVisionRaftModel,
)

logger = logging.getLogger(__name__)


class OpenCVFlowEngine:
    """
    An engine for computing optical flow using OpenCV methods (DIS, Farneback).
    """

    def __init__(self, method: str = "DIS"):
        self.method = method.upper()
        logger.info(f"Initializing OpenCV Flow Engine (method: {self.method})...")
        if self.method not in {"DIS", "FARNEBACK"}:
            raise NotImplementedError(
                f"OpenCV flow method '{method}' not implemented. Use 'DIS' or 'FARNEBACK'."
            )
        logger.info("OpenCV Flow Engine initialized.")

    def compute(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return compute_opencv_flow_sequence(frames, cast(OpenCvFlowMethod, self.method))


class TorchVisionFlowEngine:
    """
    An engine for computing optical flow using TorchVision's RAFT models.
    """

    def __init__(self, model_name: str = "raft_large"):
        logger.info(f"Initializing TorchVision Engine (model: {model_name})...")
        if model_name not in {"raft_large", "raft_small"}:
            raise NotImplementedError(
                f"TorchVision model '{model_name}' not supported. Use 'raft_large' or 'raft_small'."
            )
        self.model_name = model_name
        logger.info("TorchVision Flow Engine initialized.")

    def compute(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return compute_torchvision_raft_sequence(
            frames,
            cast(TorchVisionRaftModel, self.model_name),
        )


class RAFTFlowEngine:
    """
    An engine for computing optical flow between frames using the RAFT model.
    """

    def __init__(self, model_name: str = "sintel", arch: str = "RAFT"):
        logger.info(f"Initializing RAFT Flow Engine (model: {model_name})...")
        if arch.upper() != "RAFT":
            raise NotImplementedError(f"Flow architecture '{arch}' not implemented.")
        self.model_name = model_name
        logger.info("RAFT Flow Engine initialized.")

    def compute(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return compute_custom_raft_sequence(
            frames, cast(RaftCheckpointName, self.model_name)
        )


class NeuFlowEngine:
    """
    An engine for computing optical flow using the NeuFlow model.
    """

    def __init__(self, model_name: str = "neuflow_sintel"):
        logger.info(f"Initializing NeuFlow Engine (model: {model_name})...")
        self.model_name = model_name
        logger.info("NeuFlow Engine initialized.")

    def compute(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return compute_neuflow_sequence(
            frames, cast(NeuFlowCheckpointName, self.model_name)
        )
