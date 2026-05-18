"""Concrete ``Literal`` aliases for optical flow engines and checkpoints."""

from __future__ import annotations

from typing import Literal, TypeAlias

FlowEngineName: TypeAlias = Literal["RAFT", "NeuFlow", "OpenCV", "TorchVision"]

RaftCheckpointName: TypeAlias = Literal["sintel", "kitti", "small"]

NeuFlowCheckpointName: TypeAlias = Literal[
    "neuflow_mixed",
    "neuflow_sintel",
    "neuflow_things",
]

FlowModelName: TypeAlias = RaftCheckpointName | NeuFlowCheckpointName

RAFT_FLOW_MODELS: tuple[RaftCheckpointName, ...] = ("sintel", "kitti", "small")
NEUFLOW_FLOW_MODELS: tuple[NeuFlowCheckpointName, ...] = (
    "neuflow_mixed",
    "neuflow_sintel",
    "neuflow_things",
)

OpenCvFlowMethod: TypeAlias = Literal["DIS", "FARNEBACK"]

TorchVisionRaftModel: TypeAlias = Literal["raft_large", "raft_small"]

FlowTorchDevice: TypeAlias = Literal["auto", "cpu", "cuda", "mps"]

__all__ = [
    "FlowEngineName",
    "FlowModelName",
    "FlowTorchDevice",
    "NEUFLOW_FLOW_MODELS",
    "NeuFlowCheckpointName",
    "OpenCvFlowMethod",
    "RAFT_FLOW_MODELS",
    "RaftCheckpointName",
    "TorchVisionRaftModel",
]
