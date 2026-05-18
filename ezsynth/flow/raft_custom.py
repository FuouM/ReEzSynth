"""Project RAFT checkpoint optical flow."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..engines.backends.common import get_auto_torch_device
from ..raft.raft import RAFT
from ..raft.utils import InputPadder
from .types import FlowTorchDevice, RaftCheckpointName


def _raft_model_path(model_name: RaftCheckpointName) -> str:
    return f"models/raft/raft-{model_name}.pth"


def compute_custom_raft_sequence(
    frames: List[np.ndarray],
    model_name: RaftCheckpointName = "sintel",
    device: FlowTorchDevice = "auto",
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    model, dev = _load_raft_model(model_name, device)
    return _compute_custom_raft_sequence_with_model(frames, model, dev)


def compute_custom_raft_bidirectional(
    frames: List[np.ndarray],
    model_name: RaftCheckpointName = "sintel",
    device: FlowTorchDevice = "auto",
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    if len(frames) < 2:
        return [], []

    model, dev = _load_raft_model(model_name, device)
    fwd = _compute_custom_raft_sequence_with_model(frames, model, dev)
    bwd_reversed = _compute_custom_raft_sequence_with_model(
        list(reversed(frames)),
        model,
        dev,
        desc="Computing Backward Optical Flow (RAFT)",
    )
    return fwd, list(reversed(bwd_reversed))


def _load_raft_model(
    model_name: RaftCheckpointName,
    device: FlowTorchDevice,
) -> tuple[nn.DataParallel, torch.device]:
    print(f"Initializing RAFT Flow Engine (model: {model_name})...")
    if device == "auto":
        dev = torch.device(get_auto_torch_device())
    else:
        dev = torch.device(device)
    args = Namespace(model=model_name, small=False, mixed_precision=False)
    model = nn.DataParallel(RAFT(args))
    model_path = _raft_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RAFT model file not found: '{model_path}'")
    state_dict = torch.load(model_path, map_location=dev)
    model.load_state_dict(state_dict)
    model.to(dev).eval()
    print("RAFT Flow Engine initialized.")
    return model, dev


def _compute_custom_raft_sequence_with_model(
    frames: List[np.ndarray],
    model: nn.DataParallel,
    dev: torch.device,
    *,
    desc: str = "Computing Optical Flow (RAFT)",
    show_progress: bool = True,
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    def preprocess_image(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev).float()

    optical_flows: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(
            range(len(frames) - 1),
            desc=desc,
            disable=not show_progress,
        ):
            img1 = preprocess_image(frames[i])
            img2 = preprocess_image(frames[i + 1])
            padder = InputPadder(img1.shape)
            img1_padded, img2_padded = padder.pad(img1, img2)
            _, flow_up = model(img1_padded, img2_padded, iters=20, test_mode=True)
            flow_unpadded = padder.unpad(flow_up[0])
            flow_np = flow_unpadded.permute(1, 2, 0).cpu().numpy()
            optical_flows.append(flow_np)

    return optical_flows
