"""TorchVision RAFT optical flow."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)
from tqdm import tqdm

from ..engines.backends.common import get_auto_torch_device
from .types import FlowTorchDevice, TorchVisionRaftModel


def _load_torchvision_raft(
    model_name: TorchVisionRaftModel,
    device: FlowTorchDevice = "auto",
) -> tuple[torch.nn.Module, torch.device, object]:
    print(f"Initializing TorchVision RAFT (model: {model_name})...")
    if device == "auto":
        dev = torch.device(get_auto_torch_device())
    else:
        dev = torch.device(device)

    if model_name == "raft_large":
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(dev)
        transforms = Raft_Large_Weights.DEFAULT.transforms()
    else:
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(dev)
        transforms = Raft_Small_Weights.DEFAULT.transforms()
    model.eval()
    return model, dev, transforms


def _compute_torchvision_raft_sequence_with_model(
    frames: List[np.ndarray],
    model: torch.nn.Module,
    dev: torch.device,
    transforms: object,
    *,
    desc: str = "Computing Optical Flow (TV RAFT)",
    show_progress: bool = True,
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    optical_flows: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(
            range(len(frames) - 1),
            desc=desc,
            disable=not show_progress,
        ):
            img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(dev)
            img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).unsqueeze(0).to(dev)
            img1 = img1.flip(1)
            img2 = img2.flip(1)
            img1, img2 = transforms(img1, img2)
            list_of_flows = model(img1, img2)
            flow_up = list_of_flows[-1]
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
            optical_flows.append(flow_np)

    return optical_flows


def compute_torchvision_raft_sequence(
    frames: List[np.ndarray],
    model_name: TorchVisionRaftModel,
    device: FlowTorchDevice = "auto",
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    model, dev, transforms = _load_torchvision_raft(model_name, device)
    return _compute_torchvision_raft_sequence_with_model(frames, model, dev, transforms)
