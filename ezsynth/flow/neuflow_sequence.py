"""NeuFlow optical flow using project checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ..engines.backends.common import get_auto_torch_device
from ..neuflow.neuflow import NeuFlow
from .pad import pad_bgr_to_stride
from .types import FlowTorchDevice, NeuFlowCheckpointName

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _neuflow_model_path(model_name: NeuFlowCheckpointName) -> Path:
    return _PROJECT_ROOT / "models" / "neuflow" / f"{model_name}.pth"


def compute_neuflow_sequence(
    frames: List[np.ndarray],
    model_name: NeuFlowCheckpointName = "neuflow_sintel",
    device: FlowTorchDevice = "auto",
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    model, dev, dtype, use_amp = _load_neuflow_model(model_name, device)
    return _compute_neuflow_sequence_with_model(frames, model, dev, dtype, use_amp)


def compute_neuflow_bidirectional(
    frames: List[np.ndarray],
    model_name: NeuFlowCheckpointName = "neuflow_sintel",
    device: FlowTorchDevice = "auto",
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    if len(frames) < 2:
        return [], []

    model, dev, dtype, use_amp = _load_neuflow_model(model_name, device)
    fwd = _compute_neuflow_sequence_with_model(frames, model, dev, dtype, use_amp)
    bwd_reversed = _compute_neuflow_sequence_with_model(
        list(reversed(frames)),
        model,
        dev,
        dtype,
        use_amp,
        desc="Computing Backward Optical Flow (NeuFlow)",
    )
    return fwd, list(reversed(bwd_reversed))


def _load_neuflow_model(
    model_name: NeuFlowCheckpointName,
    device: FlowTorchDevice,
) -> tuple[NeuFlow, torch.device, torch.dtype, bool]:
    print(f"Initializing NeuFlow Engine (model: {model_name})...")
    if device == "auto":
        dev = torch.device(get_auto_torch_device())
    else:
        dev = torch.device(device)

    use_amp = dev.type == "cuda"
    use_half = dev.type == "cuda"
    model = NeuFlow()
    model_path = _neuflow_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"NeuFlow model file not found: '{model_path}'")
    checkpoint = torch.load(model_path, map_location=dev)
    model.load_state_dict(checkpoint["model"], strict=True)
    dtype = torch.half if use_half else torch.float32
    model.to(dev, dtype=dtype).eval()
    return model, dev, dtype, use_amp


def _compute_neuflow_sequence_with_model(
    frames: List[np.ndarray],
    model: NeuFlow,
    dev: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    *,
    desc: str = "Computing Optical Flow (NeuFlow)",
    show_progress: bool = True,
) -> List[np.ndarray]:
    if len(frames) < 2:
        return []

    _, (h0, w0, pb, pr) = pad_bgr_to_stride(frames[0])
    ph, pw = h0 + pb, w0 + pr
    init_key = (ph, pw, str(dev), use_amp)
    prev_key = getattr(model, "_ezsynth_bhwd_key", None)
    if prev_key != init_key:
        print(
            f"  - Initializing NeuFlow for resolution: {w0}x{h0} "
            f"(padded to {pw}x{ph} for alignment)"
        )
        model.init_bhwd(1, ph, pw, dev, amp=use_amp)
        setattr(model, "_ezsynth_bhwd_key", init_key)

    def preprocess_image(img: np.ndarray) -> torch.Tensor:
        padded, _ = pad_bgr_to_stride(img)
        t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
        return t.to(dev, dtype=dtype)

    optical_flows: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(
            range(len(frames) - 1),
            desc=desc,
            disable=not show_progress,
        ):
            img1 = preprocess_image(frames[i])
            img2 = preprocess_image(frames[i + 1])
            flow_pred = model(img1, img2)[-1]
            flow_hwc = flow_pred[0].permute(1, 2, 0).float().cpu().numpy()
            flow_np = flow_hwc[:h0, :w0, :]
            optical_flows.append(flow_np)

    return optical_flows
