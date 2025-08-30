# ezsynth/engines/flow_engine.py
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ezsynth.vendor.raft.core.core.raft import RAFT
from ezsynth.vendor.raft.core.core.utils.utils import InputPadder

from .base import BaseEngine


class RAFTFlowEngine(BaseEngine):
    def __init__(self, model_name="sintel", arch="RAFT"):
        print(f"Initializing RAFT Flow Engine (model: {model_name})...")
        self.rafter = RAFT_flow(model_name=model_name, arch=arch)
        print("RAFT Flow Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) < 2:
            return []

        optical_flows = []
        for i in tqdm(range(len(frames) - 1), desc="Computing Optical Flow"):
            flow = self.rafter._compute_flow(frames[i], frames[i + 1])
            optical_flows.append(flow)

        return optical_flows

    def compute_reverse(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Computes reverse optical flow (frame i+1 to i)."""
        if len(frames) < 2:
            return []

        optical_flows = []
        for i in tqdm(range(len(frames) - 1), desc="Computing Reverse Optical Flow"):
            flow = self.rafter._compute_flow(frames[i + 1], frames[i])
            optical_flows.append(flow)

        return optical_flows


class RAFT_flow:
    def __init__(self, model_name="sintel", arch="RAFT"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch

        base_path = os.path.join(os.path.dirname(__file__), "..", "vendor", "raft")

        if self.arch == "RAFT":
            model_path = os.path.join(base_path, "models", f"raft-{model_name}.pth")
            args = self._get_args(model_name)
            self.model = torch.nn.DataParallel(RAFT(args))
        else:
            raise NotImplementedError(f"Flow architecture '{arch}' not implemented.")

        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: '{model_path}'")

        state_dict = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.to(self.DEVICE)
        self.model.eval()

    def _get_args(self, model_name):
        from argparse import Namespace

        args = Namespace()
        args.model = model_name
        args.small = False
        args.mixed_precision = False
        return args

    def _load_tensor_from_numpy(self, np_array: np.ndarray):
        return (
            torch.from_numpy(np_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.DEVICE)
            .float()
        )

    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray):
        # --- START OF MODIFICATION ---
        # NOTE: The original ezsynth code fed BGR images directly to RAFT.
        # While RAFT expects RGB, to be 100% faithful to the original output,
        # we must replicate this behavior. The "fix" is commented out.
        # img1_rgb = img1[..., ::-1].copy()
        # img2_rgb = img2[..., ::-1].copy()

        with torch.no_grad():
            img1_tensor = self._load_tensor_from_numpy(img1)
            img2_tensor = self._load_tensor_from_numpy(img2)

            padder = InputPadder(img1_tensor.shape)
            img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)

            _, flow_up = self.model(img1_padded, img2_padded, iters=20, test_mode=True)

            flow_unpadded = padder.unpad(flow_up[0])

            result = flow_unpadded.permute(1, 2, 0).cpu().numpy()

            return result
        # --- END OF MODIFICATION ---
