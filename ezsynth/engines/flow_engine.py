# ezsynth/engines/flow_engine.py
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ..raft.raft import RAFT
from ..raft.utils import InputPadder
from .base import BaseEngine


class RAFTFlowEngine(BaseEngine):
    """
    An engine for computing optical flow between frames using the RAFT model.
    """

    def __init__(self, model_name: str = "sintel", arch: str = "RAFT"):
        """
        Initializes the RAFTFlowEngine.

        Args:
            model_name (str): The name of the pre-trained model to use (e.g., 'sintel', 'kitti').
            arch (str): The model architecture (only 'RAFT' is currently supported).
        """
        print(f"Initializing RAFT Flow Engine (model: {model_name})...")
        self.rafter = RAFT_flow(model_name=model_name, arch=arch)
        print("RAFT Flow Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes forward optical flow for a sequence (from frame i to i+1).

        Args:
            frames (List[np.ndarray]): A list of BGR frames.

        Returns:
            List[np.ndarray]: A list of [H, W, 2] flow fields.
        """
        if len(frames) < 2:
            return []

        optical_flows = []
        for i in tqdm(range(len(frames) - 1), desc="Computing Optical Flow"):
            flow = self.rafter._compute_flow(frames[i], frames[i + 1])
            optical_flows.append(flow)

        return optical_flows


class RAFT_flow:
    def __init__(self, model_name="sintel", arch="RAFT"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch

        if self.arch == "RAFT":
            model_path = f"models/raft/raft-{model_name}.pth"
            args = self._get_args(model_name)
            self.model = torch.nn.DataParallel(RAFT(args))
        else:
            raise NotImplementedError(f"Flow architecture '{arch}' not implemented.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RAFT model file not found: '{model_path}'")

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

    def _compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        # NOTE: The engine expects BGR format frames, consistent with the rest of the pipeline.
        # This matches the behavior of the original ezsynth_old codebase.
        with torch.no_grad():
            img1_tensor = self._load_tensor_from_numpy(img1)
            img2_tensor = self._load_tensor_from_numpy(img2)

            padder = InputPadder(img1_tensor.shape)
            img1_padded, img2_padded = padder.pad(img1_tensor, img2_tensor)

            _, flow_up = self.model(img1_padded, img2_padded, iters=20, test_mode=True)

            flow_unpadded = padder.unpad(flow_up[0])
            return flow_unpadded.permute(1, 2, 0).cpu().numpy()
