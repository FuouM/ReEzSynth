# ezsynth/engines/flow_engine.py
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ..neuflow.neuflow import NeuFlow
from ..raft.raft import RAFT
from ..raft.utils import InputPadder
from .base import BaseEngine


class RAFTFlowEngine(BaseEngine):
    """
    An engine for computing optical flow between frames using the RAFT model.
    """

    def __init__(self, model_name: str = "sintel", arch: str = "RAFT"):
        print(f"Initializing RAFT Flow Engine (model: {model_name})...")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if arch.upper() != "RAFT":
            raise NotImplementedError(f"Flow architecture '{arch}' not implemented.")

        from argparse import Namespace

        args = Namespace(model=model_name, small=False, mixed_precision=False)
        self.model = torch.nn.DataParallel(RAFT(args))

        model_path = f"models/raft/raft-{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RAFT model file not found: '{model_path}'")

        state_dict = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.to(self.DEVICE).eval()
        print("RAFT Flow Engine initialized.")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        # RAFT expects BGR, [0, 255] range, CHW format
        return (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE).float()
        )

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) < 2:
            return []

        optical_flows = []
        with torch.no_grad():
            for i in tqdm(range(len(frames) - 1), desc="Computing Optical Flow (RAFT)"):
                img1 = self._preprocess_image(frames[i])
                img2 = self._preprocess_image(frames[i + 1])

                padder = InputPadder(img1.shape)
                img1_padded, img2_padded = padder.pad(img1, img2)

                _, flow_up = self.model(
                    img1_padded, img2_padded, iters=20, test_mode=True
                )

                flow_unpadded = padder.unpad(flow_up[0])
                flow_np = flow_unpadded.permute(1, 2, 0).cpu().numpy()
                optical_flows.append(flow_np)
        return optical_flows


class NeuFlowEngine(BaseEngine):
    """
    An engine for computing optical flow using the NeuFlow model.
    """

    def __init__(self, model_name: str = "neuflow_sintel"):
        """
        Initializes the NeuFlowEngine.
        Args:
            model_name (str): Name of the pre-trained model file (e.g., 'neuflow_sintel').
        """
        print(f"Initializing NeuFlow Engine (model: {model_name})...")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.DEVICE.type == "cpu":
            raise RuntimeError("NeuFlowEngine requires a CUDA-enabled device.")

        self.model = NeuFlow()
        model_path = f"models/neuflow/{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NeuFlow model file not found: '{model_path}'")

        checkpoint = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.DEVICE).eval().half()

        self.initialized_dims = None
        print("NeuFlow Engine initialized.")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        # NeuFlow expects BGR, [0, 1] range, CHW format, FP16
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.DEVICE).half()
        return img_tensor

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes forward optical flow for a sequence (from frame i to i+1).
        """
        if len(frames) < 2:
            return []

        h, w, _ = frames[0].shape
        current_dims = (h, w)

        # Initialize the model with image dimensions if they have changed
        if self.initialized_dims != current_dims:
            print(f"  - Initializing NeuFlow for resolution: {w}x{h}")
            self.model.init_bhwd(1, h, w, self.DEVICE, amp=True)
            self.initialized_dims = current_dims

        optical_flows = []
        with torch.no_grad():
            for i in tqdm(
                range(len(frames) - 1), desc="Computing Optical Flow (NeuFlow)"
            ):
                img1 = self._preprocess_image(frames[i])
                img2 = self._preprocess_image(frames[i + 1])

                # NeuFlow returns a list of flow preds, we take the last (most refined)
                flow_pred = self.model(img1, img2)[-1]

                # Postprocess: remove batch dim, permute HWC, convert to float32 numpy
                flow_np = flow_pred[0].permute(1, 2, 0).float().cpu().numpy()
                optical_flows.append(flow_np)

        return optical_flows
