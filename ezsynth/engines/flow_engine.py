# ezsynth/engines/flow_engine.py
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)
from tqdm import tqdm

from ..neuflow.neuflow import NeuFlow
from ..raft.raft import RAFT
from ..raft.utils import InputPadder
from .base import BaseEngine


class OpenCVFlowEngine(BaseEngine):
    """
    An engine for computing optical flow using OpenCV methods (DIS, Farneback).
    """

    def __init__(self, method: str = "DIS"):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for OpenCVFlowEngine.")

        self.method = method.upper()
        print(f"Initializing OpenCV Flow Engine (method: {self.method})...")

        if self.method == "DIS":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.dis.setUseSpatialPropagation(True)
        elif self.method == "FARNEBACK":
            pass  # No specific initialization needed for user-level object
        else:
            raise NotImplementedError(
                f"OpenCV flow method '{method}' not implemented. Use 'DIS' or 'FARNEBACK'."
            )

        print("OpenCV Flow Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) < 2:
            return []

        optical_flows = []
        for i in tqdm(
            range(len(frames) - 1), desc=f"Computing Optical Flow (CV2 {self.method})"
        ):
            # OpenCV flow methods typically work on grayscale images
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            if self.method == "DIS":
                flow = self.dis.calc(prev_gray, curr_gray, None)
            elif self.method == "FARNEBACK":
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

            # Flow is already HxWx2, numpy
            optical_flows.append(flow)

        return optical_flows


class TorchVisionFlowEngine(BaseEngine):
    """
    An engine for computing optical flow using TorchVision's RAFT models.
    Better support for standard PyTorch optimization and potentially MPS.
    """

    def __init__(self, model_name: str = "raft_large"):
        print(f"Initializing TorchVision Engine (model: {model_name})...")

        # Determine device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            self.DEVICE = torch.device("mps")
        else:
            self.DEVICE = torch.device("cpu")

        print(f"Using device: {self.DEVICE}")

        if model_name == "raft_large":
            self.model = raft_large(
                weights=Raft_Large_Weights.DEFAULT, progress=False
            ).to(self.DEVICE)
            self.transforms = Raft_Large_Weights.DEFAULT.transforms()
        elif model_name == "raft_small":
            self.model = raft_small(
                weights=Raft_Small_Weights.DEFAULT, progress=False
            ).to(self.DEVICE)
            self.transforms = Raft_Small_Weights.DEFAULT.transforms()
        else:
            raise NotImplementedError(
                f"TorchVision model '{model_name}' not supported. Use 'raft_large' or 'raft_small'."
            )

        self.model.eval()
        print("TorchVision Flow Engine initialized.")

    def _preprocess(self, img1_batch, img2_batch):
        return self.transforms(img1_batch, img2_batch)

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if len(frames) < 2:
            return []

        optical_flows = []
        with torch.no_grad():
            for i in tqdm(
                range(len(frames) - 1), desc="Computing Optical Flow (TV RAFT)"
            ):
                # Frames are typically numpy [H, W, C] (BGR or RGB? BaseEngine implies BGR usually in cv2 context but let's assume consistent with others)
                # TorchVision models expect [C, H, W] tensors, 0-255 or 0-1 depending on transforms.
                # The weights transforms handle normalization. We need to provide [B, C, H, W] uint8 tensor generally or float.

                # Assuming frames are BGR [H, W, C] uint8 numpy arrays from cv2.imread usually.
                # TorchVision transforms often expect RGB.

                # convert to tensor [C, H, W]
                img1 = (
                    torch.from_numpy(frames[i])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.DEVICE)
                )  # [1, C, H, W]
                img2 = (
                    torch.from_numpy(frames[i + 1])
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.DEVICE)
                )

                # If frames are BGR (common in cv2), we might need to convert to RGB if the model expects it.
                # RAFT weights usually expect RGB.
                # Let's flip channels.
                img1 = img1.flip(1)  # BGR to RGB
                img2 = img2.flip(1)

                img1, img2 = self._preprocess(img1, img2)

                list_of_flows = self.model(img1, img2)
                flow_up = list_of_flows[-1]  # The last output is the final flow

                # flow_up is [1, 2, H, W]
                flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
                optical_flows.append(flow_np)

        return optical_flows


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

        # Determine device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            self.use_amp = True
            self.use_half = True
        elif hasattr(torch, "mps") and torch.mps.is_available():
            self.DEVICE = torch.device("mps")
            self.use_amp = False  # MPS doesn't support AMP well
            self.use_half = False  # MPS doesn't support half precision well
        else:
            self.DEVICE = torch.device("cpu")
            self.use_amp = False  # CPU doesn't benefit from AMP
            self.use_half = False  # Use full precision on CPU

        print(
            f"Using device: {self.DEVICE}, AMP: {self.use_amp}, Half: {self.use_half}"
        )

        self.model = NeuFlow()
        model_path = f"models/neuflow/{model_name}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NeuFlow model file not found: '{model_path}'")

        checkpoint = torch.load(model_path, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint["model"], strict=True)

        # Use half precision only on CUDA
        dtype = torch.half if self.use_half else torch.float32
        self.model.to(self.DEVICE, dtype=dtype).eval()

        self.initialized_dims = None
        print("NeuFlow Engine initialized.")

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        # NeuFlow expects BGR, [0, 1] range, CHW format
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        dtype = torch.half if self.use_half else torch.float32
        img_tensor = img_tensor.to(self.DEVICE, dtype=dtype)
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
            self.model.init_bhwd(1, h, w, self.DEVICE, amp=self.use_amp)
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
