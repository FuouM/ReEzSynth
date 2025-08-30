# ezsynth/engines/edge_engine.py
from typing import List

# Self-contained version of your EdgeDetector
import cv2
import numpy as np
import torch
from phycv import PAGE_GPU, PST_GPU
from tqdm import tqdm

from .base import BaseEngine


def replace_zeros_tensor(image: torch.Tensor, replace_value: int = 1) -> torch.Tensor:
    zero_mask = image == 0
    replace_tensor = torch.full_like(image, replace_value)
    return torch.where(zero_mask, replace_tensor, image)


class EdgeEngine(BaseEngine):
    def __init__(self, method: str = "Classic"):
        print(f"Initializing Edge Engine (method: {method})...")
        self.edge_detector = EdgeDetector(method=method)
        print("Edge Engine initialized.")

    def compute(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        edge_maps = []
        for frame in tqdm(frames, desc="Computing Edge Maps"):
            edge_map = self.edge_detector.compute_edge(frame)
            if len(edge_map.shape) == 2:
                edge_map = np.stack([edge_map] * 3, axis=-1)
            edge_maps.append(edge_map)
        return edge_maps


class EdgeConfig:
    # PST
    PST_S = 0.3
    PST_W = 15
    PST_SIG_LPF = 0.15
    PST_MIN = 0.05
    PST_MAX = 0.9

    # PAGE
    PAGE_M1 = 0
    PAGE_M2 = 0.35
    PAGE_SIG1 = 0.05
    PAGE_SIG2 = 0.8
    PAGE_S1 = 0.8
    PAGE_S2 = 0.8
    PAGE_SIG_LPF = 0.1
    PAGE_MIN = 0.0
    PAGE_MAX = 0.9

    MORPH_FLAG = 1

    def __init__(self, **kwargs):
        # PST attributes
        self.pst_s = kwargs.get("S", self.PST_S)
        self.pst_w = kwargs.get("W", self.PST_W)
        self.pst_sigma_lpf = kwargs.get("sigma_LPF", self.PST_SIG_LPF)
        self.pst_thresh_min = kwargs.get("thresh_min", self.PST_MIN)
        self.pst_thresh_max = kwargs.get("thresh_max", self.PST_MAX)

        # PAGE attributes
        self.page_mu_1 = kwargs.get("mu_1", self.PAGE_M1)
        self.page_mu_2 = kwargs.get("mu_2", self.PAGE_M2)
        self.page_sigma_1 = kwargs.get("sigma_1", self.PAGE_SIG1)
        self.page_sigma_2 = kwargs.get("sigma_2", self.PAGE_SIG2)
        self.page_s1 = kwargs.get("S1", self.PAGE_S1)
        self.page_s2 = kwargs.get("S2", self.PAGE_S2)
        self.page_sigma_lpf = kwargs.get("sigma_LPF", self.PAGE_SIG_LPF)
        self.page_thresh_min = kwargs.get("thresh_min", self.PAGE_MIN)
        self.page_thresh_max = kwargs.get("thresh_max", self.PAGE_MAX)

        self.morph_flag = kwargs.get("morph_flag", self.MORPH_FLAG)

    @classmethod
    def get_pst_default(cls) -> dict:
        return {
            "S": cls.PST_S,
            "W": cls.PST_W,
            "sigma_LPF": cls.PST_SIG_LPF,
            "thresh_min": cls.PST_MIN,
            "thresh_max": cls.PST_MAX,
            "morph_flag": cls.MORPH_FLAG,
        }

    @classmethod
    def get_page_default(cls) -> dict:
        return {
            "mu_1": cls.PAGE_M1,
            "mu_2": cls.PAGE_M2,
            "sigma_1": cls.PAGE_SIG1,
            "sigma_2": cls.PAGE_SIG2,
            "S1": cls.PAGE_S1,
            "S2": cls.PAGE_S2,
            "sigma_LPF": cls.PAGE_SIG_LPF,
            "thresh_min": cls.PAGE_MIN,
            "thresh_max": cls.PAGE_MAX,
            "morph_flag": cls.MORPH_FLAG,
        }

    def get_pst_current(self) -> dict:
        return {
            "S": self.pst_s,
            "W": self.pst_w,
            "sigma_LPF": self.pst_sigma_lpf,
            "thresh_min": self.pst_thresh_min,
            "thresh_max": self.pst_thresh_max,
            "morph_flag": self.morph_flag,
        }

    def get_page_current(self) -> dict:
        return {
            "mu_1": self.page_mu_1,
            "mu_2": self.page_mu_2,
            "sigma_1": self.page_sigma_1,
            "sigma_2": self.page_sigma_2,
            "S1": self.page_s1,
            "S2": self.page_s2,
            "sigma_LPF": self.page_sigma_lpf,
            "thresh_min": self.page_thresh_min,
            "thresh_max": self.page_thresh_max,
            "morph_flag": self.morph_flag,
        }


# (This is a copy-paste from your original 'edge_detection.py')
class EdgeDetector:
    def __init__(self, method="PAGE"):
        """
        Initialize the edge detector.

        :param method: Edge detection method. Choose from 'PST', 'Classic', or 'PAGE'.
        :PST: Phase Stretch Transform (PST) edge detector. - Good overall structure,
        but not very detailed.
        :Classic: Classic edge detector. - A good balance between structure and detail.
        :PAGE: Phase and Gradient Estimation (PAGE) edge detector. -
        Great detail, great structure, but slow.
        """
        self.method = method
        self.device = "cuda"
        if method == "PST":
            self.pst_gpu = PST_GPU(device=self.device)
        elif method == "PAGE":
            self.page_gpu = PAGE_GPU(direction_bins=10, device=self.device)
        elif method == "Classic":
            size, sigma = 5, 6.0
            self.kernel = self.create_gaussian_kernel(size, sigma)
        self.pad_size = 16

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
        return g / g.sum()

    def pad_image(self, img):
        return cv2.copyMakeBorder(
            img,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            self.pad_size,
            cv2.BORDER_REFLECT,
        )

    def unpad_image(self, img):
        return img[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]

    def classic_preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.filter2D(gray, -1, self.kernel)
        edge_map = cv2.subtract(gray, blurred)
        edge_map = np.clip(edge_map + 128, 0, 255)
        return edge_map.astype(np.uint8)

    def pst_page_postprocess(self, edge_map: np.ndarray):
        edge_map = cv2.GaussianBlur(edge_map, (5, 5), 3)
        edge_map = edge_map * 255
        return edge_map.astype(np.uint8)

    def pst_run(
        self,
        input_data: np.ndarray,
        S,
        W,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        input_img = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)

        padded_img = self.pad_image(input_img)

        self.pst_gpu.h = padded_img.shape[0]
        self.pst_gpu.w = padded_img.shape[1]

        self.pst_gpu.img = torch.from_numpy(padded_img).to(self.pst_gpu.device)
        # If input has too many zeros the model returns NaNs for some reason
        self.pst_gpu.img = replace_zeros_tensor(self.pst_gpu.img, 1)

        self.pst_gpu.init_kernel(S, W)
        self.pst_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)

        edge_map = self.pst_gpu.pst_output.cpu().numpy()
        edge_map = self.unpad_image(edge_map)

        return edge_map

    def page_run(
        self,
        input_data: np.ndarray,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    ):
        input_img = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
        padded_img = self.pad_image(input_img)

        self.page_gpu.h = padded_img.shape[0]
        self.page_gpu.w = padded_img.shape[1]

        self.page_gpu.img = torch.from_numpy(padded_img).to(self.page_gpu.device)
        # If input has too many zeros the model returns NaNs for some reason
        self.page_gpu.img = replace_zeros_tensor(self.page_gpu.img, 1)

        self.page_gpu.init_kernel(mu_1, mu_2, sigma_1, sigma_2, S1, S2)
        self.page_gpu.apply_kernel(sigma_LPF, thresh_min, thresh_max, morph_flag)
        self.page_gpu.create_page_edge()

        edge_map = self.page_gpu.page_edge.cpu().numpy()
        edge_map = self.unpad_image(edge_map)
        return edge_map

    def compute_edge(self, input_data: np.ndarray):
        edge_map = None
        if self.method == "PST":
            edge_map = self.pst_run(input_data, **EdgeConfig.get_pst_default())
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map

        if self.method == "Classic":
            edge_map = self.classic_preprocess(input_data)
            return edge_map

        if self.method == "PAGE":
            edge_map = self.page_run(input_data, **EdgeConfig.get_page_default())
            edge_map = self.pst_page_postprocess(edge_map)
            return edge_map
        return edge_map
