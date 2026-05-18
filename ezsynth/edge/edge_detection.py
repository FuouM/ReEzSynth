from typing import cast

import numpy as np

from .batch import EdgeMethod, compute_edge_frame
from .classic import compute_classic_edge
from .ops import (
    PageEdgeParams,
    PstEdgeParams,
    create_gaussian_kernel,
    pad_gray_reflect,
    postprocess_pst_page,
    unpad_gray,
)
from .ops import (
    replace_zeros_tensor as replace_zeros_tensor,
)
from .phycv import compute_page_edge, compute_pst_edge


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
        self.pad_size = 16

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        return create_gaussian_kernel(size, sigma)

    def pad_image(self, img):
        return pad_gray_reflect(img, self.pad_size)

    def unpad_image(self, img):
        return unpad_gray(img, self.pad_size)

    def classic_preprocess(self, img):
        return compute_classic_edge(img)

    def pst_page_postprocess(self, edge_map: np.ndarray):
        return postprocess_pst_page(edge_map)

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
        return compute_pst_edge(
            input_data,
            self.device,
            PstEdgeParams(S, W, sigma_LPF, thresh_min, thresh_max, morph_flag),
            self.pad_size,
        )

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
        return compute_page_edge(
            input_data,
            self.device,
            PageEdgeParams(
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
            ),
            self.pad_size,
        )

    def compute_edge(self, input_data: np.ndarray):
        return compute_edge_frame(input_data, cast(EdgeMethod, self.method))
