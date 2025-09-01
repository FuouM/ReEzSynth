# ezsynth/utils/blend_utils.py
from typing import List

import numpy as np
from tqdm import tqdm

from .blend_logic import Reconstructor, hist_blender
from .warp_utils import Warp


class Blender:
    def __init__(
        self,
        height,
        width,
        poisson_solver="lsqr",
        poisson_maxiter=None,
        poisson_grad_weight_l=2.5,
        poisson_grad_weight_ab=0.5,
    ):
        self.warp = Warp(height, width)
        self.reconstructor = Reconstructor(
            solver=poisson_solver,
            poisson_maxiter=poisson_maxiter,
            grad_weights=[
                poisson_grad_weight_l,
                poisson_grad_weight_ab,
                poisson_grad_weight_ab,
            ],
        )

    def create_selection_masks(
        self, err_fwd: List[np.ndarray], err_bwd: List[np.ndarray]
    ) -> List[np.ndarray]:
        err_fwd_arr = np.array(err_fwd)
        err_bwd_arr = np.array(err_bwd)
        selection_masks = np.where(err_fwd_arr < err_bwd_arr, 0, 1).astype(np.uint8)
        return [mask for mask in selection_masks]

    def warp_masks(
        self, flows_fwd: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        warped_masks = []
        prev_mask = np.zeros_like(masks[0])
        for i, mask in tqdm(
            enumerate(masks), total=len(masks), desc="Warping blend masks"
        ):
            # The original logic used the previous frame's flow to warp the current mask.
            # We replicate this for perfect consistency. For the first mask, use a zero flow.
            flow = flows_fwd[i - 1] if i > 0 else np.zeros_like(flows_fwd[0])

            warped_prev_mask = self.warp.run_warping(prev_mask.astype(np.float32), flow)

            final_mask = np.where(
                (warped_prev_mask > 0.5) & (mask == 0), 1, mask
            ).astype(np.uint8)

            prev_mask = final_mask.copy()
            warped_masks.append(final_mask)
        return warped_masks

    def run(self, fwd_frames, bwd_frames, fwd_errors, bwd_errors, fwd_flows):
        print("Starting blend process...")

        num_blend_frames = len(fwd_errors)
        if num_blend_frames <= 0:
            return []

        selection_masks = self.create_selection_masks(fwd_errors, bwd_errors)

        warped_selection_masks = self.warp_masks(fwd_flows, selection_masks)

        hist_blends = []
        for i in tqdm(range(num_blend_frames), desc="Histogram Blending"):
            # --- REPLICATING OLD LOGIC ---
            # The old code paired the error/mask for frame `i+1` with the styled result from frame `i`.
            # This creates a blended result for the keyframe itself.
            hist_blends.append(
                hist_blender(fwd_frames[i], bwd_frames[i], warped_selection_masks[i])
            )

        # The reconstructor also gets the full, misaligned frame lists.
        # It will produce N-1 blended frames, starting with a blend for the keyframe's index.
        final_blends = self.reconstructor.run(
            hist_blends,
            style_fwd=fwd_frames,
            style_bwd=bwd_frames,
            err_masks=warped_selection_masks,
        )

        print("Blending complete.")
        return final_blends
