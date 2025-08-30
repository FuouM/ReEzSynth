from typing import List, Tuple

import numpy as np

from ..vendor._ebsynth import ebsynth
from ..config import EbsynthParamsConfig


class EbsynthEngine:
    def __init__(self, config: EbsynthParamsConfig):
        print("Initializing Ebsynth Engine...")
        self.eb = ebsynth(
            uniformity=config.uniformity,
            patchsize=config.patch_size,
            searchvoteiters=config.search_vote_iters,
            patchmatchiters=config.patch_match_iters,
            extrapass3x3=True,  # This was a fixed value in the old code
        )
        self.eb.runner.initialize_libebsynth()
        print("Ebsynth Engine initialized.")

    def run_pass(
        self,
        source_frame: np.ndarray,
        target_frame: np.ndarray,
        source_style: np.ndarray,
        guides: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> np.ndarray:
        """
        Runs a single synthesis pass from a source to a target.

        Args:
            source_frame: The original frame corresponding to the source_style.
            target_frame: The frame we want to stylize.
            source_style: The stylized version of the source_frame.
            guides: A list of guides, each a tuple of (source_guide, target_guide, weight).

        Returns:
            The stylized target_frame.
        """
        # The main image guide is always present.
        all_guides = [
            (source_frame, target_frame, 6.0)
        ]  # Using a default, should be from config
        all_guides.extend(guides)

        stylized_img, _ = self.eb.run(source_style, guides=all_guides)
        return stylized_img
