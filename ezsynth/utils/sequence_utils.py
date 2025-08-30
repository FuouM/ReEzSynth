# ezsynth/utils/sequence_utils.py
from typing import List


class SynthesisSequence:
    MODE_FWD = "forward"
    MODE_REV = "reverse"
    MODE_BLN = "blend"

    def __init__(
        self, start_frame: int, end_frame: int, mode: str, style_indices: List[int]
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.mode = mode
        self.style_indices = style_indices

    def __repr__(self) -> str:
        return f"Sequence(start={self.start_frame}, end={self.end_frame}, mode='{self.mode}', styles={self.style_indices})"


def create_sequences(
    num_frames: int, style_indices: List[int]
) -> List[SynthesisSequence]:
    """Creates a list of SynthesisSequence objects to cover the entire frame range."""
    sequences: List[SynthesisSequence] = []

    # Ensure style indices are sorted and within bounds
    style_indices = sorted(list(set(style_indices)))
    if not style_indices or style_indices[0] < 0 or style_indices[-1] >= num_frames:
        # Handle case with no valid styles - just do a single forward pass
        print(
            "Warning: No valid style frames provided or indices out of bounds. Defaulting to a single forward pass."
        )
        sequences.append(
            SynthesisSequence(0, num_frames - 1, SynthesisSequence.MODE_FWD, [0])
        )
        return sequences

    # Sequence before the first style frame (reverse)
    if style_indices[0] > 0:
        sequences.append(
            SynthesisSequence(0, style_indices[0], SynthesisSequence.MODE_REV, [0])
        )

    # Sequences between style frames (blend)
    for i in range(len(style_indices) - 1):
        start_idx = style_indices[i]
        end_idx = style_indices[i + 1]
        sequences.append(
            SynthesisSequence(
                start_idx, end_idx, SynthesisSequence.MODE_BLN, [i, i + 1]
            )
        )

    # Sequence after the last style frame (forward)
    if style_indices[-1] < num_frames - 1:
        last_style_map_idx = len(style_indices) - 1
        sequences.append(
            SynthesisSequence(
                style_indices[-1],
                num_frames - 1,
                SynthesisSequence.MODE_FWD,
                [last_style_map_idx],
            )
        )

    print("Defined sequences:")
    for seq in sequences:
        print(f"  - {seq}")

    return sequences
