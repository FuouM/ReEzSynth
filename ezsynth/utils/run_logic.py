# ezsynth/utils/run_logic.py

# This is a placeholder to migrate your sequence definition logic from sequences.py
def define_sequences(num_frames, style_indices):
    # This is a simplified version of your SequenceManager
    # You should migrate the full logic here.
    if not style_indices:
        return [{"start": 0, "end": num_frames - 1, "mode": "forward"}]
    # Add more complex logic for multiple styles and blending later
    return [{"start": 0, "end": num_frames - 1, "mode": "forward", "style_idx": 0}]


# You can also move the logic from aux_run.py here
def run_sequence_pass():
    pass
