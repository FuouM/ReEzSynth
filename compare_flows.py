import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ezsynth.engines.flow_engine import (
    NeuFlowEngine,
    OpenCVFlowEngine,
    RAFTFlowEngine,
    TorchVisionFlowEngine,
)


def flow_to_color(flow):
    """
    Converts flow to RGB image using HSV color wheel.
    flow: [H, W, 2]
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    video_path = "examples/cat_full.mp4"
    output_path = "output/flow_comparison.mp4"

    if not os.path.exists("output"):
        os.makedirs("output")

    print(f"Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Read first 60 frames to keep it quick but sufficient
    MAX_FRAMES = 60
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to smaller resolution for speed if needed, but let's keep original or standard
        # For display, maybe 512 width max?
        h, w = frame.shape[:2]
        new_w = 480
        new_h = int(h * (new_w / w))
        frame = cv2.resize(frame, (new_w, new_h))
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames. Resolution: {frames[0].shape}")

    engines = [
        ("Original", None),
        ("CV2 DIS", OpenCVFlowEngine(method="DIS")),
        ("CV2 Farneback", OpenCVFlowEngine(method="FARNEBACK")),
        ("TV RAFT Small", TorchVisionFlowEngine(model_name="raft_small")),
        # ("TV RAFT Large", TorchVisionFlowEngine(model_name="raft_large")), # Skip large for speed if preferred, or keep
    ]

    try:
        engines.append(("Our RAFT (Sintel)", RAFTFlowEngine(model_name="sintel")))
    except Exception as e:
        print(f"Skipping Our RAFT: {e}")

    # Try adding NeuFlow if possible
    try:
        engines.append(("NeuFlow", NeuFlowEngine(model_name="neuflow_mixed")))
    except Exception as e:
        print(f"Skipping NeuFlow: {e}")

    # Compute flows
    results = {}  # label -> list of flow images (or original frames)

    for label, engine in engines:
        if engine is None:
            results[label] = frames[1:]  # Align length
            continue

        print(f"Running {label}...")
        try:
            flows = engine.compute(frames)
            # Visualize flows
            viz_frames = [flow_to_color(f) for f in flows]
            results[label] = viz_frames
        except Exception as e:
            print(f"Error computing {label}: {e}")
            # Fill with blanks
            results[label] = [np.zeros_like(frames[0])] * (len(frames) - 1)

    # Compose grid
    # We have N engines. Let's make a grid roughly square.
    n_items = len(engines)
    cols = 3
    rows = (n_items + cols - 1) // cols

    print("creating grid video...")
    h, w, _ = frames[0].shape

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (w * cols, h * rows))

    num_frames = len(frames) - 1
    for i in range(num_frames):
        grid_img = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        for idx, (label, _) in enumerate(engines):
            r = idx // cols
            c = idx % cols

            # Get content
            if label in results and i < len(results[label]):
                img = results[label][i].copy()
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)

            draw_label(img, label)

            y_off = r * h
            x_off = c * w
            grid_img[y_off : y_off + h, x_off : x_off + w] = img

        out.write(grid_img)

    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
