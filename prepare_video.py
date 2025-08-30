import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str):
    """
    Extracts all frames from a video file and saves them as PNGs in an output directory.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory where frames will be saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting frames from '{video_path.name}' to '{output_dir}'...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc="Extracting Frames") as pbar:
        frame_num = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            output_filename = output_dir / f"{frame_num:05d}.png"
            cv2.imwrite(str(output_filename), frame)

            frame_num += 1
            pbar.update(1)

    cap.release()
    print(f"\nSuccessfully extracted {frame_num} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for Ezsynth."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (e.g., my_video.mp4).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output directory for frames (e.g., projects/my_video/content).",
    )

    args = parser.parse_args()

    extract_frames(args.video, args.output)
