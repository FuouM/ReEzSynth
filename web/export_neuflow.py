"""Export NeuFlow v2 to ONNX for the web app's optical flow (web/public/models/).

The ONNX graph has a fixed input size (a multiple of 32); the web app pads or resizes
frames to it. Needs the ezsynth environment, the NeuFlow checkpoint in models/neuflow/
and the ONNX packages:

  uv run --with onnx --with onnxscript python web/export_neuflow.py --size 960x544
  uv run --with onnx --with onnxscript python web/export_neuflow.py --size 960x544 --fp16

fp16 halves the download and memory (needs WebGPU shader-f16) and is traced on CUDA.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ezsynth.logging_setup import configure_cli_logging
from ezsynth.neuflow.neuflow import NeuFlow

logger = logging.getLogger(__name__)


class FlowOnly(torch.nn.Module):
    """(img0, img1) float32 BGR in 0..255, (1, 3, H, W) -> final flow (1, 2, H, W)."""

    def __init__(self, model: NeuFlow, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        flow = self.model(img0.to(self.dtype), img1.to(self.dtype))[-1]
        return flow.float()


def parse_size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    if w <= 0 or h <= 0 or w % 32 or h % 32:
        raise argparse.ArgumentTypeError("size must be WxH with multiples of 32")
    return w, h


def main(argv: list[str] | None = None) -> None:
    configure_cli_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=parse_size, default=(960, 544))
    parser.add_argument("--model", default="neuflow_mixed")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "web" / "public" / "models"
    )
    args = parser.parse_args(argv)

    w, h = args.size
    dtype = torch.half if args.fp16 else torch.float
    device = torch.device("cuda" if args.fp16 else "cpu")
    ckpt = ROOT / "models" / "neuflow" / f"{args.model}.pth"
    model = NeuFlow()
    model.load_state_dict(
        torch.load(ckpt, map_location="cpu", weights_only=True)["model"], strict=True
    )
    model.to(device, dtype).eval()
    model.init_bhwd(1, h, w, device, amp=args.fp16)

    dummy = torch.rand(1, 3, h, w, device=device) * 255
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.model}_{w}x{h}{'_fp16' if args.fp16 else ''}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            FlowOnly(model, dtype).eval(),
            (dummy, dummy.clone()),
            str(out),
            input_names=["img0", "img1"],
            output_names=["flow"],
            opset_version=17,
            dynamo=False,
        )
    logger.info(f"{out} ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
