// Export a frame sequence as a video (WebCodecs through mediabunny) or a zip of PNGs.
// mediabunny and fflate are imported on first use so they stay out of the initial bundle.
import type { Zippable } from "fflate";
import type { VideoCodec } from "mediabunny";
import { Cancelled } from "./fastblend";
import type { FrameSet } from "./frames";
import { pngBlob, rgbToRgba } from "./image";

export interface ExportProgress {
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

/** First encodable codec: H.264 in MP4, else VP9/VP8 in WebM. */
async function pickCodec(width: number, height: number): Promise<{ codec: VideoCodec; mp4: boolean } | null> {
  const { canEncodeVideo, QUALITY_HIGH } = await import("mediabunny");
  for (const [codec, mp4] of [["avc", true], ["vp9", false], ["vp8", false]] as const) {
    if (await canEncodeVideo(codec, { width, height, bitrate: QUALITY_HIGH })) return { codec, mp4 };
  }
  return null;
}

export async function exportVideo(set: FrameSet, p: ExportProgress = {}): Promise<{ blob: Blob; ext: string }> {
  const { BufferTarget, CanvasSource, Mp4OutputFormat, Output, QUALITY_HIGH, WebMOutputFormat } = await import("mediabunny");
  const choice = await pickCodec(set.width, set.height);
  if (!choice) throw new Error("This browser cannot encode video (no WebCodecs H.264/VP9/VP8 encoder); export PNGs instead.");
  const output = new Output({
    format: choice.mp4 ? new Mp4OutputFormat({ fastStart: "in-memory" }) : new WebMOutputFormat(),
    target: new BufferTarget(),
  });
  const canvas = new OffscreenCanvas(set.width, set.height);
  const ctx = canvas.getContext("2d")!;
  const source = new CanvasSource(canvas, { codec: choice.codec, bitrate: QUALITY_HIGH });
  output.addVideoTrack(source, { frameRate: set.fps });
  await output.start();
  try {
    for (const [i, f] of set.frames.entries()) {
      if (p.signal?.aborted) throw new Cancelled();
      ctx.putImageData(new ImageData(rgbToRgba(f), set.width, set.height), 0, 0);
      await source.add(i / set.fps, 1 / set.fps);
      p.onProgress?.(i + 1, set.frames.length);
    }
    await output.finalize();
  } catch (e) {
    await output.cancel();
    throw e;
  }
  const buffer = output.target.buffer!;
  return choice.mp4 ? { blob: new Blob([buffer], { type: "video/mp4" }), ext: "mp4" } : { blob: new Blob([buffer], { type: "video/webm" }), ext: "webm" };
}

export async function exportPngZip(set: FrameSet, prefix: string, p: ExportProgress = {}): Promise<Blob> {
  const { zip } = await import("fflate");
  const files: Zippable = {};
  const digits = Math.max(4, String(set.frames.length).length);
  for (const [i, f] of set.frames.entries()) {
    if (p.signal?.aborted) throw new Cancelled();
    const png = await pngBlob({ width: set.width, height: set.height, data: f });
    // PNGs are already compressed; store them.
    files[`${prefix}_${String(i).padStart(digits, "0")}.png`] = [new Uint8Array(await png.arrayBuffer()), { level: 0 }];
    p.onProgress?.(i + 1, set.frames.length);
  }
  const data = await new Promise<Uint8Array>((resolve, reject) => zip(files, (err, out) => (err ? reject(err) : resolve(out))));
  return new Blob([data as Uint8Array<ArrayBuffer>], { type: "application/zip" });
}
