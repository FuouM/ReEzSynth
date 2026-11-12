// Frame sequences: decode a video file (mediabunny + WebCodecs) or a set of image files into
// packed RGB frames at one working size.
import { Cancelled } from "./fastblend";
import { decodeImage, fitSize, naturalCompare, rasterize, trailingNumber } from "./image";

export interface FrameSet {
  name: string;
  width: number;
  height: number;
  fps: number;
  /** Packed RGB frames, width*height*3 each. */
  frames: Uint8Array[];
  /** Number in each source file name (image sequences), for matching keyframes by name. */
  numbers?: (number | null)[];
}

export interface LoadOptions {
  /** Shrink so the longer side is at most this (0 = keep). Ignored when `size` is given. */
  maxSide: number;
  /** Force this size (used to match a second sequence to the first). */
  size?: { width: number; height: number };
  /** Stop after this many frames (0 = all). */
  maxFrames: number;
  /** Frame rate for image sequences. */
  fps: number;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number | null) => void;
}

const VIDEO_EXT = /\.(mp4|m4v|mov|webm|mkv|ogv)$/i;

export function isVideoFile(f: File): boolean {
  return f.type.startsWith("video/") || VIDEO_EXT.test(f.name);
}

export async function loadVideo(file: File, opts: LoadOptions): Promise<FrameSet> {
  // Loaded on first use so image-only sessions never fetch the video stack.
  const { ALL_FORMATS, BlobSource, CanvasSink, Input } = await import("mediabunny");
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error(`${file.name} has no video track`);
    if (!(await track.canDecode())) {
      throw new Error(`This browser cannot decode the ${track.codec ?? "unknown"} video in ${file.name}; convert it to H.264 MP4 or export an image sequence.`);
    }
    const dw = await track.getDisplayWidth();
    const dh = await track.getDisplayHeight();
    const { width, height } = opts.size ?? fitSize(dw, dh, opts.maxSide);
    const stats = await track.computePacketStats(120);
    const fps = stats.averagePacketRate > 0 ? Math.round(stats.averagePacketRate * 1000) / 1000 : opts.fps;
    const total = opts.maxFrames > 0 ? Math.min(opts.maxFrames, stats.packetCount) : null;
    const sink = new CanvasSink(track, { width, height, fit: "fill", poolSize: 2 });
    const frames: Uint8Array[] = [];
    for await (const wrapped of sink.canvases()) {
      if (opts.signal?.aborted) throw new Cancelled();
      frames.push(rasterize(wrapped.canvas, width, height));
      opts.onProgress?.(frames.length, total);
      if (opts.maxFrames > 0 && frames.length >= opts.maxFrames) break;
    }
    if (frames.length === 0) throw new Error(`No frames decoded from ${file.name}`);
    return { name: file.name, width, height, fps, frames };
  } finally {
    input.dispose();
  }
}

export async function loadImageSequence(files: File[], opts: LoadOptions): Promise<FrameSet> {
  const sorted = [...files].filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|avif)$/i.test(f.name)).sort((a, b) => naturalCompare(a.name, b.name));
  if (sorted.length === 0) throw new Error("No image files selected");
  const picked = opts.maxFrames > 0 ? sorted.slice(0, opts.maxFrames) : sorted;
  const first = await decodeImage(picked[0], opts.size ?? { maxSide: opts.maxSide });
  // Video codecs need even sizes; later frames are scaled to the first one's size.
  const size = opts.size ?? fitSize(first.width, first.height, 0);
  const frames: Uint8Array[] = [];
  for (const [i, f] of picked.entries()) {
    if (opts.signal?.aborted) throw new Cancelled();
    frames.push((await decodeImage(f, size)).data);
    opts.onProgress?.(i + 1, picked.length);
  }
  const prefix = picked[0].name.replace(/\d*\.[^.]*$/, "") || "frames";
  return {
    name: `${prefix}… (${picked.length} images)`,
    width: size.width,
    height: size.height,
    fps: opts.fps,
    frames,
    numbers: picked.map((f) => trailingNumber(f.name)),
  };
}

/** A video file, or several images as a sequence. */
export async function loadFrames(files: File[], opts: LoadOptions): Promise<FrameSet> {
  if (files.length === 1 && isVideoFile(files[0])) return loadVideo(files[0], opts);
  return loadImageSequence(files, opts);
}

/**
 * Frame index for a keyframe file: its number matched against the sequence's file numbers
 * when the frames came from images, otherwise the number itself as a 0-based frame index.
 */
export function keyframeIndex(set: FrameSet, fileName: string): number | null {
  const num = trailingNumber(fileName);
  if (num === null) return null;
  if (set.numbers) {
    const i = set.numbers.indexOf(num);
    if (i >= 0) return i;
  }
  return num >= 0 && num < set.frames.length ? num : null;
}

export function memoryMiB(set: { frames: Uint8Array[] }): number {
  return Math.round(set.frames.reduce((s, f) => s + f.byteLength, 0) / 2 ** 20);
}
