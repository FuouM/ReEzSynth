// EbSynth video in the browser, after ezsynth/scheduler.py and engines/pass_runner.py.
//
// Every frame is synthesized from a keyframe's style with guides that tie the keyframe to the
// frame: an edge map and the frame itself, and, when optical flow is available, a positional
// guide and the previous stylized frame warped onto this one. Frames before the first keyframe
// run backwards from it, frames after the last run forwards, and frames between two keyframes
// run both ways and are blended by distance (Python blends with error maps and Poisson
// reconstruction; that can replace blendByDistance later).
//
// No imports: the synthesizer and the flow source are passed in, so this runs in Node tests.

/** RGB frames, width*height*3 bytes each. */
export interface Frames {
  width: number;
  height: number;
  frames: Uint8Array[];
}

/** Dense flow from frame i to frame i+1: (dx, dy) per pixel in pixels, width*height*2. */
export interface FlowField {
  width: number;
  height: number;
  data: Float32Array;
}

/** Where flows come from (an ONNX model in the browser, precomputed files …). */
export interface FlowProvider {
  /** Flow from frame `i` to frame `i + 1`, at the frames' size. */
  forward(i: number): Promise<FlowField>;
}

export interface GuideInput {
  /** RGB at the style's size. */
  source: Uint8Array;
  /** RGB at the frame's size. */
  target: Uint8Array;
  weight: number;
}

/** Stylize one frame: style and guides are RGB of the frame size; resolves to RGB. */
export type SynthFn = (style: Uint8Array, guides: GuideInput[]) => Promise<Uint8Array>;

export interface GuideWeights {
  edge: number;
  image: number;
  /** Used only with flow. */
  pos: number;
  /** Used only with flow. */
  warp: number;
}

/** ezsynth's EbsynthParamsConfig defaults. */
export const DEFAULT_WEIGHTS: GuideWeights = { edge: 1, image: 6, pos: 2, warp: 0.5 };

export class Cancelled extends Error {
  constructor() {
    super("Cancelled");
    this.name = "Cancelled";
  }
}

// ── Sequences ──────────────────────────────────────────────────────────────────────────────

export type Sequence =
  /** Frames from..to (inclusive) synthesized from keyframe `from`, walking towards `to`. */
  | { mode: "forward" | "reverse"; from: number; to: number }
  /** Frames between keyframes `from` and `to`: both directions, then blended. */
  | { mode: "blend"; from: number; to: number };

/** The schedule of ezsynth.utils.sequence_utils.create_sequences, in frame order. */
export function planSequences(keyIndices: number[], n: number): Sequence[] {
  const keys = [...new Set(keyIndices)].sort((a, b) => a - b);
  if (keys.length === 0) throw new Error("No keyframes");
  if (keys[0] < 0 || keys[keys.length - 1] >= n) throw new Error(`Keyframe index out of range 0..${n - 1}`);
  const out: Sequence[] = [];
  if (keys[0] > 0) out.push({ mode: "reverse", from: keys[0], to: 0 });
  for (let i = 0; i + 1 < keys.length; i++) {
    if (keys[i + 1] - keys[i] > 1) out.push({ mode: "blend", from: keys[i], to: keys[i + 1] });
  }
  const last = keys[keys.length - 1];
  if (last < n - 1) out.push({ mode: "forward", from: last, to: n - 1 });
  return out;
}

/** Frames each sequence synthesizes (blend sequences count both passes). */
export function synthesisCount(seqs: Sequence[]): number {
  // A blend runs both ways over the frames strictly between its keyframes.
  return seqs.reduce((s, q) => s + (q.mode === "blend" ? 2 * (q.to - q.from - 1) : Math.abs(q.to - q.from)), 0);
}

// ── Image operations ───────────────────────────────────────────────────────────────────────

/** Normalized Gaussian kernel (ezsynth.edge.ops.create_gaussian_kernel). */
export function gaussianKernel(size: number, sigma: number): Float32Array {
  const k = new Float32Array(size * size);
  // np.mgrid[-size//2+1 : size//2+1]: -2..2 for size 5.
  const start = Math.floor(-size / 2) + 1;
  let sum = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = start + x;
      const dy = start + y;
      const v = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      k[y * size + x] = v;
      sum += v;
    }
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

/** OpenCV BORDER_REFLECT_101 (filter2D's default): dcb|abcd|cba. */
function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** OpenCV BORDER_REFLECT (remap): cba|abcd|dcb. */
function reflect(i: number, n: number): number {
  while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i;
  return i;
}

/**
 * Classic edge guide (ezsynth.edge.classic.compute_classic_edge): gray minus its 5x5 Gaussian
 * blur (sigma 6), saturated at 0, plus 128 with uint8 wrap-around exactly as NumPy does.
 * Returned as 3-channel RGB.
 */
export function classicEdge(rgb: Uint8Array, width: number, height: number, size = 5, sigma = 6): Uint8Array {
  const n = width * height;
  const gray = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    // cv2.COLOR_BGR2GRAY on the BGR frame = 0.299 R + 0.587 G + 0.114 B, rounded.
    gray[i] = Math.round(0.299 * rgb[j] + 0.587 * rgb[j + 1] + 0.114 * rgb[j + 2]);
  }
  const k = gaussianKernel(size, sigma);
  const r = Math.floor(size / 2);
  const out = new Uint8Array(n * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let ky = 0; ky < size; ky++) {
        const yy = reflect101(y + ky - r, height) * width;
        for (let kx = 0; kx < size; kx++) acc += k[ky * size + kx] * gray[yy + reflect101(x + kx - r, width)];
      }
      const blurred = Math.min(255, Math.max(0, Math.round(acc)));
      const g = gray[y * width + x];
      const e = (Math.max(0, g - blurred) + 128) & 255;
      const o = (y * width + x) * 3;
      out[o] = out[o + 1] = out[o + 2] = e;
    }
  }
  return out;
}

/**
 * Backward warp (ezsynth Warp.run_warping on float data): out(x, y) = img(x + fx, y + fy),
 * bilinear, reflected borders. `img` has `channels` floats per pixel.
 */
export function warpFloat(img: Float32Array, channels: number, flow: FlowField, scale = 1): Float32Array {
  const { width: w, height: h, data: f } = flow;
  const out = new Float32Array(w * h * channels);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      const sx = x + scale * f[p * 2];
      const sy = y + scale * f[p * 2 + 1];
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const ax = sx - x0;
      const ay = sy - y0;
      const xa = reflect(x0, w);
      const xb = reflect(x0 + 1, w);
      const ya = reflect(y0, h) * w;
      const yb = reflect(y0 + 1, h) * w;
      for (let c = 0; c < channels; c++) {
        const v00 = img[(ya + xa) * channels + c];
        const v01 = img[(ya + xb) * channels + c];
        const v10 = img[(yb + xa) * channels + c];
        const v11 = img[(yb + xb) * channels + c];
        out[p * channels + c] = (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay;
      }
    }
  }
  return out;
}

/** Warp RGB bytes like Warp.run_warping on uint8 (via 0..1 floats, truncated back). */
export function warpRgb(rgb: Uint8Array, flow: FlowField, scale = 1): Uint8Array {
  const f = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) f[i] = rgb[i] / 255;
  const w = warpFloat(f, 3, flow, scale);
  const out = new Uint8Array(rgb.length);
  for (let i = 0; i < w.length; i++) out[i] = Math.min(255, Math.max(0, Math.trunc(w[i] * 255)));
  return out;
}

/** Positional guide coordinates (PositionalGuide.pristine_coord_map): (x, y, 0) in 0..1. */
export function pristineCoords(width: number, height: number): Float32Array {
  const out = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 3;
      out[o] = width > 1 ? x / (width - 1) : 0;
      out[o + 1] = height > 1 ? y / (height - 1) : 0;
    }
  }
  return out;
}

/** Coordinates as a guide image, truncated to bytes like `(map * 255).astype(uint8)`. */
export function coordsToRgb(coords: Float32Array): Uint8Array {
  const out = new Uint8Array(coords.length);
  for (let i = 0; i < coords.length; i++) out[i] = Math.min(255, Math.max(0, Math.trunc(coords[i] * 255)));
  return out;
}

/** PositionalGuide.create_from_flow: the pristine map warped by `flow`, wrapped into 0..1. */
export function positionalFromFlow(pristine: Float32Array, flow: FlowField): Uint8Array {
  const warped = warpFloat(pristine, 3, flow);
  for (let i = 0; i < warped.length; i += 3) {
    warped[i] = ((warped[i] % 1) + 1) % 1;
    warped[i + 1] = ((warped[i + 1] % 1) + 1) % 1;
  }
  return coordsToRgb(warped);
}

export function blendByDistance(a: Uint8Array, b: Uint8Array, wa: number): Uint8Array {
  const out = new Uint8Array(a.length);
  const wb = 1 - wa;
  for (let i = 0; i < a.length; i++) out[i] = Math.round(a[i] * wa + b[i] * wb);
  return out;
}

// ── Passes ─────────────────────────────────────────────────────────────────────────────────

export interface VideoOptions {
  weights: GuideWeights;
  flow?: FlowProvider;
  signal?: AbortSignal;
  /** Frames synthesized so far, out of the total for this run. */
  onProgress?: (done: number, total: number) => void;
  /** Called with each finished output frame (for live preview). */
  onFrame?: (index: number, rgb: Uint8Array) => void;
}

interface PassContext {
  video: Frames;
  keyframes: Map<number, Uint8Array>;
  synth: SynthFn;
  opts: VideoOptions;
  edges: (i: number) => Uint8Array;
  pristine: Uint8Array | null;
  pristineCoords: Float32Array | null;
  tick: () => void;
}

/**
 * One pass (pass_runner.run): walk from keyframe `key` towards `to`, synthesizing every frame
 * strictly after the keyframe up to and including `to`. Returns frames in walk order.
 */
async function runPass(ctx: PassContext, key: number, to: number): Promise<Uint8Array[]> {
  const step = to >= key ? 1 : -1;
  const style = ctx.keyframes.get(key)!;
  const { video, opts } = ctx;
  const out: Uint8Array[] = [];
  let previous = style;
  for (let source = key; source !== to; source += step) {
    if (opts.signal?.aborted) throw new Cancelled();
    const target = source + step;
    const guides: GuideInput[] = [
      { source: ctx.edges(key), target: ctx.edges(target), weight: opts.weights.edge },
      { source: video.frames[key], target: video.frames[target], weight: opts.weights.image },
    ];
    if (opts.flow && ctx.pristine && ctx.pristineCoords) {
      // fwd_flows[transition] is the flow between the two frames, from the lower index.
      const transition = step > 0 ? source : target;
      const flow = await opts.flow.forward(transition);
      if (flow.width !== video.width || flow.height !== video.height) throw new Error(`Flow ${transition} is ${flow.width}×${flow.height}, frames are ${video.width}×${video.height}`);
      // directed_flow = flow * (-step); the positional guide uses the undirected flow.
      guides.push({ source: ctx.pristine, target: positionalFromFlow(ctx.pristineCoords, flow), weight: opts.weights.pos });
      guides.push({ source: style, target: warpRgb(previous, flow, -step), weight: opts.weights.warp });
    }
    const frame = await ctx.synth(style, guides);
    out.push(frame);
    previous = frame;
    ctx.tick();
  }
  return out;
}

/** Stylize every frame of `video` from the keyframes. Keyframes are returned unchanged. */
export async function synthesizeVideo(video: Frames, keyframes: Map<number, Uint8Array>, synth: SynthFn, opts: VideoOptions): Promise<Uint8Array[]> {
  const n = video.frames.length;
  const frameLen = video.width * video.height * 3;
  for (const [i, k] of keyframes) if (k.length !== frameLen) throw new Error(`Keyframe ${i} is not the size of the video frames`);
  const seqs = planSequences([...keyframes.keys()], n);
  const total = synthesisCount(seqs);
  let done = 0;
  const edgeCache = new Map<number, Uint8Array>();
  const coords = opts.flow ? pristineCoords(video.width, video.height) : null;
  const ctx: PassContext = {
    video,
    keyframes,
    synth,
    opts,
    edges: (i) => {
      let e = edgeCache.get(i);
      if (!e) edgeCache.set(i, (e = classicEdge(video.frames[i], video.width, video.height)));
      return e;
    },
    pristine: coords ? coordsToRgb(coords) : null,
    pristineCoords: coords,
    tick: () => opts.onProgress?.(++done, total),
  };
  const output: (Uint8Array | null)[] = new Array(n).fill(null);
  const put = (i: number, f: Uint8Array) => {
    output[i] = f;
    opts.onFrame?.(i, f);
  };
  for (const [i, k] of keyframes) put(i, k);

  for (const seq of seqs) {
    if (seq.mode !== "blend") {
      const frames = await runPass(ctx, seq.from, seq.to);
      const step = seq.to >= seq.from ? 1 : -1;
      frames.forEach((f, j) => put(seq.from + step * (j + 1), f));
      continue;
    }
    const { from: l, to: r } = seq;
    // Both passes stop one short of the opposite keyframe, which stays as painted.
    const fwd = await runPass(ctx, l, r - 1);
    const bwd = await runPass(ctx, r, l + 1);
    for (let m = l + 1; m < r; m++) {
      const a = fwd[m - l - 1];
      const b = bwd[r - m - 1];
      put(m, blendByDistance(a, b, (r - m) / (r - l)));
    }
  }
  return output.map((f, i) => f ?? video.frames[i]);
}
