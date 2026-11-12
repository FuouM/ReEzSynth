// Optical flow for EbSynth video: NeuFlow v2 exported to ONNX by web/export_neuflow.py, run by
// ONNX Runtime Web on WebGPU. Implements ebsynthVideo's FlowProvider.
import type { FlowField, FlowProvider, Frames } from "./ebsynthVideo";

/** ONNX Runtime Web with the WebGPU execution provider, loaded on first use. */
export const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

/** Where web/export_neuflow.py writes the model (web/public/models, served next to the app). */
export const DEFAULT_FLOW_MODEL = "models/neuflow_mixed_960x544.onnx";

// The subset of the onnxruntime-web API used here.
interface OrtTensor {
  getData(): Promise<Float32Array>;
  dispose?(): void;
}
interface OrtSession {
  inputNames: readonly string[];
  inputMetadata?: unknown;
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensor>>;
  release(): Promise<void>;
}
interface Ort {
  InferenceSession: { create(model: string | ArrayBuffer | Uint8Array, options: object): Promise<OrtSession> };
  Tensor: new (type: "float32", data: Float32Array, dims: number[]) => unknown;
}

let ortPending: Promise<Ort> | null = null;

function loadOrt(): Promise<Ort> {
  ortPending ??= import(/* @vite-ignore */ ORT_URL) as Promise<Ort>;
  ortPending.catch(() => (ortPending = null));
  return ortPending;
}

/** Model input size from session metadata, else from the exported name `..._<W>x<H>[_fp16].onnx`. */
function modelSize(session: OrtSession, name: string): { w: number; h: number } {
  const meta = session.inputMetadata as { shape?: unknown[] }[] | Record<string, { shape?: unknown[] }> | undefined;
  const first = Array.isArray(meta) ? meta[0] : meta?.[session.inputNames[0]];
  const shape = first?.shape;
  if (shape?.length === 4 && shape.every((d) => typeof d === "number")) return { w: shape[3] as number, h: shape[2] as number };
  const m = /_(\d+)x(\d+)(?:_fp16)?\.onnx$/.exec(name);
  if (!m) throw new Error("Cannot tell the NeuFlow model's input size; keep the exported file name (…_<W>x<H>.onnx).");
  return { w: Number(m[1]), h: Number(m[2]) };
}

/** Bilinear resize of an (h, w, c) float image (cv2.INTER_LINEAR pixel centres). */
export function resizeFloat(src: Float32Array, w: number, h: number, c: number, nw: number, nh: number): Float32Array {
  const out = new Float32Array(nw * nh * c);
  const sx = w / nw;
  const sy = h / nh;
  for (let y = 0; y < nh; y++) {
    const fy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), h - 1);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, h - 1);
    const ty = fy - y0;
    for (let x = 0; x < nw; x++) {
      const fx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), w - 1);
      const x0 = Math.floor(fx);
      const x1 = Math.min(x0 + 1, w - 1);
      const tx = fx - x0;
      for (let ch = 0; ch < c; ch++) {
        const a = src[(y0 * w + x0) * c + ch];
        const b = src[(y0 * w + x1) * c + ch];
        const d = src[(y1 * w + x0) * c + ch];
        const e = src[(y1 * w + x1) * c + ch];
        out[(y * nw + x) * c + ch] = (a * (1 - tx) + b * tx) * (1 - ty) + (d * (1 - tx) + e * tx) * ty;
      }
    }
  }
  return out;
}

/**
 * NeuFlow flow between consecutive frames. Like the Python path, frames within 32 px of the
 * model size are edge-padded to it; other sizes are resized and the flow vectors rescaled.
 * Flows are cached: the passes and the pass blending ask for the same transitions.
 */
export class NeuFlow implements FlowProvider {
  private cache = new Map<number, FlowField>();

  private constructor(
    private ort: Ort,
    private session: OrtSession,
    readonly modelWidth: number,
    readonly modelHeight: number,
    private video: Frames,
  ) {}

  /** `model`: URL or bytes of the ONNX file; `name` is its file name (for the input size). */
  static async create(model: string | ArrayBuffer, name: string, video: Frames): Promise<NeuFlow> {
    if (typeof model === "string" && !(await fetch(model, { method: "HEAD" })).ok) {
      throw new Error(`No NeuFlow model at ${model}: export it with web/export_neuflow.py, or choose an .onnx file.`);
    }
    const ort = await loadOrt();
    const session = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
    const { w, h } = modelSize(session, name);
    return new NeuFlow(ort, session, w, h, video);
  }

  async forward(i: number): Promise<FlowField> {
    const hit = this.cache.get(i);
    if (hit) return hit;
    const { frames, width, height } = this.video;
    if (i < 0 || i + 1 >= frames.length) throw new Error(`No flow from frame ${i}: the video has ${frames.length} frames`);
    const flow = { width, height, data: await this.flow(frames[i], frames[i + 1]) };
    this.cache.set(i, flow);
    return flow;
  }

  async release(): Promise<void> {
    this.cache.clear();
    await this.session.release();
  }

  private get padded(): boolean {
    const { width: w, height: h } = this.video;
    const { modelWidth: mw, modelHeight: mh } = this;
    return w <= mw && h <= mh && mw - w < 32 && mh - h < 32;
  }

  /** RGB frame -> model input (1, 3, H, W) BGR floats in 0..255. */
  private input(rgb: Uint8Array): unknown {
    const { width: w, height: h } = this.video;
    const { modelWidth: mw, modelHeight: mh } = this;
    const pad = this.padded;
    let src: Uint8Array | Uint8ClampedArray = rgb;
    let channels = 3;
    let sw = w;
    if (!pad) {
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let p = 0; p < w * h; p++) {
        rgba[4 * p] = rgb[3 * p];
        rgba[4 * p + 1] = rgb[3 * p + 1];
        rgba[4 * p + 2] = rgb[3 * p + 2];
        rgba[4 * p + 3] = 255;
      }
      const canvas = new OffscreenCanvas(w, h);
      canvas.getContext("2d")!.putImageData(new ImageData(rgba, w, h), 0, 0);
      const scaled = new OffscreenCanvas(mw, mh);
      const ctx = scaled.getContext("2d", { willReadFrequently: true })!;
      ctx.drawImage(canvas, 0, 0, mw, mh);
      src = ctx.getImageData(0, 0, mw, mh).data;
      channels = 4;
      sw = mw;
    }
    const plane = mw * mh;
    const data = new Float32Array(3 * plane);
    for (let y = 0; y < mh; y++) {
      // Edge padding (np.pad mode="edge") on the bottom and right.
      const yy = pad ? Math.min(y, h - 1) : y;
      for (let x = 0; x < mw; x++) {
        const xx = pad ? Math.min(x, w - 1) : x;
        const s = channels * (yy * sw + xx);
        const d = y * mw + x;
        data[d] = src[s + 2];
        data[plane + d] = src[s + 1];
        data[2 * plane + d] = src[s];
      }
    }
    return new this.ort.Tensor("float32", data, [1, 3, mh, mw]);
  }

  /** (h, w, 2) flow from frame `a` to frame `b`. */
  private async flow(a: Uint8Array, b: Uint8Array): Promise<Float32Array> {
    const { width: w, height: h } = this.video;
    const { modelWidth: mw, modelHeight: mh } = this;
    const out = await this.session.run({ img0: this.input(a), img1: this.input(b) });
    const planes = await out.flow.getData();
    out.flow.dispose?.();
    const full = new Float32Array(mw * mh * 2);
    for (let i = 0; i < mw * mh; i++) {
      full[2 * i] = planes[i];
      full[2 * i + 1] = planes[mw * mh + i];
    }
    if (this.padded) {
      const flow = new Float32Array(w * h * 2);
      for (let y = 0; y < h; y++) flow.set(full.subarray(2 * y * mw, 2 * (y * mw + w)), 2 * y * w);
      return flow;
    }
    const flow = resizeFloat(full, mw, mh, 2, w, h);
    const fx = w / mw;
    const fy = h / mh;
    for (let i = 0; i < w * h; i++) {
      flow[2 * i] *= fx;
      flow[2 * i + 1] *= fy;
    }
    return flow;
  }
}
