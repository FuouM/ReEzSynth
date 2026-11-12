// The WebGPU engine: one wasm module and one device, shared by every mode.
import init, { WebSynth, type WebFastBlend } from "../pkg/ebsynth_web.js";

export interface Engine {
  synth: WebSynth;
  /** "<adapter> (<backend>)" */
  adapter: string;
  /** Largest storage-buffer binding, in bytes. */
  maxBinding: number;
}

let pending: Promise<Engine> | null = null;

export function loadEngine(): Promise<Engine> {
  pending ??= (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("This browser has no WebGPU. Use a current Chrome or Edge, or Firefox/Safari with WebGPU enabled; the page must be served from localhost or https.");
    }
    await init();
    const synth = await WebSynth.create();
    return { synth, adapter: synth.adapter, maxBinding: synth.maxStorageBinding };
  })();
  pending.catch(() => (pending = null));
  return pending;
}

export interface FastBlendOptions {
  minimumPatchSize: number;
  numIter: number;
  guideWeight: number;
  usePairwisePatchError?: boolean;
  useMeanTargetStyle?: boolean;
  initialize?: "identity" | "random";
}

export async function createFastBlend(engine: Engine, height: number, width: number, options: FastBlendOptions): Promise<WebFastBlend> {
  return (await engine.synth.fastBlend(height, width, options)) as WebFastBlend;
}

/**
 * Frames per FastBlend call that fit both the device's binding limit and a ~1 GiB wasm
 * working set (three u8 inputs, their f32 copies and the f32 output per frame).
 */
export function fastBlendBatchLimit(engine: Engine, width: number, height: number): number {
  const pad = 32; // reflect padding of the coarsest patch, rounded up
  const paddedFrame = (width + 2 * pad) * (height + 2 * pad) * 3 * 4;
  const byBinding = Math.floor(engine.maxBinding / paddedFrame);
  const byWasm = Math.floor(2 ** 30 / (width * height * 3 * 19));
  return Math.max(1, Math.min(64, byBinding, byWasm));
}

function rgba(rgb: Uint8Array): Uint8Array {
  const n = rgb.length / 3;
  const out = new Uint8Array(n * 4);
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    out[i * 4] = rgb[j];
    out[i * 4 + 1] = rgb[j + 1];
    out[i * 4 + 2] = rgb[j + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

function rgb(rgbaBytes: Uint8Array): Uint8Array {
  const n = rgbaBytes.length >> 2;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i * 3] = rgbaBytes[i * 4];
    out[i * 3 + 1] = rgbaBytes[i * 4 + 1];
    out[i * 3 + 2] = rgbaBytes[i * 4 + 2];
  }
  return out;
}

export interface RgbGuide {
  source: Uint8Array;
  target: Uint8Array;
  weight: number;
}

/**
 * EbSynth on RGB bytes: `style` and guide sources are `sw x sh`, guide targets `tw x th`.
 * Resolves to the `tw x th` RGB result.
 */
export async function synthesizeRgb(
  engine: Engine,
  style: Uint8Array,
  sw: number,
  sh: number,
  guides: RgbGuide[],
  tw: number,
  th: number,
  params: object,
  progress?: (level: number, levels: number) => void,
): Promise<Uint8Array> {
  const out: Uint8Array = await engine.synth.synthesize(
    rgba(style),
    sw,
    sh,
    guides.map((g) => rgba(g.source)),
    guides.map((g) => rgba(g.target)),
    new Float32Array(guides.map((g) => g.weight)),
    tw,
    th,
    { ...params },
    progress ?? null,
  );
  return rgb(out);
}

/** synthesizeRgb that also returns the per-pixel patch error (target size). */
export async function synthesizeRgbWithError(
  engine: Engine,
  style: Uint8Array,
  sw: number,
  sh: number,
  guides: RgbGuide[],
  tw: number,
  th: number,
  params: object,
): Promise<{ image: Uint8Array; error: Float32Array }> {
  const out = (await engine.synth.synthesizeWithError(
    rgba(style),
    sw,
    sh,
    guides.map((g) => rgba(g.source)),
    guides.map((g) => rgba(g.target)),
    new Float32Array(guides.map((g) => g.weight)),
    tw,
    th,
    { ...params },
    null,
  )) as { image: Uint8Array; error: Float32Array };
  return { image: rgb(out.image), error: out.error };
}
