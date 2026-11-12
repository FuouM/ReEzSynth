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
