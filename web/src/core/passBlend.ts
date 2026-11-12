// Blending of the forward and backward EbSynth passes between two keyframes, ported from
// ezsynth/utils/blend_utils.py (Blender) and blend_logic.py (hist_blender, poisson_fusion_cpu):
//
// 1. Per pixel, pick the pass with the lower synthesis error (0 = forward, 1 = backward).
// 2. With optical flow, carry "backward" choices forward along the flow so a region that
//    switched to the backward pass stays switched.
// 3. Histogram-blend the two passes in Lab space under that mask.
// 4. Poisson-fuse: keep the gradients of the chosen pass and the colours of the histogram
//    blend (a least-squares solve; here conjugate gradients on the normal equations, which
//    reach the same solution as SciPy's lsqr).
//
// One deliberate difference: Blender.run pairs frame i with the error maps of its neighbour
// ("replicating old logic") and blends the left keyframe itself; here every frame is paired
// with its own error maps and keyframes are left as painted.

import { labToRgb, rgbToLab } from "./lab.ts";

/** 1 where the backward pass has the lower (or equal) error, like np.where(fwd < bwd, 0, 1). */
export function selectionMask(errFwd: Float32Array, errBwd: Float32Array): Uint8Array {
  const out = new Uint8Array(errFwd.length);
  for (let i = 0; i < out.length; i++) out[i] = errFwd[i] < errBwd[i] ? 0 : 1;
  return out;
}

function meanStd(lab: ArrayLike<number>, c: number): [number, number] {
  let s = 0;
  let n = 0;
  for (let i = c; i < lab.length; i += 3, n++) s += lab[i];
  const mean = s / n;
  let v = 0;
  for (let i = c; i < lab.length; i += 3) v += (lab[i] - mean) ** 2;
  return [mean, Math.sqrt(v / n)];
}

/** blend_logic.hist_blender (CPU path) on RGB bytes; `mask` is per pixel (0 = a, 1 = b). */
export function histBlend(a: Uint8Array, b: Uint8Array, mask: Uint8Array, w1 = 0.5, w2 = 0.5): Uint8Array {
  const aLab = rgbToLab(a);
  const bLab = rgbToLab(b);
  const minLab = new Uint8Array(aLab.length);
  for (let p = 0, i = 0; p < mask.length; p++, i += 3) {
    const src = mask[p] === 0 ? aLab : bLab;
    minLab[i] = src[i];
    minLab[i + 1] = src[i + 1];
    minLab[i + 2] = src[i + 2];
  }
  const tMean = 0.5 * 256;
  const tStd = (1 / 36) * 256;
  const ab = new Float32Array(aLab.length);
  const out = new Uint8Array(aLab.length);
  for (let c = 0; c < 3; c++) {
    const [am, as] = meanStd(aLab, c);
    const [bm, bs] = meanStd(bLab, c);
    for (let i = c; i < ab.length; i += 3) {
      const an = Math.fround(((aLab[i] - am) * tStd) / (as + 1e-6) + tMean);
      const bn = Math.fround(((bLab[i] - bm) * tStd) / (bs + 1e-6) + tMean);
      ab[i] = (an * w1 + bn * w2 - 128) / 0.5 + 128;
    }
    const [abm, abs] = meanStd(ab, c);
    const [mm, ms] = meanStd(minLab, c);
    for (let i = c; i < ab.length; i += 3) {
      const v = ((ab[i] - abm) * ms) / (abs + 1e-6) + mm;
      out[i] = Math.min(255, Math.max(0, Math.round(v)));
    }
  }
  return labToRgb(out);
}

/**
 * Solve (k^2 (Gx'Gx + Gy'Gy) + I) u = k^2 (Gx'gx + Gy'gy) + c for one channel with conjugate
 * gradients. Gx and Gy are built on flat indices exactly like construct_A_cpu: Gx row i
 * (i < (h-1)w) is u[i] - u[i+w]; Gy row i (i < h(w-1)) is u[i] - u[i+1].
 */
export function solveChannel(gx: Float32Array, gy: Float32Array, c: Float32Array, k: number, h: number, w: number, maxIter: number, tol: number): Float32Array {
  const n = h * w;
  const nx = (h - 1) * w;
  const ny = h * (w - 1);
  const k2 = k * k;
  const apply = (u: Float64Array, out: Float64Array) => {
    for (let i = 0; i < n; i++) out[i] = u[i];
    for (let i = 0; i < nx; i++) {
      const d = k2 * (u[i] - u[i + w]);
      out[i] += d;
      out[i + w] -= d;
    }
    for (let i = 0; i < ny; i++) {
      const d = k2 * (u[i] - u[i + 1]);
      out[i] += d;
      out[i + 1] -= d;
    }
  };
  const rhs = new Float64Array(n);
  for (let i = 0; i < n; i++) rhs[i] = c[i];
  for (let i = 0; i < nx; i++) {
    rhs[i] += k2 * gx[i];
    rhs[i + w] -= k2 * gx[i];
  }
  for (let i = 0; i < ny; i++) {
    rhs[i] += k2 * gy[i];
    rhs[i + 1] -= k2 * gy[i];
  }
  // Start from the target colours: the identity rows already pull towards them.
  const u = Float64Array.from(c);
  const r = new Float64Array(n);
  const p = new Float64Array(n);
  const q = new Float64Array(n);
  apply(u, q);
  let rr = 0;
  let bb = 0;
  for (let i = 0; i < n; i++) {
    r[i] = rhs[i] - q[i];
    p[i] = r[i];
    rr += r[i] * r[i];
    bb += rhs[i] * rhs[i];
  }
  const stop = tol * tol * Math.max(bb, 1e-30);
  for (let it = 0; it < maxIter && rr > stop; it++) {
    apply(p, q);
    let pq = 0;
    for (let i = 0; i < n; i++) pq += p[i] * q[i];
    const alpha = rr / pq;
    let rr2 = 0;
    for (let i = 0; i < n; i++) {
      u[i] += alpha * p[i];
      r[i] -= alpha * q[i];
      rr2 += r[i] * r[i];
    }
    const beta = rr2 / rr;
    rr = rr2;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
  }
  return Float32Array.from(u);
}

export interface PoissonOptions {
  /** Gradient weights for L, a, b (BlendingConfig.poisson_grad_weight_l/_ab). */
  gradWeights?: [number, number, number];
  maxIter?: number;
  /** Relative residual at which CG stops. */
  tol?: number;
}

/** blend_logic.poisson_fusion_cpu on RGB bytes. */
export function poissonFusion(blend: Uint8Array, a: Uint8Array, b: Uint8Array, mask: Uint8Array, width: number, height: number, opts: PoissonOptions = {}): Uint8Array {
  const weights = opts.gradWeights ?? [2.5, 0.5, 0.5];
  const maxIter = opts.maxIter ?? 2000;
  const tol = opts.tol ?? 1e-6;
  // Python converts the uint8 Lab images to float32 (the rounded 8-bit values).
  const lab = (x: Uint8Array) => Float32Array.from(rgbToLab(x));
  const Iab = lab(blend);
  const Ia = lab(a);
  const Ib = lab(b);
  const n = width * height;
  const out = new Uint8Array(n * 3);
  for (let ch = 0; ch < 3; ch++) {
    const gx = new Float32Array(n);
    const gy = new Float32Array(n);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const p = y * width + x;
        const m = mask[p] > 0 ? 1 : 0;
        const src = m ? Ib : Ia;
        if (y < height - 1) gx[p] = Math.min(100, Math.max(-100, src[p * 3 + ch] - src[(p + width) * 3 + ch]));
        if (x < width - 1) gy[p] = Math.min(100, Math.max(-100, src[p * 3 + ch] - src[(p + 1) * 3 + ch]));
      }
    }
    let mean = 0;
    for (let p = 0; p < n; p++) mean += Iab[p * 3 + ch];
    mean /= n;
    const centered = new Float32Array(n);
    for (let p = 0; p < n; p++) centered[p] = Iab[p * 3 + ch] - mean;
    const u = solveChannel(gx, gy, centered, weights[ch], height, width, maxIter, tol);
    // np.clip(final, 0, 255).astype(np.uint8) truncates.
    for (let p = 0; p < n; p++) out[p * 3 + ch] = Math.trunc(Math.min(255, Math.max(0, u[p] + mean)));
  }
  return labToRgb(out);
}

/** Bilinear backward warp of a one-channel float map (Warp.run_warping_float_map). */
export type MaskWarp = (prev: Float32Array, transition: number) => Promise<Float32Array>;

export interface BlendInput {
  width: number;
  height: number;
  /** Frames from+1 .. to-1 of each pass, in frame order. */
  fwd: Uint8Array[];
  bwd: Uint8Array[];
  errFwd: Float32Array[];
  errBwd: Float32Array[];
  /** First frame index of the lists (the frame after the left keyframe). */
  first: number;
  /** Warp the previous frame's mask along flow `first + i - 1` (omit without flow). */
  warpMask?: MaskWarp;
  poisson?: PoissonOptions | false;
}

/** Blender.run for the frames between two keyframes. */
export async function blendPasses(input: BlendInput, onFrame?: (i: number) => void): Promise<Uint8Array[]> {
  const { width, height, fwd, bwd, errFwd, errBwd } = input;
  const out: Uint8Array[] = [];
  let prev: Uint8Array | null = null;
  for (let i = 0; i < fwd.length; i++) {
    let mask = selectionMask(errFwd[i], errBwd[i]);
    if (prev && input.warpMask) {
      const warped = await input.warpMask(Float32Array.from(prev), input.first + i - 1);
      const m = new Uint8Array(mask.length);
      for (let p = 0; p < m.length; p++) m[p] = warped[p] > 0.5 && mask[p] === 0 ? 1 : mask[p];
      mask = m;
    }
    prev = mask;
    const hist = histBlend(fwd[i], bwd[i], mask);
    out.push(input.poisson === false ? hist : poissonFusion(hist, fwd[i], bwd[i], mask, width, height, input.poisson ?? {}));
    onFrame?.(i);
  }
  return out;
}

