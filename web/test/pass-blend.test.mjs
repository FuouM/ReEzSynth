// Pass blending against ezsynth's own hist_blender / poisson_fusion_cpu / construct_A_cpu
// (fixtures from scripts/gen-parity-fixtures.py).
import assert from "node:assert/strict";
import { test } from "node:test";
import fx from "./fixtures-pass-blend.json" with { type: "json" };
import { labToRgb, rgbToLab } from "../src/core/lab.ts";
import { blendPasses, histBlend, poissonFusion, selectionMask, solveChannel } from "../src/core/passBlend.ts";

function diff(a, b) {
  let max = 0;
  let sum = 0;
  for (let i = 0; i < b.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    max = Math.max(max, d);
    sum += d;
  }
  return { max, mean: sum / b.length };
}

test("Poisson solve matches the exact least-squares solution of construct_A_cpu", () => {
  const s = fx.solver;
  const u = solveChannel(Float32Array.from(s.gx), Float32Array.from(s.gy), Float32Array.from(s.c), s.k, s.h, s.w, 5000, 1e-10);
  assert.ok(diff(u, s.u).max < 1e-4);
});

test("8-bit Lab is within OpenCV's lookup-table rounding", () => {
  const d1 = diff(rgbToLab(Uint8Array.from(fx.labRgb)), fx.lab);
  const d2 = diff(labToRgb(Uint8Array.from(fx.labIn)), fx.labBack);
  assert.ok(d1.max <= 2 && d1.mean < 0.15, JSON.stringify(d1));
  assert.ok(d2.max <= 1 && d2.mean < 0.05, JSON.stringify(d2));
});

const a = Uint8Array.from(fx.a);
const b = Uint8Array.from(fx.b);
const mask = Uint8Array.from(fx.mask);

test("histogram blend and Poisson fusion stay within a few levels of ezsynth", () => {
  const hist = histBlend(a, b, mask);
  const dh = diff(hist, fx.hist);
  assert.ok(dh.max <= 6 && dh.mean < 1, JSON.stringify(dh));
  const fused = poissonFusion(Uint8Array.from(fx.hist), a, b, mask, fx.width, fx.height);
  const df = diff(fused, fx.fused);
  assert.ok(df.max <= 6 && df.mean < 1, JSON.stringify(df));
});

test("selection picks the pass with the lower error; ties go backward", () => {
  assert.deepEqual([...selectionMask(Float32Array.of(1, 5, 2), Float32Array.of(2, 4, 2))], [0, 1, 1]);
});

test("with flow, a backward choice carries forward until the forward pass wins outright", async () => {
  const one = (v) => new Uint8Array(12).fill(v);
  const errs = (f, bw) => [Float32Array.of(...f), Float32Array.of(...bw)];
  const [f0, b0] = errs([5, 0, 0, 0], [1, 9, 9, 9]); // frame 0: pixel 0 backward
  const [f1, b1] = errs([0, 0, 0, 0], [9, 9, 9, 9]); // frame 1: all forward on their own
  const masks = [];
  const identity = async (prev) => prev; // zero flow
  const input = { width: 2, height: 2, fwd: [one(10), one(10)], bwd: [one(200), one(200)], errFwd: [f0, f1], errBwd: [b0, b1], first: 1, warpMask: identity, poisson: false };
  const out = await blendPasses(input, (i) => masks.push(i));
  assert.equal(out.length, 2);
  // Without the warp the second frame would be all forward (10); the carried pixel keeps it from being uniform.
  const noWarp = await blendPasses({ ...input, warpMask: undefined });
  assert.notDeepEqual([...out[1]], [...noWarp[1]]);
});
