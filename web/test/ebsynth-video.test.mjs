import assert from "node:assert/strict";
import { test } from "node:test";
import fx from "./fixtures-ebsynth-video.json" with { type: "json" };
import {
  Cancelled,
  classicEdge,
  coordsToRgb,
  DEFAULT_WEIGHTS,
  planSequences,
  positionalFromFlow,
  pristineCoords,
  synthesisCount,
  synthesizeVideo,
  warpRgb,
} from "../src/core/ebsynthVideo.ts";

// ── Parity with ezsynth (fixtures from scripts/gen-parity-fixtures.py) ──────────────────────
const { width: W, height: H } = fx;
const rgb = Uint8Array.from(fx.rgb);
const flow = { width: W, height: H, data: Float32Array.from(fx.flow) };

test("classic edge matches ezsynth.edge.classic", () => {
  const e = classicEdge(rgb, W, H);
  assert.deepEqual([...e].filter((_, i) => i % 3 === 0), fx.edge);
});

test("warp matches Warp.run_warping", () => {
  assert.deepEqual([...warpRgb(rgb, flow, -1)], fx.warpedRgbNegFlow);
});

test("positional guides match PositionalGuide", () => {
  assert.deepEqual([...coordsToRgb(pristineCoords(W, H))], fx.pristine);
  assert.deepEqual([...positionalFromFlow(pristineCoords(W, H), flow)], fx.posFromFlow);
});

// ── Scheduling ────────────────────────────────────────────────────────────────────────────
test("sequences follow create_sequences", () => {
  assert.deepEqual(planSequences([3, 6], 10), [
    { mode: "reverse", from: 3, to: 0 },
    { mode: "blend", from: 3, to: 6 },
    { mode: "forward", from: 6, to: 9 },
  ]);
  assert.deepEqual(planSequences([0], 4), [{ mode: "forward", from: 0, to: 3 }]);
  // Adjacent keyframes leave nothing to blend.
  assert.deepEqual(planSequences([0, 1], 2), []);
  assert.throws(() => planSequences([], 3));
  assert.throws(() => planSequences([5], 3));
});

test("synthesis count covers both passes of a blend", () => {
  assert.equal(synthesisCount(planSequences([3, 6], 10)), 3 + 2 * 2 + 3);
});

// ── Video synthesis against a fake synthesizer ────────────────────────────────────────────
const frame = (v, len = 12) => new Uint8Array(len).fill(v);
const video = (n) => ({ width: 2, height: 2, frames: Array.from({ length: n }, (_, i) => frame(i)) });

test("each frame comes from the keyframe of its pass, blended by distance between keyframes", async () => {
  const calls = [];
  // The fake returns the style, so every frame equals its keyframe's colour.
  const synth = async (style, guides) => {
    calls.push(guides.map((g) => g.weight));
    return { image: style };
  };
  const keys = new Map([
    [1, frame(100)],
    [5, frame(200)],
  ]);
  const seen = [];
  const out = await synthesizeVideo(video(7), keys, synth, { weights: DEFAULT_WEIGHTS, onProgress: (d, t) => seen.push([d, t]) });
  assert.deepEqual(out.map((f) => f[0]), [100, 100, 125, 150, 175, 200, 200]);
  assert.equal(out[1], keys.get(1));
  // Without flow: edge and image guides only.
  assert.ok(calls.every((w) => w.length === 2 && w[0] === 1 && w[1] === 6));
  assert.deepEqual(seen.at(-1), [synthesisCount(planSequences([1, 5], 7)), synthesisCount(planSequences([1, 5], 7))]);
});

test("with flow, passes add positional and warp guides and ask for the right transitions", async () => {
  const asked = [];
  const zero = { forward: async (i) => (asked.push(i), { width: 2, height: 2, data: new Float32Array(8) }) };
  let guides = 0;
  const synth = async (style, g) => ((guides = g.length), { image: style });
  await synthesizeVideo(video(4), new Map([[1, frame(9)]]), synth, { weights: DEFAULT_WEIGHTS, flow: zero });
  assert.equal(guides, 4);
  // Reverse pass 1→0 uses flow 0 (0→1); forward pass 1→2→3 uses flows 1 and 2.
  assert.deepEqual(asked, [0, 1, 2]);
});

test("the warp guide is the previous stylized frame, not the keyframe", async () => {
  const targets = [];
  let n = 0;
  const synth = async (style, g) => {
    targets.push(g[3].target[0]);
    return { image: frame(50 + ++n) };
  };
  const zero = { forward: async () => ({ width: 2, height: 2, data: new Float32Array(8) }) };
  await synthesizeVideo(video(4), new Map([[0, frame(10)]]), synth, { weights: DEFAULT_WEIGHTS, flow: zero });
  // Zero flow leaves the image unchanged up to the 0..1 round trip (truncation).
  assert.deepEqual(targets, [10, 51, 52].map((v) => Math.trunc((v / 255) * 255)));
});

test("cancelling stops between frames", async () => {
  const ac = new AbortController();
  const synth = async (style) => (ac.abort(), { image: style });
  await assert.rejects(synthesizeVideo(video(5), new Map([[0, frame(1)]]), synth, { weights: DEFAULT_WEIGHTS, signal: ac.signal }), Cancelled);
});

test("with error maps, frames between keyframes take the lower-error pass", async () => {
  // Forward pass outputs 40 with error 1; backward outputs 220 with error 5: forward wins.
  const synth = async (style) => ({ image: style[0] === 10 ? frame(40) : frame(220), error: new Float32Array(4).fill(style[0] === 10 ? 1 : 5) });
  const keys = new Map([
    [0, frame(10)],
    [3, frame(250)],
  ]);
  const out = await synthesizeVideo(video(4), keys, synth, { weights: DEFAULT_WEIGHTS, blend: "histogram" });
  // Histogram blending keeps the chosen pass's statistics: a uniform 40 stays 40.
  assert.deepEqual([out[1][0], out[2][0]], [40, 40]);
  const dist = await synthesizeVideo(video(4), keys, synth, { weights: DEFAULT_WEIGHTS, blend: "distance" });
  assert.notEqual(dist[1][0], 40);
});
