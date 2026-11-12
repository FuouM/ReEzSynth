// Runner logic against a fake matcher that returns each source style unchanged, so blends are
// plain averages that can be checked exactly.
import assert from "node:assert/strict";
import { test } from "node:test";
import { blendSequence, interpolateKeyframes, interpolationTasks, keyframeWeights, Cancelled } from "../src/core/fastblend.ts";

const frame = (v, len = 6) => new Uint8Array(len).fill(v);

function fakeMatcher(log = []) {
  return {
    async estimate(batch, sg, tg, ss) {
      assert.equal(sg.length, tg.length);
      assert.equal(sg.length, ss.length);
      assert.equal(sg.length % batch, 0);
      log.push(batch);
      return Float32Array.from(ss);
    },
  };
}

test("blend averages each frame with its window", async () => {
  const styles = [0, 30, 60, 90].map((v) => frame(v));
  const guides = styles.map(() => frame(0));
  const out = await blendSequence(fakeMatcher(), guides, styles, 1, { batchSize: 3 });
  assert.deepEqual([...out].map((f) => f[0]), [15, 30, 60, 75]);
});

test("blend covers every frame once with progress", async () => {
  const n = 7;
  const styles = Array.from({ length: n }, (_, i) => frame(i * 10));
  const seen = [];
  const out = await blendSequence(fakeMatcher(), styles, styles, 2, { batchSize: 4, onProgress: (d, t) => seen.push([d, t]) });
  assert.equal(out.length, n);
  assert.deepEqual(seen.at(-1), [n, n]);
  // Frame 3 averages frames 1..5 = (10+20+30+40+50)/5.
  assert.equal(out[3][0], 30);
});

test("single-frame blend returns the frame", async () => {
  const out = await blendSequence(fakeMatcher(), [frame(1)], [frame(42)], 5, { batchSize: 4 });
  assert.equal(out[0][0], 42);
});

test("blend stops when cancelled", async () => {
  const ac = new AbortController();
  ac.abort();
  const styles = [frame(0), frame(1)];
  await assert.rejects(blendSequence(fakeMatcher(), styles, styles, 1, { batchSize: 1, signal: ac.signal }), Cancelled);
});

test("keyframe weights favour the closer keyframe", () => {
  assert.deepEqual(keyframeWeights(0, 0, 10), [1, 0]);
  assert.deepEqual(keyframeWeights(0, 5, 10), [0.5, 0.5]);
  assert.deepEqual(keyframeWeights(4, 4, 4), [0.5, 0.5]);
  const [l, r] = keyframeWeights(0, 2, 10);
  assert.ok(Math.abs(l - 0.8) < 1e-9 && Math.abs(r - 0.2) < 1e-9);
});

test("interpolation tasks skip keyframes and clamp the ends", () => {
  assert.deepEqual(interpolationTasks([2, 5], 7), [
    [2, 0, 2],
    [2, 1, 2],
    [2, 3, 5],
    [2, 4, 5],
    [5, 6, 5],
  ]);
  assert.throws(() => interpolationTasks([9], 5));
  assert.throws(() => interpolationTasks([], 5));
});

test("interpolation blends keyframes by distance and keeps keyframes", async () => {
  const guides = Array.from({ length: 5 }, () => frame(0));
  const keys = new Map([
    [0, frame(0)],
    [4, frame(200)],
  ]);
  const batches = [];
  const out = await interpolateKeyframes(fakeMatcher(batches), guides, keys, { batchSize: 4 });
  assert.deepEqual([...out].map((f) => f[0]), [0, 50, 100, 150, 200]);
  assert.equal(out[4], keys.get(4));
  assert.ok(batches.every((b) => b % 2 === 0), "pairwise matching needs even batches");
});

test("interpolation rejects keyframes of the wrong size", async () => {
  await assert.rejects(interpolateKeyframes(fakeMatcher(), [frame(0), frame(0)], new Map([[0, frame(1, 3)]]), { batchSize: 2 }));
});
