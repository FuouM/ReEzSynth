import assert from "node:assert/strict";
import { test } from "node:test";
import { fitSize, naturalCompare, rgbaToRgb, rgbToRgba, trailingNumber } from "../src/core/image.ts";

test("fitSize keeps aspect, shrinks only, and rounds to even", () => {
  assert.deepEqual(fitSize(1920, 1080, 768), { width: 768, height: 432 });
  assert.deepEqual(fitSize(640, 360, 768), { width: 640, height: 360 });
  assert.deepEqual(fitSize(1001, 501, 0), { width: 1000, height: 500 });
  assert.deepEqual(fitSize(1001, 501, 0, false), { width: 1001, height: 501 });
});

test("RGB/RGBA round trip", () => {
  const rgba = new Uint8Array([1, 2, 3, 9, 4, 5, 6, 9]);
  const rgb = rgbaToRgb(rgba);
  assert.deepEqual([...rgb], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...rgbToRgba(rgb)], [1, 2, 3, 255, 4, 5, 6, 255]);
});

test("file name numbers and natural order", () => {
  assert.equal(trailingNumber("shot_0012.png"), 12);
  assert.equal(trailingNumber("v2_frame0003.jpg"), 3);
  assert.equal(trailingNumber("style.png"), null);
  assert.deepEqual(["f10.png", "f2.png", "f1.png"].sort(naturalCompare), ["f1.png", "f2.png", "f10.png"]);
});
