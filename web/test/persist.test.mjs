import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeSaved } from "../src/core/persist.ts";

const defaults = { patchSize: 7, costFunction: "ssd", useBilateral: false };

test("saved values of the right type override defaults", () => {
  assert.deepEqual(mergeSaved(defaults, { patchSize: 9, costFunction: "ncc", useBilateral: true }), { patchSize: 9, costFunction: "ncc", useBilateral: true });
});

test("wrong types, unknown keys, non-finite numbers and junk are ignored", () => {
  assert.deepEqual(mergeSaved(defaults, { patchSize: "9", extra: 1, useBilateral: null }), defaults);
  assert.deepEqual(mergeSaved(defaults, { patchSize: null }), defaults);
  assert.deepEqual(mergeSaved(defaults, "garbage"), defaults);
  assert.deepEqual(mergeSaved(defaults, null), defaults);
});

test("missing keys keep their defaults (settings added in a later version)", () => {
  assert.deepEqual(mergeSaved(defaults, { patchSize: 5 }), { ...defaults, patchSize: 5 });
});
