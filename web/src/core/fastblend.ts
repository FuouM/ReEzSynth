// FastBlend runners, ported from FastBlend/src/balanced_runner.py and
// interpolation_runner.py. The patch matching itself runs in the wasm module (WebFastBlend);
// this file only plans the frame pairs and blends the remapped results. It has no imports so
// it can be unit-tested in Node against a fake matcher.

/** What the runners need from a matcher: remap each source style onto its target guide. */
export interface Matcher {
  /** `batch` RGB frames each, concatenated; resolves to batch*frame float32 values (0..255). */
  estimate(batch: number, sourceGuide: Uint8Array, targetGuide: Uint8Array, sourceStyle: Uint8Array): Promise<Float32Array>;
}

export interface RunOptions {
  /** Frames per matcher call. */
  batchSize: number;
  signal?: AbortSignal;
  /** Finished output frames so far, out of the frames this run produces. */
  onProgress?: (done: number, total: number) => void;
}

export class Cancelled extends Error {
  constructor() {
    super("Cancelled");
    this.name = "Cancelled";
  }
}

function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Cancelled();
}

function pack(frames: Uint8Array[], frameLen: number): Uint8Array {
  const out = new Uint8Array(frames.length * frameLen);
  frames.forEach((f, i) => out.set(f, i * frameLen));
  return out;
}

function toBytes(values: Float32Array): Uint8Array {
  // Uint8ClampedArray clamps to 0..255 and rounds.
  return new Uint8Array(Uint8ClampedArray.from(values).buffer);
}

/**
 * Deflicker a stylized sequence (FastBlend "balanced" mode): every frame becomes the average
 * of itself and the styles of its `windowSize` neighbours on each side, remapped onto it.
 */
export async function blendSequence(
  matcher: Matcher,
  guides: Uint8Array[],
  styles: Uint8Array[],
  windowSize: number,
  opts: RunOptions,
): Promise<Uint8Array[]> {
  const n = styles.length;
  if (guides.length !== n) throw new Error(`${guides.length} guide frames but ${n} style frames`);
  if (n === 0) return [];
  const frameLen = styles[0].length;
  const tasks: [source: number, target: number][] = [];
  for (let target = 0; target < n; target++) {
    for (let source = target - windowSize; source <= target + windowSize; source++) {
      if (source >= 0 && source < n && source !== target) tasks.push([source, target]);
    }
  }
  const output: (Uint8Array | null)[] = new Array(n).fill(null);
  const sums = new Map<number, { sum: Float32Array; count: number }>();
  let done = 0;
  const expected = (t: number) => Math.min(n, t + windowSize + 1) - Math.max(0, t - windowSize) - 1;
  const finish = (t: number) => {
    const acc = sums.get(t);
    const base = styles[t];
    const out = new Float32Array(frameLen);
    const k = 1 + (acc?.count ?? 0);
    for (let i = 0; i < frameLen; i++) out[i] = (base[i] + (acc ? acc.sum[i] : 0)) / k;
    sums.delete(t);
    output[t] = toBytes(out);
    done++;
    opts.onProgress?.(done, n);
  };
  // Frames with no neighbours (a one-frame sequence) are already final.
  for (let t = 0; t < n; t++) if (expected(t) === 0) finish(t);

  const batchSize = Math.max(1, opts.batchSize);
  for (let b = 0; b < tasks.length; b += batchSize) {
    checkAbort(opts.signal);
    const batch = tasks.slice(b, b + batchSize);
    const result = await matcher.estimate(
      batch.length,
      pack(batch.map(([s]) => guides[s]), frameLen),
      pack(batch.map(([, t]) => guides[t]), frameLen),
      pack(batch.map(([s]) => styles[s]), frameLen),
    );
    batch.forEach(([, t], i) => {
      let acc = sums.get(t);
      if (!acc) sums.set(t, (acc = { sum: new Float32Array(frameLen), count: 0 }));
      const r = result.subarray(i * frameLen, (i + 1) * frameLen);
      for (let j = 0; j < frameLen; j++) acc.sum[j] += r[j];
      acc.count++;
      if (acc.count >= expected(t)) finish(t);
    });
  }
  return output.map((f, i) => f ?? styles[i]);
}

/** Distance weights of the left and right keyframe for frame `m` (closer keyframe weighs more). */
export function keyframeWeights(l: number, m: number, r: number): [number, number] {
  const wl = Math.abs(m - r);
  const wr = Math.abs(m - l);
  return wl + wr === 0 ? [0.5, 0.5] : [wl / (wl + wr), wr / (wl + wr)];
}

/** (left keyframe, frame, right keyframe) for every frame that is not a keyframe. */
export function interpolationTasks(keyIndices: number[], n: number): [number, number, number][] {
  const keys = [...new Set(keyIndices)].sort((a, b) => a - b);
  if (keys.length === 0) throw new Error("No keyframes");
  if (keys[0] < 0 || keys[keys.length - 1] >= n) throw new Error(`Keyframe index out of range 0..${n - 1}`);
  const isKey = new Set(keys);
  const tasks: [number, number, number][] = [];
  for (let m = 0; m < n; m++) {
    if (isKey.has(m)) continue;
    let l = keys[0];
    let r = keys[keys.length - 1];
    for (const k of keys) {
      if (k <= m) l = k;
      if (k >= m) {
        r = k;
        break;
      }
    }
    // Before the first or after the last keyframe both sides are the same keyframe.
    if (m < keys[0]) l = r = keys[0];
    if (m > keys[keys.length - 1]) l = r = keys[keys.length - 1];
    tasks.push([l, m, r]);
  }
  return tasks;
}

/**
 * Propagate stylized keyframes to every frame (FastBlend interpolation mode): each frame is
 * the distance-weighted blend of its left and right keyframes remapped onto it. The matcher
 * must use pairwise patch error (the left/right pair is matched jointly), so each call gets an
 * even batch. Keyframes are returned unchanged.
 */
export async function interpolateKeyframes(
  matcher: Matcher,
  guides: Uint8Array[],
  keyframes: Map<number, Uint8Array>,
  opts: RunOptions,
): Promise<Uint8Array[]> {
  const n = guides.length;
  if (n === 0) return [];
  const frameLen = guides[0].length;
  for (const [i, k] of keyframes) {
    if (k.length !== frameLen) throw new Error(`Keyframe ${i} is not the size of the video frames`);
  }
  const tasks = interpolationTasks([...keyframes.keys()], n);
  const output: (Uint8Array | null)[] = new Array(n).fill(null);
  for (const [i, k] of keyframes) output[i] = k;
  const perCall = Math.max(1, Math.floor(opts.batchSize / 2));
  let done = 0;
  for (let b = 0; b < tasks.length; b += perCall) {
    checkAbort(opts.signal);
    const batch = tasks.slice(b, b + perCall);
    const sg: Uint8Array[] = [];
    const tg: Uint8Array[] = [];
    const ss: Uint8Array[] = [];
    for (const [l, m, r] of batch) {
      sg.push(guides[l], guides[r]);
      tg.push(guides[m], guides[m]);
      ss.push(keyframes.get(l)!, keyframes.get(r)!);
    }
    const result = await matcher.estimate(batch.length * 2, pack(sg, frameLen), pack(tg, frameLen), pack(ss, frameLen));
    batch.forEach(([l, m, r], i) => {
      const [wl, wr] = keyframeWeights(l, m, r);
      const a = result.subarray(2 * i * frameLen, (2 * i + 1) * frameLen);
      const c = result.subarray((2 * i + 1) * frameLen, (2 * i + 2) * frameLen);
      const out = new Float32Array(frameLen);
      for (let j = 0; j < frameLen; j++) out[j] = a[j] * wl + c[j] * wr;
      output[m] = toBytes(out);
      done++;
      opts.onProgress?.(done, tasks.length);
    });
  }
  return output as Uint8Array[];
}
