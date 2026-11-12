// Image decoding, resizing and conversion. Frames are packed RGB bytes (width*height*3).

export interface Rgb {
  width: number;
  height: number;
  /** Packed RGB, width*height*3. */
  data: Uint8Array;
}

export function rgbaToRgb(rgba: Uint8Array | Uint8ClampedArray): Uint8Array {
  const n = rgba.length >> 2;
  const out = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const k = i << 2;
    out[j] = rgba[k];
    out[j + 1] = rgba[k + 1];
    out[j + 2] = rgba[k + 2];
  }
  return out;
}

export function rgbToRgba(rgb: Uint8Array): Uint8ClampedArray<ArrayBuffer> {
  const n = rgb.length / 3;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    const k = i << 2;
    out[k] = rgb[j];
    out[k + 1] = rgb[j + 1];
    out[k + 2] = rgb[j + 2];
    out[k + 3] = 255;
  }
  return out;
}

/** Size that fits in `maxSide` (keeping the aspect), rounded to even numbers for video codecs. */
export function fitSize(width: number, height: number, maxSide: number, even = true): { width: number; height: number } {
  const s = maxSide > 0 ? Math.min(1, maxSide / Math.max(width, height)) : 1;
  let w = Math.max(2, Math.round(width * s));
  let h = Math.max(2, Math.round(height * s));
  if (even) {
    w -= w % 2;
    h -= h % 2;
  }
  return { width: w, height: h };
}

/** Pixels of a drawable at `width x height` as RGB. */
export function rasterize(source: CanvasImageSource, width: number, height: number): Uint8Array {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, width, height);
  return rgbaToRgb(ctx.getImageData(0, 0, width, height).data);
}

export type SizeRule = { maxSide: number } | { width: number; height: number } | null;

/** Decode an image file. `size`: null keeps it, maxSide shrinks it, width/height forces it. */
export async function decodeImage(file: Blob, size: SizeRule = null): Promise<Rgb> {
  const bitmap = await createImageBitmap(file);
  try {
    let { width, height } = bitmap;
    if (size && "maxSide" in size) ({ width, height } = fitSize(width, height, size.maxSide, false));
    else if (size) ({ width, height } = size);
    return { width, height, data: rasterize(bitmap, width, height) };
  } finally {
    bitmap.close();
  }
}

export function drawRgb(canvas: HTMLCanvasElement, img: Rgb): void {
  if (canvas.width !== img.width) canvas.width = img.width;
  if (canvas.height !== img.height) canvas.height = img.height;
  canvas.getContext("2d")!.putImageData(new ImageData(rgbToRgba(img.data), img.width, img.height), 0, 0);
}

export async function pngBlob(img: Rgb): Promise<Blob> {
  const canvas = new OffscreenCanvas(img.width, img.height);
  canvas.getContext("2d")!.putImageData(new ImageData(rgbToRgba(img.data), img.width, img.height), 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

export function download(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** Natural sort ("frame2" before "frame10"). */
export function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

/** Last run of digits in a file name ("shot_0012.png" -> 12), or null. */
export function trailingNumber(name: string): number | null {
  const m = name.replace(/\.[^.]*$/, "").match(/(\d+)(?!.*\d)/);
  return m ? Number(m[1]) : null;
}
