// OpenCV's 8-bit sRGB <-> CIE Lab (cv2.COLOR_BGR2Lab / COLOR_Lab2BGR on uint8 images):
// L scaled to 0..255, a and b offset by 128. Computed in floating point; OpenCV uses lookup
// tables and fixed point, so a few values differ by one (see test/lab.test.mjs).

const T = 0.008856;
const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const toGamma = (v: number) => (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);
const f = (t: number) => (t > T ? Math.cbrt(t) : 7.787 * t + 16 / 116);

/** RGB bytes to Lab floats (unrounded, 0..255 scale like OpenCV's 8-bit Lab). */
export function rgbToLabF(rgb: Uint8Array): Float32Array {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i += 3) {
    const r = toLinear(rgb[i] / 255);
    const g = toLinear(rgb[i + 1] / 255);
    const b = toLinear(rgb[i + 2] / 255);
    const x = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.950456;
    const y = 0.212671 * r + 0.71516 * g + 0.072169 * b;
    const z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
    const fy = f(y);
    const L = y > T ? 116 * fy - 16 : 903.3 * y;
    out[i] = (L * 255) / 100;
    out[i + 1] = 500 * (f(x) - fy) + 128;
    out[i + 2] = 200 * (fy - f(z)) + 128;
  }
  return out;
}

/** RGB bytes to 8-bit Lab (rounded and clamped). */
export function rgbToLab(rgb: Uint8Array): Uint8Array {
  const lab = rgbToLabF(rgb);
  const out = new Uint8Array(rgb.length);
  for (let i = 0; i < lab.length; i++) out[i] = Math.min(255, Math.max(0, Math.round(lab[i])));
  return out;
}

/** 8-bit Lab to RGB bytes. */
export function labToRgb(lab: Uint8Array): Uint8Array {
  const out = new Uint8Array(lab.length);
  for (let i = 0; i < lab.length; i += 3) {
    const L = (lab[i] * 100) / 255;
    const a = lab[i + 1] - 128;
    const bb = lab[i + 2] - 128;
    const fy = (L + 16) / 116;
    const y = L > 903.3 * T ? fy ** 3 : L / 903.3;
    const fyy = y > T ? fy : 7.787 * y + 16 / 116;
    const fx = a / 500 + fyy;
    const fz = fyy - bb / 200;
    const x = (fx ** 3 > T ? fx ** 3 : (fx - 16 / 116) / 7.787) * 0.950456;
    const z = (fz ** 3 > T ? fz ** 3 : (fz - 16 / 116) / 7.787) * 1.088754;
    const r = 3.240479 * x - 1.53715 * y - 0.498535 * z;
    const g = -0.969256 * x + 1.875991 * y + 0.041556 * z;
    const b = 0.055648 * x - 0.204043 * y + 1.057311 * z;
    out[i] = Math.min(255, Math.max(0, Math.round(toGamma(Math.min(1, Math.max(0, r))) * 255)));
    out[i + 1] = Math.min(255, Math.max(0, Math.round(toGamma(Math.min(1, Math.max(0, g))) * 255)));
    out[i + 2] = Math.min(255, Math.max(0, Math.round(toGamma(Math.min(1, Math.max(0, b))) * 255)));
  }
  return out;
}
