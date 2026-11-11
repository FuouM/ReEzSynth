// ReEzSynth Web: loads images, runs WebSynth (Rust + WebGPU, see rust/ebsynth-web), shows the result.
import init, { WebSynth } from "./pkg/ebsynth_web.js";

const $ = (id) => document.getElementById(id);
const status = $("status");
const runButton = $("run");
let synth = null;
let busy = false;

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

/** Decode an image file to {width, height, data: Uint8Array RGBA}. */
async function loadImage(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return { width: canvas.width, height: canvas.height, data: new Uint8Array(data.buffer) };
}

function drawImage(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(img.data.buffer), img.width, img.height), 0, 0);
  canvas.hidden = false;
}

function addGuide() {
  const row = $("guide-row").content.firstElementChild.cloneNode(true);
  row.querySelector(".remove").addEventListener("click", () => row.remove());
  $("guides").append(row);
  return row;
}

function readParams() {
  const form = new FormData($("params"));
  const options = {};
  for (const [key, value] of form.entries()) {
    const input = $("params").elements[key];
    options[key] = input.type === "number" ? Number(value) : value;
  }
  options.useBilateral = $("params").elements.useBilateral.checked;
  return options;
}

async function run() {
  if (busy) return;
  const styleFile = $("style-file").files[0];
  if (!styleFile) return setStatus("Choose a style image.", true);
  const rows = [...document.querySelectorAll(".guide")];
  if (rows.length === 0) return setStatus("Add at least one guide.", true);

  busy = true;
  runButton.disabled = true;
  const progress = $("progress");
  progress.hidden = false;
  progress.value = 0;
  try {
    const style = await loadImage(styleFile);
    const sources = [];
    const targets = [];
    const weights = [];
    let size = null;
    for (const [i, row] of rows.entries()) {
      const src = row.querySelector(".guide-source").files[0];
      const tgt = row.querySelector(".guide-target").files[0];
      if (!src || !tgt) throw new Error(`Guide ${i + 1}: choose both a source and a target image.`);
      const [s, t] = await Promise.all([loadImage(src), loadImage(tgt)]);
      if (s.width !== style.width || s.height !== style.height) {
        throw new Error(`Guide ${i + 1}: the source is ${s.width}x${s.height}; the style is ${style.width}x${style.height}.`);
      }
      size ??= { width: t.width, height: t.height };
      if (t.width !== size.width || t.height !== size.height) {
        throw new Error(`Guide ${i + 1}: every target must be ${size.width}x${size.height}.`);
      }
      sources.push(s.data);
      targets.push(t.data);
      weights.push(Number(row.querySelector(".guide-weight").value));
    }

    setStatus("Synthesizing…");
    const t0 = performance.now();
    const rgba = await synth.synthesize(
      style.data, style.width, style.height, sources, targets, new Float32Array(weights),
      size.width, size.height, readParams(),
      (level, levels) => { progress.value = level / levels; },
    );
    const seconds = (performance.now() - t0) / 1000;
    const result = { width: size.width, height: size.height, data: rgba };
    drawImage($("result"), result);
    $("result").toBlob((blob) => {
      const link = $("download");
      URL.revokeObjectURL(link.href);
      link.href = URL.createObjectURL(blob);
      link.hidden = false;
    });
    setStatus(`Done in ${seconds.toFixed(2)} s (${size.width}x${size.height}).`);
  } catch (err) {
    setStatus(String(err?.message ?? err), true);
  } finally {
    busy = false;
    runButton.disabled = false;
    progress.hidden = true;
  }
}

$("style-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) drawImage($("style-preview"), await loadImage(file));
});
$("add-guide").addEventListener("click", addGuide);
runButton.addEventListener("click", run);
addGuide();

try {
  if (!navigator.gpu) throw new Error("This browser has no WebGPU support (try a current Chrome, Edge or Firefox).");
  await init();
  synth = await WebSynth.create();
  $("adapter").textContent = `GPU: ${synth.adapter}`;
  runButton.disabled = false;
} catch (err) {
  $("adapter").textContent = `WebGPU unavailable: ${err?.message ?? err}`;
  $("adapter").classList.add("error");
}
