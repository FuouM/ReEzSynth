// FastBlend blending: deflicker a frame-by-frame stylized video, guided by the original.
import { createMemo, createResource, createSignal, onMount } from "solid-js";
import { BlendSettingsRows, createBlendSettings, createLoadSettings, ExportBox, FrameSetBox, loadAsJob, LoadSettingsRows } from "../components/VideoPanels";
import { GroupBox } from "soulwindow/solid";
import { Viewer, ViewerEmpty, type Layer } from "../components/Viewer";
import { createFastBlend, fastBlendBatchLimit, loadEngine } from "../core/engine";
import { blendSequence } from "../core/fastblend";
import type { FrameSet } from "../core/frames";
import { elapsed, runJob } from "../core/job";
import type { ModeProps } from "./types";

export function BlendMode(props: ModeProps) {
  const [original, setOriginal] = createSignal<FrameSet | null>(null);
  const [stylized, setStylized] = createSignal<FrameSet | null>(null);
  const [load, setLoad] = createLoadSettings();
  const [blend, setBlend] = createBlendSettings("balanced");
  const [result, setResult] = createSignal<FrameSet | null>(null);
  const [frame, setFrame] = createSignal(0);
  const [layer, setLayer] = createSignal("stylized");
  const [engine] = createResource(() => loadEngine().catch(() => null));
  const autoBatch = () => {
    const v = original();
    const e = engine();
    return v && e ? Math.min(16, fastBlendBatchLimit(e, v.width, v.height)) : null;
  };

  const loadOriginal = async (files: File[]) => {
    // A stylized sequence already loaded fixes the working size.
    const s = stylized();
    const set = await loadAsJob(files, load, s ? { width: s.width, height: s.height } : undefined);
    if (!set) return;
    setOriginal(set);
    setResult(null);
    if (!s) setLayer("original");
  };
  const loadStylized = async (files: File[]) => {
    const o = original();
    const set = await loadAsJob(files, load, o ? { width: o.width, height: o.height } : undefined);
    if (!set) return;
    setStylized(set);
    setResult(null);
    setLayer("stylized");
  };

  const frameCount = () => Math.min(original()?.frames.length ?? 0, stylized()?.frames.length ?? 0);
  const countWarning = () => {
    const o = original();
    const s = stylized();
    return o && s && o.frames.length !== s.frames.length ? `The original has ${o.frames.length} frames and the stylized video ${s.frames.length}; the first ${frameCount()} are used.` : null;
  };

  const blocker = createMemo((): string | null => {
    if (!original()) return "Load the original video.";
    if (!stylized()) return "Load the stylized video.";
    if (frameCount() < 2) return "Need at least two frames.";
    return null;
  });

  const run = () =>
    runJob("Blending", async (c) => {
      const o = original()!;
      const s = stylized()!;
      const n = frameCount();
      const e = await loadEngine();
      const t0 = performance.now();
      const matcher = await createFastBlend(e, o.height, o.width, {
        minimumPatchSize: blend.minimumPatchSize,
        numIter: blend.numIter,
        guideWeight: blend.guideWeight,
      });
      try {
        const frames = await blendSequence(matcher, o.frames.slice(0, n), s.frames.slice(0, n), blend.windowSize, {
          batchSize: blend.batchSize || autoBatch() || 8,
          signal: c.signal,
          onProgress: c.progress,
        });
        setResult({ ...s, name: "result", frames });
      } finally {
        matcher.free();
      }
      setLayer("result");
      return `Blended ${n} frames (window ${blend.windowSize}) in ${elapsed(performance.now() - t0)}.`;
    });

  onMount(() => props.setAction({ label: "Deflicker", icon: "magic", blocker, run }));

  const layers = createMemo((): Layer[] => {
    const out: Layer[] = [];
    for (const [id, label, set] of [
      ["result", "Result", result()],
      ["stylized", "Stylized", stylized()],
      ["original", "Original", original()],
    ] as const) {
      if (set) out.push({ id, label, width: set.width, height: set.height, frames: set.frames, fps: set.fps });
    }
    return out;
  });

  return (
    <div class="daw-upper">
      <div class="daw-dock">
        <div class="daw-panel-title">Inputs</div>
        <div class="daw-panel-body rz-dock-body">
          <FrameSetBox title="Original" icon="camera-video" help="The unstylized footage; it guides the matching between frames." set={original()} onFiles={loadOriginal} />
          <FrameSetBox
            title="Stylized"
            icon="palette"
            help="The same footage stylized frame by frame (for example by an image model). It flickers; blending smooths it."
            set={stylized()}
            onFiles={loadStylized}
            warning={countWarning()}
          />
        </div>
      </div>

      <Viewer
        title="Monitor"
        layers={layers()}
        layer={layer()}
        onLayer={setLayer}
        frame={frame()}
        onFrame={setFrame}
        empty={
          <ViewerEmpty icon="layers">
            <div class="rz-empty">
              <strong>Deflicker video</strong>
              <ol>
                <li>Load the original video (or its frames as images).</li>
                <li>Load the stylized version of the same frames.</li>
                <li>Press Deflicker: each frame is blended with its neighbours' styles, matched through the original.</li>
              </ol>
            </div>
          </ViewerEmpty>
        }
      />

      <div class="daw-dock rz-dock-right">
        <div class="daw-panel-title">Settings</div>
        <div class="daw-panel-body rz-dock-body">
          <GroupBox title="Loading">
            <LoadSettingsRows s={load} set={setLoad} />
          </GroupBox>
          <GroupBox title="FastBlend">
            <BlendSettingsRows s={blend} set={setBlend} window autoBatch={autoBatch()} />
          </GroupBox>
          <ExportBox result={result()} prefix="deflickered" frame={frame()} />
        </div>
      </div>
    </div>
  );
}
