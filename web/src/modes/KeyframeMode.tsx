// FastBlend interpolation: stylized keyframes spread over every frame of a video.
import { createMemo, createResource, createSignal, For, onMount, Show } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { Banner, Button, GroupBox, Icon, IconText } from "soulwindow/solid";
import { FileButton } from "../components/Fields";
import { BlendSettingsRows, createBlendSettings, createLoadSettings, ExportBox, FrameSetBox, loadAsJob, LoadSettingsRows } from "../components/VideoPanels";
import { Thumb, Viewer, ViewerEmpty, type Layer } from "../components/Viewer";
import { createFastBlend, fastBlendBatchLimit, loadEngine } from "../core/engine";
import { interpolateKeyframes } from "../core/fastblend";
import { keyframeIndex, type FrameSet } from "../core/frames";
import { decodeImage, download, pngBlob } from "../core/image";
import { elapsed, job, runJob, setMessage } from "../core/job";
import type { ModeProps } from "./types";

interface Keyframe {
  id: number;
  index: number;
  name: string;
  data: Uint8Array;
}

let nextId = 1;

export function KeyframeMode(props: ModeProps) {
  const [video, setVideo] = createSignal<FrameSet | null>(null);
  const [keys, setKeys] = createStore<Keyframe[]>([]);
  const [load, setLoad] = createLoadSettings();
  const [blend, setBlend] = createBlendSettings("balanced");
  const [result, setResult] = createSignal<FrameSet | null>(null);
  const [frame, setFrame] = createSignal(0);
  const [layer, setLayer] = createSignal("original");
  const [engine] = createResource(() => loadEngine().catch(() => null));
  const autoBatch = () => {
    const v = video();
    const e = engine();
    return v && e ? Math.min(16, fastBlendBatchLimit(e, v.width, v.height)) : null;
  };

  const loadVideo = async (files: File[]) => {
    const set = await loadAsJob(files, load);
    if (!set) return;
    setVideo(set);
    setKeys([]);
    setResult(null);
    setFrame(0);
    setLayer("original");
  };

  const addKeyframes = async (files: File[], at?: number) => {
    const v = video();
    if (!v) return;
    const skipped: string[] = [];
    for (const f of files) {
      const index = at ?? keyframeIndex(v, f.name) ?? frame();
      try {
        const bitmap = await createImageBitmap(f);
        const aspect = bitmap.width / bitmap.height;
        bitmap.close();
        if (Math.abs(aspect / (v.width / v.height) - 1) > 0.02) skipped.push(`${f.name} (different aspect ratio)`);
        const img = await decodeImage(f, { width: v.width, height: v.height });
        setKeys(
          produce((ks) => {
            const same = ks.findIndex((k) => k.index === index);
            const k = { id: nextId++, index, name: f.name, data: img.data };
            if (same >= 0) ks[same] = k;
            else ks.push(k);
            ks.sort((a, b) => a.index - b.index);
          }),
        );
      } catch {
        skipped.push(`${f.name} (not an image)`);
      }
    }
    if (skipped.length) setMessage(`Check: ${skipped.join(", ")}`, true);
    setLayer("original");
  };

  const duplicates = createMemo(() => {
    const seen = new Set<number>();
    return keys.filter((k) => (seen.has(k.index) ? true : (seen.add(k.index), false))).map((k) => k.index);
  });

  const blocker = createMemo((): string | null => {
    const v = video();
    if (!v) return "Load the video.";
    if (keys.length === 0) return "Add at least one stylized keyframe.";
    if (keys.some((k) => k.index < 0 || k.index >= v.frames.length)) return `Keyframe frame numbers must be 0–${v.frames.length - 1}.`;
    if (duplicates().length) return `Two keyframes share frame ${duplicates()[0]}.`;
    if (keys.length === v.frames.length) return "Every frame is already a keyframe.";
    return null;
  });

  const run = () =>
    runJob("Matching", async (c) => {
      const v = video()!;
      const e = await loadEngine();
      const t0 = performance.now();
      const batchSize = blend.batchSize || autoBatch() || 8;
      const matcher = await createFastBlend(e, v.height, v.width, {
        minimumPatchSize: blend.minimumPatchSize,
        numIter: blend.numIter,
        guideWeight: blend.guideWeight,
        usePairwisePatchError: true,
      });
      try {
        const frames = await interpolateKeyframes(matcher, v.frames, new Map(keys.map((k) => [k.index, k.data])), { batchSize, signal: c.signal, onProgress: c.progress });
        setResult({ ...v, name: "result", frames });
      } finally {
        matcher.free();
      }
      setLayer("result");
      return `Interpolated ${v.frames.length - keys.length} frames from ${keys.length} keyframe${keys.length > 1 ? "s" : ""} in ${elapsed(performance.now() - t0)}.`;
    });

  onMount(() => props.setAction({ label: "Interpolate", icon: "magic", blocker, run }));

  const layers = createMemo((): Layer[] => {
    const out: Layer[] = [];
    const r = result();
    if (r) out.push({ id: "result", label: "Result", width: r.width, height: r.height, frames: r.frames, fps: r.fps });
    const v = video();
    if (v) out.push({ id: "original", label: "Original", width: v.width, height: v.height, frames: v.frames, fps: v.fps });
    return out;
  });

  const saveFrameForPainting = async () => {
    const v = video()!;
    const i = Math.min(frame(), v.frames.length - 1);
    download(await pngBlob({ width: v.width, height: v.height, data: v.frames[i] }), `frame_${String(i).padStart(4, "0")}.png`);
  };

  return (
    <div class="daw-upper">
      <div class="daw-dock">
        <div class="daw-panel-title">Inputs</div>
        <div class="daw-panel-body rz-dock-body">
          <FrameSetBox title="Video" icon="film" help="The original footage. Every frame is matched against it." set={video()} onFiles={loadVideo} />

          <GroupBox
            title={<IconText icon={<Icon name="key" />}>Keyframes</IconText>}
            actions={<FileButton text="Add…" icon="plus-lg" accept="image/*" multiple disabled={!video() || job.busy} title="Stylized frames; the number in each file name picks its frame" onFiles={(f) => addKeyframes(f)} />}
          >
            <p class="ds-muted rz-help">
              Stylize a few frames yourself (paint over them, or use any image tool), then add them here. A number in the file name (<code>frame_0012.png</code>) places it on that frame; otherwise it goes on the frame shown.
            </p>
            <Show when={video()}>
              <div class="sw-row sw-gap-3 sw-wrap rz-pick">
                <Button size="xs" icon={<Icon name="download" />} text={`Save frame ${frame()} to paint`} disabled={job.busy} onClick={saveFrameForPainting} />
                <FileButton text={`Set frame ${frame()}…`} icon="key" accept="image/*" disabled={job.busy} onFiles={(f) => addKeyframes(f.slice(0, 1), frame())} />
              </div>
            </Show>
            <For each={keys}>
              {(k, i) => (
                <div class="rz-key" classList={{ "rz-key--current": k.index === frame() }}>
                  <button type="button" class="rz-key-thumb" title="Show this frame" onClick={() => setFrame(k.index)}>
                    <Thumb width={video()!.width} height={video()!.height} data={k.data} />
                  </button>
                  <div class="rz-key-meta">
                    <span class="sw-truncate" title={k.name}>
                      {k.name}
                    </span>
                    <label class="rz-inline">
                      Frame
                      <input
                        class="input-field rz-number rz-number--sm"
                        type="number"
                        min={0}
                        max={(video()?.frames.length ?? 1) - 1}
                        value={k.index}
                        disabled={job.busy}
                        onChange={(e) => Number.isFinite(e.currentTarget.valueAsNumber) && setKeys(i(), "index", Math.round(e.currentTarget.valueAsNumber))}
                      />
                    </label>
                  </div>
                  <Button size="icon-sm" icon={<Icon name="x-lg" />} title="Remove keyframe" disabled={job.busy} onClick={() => setKeys(produce((ks) => void ks.splice(i(), 1)))} />
                </div>
              )}
            </For>
          </GroupBox>
          <Show when={video() && keys.length > 0 && blocker()}>{(b) => <Banner tone="warn">{b()}</Banner>}</Show>
        </div>
      </div>

      <Viewer
        title="Monitor"
        layers={layers()}
        layer={layer()}
        onLayer={setLayer}
        frame={frame()}
        onFrame={setFrame}
        marks={keys.map((k) => k.index)}
        empty={
          <ViewerEmpty icon="film">
            <div class="rz-empty">
              <strong>Keyframes → video</strong>
              <ol>
                <li>Load the video (or its frames as images).</li>
                <li>Save a frame or two, stylize them in any tool, and add them back as keyframes.</li>
                <li>Press Interpolate: FastBlend carries the keyframes' style through every frame.</li>
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
            <BlendSettingsRows s={blend} set={setBlend} autoBatch={autoBatch()} />
          </GroupBox>
          <ExportBox result={result()} prefix="keyframes" frame={frame()} />
        </div>
      </div>
    </div>
  );
}
