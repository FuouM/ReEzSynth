// FastBlend interpolation: stylized keyframes spread over every frame of a video.
import { createEffect, createMemo, createResource, createSignal, For, Show } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { Banner, Button, GroupBox, Icon, IconText } from "soulwindow/solid";
import { DropZone } from "../components/Drop";
import { FileButton, NumberRow, SelectRow, ToggleRow } from "../components/Fields";
import { SYNTH_DEFAULTS, SynthParamRows, type SynthParams } from "../components/SynthParams";
import { BlendSettingsRows, createBlendSettings, createLoadSettings, ExportBox, FrameSetBox, loadAsJob, LoadSettingsRows } from "../components/VideoPanels";
import { Thumb, Viewer, ViewerEmpty, type Layer } from "../components/Viewer";
import { createFastBlend, fastBlendBatchLimit, loadEngine, synthesizeRgbWithError } from "../core/engine";
import { DEFAULT_WEIGHTS, synthesizeVideo, type PassBlend } from "../core/ebsynthVideo";
import { DEFAULT_FLOW_MODEL, NeuFlow } from "../core/flow";
import { createPersisted } from "../core/persist";
import { blendSequence, interpolateKeyframes } from "../core/fastblend";
import { isVideoFile, keyframeIndex, type FrameSet } from "../core/frames";
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
  const [load, setLoad] = createLoadSettings("keyframes");
  const [blend, setBlend] = createBlendSettings("keyframes", "balanced");
  const [method, setMethod] = createPersisted("keyframes-method", {
    method: "fastblend" as "fastblend" | "ebsynth",
    edgeWeight: 1,
    imageWeight: 6,
    deflicker: true,
    deflickerWindow: 3,
    passBlend: "poisson" as PassBlend,
    opticalFlow: true,
    posWeight: DEFAULT_WEIGHTS.pos,
    warpWeight: DEFAULT_WEIGHTS.warp,
  });
  // Video Synth Config defaults (reezsynth_nodes): NCC cost, unlike single-image synthesis.
  const [synthParams, setSynthParams] = createPersisted("keyframes-synth", { ...SYNTH_DEFAULTS, costFunction: "ncc" as SynthParams["costFunction"] });
  const [result, setResult] = createSignal<FrameSet | null>(null);
  const [frame, setFrame] = createSignal(0);
  const [synthCollapsed, setSynthCollapsed] = createSignal(true);
  const [layer, setLayer] = createSignal("original");
  const [flowModel, setFlowModel] = createSignal<File | null>(null);
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

  /** Monitor drop: a video (or images before any video) loads the footage; later images are keyframes. */
  const dropAnywhere = (files: File[]) => {
    if (!video() || (files.length === 1 && isVideoFile(files[0]))) void loadVideo(files);
    else void addKeyframes(files);
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

  const fastBlendOptions = () => ({ minimumPatchSize: blend.minimumPatchSize, numIter: blend.numIter, guideWeight: blend.guideWeight });

  const runFastBlend = () =>
    runJob("Matching", async (c) => {
      const v = video()!;
      const e = await loadEngine();
      const t0 = performance.now();
      const batchSize = blend.batchSize || autoBatch() || 8;
      const matcher = await createFastBlend(e, v.height, v.width, { ...fastBlendOptions(), usePairwisePatchError: true });
      try {
        const frames = await interpolateKeyframes(matcher, v.frames, new Map(keys.map((k) => [k.index, k.data])), { batchSize, signal: c.signal, onProgress: c.progress });
        setResult({ ...v, name: "result", frames });
      } finally {
        matcher.free();
      }
      setLayer("result");
      return `Interpolated ${v.frames.length - keys.length} frames from ${keys.length} keyframe${keys.length > 1 ? "s" : ""} in ${elapsed(performance.now() - t0)}.`;
    });

  const runEbSynth = () =>
    runJob("Synthesizing", async (c) => {
      const v = video()!;
      const e = await loadEngine();
      const t0 = performance.now();
      // Live preview: unfinished frames show the original until they are done.
      const live = [...v.frames];
      const show = () => setResult({ ...v, name: "result", frames: [...live] });
      show();
      setLayer("result");
      let lastShown = 0;
      const params = { ...synthParams };
      let flow: NeuFlow | null = null;
      if (method.opticalFlow) {
        c.stage("Loading NeuFlow");
        const file = flowModel();
        flow = file ? await NeuFlow.create(await file.arrayBuffer(), file.name, v) : await NeuFlow.create(DEFAULT_FLOW_MODEL, DEFAULT_FLOW_MODEL, v);
        c.stage("Synthesizing");
      }
      let frames: Uint8Array[];
      try {
        frames = await synthesizeVideo(
          v,
          new Map(keys.map((k) => [k.index, k.data])),
          (style, guides) => synthesizeRgbWithError(e, style, v.width, v.height, guides, v.width, v.height, params),
          {
            weights: { edge: method.edgeWeight, image: method.imageWeight, pos: method.posWeight, warp: method.warpWeight },
            blend: method.passBlend,
            flow: flow ?? undefined,
            signal: c.signal,
            onProgress: c.progress,
            onFrame: (i, f) => {
              live[i] = f;
              const now = performance.now();
              if (now - lastShown > 250) {
                lastShown = now;
                show();
              }
            },
          },
        );
      } finally {
        await flow?.release();
      }
      setResult({ ...v, name: "result", frames });
      let note = "";
      if (method.deflicker && frames.length > 1) {
        c.stage("Deflickering");
        const matcher = await createFastBlend(e, v.height, v.width, fastBlendOptions());
        try {
          const smooth = await blendSequence(matcher, v.frames, frames, method.deflickerWindow, {
            batchSize: blend.batchSize || autoBatch() || 8,
            signal: c.signal,
            onProgress: c.progress,
          });
          // Keyframes stay exactly as painted.
          for (const k of keys) smooth[k.index] = k.data;
          setResult({ ...v, name: "result", frames: smooth });
        } finally {
          matcher.free();
        }
        note = `, deflickered (window ${method.deflickerWindow})`;
      }
      return `Synthesized ${v.frames.length - keys.length} frames with EbSynth${note} in ${elapsed(performance.now() - t0)}.`;
    });

  const run = () => (method.method === "ebsynth" ? runEbSynth() : runFastBlend());

  createEffect(() => props.setAction({ label: method.method === "ebsynth" ? "Synthesize" : "Interpolate", icon: "magic", blocker, run }));

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

          <DropZone onFiles={(f) => addKeyframes(f)} disabled={!video() || job.busy}>
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
          </DropZone>
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
        onDropFiles={dropAnywhere}
        dropDisabled={job.busy}
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
          <GroupBox title="Method">
            <SelectRow
              label="Method"
              desc={method.method === "ebsynth" ? "EbSynth from the nearest keyframes with edge and colour guides, then optional deflicker" : "FastBlend: remap the nearest keyframes onto every frame (fastest)"}
              value={method.method}
              options={[
                { value: "fastblend", label: "FastBlend" },
                { value: "ebsynth", label: "EbSynth" },
              ]}
              onChange={(v) => setMethod("method", v)}
              disabled={job.busy}
            />
          </GroupBox>
          <Show when={method.method === "ebsynth"}>
            <GroupBox title="EbSynth guides">
              <NumberRow label="Edge weight" value={method.edgeWeight} min={0} step={0.5} onChange={(v) => setMethod("edgeWeight", Math.max(0, v))} />
              <NumberRow divider label="Colour weight" desc="How closely each frame's colours steer patch choice" value={method.imageWeight} min={0} step={0.5} onChange={(v) => setMethod("imageWeight", Math.max(0, v))} />
              <ToggleRow divider label="Optical flow (NeuFlow)" desc="Adds the positional and temporal guides and carries blend masks along the motion" checked={method.opticalFlow} onChange={(v) => setMethod("opticalFlow", v)} />
              <Show when={method.opticalFlow}>
                <NumberRow divider label="Position weight" value={method.posWeight} min={0} step={0.5} onChange={(v) => setMethod("posWeight", Math.max(0, v))} />
                <NumberRow divider label="Temporal weight" desc="Previous result moved along the flow" value={method.warpWeight} min={0} step={0.1} onChange={(v) => setMethod("warpWeight", Math.max(0, v))} />
                <p class="ds-muted rz-help rz-help--row">
                  Model: {flowModel()?.name ?? <code>{DEFAULT_FLOW_MODEL}</code>} (export with <code>web/export_neuflow.py</code>).
                </p>
                <div class="sw-row sw-gap-3 sw-wrap rz-pick">
                  <FileButton text="Choose model…" icon="cpu" accept=".onnx" disabled={job.busy} onFiles={(f) => setFlowModel(f[0] ?? null)} />
                </div>
              </Show>
              <SelectRow
                divider
                label="Pass blending"
                desc="How the forward and backward passes between two keyframes are combined"
                value={method.passBlend}
                options={[
                  { value: "poisson", label: "Error maps + Poisson" },
                  { value: "histogram", label: "Error maps + histogram" },
                  { value: "distance", label: "By distance" },
                ]}
                onChange={(v) => setMethod("passBlend", v)}
              />
              <ToggleRow divider label="Deflicker afterwards" checked={method.deflicker} onChange={(v) => setMethod("deflicker", v)} />
              <Show when={method.deflicker}>
                <NumberRow divider label="Deflicker window" desc="Neighbours on each side (FastBlend)" value={method.deflickerWindow} min={1} max={30} onChange={(v) => setMethod("deflickerWindow", Math.max(1, v))} />
              </Show>
            </GroupBox>
            <GroupBox title="EbSynth" collapsible collapsed={synthCollapsed()} onToggle={() => setSynthCollapsed(!synthCollapsed())}>
              <SynthParamRows p={synthParams} set={setSynthParams} compact />
            </GroupBox>
          </Show>
          <GroupBox title="FastBlend">
            <BlendSettingsRows s={blend} set={setBlend} autoBatch={autoBatch()} />
          </GroupBox>
          <ExportBox result={result()} prefix="keyframes" frame={frame()} />
        </div>
      </div>
    </div>
  );
}
