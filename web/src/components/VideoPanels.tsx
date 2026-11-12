// Pieces shared by the two video modes: picking a frame source, working-size settings,
// FastBlend settings and export.
import { Show, type JSX } from "solid-js";
import { createPersisted } from "../core/persist";
import { Banner, Button, GroupBox, Icon, IconText } from "soulwindow/solid";
import { exportPngZip, exportVideo } from "../core/export";
import { loadFrames, memoryMiB, type FrameSet } from "../core/frames";
import { download, pngBlob } from "../core/image";
import { job, runJob } from "../core/job";
import { Facts, FileButton, NumberRow, SelectRow } from "./Fields";
import { DropZone } from "./Drop";
import { Thumb } from "./Viewer";

/** Working size and limits applied when a sequence is loaded. */
export function createLoadSettings(key: string) {
  return createPersisted(`${key}-load`, { maxSide: 768, maxFrames: 0, fps: 24 });
}
export type LoadSettings = ReturnType<typeof createLoadSettings>[0];

export function LoadSettingsRows(props: { s: LoadSettings; set: ReturnType<typeof createLoadSettings>[1] }) {
  return (
    <>
      <NumberRow label="Working size" desc="Longest side in pixels when loading (0 = original). Smaller is much faster." value={props.s.maxSide} min={0} step={16} onChange={(v) => props.set("maxSide", Math.max(0, v))} />
      <NumberRow divider label="Frame limit" desc="Load at most this many frames (0 = all)" value={props.s.maxFrames} min={0} onChange={(v) => props.set("maxFrames", Math.max(0, v))} />
      <NumberRow divider label="Sequence FPS" desc="Frame rate of image sequences (videos keep their own)" value={props.s.fps} min={1} max={240} step={0.001} onChange={(v) => props.set("fps", Math.max(1, v))} />
    </>
  );
}

const VIDEO_ACCEPT = "video/*,.mp4,.m4v,.mov,.webm,.mkv";
const IMAGE_ACCEPT = "image/*";

/** Load files as a frame set through the job queue. */
export async function loadAsJob(files: File[], settings: LoadSettings, size?: { width: number; height: number }): Promise<FrameSet | null> {
  let set: FrameSet | null = null;
  await runJob(files.length === 1 ? "Decoding video" : "Loading images", async (c) => {
    set = await loadFrames(files, { maxSide: settings.maxSide, maxFrames: settings.maxFrames, fps: settings.fps, size, signal: c.signal, onProgress: c.progress });
    return `Loaded ${set.frames.length} frames at ${set.width}×${set.height} (${memoryMiB(set)} MiB).`;
  });
  return set;
}

export function FrameSetBox(props: {
  title: string;
  icon: string;
  help: JSX.Element;
  set: FrameSet | null;
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  warning?: string | null;
}) {
  return (
    <DropZone onFiles={props.onFiles} disabled={props.disabled || job.busy}>
    <GroupBox title={<IconText icon={<Icon name={props.icon} />}>{props.title}</IconText>}>
      <p class="ds-muted rz-help">{props.help} Or drop files here.</p>
      <div class="sw-row sw-gap-3 sw-wrap rz-pick">
        <FileButton text="Video file…" icon="film" accept={VIDEO_ACCEPT} disabled={props.disabled || job.busy} onFiles={(f) => props.onFiles(f.slice(0, 1))} />
        <FileButton text="Image sequence…" icon="images" accept={IMAGE_ACCEPT} multiple disabled={props.disabled || job.busy} title="Select all the frames (they are sorted by name)" onFiles={props.onFiles} />
      </div>
      <Show when={props.set}>
        {(s) => (
          <div class="rz-input-card">
            <Thumb width={s().width} height={s().height} data={s().frames[0]} />
            <Facts
              items={[
                ["Source", <span class="sw-truncate" title={s().name}>{s().name}</span>],
                ["Frames", `${s().frames.length} @ ${s().fps} fps`],
                ["Size", `${s().width}×${s().height}`],
                ["Memory", `${memoryMiB(s())} MiB`],
              ]}
            />
          </div>
        )}
      </Show>
      <Show when={props.set && memoryMiB(props.set) > 1024}>
        <Banner tone="warn">These frames take over 1 GiB of memory, and processing needs several times that. Lower the working size or the frame limit and reload if the tab becomes slow or crashes.</Banner>
      </Show>
      <Show when={props.warning}>{(w) => <Banner tone="warn">{w()}</Banner>}</Show>
    </GroupBox>
    </DropZone>
  );
}

export type Accuracy = "fast" | "balanced" | "accurate";

/** FastBlend presets (FastBlend/src/config.py). */
export const PRESETS: Record<Accuracy, { windowSize: number; minimumPatchSize: number; numIter: number; guideWeight: number }> = {
  fast: { windowSize: 5, minimumPatchSize: 7, numIter: 3, guideWeight: 5 },
  balanced: { windowSize: 15, minimumPatchSize: 5, numIter: 5, guideWeight: 10 },
  accurate: { windowSize: 25, minimumPatchSize: 3, numIter: 8, guideWeight: 15 },
};

export function createBlendSettings(key: string, accuracy: Accuracy = "balanced") {
  return createPersisted(`${key}-fastblend`, { accuracy: accuracy as Accuracy | "custom", ...PRESETS[accuracy], batchSize: 0 });
}
export type BlendSettings = ReturnType<typeof createBlendSettings>[0];

export function BlendSettingsRows(props: { s: BlendSettings; set: ReturnType<typeof createBlendSettings>[1]; window?: boolean; autoBatch: number | null }) {
  const custom = <K extends "windowSize" | "minimumPatchSize" | "numIter" | "guideWeight">(k: K, v: number) => {
    props.set(k, v);
    props.set("accuracy", "custom");
  };
  return (
    <>
      <SelectRow
        label="Accuracy"
        value={props.s.accuracy}
        options={[
          { value: "fast", label: "Fast" },
          { value: "balanced", label: "Balanced" },
          { value: "accurate", label: "Accurate" },
          { value: "custom", label: "Custom" },
        ]}
        onChange={(v) => {
          if (v !== "custom") props.set({ accuracy: v, ...PRESETS[v] });
          else props.set("accuracy", "custom");
        }}
      />
      <Show when={props.window}>
        <NumberRow divider label="Window size" desc="Neighbours on each side blended into every frame" value={props.s.windowSize} min={1} max={60} onChange={(v) => custom("windowSize", Math.max(1, v))} />
      </Show>
      <NumberRow divider label="Minimum patch size" desc="Odd; smaller keeps more detail" value={props.s.minimumPatchSize} min={1} max={15} step={2} onChange={(v) => custom("minimumPatchSize", Math.max(1, v | 1))} />
      <NumberRow divider label="Iterations" value={props.s.numIter} min={1} max={20} onChange={(v) => custom("numIter", Math.max(1, v))} />
      <NumberRow divider label="Guide weight" desc="How strongly the original frames steer matching" value={props.s.guideWeight} min={0} step={0.5} onChange={(v) => custom("guideWeight", Math.max(0, v))} />
      <NumberRow
        divider
        label="Batch size"
        desc={props.autoBatch ? `Frames per GPU call (0 = auto: ${props.autoBatch})` : "Frames per GPU call (0 = auto)"}
        value={props.s.batchSize}
        min={0}
        max={64}
        onChange={(v) => props.set("batchSize", Math.max(0, v))}
      />
    </>
  );
}

export function ExportBox(props: { result: FrameSet | null; prefix: string; frame: number; extra?: JSX.Element }) {
  const saveVideo = () =>
    runJob("Encoding video", async (c) => {
      const r = props.result!;
      const { blob, ext } = await exportVideo(r, { signal: c.signal, onProgress: c.progress });
      download(blob, `${props.prefix}.${ext}`);
      return `Saved ${props.prefix}.${ext} (${(blob.size / 2 ** 20).toFixed(1)} MiB).`;
    });
  const saveZip = () =>
    runJob("Writing PNGs", async (c) => {
      const r = props.result!;
      const blob = await exportPngZip(r, props.prefix, { signal: c.signal, onProgress: c.progress });
      download(blob, `${props.prefix}_frames.zip`);
      return `Saved ${r.frames.length} PNGs in ${props.prefix}_frames.zip.`;
    });
  const saveFrame = async () => {
    const r = props.result!;
    const i = Math.min(props.frame, r.frames.length - 1);
    download(await pngBlob({ width: r.width, height: r.height, data: r.frames[i] }), `${props.prefix}_${String(i).padStart(4, "0")}.png`);
  };
  return (
    <GroupBox title={<IconText icon={<Icon name="box-arrow-up" />}>Output</IconText>}>
      <Show when={props.result} fallback={<p class="ds-muted rz-help">Run to produce the output video.</p>}>
        <div class="sw-col sw-gap-3">
          <Button icon={<Icon name="film" />} text="Save video" title="H.264 MP4 where the browser can encode it, else WebM" disabled={job.busy} onClick={saveVideo} />
          <Button icon={<Icon name="file-earmark-zip" />} text="Save PNG frames (zip)" disabled={job.busy} onClick={saveZip} />
          <Button icon={<Icon name="image" />} text="Save shown frame" disabled={job.busy} onClick={saveFrame} />
        </div>
      </Show>
      {props.extra}
    </GroupBox>
  );
}
