// Single-image EbSynth: a style image, guide pairs and the synthesis parameters.
import { createMemo, createSignal, For, onMount, Show } from "solid-js";
import { createStore, produce } from "solid-js/store";
import { Banner, Button, GroupBox, Icon, IconText } from "soulwindow/solid";
import { DropZone } from "../components/Drop";
import { Facts, FileButton, NumberRow, SelectRow, ToggleRow } from "../components/Fields";
import { Thumb, Viewer, ViewerEmpty, type Layer } from "../components/Viewer";
import { loadEngine } from "../core/engine";
import { decodeImage, download, pngBlob, rgbaToRgb, rgbToRgba, type Rgb } from "../core/image";
import { elapsed, job, runJob, setMessage } from "../core/job";
import { createPersisted } from "../core/persist";
import type { ModeProps } from "./types";

interface Named extends Rgb {
  name: string;
}

interface GuideRow {
  id: number;
  source: Named | null;
  target: Named | null;
  weight: number;
}

const DEFAULTS = {
  patchSize: 7,
  uniformity: 3500,
  pyramidLevels: 6,
  searchVoteIters: 12,
  patchMatchIters: 6,
  stopThreshold: 5,
  costFunction: "ssd" as "ssd" | "ncc" | "sad",
  searchMode: "patchmatch" as "patchmatch" | "jfa" | "multicand",
  voteMode: "weighted" as "weighted" | "plain",
  useBilateral: false,
  seed: 2024,
};

let nextId = 1;
const ACCEPT = "image/*";

export function ImageMode(props: ModeProps) {
  const [style, setStyle] = createSignal<Named | null>(null);
  const [guides, setGuides] = createStore<GuideRow[]>([{ id: nextId++, source: null, target: null, weight: 1 }]);
  const [params, setParams] = createPersisted("image-params", { ...DEFAULTS });
  const [result, setResult] = createSignal<Rgb | null>(null);
  const [layer, setLayer] = createSignal<string>("style");

  const load = async (file: File): Promise<Named | null> => {
    try {
      return { ...(await decodeImage(file)), name: file.name };
    } catch {
      setMessage(`Could not read ${file.name} as an image.`, true);
      return null;
    }
  };

  const setStyleFile = async (f: File) => {
    const img = await load(f);
    if (img) {
      setStyle(img);
      setLayer("style");
    }
  };
  const setGuideFile = async (i: number, side: "source" | "target", f: File) => {
    const img = await load(f);
    if (img) setGuides(i, side, img);
  };
  /** Monitor drop: fill the style, then empty guide slots in order, adding guides as needed. */
  const dropAnywhere = async (files: File[]) => {
    for (const f of files) {
      if (!style()) {
        await setStyleFile(f);
        continue;
      }
      let slot: [number, "source" | "target"] | null = null;
      for (const [i, g] of guides.entries()) {
        if (!g.source) slot = [i, "source"];
        else if (!g.target) slot = [i, "target"];
        if (slot) break;
      }
      if (!slot) {
        setGuides(guides.length, { id: nextId++, source: null, target: null, weight: 1 });
        slot = [guides.length - 1, "source"];
      }
      await setGuideFile(slot[0], slot[1], f);
    }
  };

  const blocker = createMemo((): string | null => {
    const s = style();
    if (!s) return "Choose a style image.";
    if (guides.length === 0) return "Add at least one guide.";
    let size: { width: number; height: number } | null = null;
    for (const [i, g] of guides.entries()) {
      if (!g.source || !g.target) return `Guide ${i + 1}: choose both a source and a target image.`;
      if (g.source.width !== s.width || g.source.height !== s.height) {
        return `Guide ${i + 1}: the source is ${g.source.width}×${g.source.height}; it must match the style (${s.width}×${s.height}).`;
      }
      size ??= g.target;
      if (g.target.width !== size.width || g.target.height !== size.height) return `Guide ${i + 1}: every target must be ${size.width}×${size.height}.`;
    }
    return null;
  });

  const run = () =>
    runJob("Synthesizing", async (c) => {
      const engine = await loadEngine();
      const s = style()!;
      const rows = guides.map((g) => ({ ...g }));
      const t = rows[0].target!;
      const t0 = performance.now();
      const rgba = (img: Rgb) => new Uint8Array(rgbToRgba(img.data).buffer);
      const out: Uint8Array = await engine.synth.synthesize(
        rgba(s),
        s.width,
        s.height,
        rows.map((g) => rgba(g.source!)),
        rows.map((g) => rgba(g.target!)),
        new Float32Array(rows.map((g) => g.weight)),
        t.width,
        t.height,
        { ...params },
        (level: number, levels: number) => c.progress(level, levels),
      );
      setResult({ width: t.width, height: t.height, data: rgbaToRgb(out) });
      setLayer("result");
      return `Synthesized ${t.width}×${t.height} in ${elapsed(performance.now() - t0)}.`;
    });

  onMount(() => props.setAction({ label: "Synthesize", icon: "magic", blocker, run }));

  const layers = createMemo((): Layer[] => {
    const out: Layer[] = [];
    const r = result();
    if (r) out.push({ id: "result", label: "Result", width: r.width, height: r.height, frames: [r.data] });
    const s = style();
    if (s) out.push({ id: "style", label: "Style", width: s.width, height: s.height, frames: [s.data] });
    guides.forEach((g, i) => {
      if (g.target) out.push({ id: `target-${g.id}`, label: `Target ${i + 1}`, width: g.target.width, height: g.target.height, frames: [g.target.data], compare: i === 0 });
    });
    return out;
  });

  const save = async () => {
    const r = result();
    if (r) download(await pngBlob(r), "reezsynth.png");
  };

  return (
    <div class="daw-upper">
      <div class="daw-dock">
        <div class="daw-panel-title">Inputs</div>
        <div class="daw-panel-body rz-dock-body">
          <DropZone onFiles={([f]) => setStyleFile(f)} disabled={job.busy}>
          <GroupBox title={<IconText icon={<Icon name="palette" />}>Style</IconText>}>
            <p class="ds-muted rz-help">The painted or stylized image whose look is transferred.</p>
            <div class="sw-row sw-gap-3 rz-pick">
              <FileButton text={style() ? "Replace…" : "Choose image…"} icon="image" accept={ACCEPT} disabled={job.busy} onFiles={([f]) => setStyleFile(f)} />
            </div>
            <Show when={style()}>
              {(s) => (
                <div class="rz-input-card">
                  <Thumb width={s().width} height={s().height} data={s().data} />
                  <Facts items={[["File", <span class="sw-truncate">{s().name}</span>], ["Size", `${s().width}×${s().height}`]]} />
                </div>
              )}
            </Show>
          </GroupBox>
          </DropZone>

          <GroupBox
            title={<IconText icon={<Icon name="intersect" />}>Guides</IconText>}
            actions={<Button size="xs" icon={<Icon name="plus-lg" />} text="Add" disabled={job.busy} onClick={() => setGuides(guides.length, { id: nextId++, source: null, target: null, weight: 1 })} />}
          >
            <p class="ds-muted rz-help">Each guide pairs an image aligned with the style (source, same size as the style) with one aligned with the result (target). Targets set the result size.</p>
            <For each={guides}>
              {(g, i) => (
                <div class="rz-guide">
                  <div class="rz-guide-head">
                    <span class="ds-label">Guide {i() + 1}</span>
                    <span class="sw-grow" />
                    <label class="rz-inline">
                      Weight
                      <input class="input-field rz-number rz-number--sm" type="number" min={0} step={0.1} value={g.weight} onChange={(e) => Number.isFinite(e.currentTarget.valueAsNumber) && setGuides(i(), "weight", e.currentTarget.valueAsNumber)} />
                    </label>
                    <Button size="icon-sm" icon={<Icon name="x-lg" />} title="Remove guide" disabled={job.busy || guides.length === 1} onClick={() => setGuides(produce((gs) => void gs.splice(i(), 1)))} />
                  </div>
                  <div class="rz-guide-pair">
                    <For each={["source", "target"] as const}>
                      {(side) => (
                        <DropZone onFiles={([f]) => setGuideFile(i(), side, f)} disabled={job.busy}>
                        <div class="rz-guide-slot">
                          <Show when={g[side]} fallback={<div class="rz-thumb rz-thumb--empty">{side === "source" ? "Source" : "Target"}</div>}>
                            {(img) => <Thumb width={img().width} height={img().height} data={img().data} title={`${img().name} (${img().width}×${img().height})`} />}
                          </Show>
                          <FileButton text={side === "source" ? "Source…" : "Target…"} accept={ACCEPT} disabled={job.busy} onFiles={([f]) => setGuideFile(i(), side, f)} />
                        </div>
                        </DropZone>
                      )}
                    </For>
                  </div>
                </div>
              )}
            </For>
          </GroupBox>
          <Show when={(style() || guides.some((g) => g.source || g.target)) && blocker()}>
            {(b) => <Banner tone="warn">{b()}</Banner>}
          </Show>
        </div>
      </div>

      <Viewer
        title="Monitor"
        layers={layers()}
        layer={layer()}
        onLayer={setLayer}
        frame={0}
        onFrame={() => {}}
        onDropFiles={dropAnywhere}
        dropDisabled={job.busy}
        actions={<Button size="xs" icon={<Icon name="download" />} text="Save PNG" disabled={!result()} onClick={save} />}
        empty={
          <ViewerEmpty icon="magic">
            <div class="rz-empty">
              <strong>Image synthesis</strong>
              <ol>
                <li>Choose the style image (or drop images here: style first, then guide sources and targets in order).</li>
                <li>For each guide, choose a source (aligned with the style) and a target (aligned with the result you want).</li>
                <li>Press Synthesize. Everything runs on your GPU; images never leave the browser.</li>
              </ol>
            </div>
          </ViewerEmpty>
        }
      />

      <div class="daw-dock rz-dock-right">
        <div class="daw-panel-title">
          Parameters
          <span class="daw-panel-title-actions">
            <Button size="xs" text="Defaults" disabled={job.busy} onClick={() => setParams({ ...DEFAULTS })} />
          </span>
        </div>
        <div class="daw-panel-body rz-dock-body">
          <NumberRow label="Patch size" desc="Odd, 3–31" value={params.patchSize} min={3} max={31} step={2} onChange={(v) => setParams("patchSize", v | 1)} />
          <NumberRow divider label="Uniformity" desc="Higher spreads style patches more evenly" value={params.uniformity} min={0} step={100} onChange={(v) => setParams("uniformity", v)} />
          <NumberRow divider label="Pyramid levels" value={params.pyramidLevels} min={1} max={12} onChange={(v) => setParams("pyramidLevels", v)} />
          <NumberRow divider label="Search/vote iterations" value={params.searchVoteIters} min={0} max={64} onChange={(v) => setParams("searchVoteIters", v)} />
          <NumberRow divider label="PatchMatch iterations" value={params.patchMatchIters} min={0} max={32} onChange={(v) => setParams("patchMatchIters", v)} />
          <NumberRow divider label="Stop threshold" desc="Stop a level early when fewer pixels change" value={params.stopThreshold} min={0} onChange={(v) => setParams("stopThreshold", v)} />
          <SelectRow divider label="Cost" value={params.costFunction} options={[{ value: "ssd", label: "SSD" }, { value: "ncc", label: "NCC" }, { value: "sad", label: "SAD" }]} onChange={(v) => setParams("costFunction", v)} />
          <SelectRow divider label="Search" value={params.searchMode} options={[{ value: "patchmatch", label: "PatchMatch" }, { value: "jfa", label: "Jump flooding" }, { value: "multicand", label: "Multi-candidate" }]} onChange={(v) => setParams("searchMode", v)} />
          <SelectRow divider label="Vote" value={params.voteMode} options={[{ value: "weighted", label: "Weighted" }, { value: "plain", label: "Plain" }]} onChange={(v) => setParams("voteMode", v)} />
          <ToggleRow divider label="Bilateral weights" checked={params.useBilateral} onChange={(v) => setParams("useBilateral", v)} />
          <NumberRow divider label="Seed" value={params.seed} min={0} onChange={(v) => setParams("seed", v)} />
        </div>
      </div>
    </div>
  );
}
