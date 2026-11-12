// Centre monitor: shows one layer (result, input, guide …) of RGB frames, with a frame
// scrubber and playback for sequences.
import { createEffect, createMemo, createSignal, on, onCleanup, Show, type JSX } from "solid-js";
import { Button, EmptyState, Icon, SegmentedSwitch } from "soulwindow/solid";
import { drawRgb } from "../core/image";

export interface Layer {
  id: string;
  label: string;
  width: number;
  height: number;
  frames: Uint8Array[];
  fps?: number;
}

export function Viewer(props: {
  title: string;
  layers: Layer[];
  /** Selected layer id; follows the first layer when unset or gone. */
  layer?: string;
  onLayer?: (id: string) => void;
  frame: number;
  onFrame: (i: number) => void;
  /** Frames to mark on the scrubber (keyframes). */
  marks?: number[];
  actions?: JSX.Element;
  empty: JSX.Element;
}) {
  let canvas!: HTMLCanvasElement;
  const current = createMemo(() => props.layers.find((l) => l.id === props.layer) ?? props.layers[0]);
  const count = () => current()?.frames.length ?? 0;
  const index = () => Math.min(Math.max(0, props.frame), Math.max(0, count() - 1));
  const [playing, setPlaying] = createSignal(false);

  createEffect(() => {
    const l = current();
    if (!l || l.frames.length === 0) return;
    drawRgb(canvas, { width: l.width, height: l.height, data: l.frames[index()] });
  });

  createEffect(
    on([playing, () => current()?.fps], ([on, fps]) => {
      if (!on) return;
      const step = 1000 / (fps || 24);
      let last = performance.now();
      let raf = requestAnimationFrame(function tick(now) {
        if (now - last >= step) {
          last = now;
          props.onFrame((index() + 1) % Math.max(1, count()));
        }
        raf = requestAnimationFrame(tick);
      });
      onCleanup(() => cancelAnimationFrame(raf));
    }),
  );
  createEffect(() => count() <= 1 && setPlaying(false));

  const onKey = (e: KeyboardEvent) => {
    if (count() <= 1) return;
    if (e.key === "ArrowRight") props.onFrame(Math.min(count() - 1, index() + 1));
    else if (e.key === "ArrowLeft") props.onFrame(Math.max(0, index() - 1));
    else if (e.key === " ") setPlaying(!playing());
    else return;
    e.preventDefault();
  };

  return (
    <div class="daw-center">
      <div class="daw-panel-title">
        {props.title}
        <Show when={props.layers.length > 1}>
          <SegmentedSwitch value={current()?.id ?? ""} options={props.layers.map((l) => ({ value: l.id, text: l.label }))} onChange={(id) => props.onLayer?.(id)} aria-label="Shown image" />
        </Show>
        <span class="daw-panel-title-actions">{props.actions}</span>
      </div>
      <div class="daw-panel-body rz-stage" tabindex="0" onKeyDown={onKey}>
        <Show when={current()} fallback={props.empty}>
          <canvas ref={canvas} class="rz-canvas" />
        </Show>
      </div>
      <Show when={count() > 1}>
        <div class="rz-scrub">
          <Button size="icon-sm" icon={<Icon name={playing() ? "pause-fill" : "play-fill"} />} title={playing() ? "Pause (Space)" : "Play (Space)"} onClick={() => setPlaying(!playing())} />
          <div class="rz-scrub-track">
            <input type="range" min={0} max={count() - 1} value={index()} aria-label="Frame" onInput={(e) => props.onFrame(e.currentTarget.valueAsNumber)} />
            <Show when={props.marks?.length}>
              <div class="rz-marks" aria-hidden="true">
                {props.marks!.filter((m) => m < count()).map((m) => (
                  <span class="rz-mark" style={{ left: `${(m / Math.max(1, count() - 1)) * 100}%` }} />
                ))}
              </div>
            </Show>
          </div>
          <span class="sw-tabular rz-scrub-label">
            {index()} / {count() - 1}
          </span>
        </div>
      </Show>
    </div>
  );
}

export function ViewerEmpty(props: { icon: string; children: JSX.Element }) {
  return <EmptyState icon={props.icon}>{props.children}</EmptyState>;
}

/** Small preview of one RGB image. */
export function Thumb(props: { width: number; height: number; data: Uint8Array; title?: string }) {
  let c!: HTMLCanvasElement;
  createEffect(() => drawRgb(c, { width: props.width, height: props.height, data: props.data }));
  return <canvas ref={c} class="rz-thumb" title={props.title} />;
}
