// Centre monitor: shows one layer (result, input, guide …) of RGB frames, with a frame
// scrubber, playback, a before/after split against the result, and a drop target.
import { createEffect, createMemo, createSignal, on, onCleanup, Show, type JSX } from "solid-js";
import { Button, EmptyState, Icon, SegmentedSwitch } from "soulwindow/solid";
import { drawRgb } from "../core/image";
import { createDrop } from "./Drop";

export interface Layer {
  id: string;
  label: string;
  width: number;
  height: number;
  frames: Uint8Array[];
  fps?: number;
  /** Default partner of the result in the before/after split. */
  compare?: boolean;
}

const sameSize = (a: Layer, b: Layer) => a.width === b.width && a.height === b.height;

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
  /** Files dropped on the monitor. */
  onDropFiles?: (files: File[]) => void;
  dropDisabled?: boolean;
}) {
  let canvas!: HTMLCanvasElement;
  let overlay!: HTMLCanvasElement;
  let frameBox!: HTMLDivElement;
  const [compare, setCompare] = createSignal(false);
  const [split, setSplit] = createSignal(0.5);
  const [playing, setPlaying] = createSignal(false);

  const result = createMemo(() => props.layers.find((l) => l.id === "result"));
  const others = createMemo(() => {
    const r = result();
    return r ? props.layers.filter((l) => l.id !== "result" && sameSize(l, r)) : [];
  });
  const comparing = () => compare() && others().length > 0;
  const selected = createMemo(() => props.layers.find((l) => l.id === props.layer) ?? props.layers[0]);
  // While comparing, the result is on the left and the selected other layer on the right.
  const right = createMemo(() => {
    const s = selected();
    return s && others().includes(s) ? s : (others().find((l) => l.compare) ?? others()[0]);
  });
  const base = () => (comparing() ? result() : selected());
  const count = () => base()?.frames.length ?? 0;
  const index = () => Math.min(Math.max(0, props.frame), Math.max(0, count() - 1));
  const frameOf = (l: Layer) => l.frames[Math.min(index(), l.frames.length - 1)];

  createEffect(() => {
    const l = base();
    if (!l || l.frames.length === 0) return;
    drawRgb(canvas, { width: l.width, height: l.height, data: frameOf(l) });
  });
  createEffect(() => {
    const r = right();
    if (!comparing() || !r) return;
    drawRgb(overlay, { width: r.width, height: r.height, data: frameOf(r) });
  });

  createEffect(
    on([playing, () => base()?.fps], ([on, fps]) => {
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
    if (e.key === "c" && others().length) setCompare(!compare());
    else if (count() <= 1) return;
    else if (e.key === "ArrowRight") props.onFrame(Math.min(count() - 1, index() + 1));
    else if (e.key === "ArrowLeft") props.onFrame(Math.max(0, index() - 1));
    else if (e.key === " ") setPlaying(!playing());
    else return;
    e.preventDefault();
  };

  const dragSplit = (e: PointerEvent) => {
    if (!comparing()) return;
    const move = (ev: PointerEvent) => {
      const r = frameBox.getBoundingClientRect();
      setSplit(Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)));
    };
    move(e);
    frameBox.setPointerCapture(e.pointerId);
    frameBox.onpointermove = move;
    frameBox.onpointerup = () => (frameBox.onpointermove = null);
  };

  const drop = createDrop(
    (files) => props.onDropFiles?.(files),
    () => !!props.onDropFiles && !props.dropDisabled,
  );

  const switchOptions = () => (comparing() ? others() : props.layers).map((l) => ({ value: l.id, text: l.label }));

  return (
    <div class="daw-center">
      <div class="daw-panel-title">
        {props.title}
        <Show when={props.layers.length > 1}>
          <Show when={comparing()}>
            <span class="ds-muted rz-compare-label">Result ◂ ▸</span>
          </Show>
          <SegmentedSwitch value={(comparing() ? right() : selected())?.id ?? ""} options={switchOptions()} onChange={(id) => props.onLayer?.(id)} aria-label="Shown image" />
        </Show>
        <span class="daw-panel-title-actions">
          <Show when={others().length > 0}>
            <Button size="xs" icon={<Icon name="layout-split" />} text="Compare" active={comparing()} title="Before/after split against the result (C)" onClick={() => setCompare(!compare())} />
          </Show>
          {props.actions}
        </span>
      </div>
      <div class="daw-panel-body rz-stage" classList={{ "rz-drop--over": drop.over() }} tabindex="0" onKeyDown={onKey} {...drop.handlers}>
        <Show when={base()} fallback={props.empty}>
          {(l) => (
            <div ref={frameBox} class="rz-frame" classList={{ "rz-frame--compare": comparing() }} style={{ "--rz-aspect": `${l().width} / ${l().height}`, "--rz-ratio": l().width / l().height }} onPointerDown={dragSplit}>
              <canvas ref={canvas} class="rz-canvas" />
              <Show when={comparing()}>
                <canvas ref={overlay} class="rz-canvas rz-canvas--overlay" style={{ "clip-path": `inset(0 0 0 ${split() * 100}%)` }} />
                <div class="rz-split" style={{ left: `${split() * 100}%` }} aria-hidden="true" />
              </Show>
            </div>
          )}
        </Show>
        <Show when={drop.over()}>
          <div class="rz-drop-hint">
            <Icon name="box-arrow-in-down" /> Drop to load
          </div>
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
