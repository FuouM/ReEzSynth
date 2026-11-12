import { createResource, createSignal, Match, Show, Switch } from "solid-js";
import type { SoulWindow } from "soulwindow/runtime";
import { Button, Icon, Modal, useSoulWindow } from "soulwindow/solid";
import { MenuBar } from "soulwindow/solid/pro";
import { loadEngine } from "./core/engine";
import { cancelJob, elapsed, job } from "./core/job";
import { BlendMode } from "./modes/BlendMode";
import { ImageMode } from "./modes/ImageMode";
import { KeyframeMode } from "./modes/KeyframeMode";
import type { ModeAction } from "./modes/types";

type Mode = "image" | "keyframes" | "blend";

const MODES: { id: Mode; label: string; icon: string; title: string }[] = [
  { id: "image", label: "Image", icon: "image", title: "EbSynth: stylize one image from a style image and guides" },
  { id: "keyframes", label: "Keyframes → video", icon: "film", title: "FastBlend interpolation: spread stylized keyframes over a whole video" },
  { id: "blend", label: "Deflicker video", icon: "layers", title: "FastBlend blending: remove flicker from a frame-by-frame stylized video" },
];

const REPO = "https://github.com/FuouM/ReEzSynth";

export function App(props: { sw: SoulWindow }) {
  const s = useSoulWindow(props.sw);
  const [mode, setMode] = createSignal<Mode>((new URLSearchParams(location.search).get("mode") as Mode) || "image");
  const [action, setAction] = createSignal<ModeAction | null>(null);
  const [engine] = createResource(loadEngine);
  const [about, setAbout] = createSignal(false);
  // Re-render the elapsed time while a job runs.
  const [now, setNow] = createSignal(performance.now());
  setInterval(() => job.busy && setNow(performance.now()), 250);

  const switchMode = (m: Mode) => {
    if (job.busy || m === mode()) return;
    setAction(null);
    setMode(m);
    const url = new URL(location.href);
    url.searchParams.set("mode", m);
    history.replaceState(null, "", url);
  };

  const blocker = () => {
    if (engine.error) return "WebGPU is not available.";
    if (!engine()) return "Starting WebGPU…";
    return action()?.blocker() ?? null;
  };

  const percent = () => (job.total ? Math.min(100, Math.round((job.done / job.total) * 100)) : null);

  return (
    <div class="daw-app">
      <MenuBar
        brand={
          <>
            <Icon name="brush" /> ReEzSynth
          </>
        }
        menus={[
          {
            label: "Mode",
            items: MODES.map((m) => ({ label: m.label, checked: mode() === m.id, disabled: job.busy, action: () => switchMode(m.id) })),
          },
          {
            label: "View",
            items: [
              { label: "Light theme", checked: s.theme() === "light", action: () => props.sw.setTheme("light") },
              { label: "Dark theme", checked: s.theme() === "dark", action: () => props.sw.setTheme("dark") },
              { label: "High contrast", checked: s.theme() === "high-contrast", action: () => props.sw.setTheme("high-contrast") },
            ],
          },
          {
            label: "Help",
            items: [
              { label: "About ReEzSynth Web", action: () => setAbout(true) },
              { label: "Source on GitHub", action: () => window.open(REPO, "_blank", "noopener") },
            ],
          },
        ]}
      >
        <span class="daw-doc-title">{MODES.find((m) => m.id === mode())!.title}</span>
      </MenuBar>

      <div class="daw-transport">
        <div class="daw-transport-group daw-tabs rz-modes" role="tablist">
          {MODES.map((m) => (
            <button
              type="button"
              role="tab"
              class="daw-tab"
              classList={{ "daw-tab--active": mode() === m.id }}
              aria-selected={mode() === m.id}
              title={m.title}
              disabled={job.busy && mode() !== m.id}
              onClick={() => switchMode(m.id)}
            >
              <Icon name={m.icon} /> {m.label}
            </button>
          ))}
        </div>
        <div class="daw-transport-spacer" />
        <div class="daw-transport-group">
          <Show
            when={engine()}
            fallback={
              <span class={`daw-chip ${engine.error ? "daw-chip--err" : ""}`} title={engine.error ? String(engine.error.message ?? engine.error) : undefined}>
                {engine.error ? "NO WEBGPU" : "GPU…"}
              </span>
            }
          >
            {(e) => (
              <span class="daw-chip daw-chip--ok" title={e().adapter}>
                GPU
              </span>
            )}
          </Show>
          <Show
            when={job.busy}
            fallback={
              <button type="button" class="daw-tbtn daw-tbtn--accent rz-run" disabled={!!blocker()} title={blocker() ?? undefined} onClick={() => action()?.run()}>
                <Icon name={action()?.icon ?? "play-fill"} /> {action()?.label ?? "Run"}
              </button>
            }
          >
            <button type="button" class="daw-tbtn rz-run" onClick={cancelJob}>
              <Icon name="stop-fill" /> Cancel
            </button>
          </Show>
        </div>
      </div>

      <Show when={engine.error}>
        <div class="rz-gpu-error" role="alert">
          <Icon name="exclamation-triangle" /> {String(engine.error?.message ?? engine.error)}
        </div>
      </Show>

      <Switch>
        <Match when={mode() === "image"}>
          <ImageMode setAction={setAction} />
        </Match>
        <Match when={mode() === "keyframes"}>
          <KeyframeMode setAction={setAction} />
        </Match>
        <Match when={mode() === "blend"}>
          <BlendMode setAction={setAction} />
        </Match>
      </Switch>

      <div class="daw-statusbar">
        <span class="daw-status-cell">
          <span class="daw-status-dot" classList={{ "daw-status-dot--busy": job.busy, "daw-status-dot--offline": !job.busy && job.error }} />
          {job.busy ? job.label : job.error ? "Error" : "Ready"}
        </span>
        <span class="daw-status-cell daw-status-cell--grow">
          <Show when={job.busy} fallback={<span class="daw-status-msg" classList={{ "rz-error": job.error }}>{job.message}</span>}>
            <Show when={percent() !== null} fallback={<progress class="rz-progress" />}>
              <progress class="rz-progress" max={100} value={percent()!} />
            </Show>
            <span class="daw-status-msg sw-tabular">
              {job.total ? `${job.done} / ${job.total}` : job.done ? `${job.done}` : ""} {(now(), elapsed(performance.now() - job.startedAt))}
            </span>
          </Show>
        </span>
        <span class="daw-status-cell sw-truncate rz-adapter">{engine() ? engine()!.adapter : engine.error ? "WebGPU unavailable" : "Starting WebGPU…"}</span>
      </div>

      <Modal open={about()} onClose={() => setAbout(false)} title="About ReEzSynth Web" width={520} footer={<Button variant="primary" text="Close" onClick={() => setAbout(false)} />}>
        <div class="rz-about">
          <p>
            Example-based stylization from <a href={REPO} target="_blank" rel="noopener">ReEzSynth</a>, running on your GPU through WebGPU. Images and videos are processed locally and never uploaded.
          </p>
          <ul>
            <li>
              <strong>Image</strong>: EbSynth patch synthesis of one image from a style image and guide pairs.
            </li>
            <li>
              <strong>Keyframes → video</strong>: FastBlend interpolation. Stylize a few frames of a video yourself, then spread them over every frame.
            </li>
            <li>
              <strong>Deflicker video</strong>: FastBlend blending of a video stylized frame by frame (for example by an image model), guided by the original.
            </li>
          </ul>
          <p class="ds-muted">Needs a browser with WebGPU (current Chrome or Edge; Firefox and Safari where enabled). Long videos use a lot of memory: lower the working size or frame limit if the tab runs out.</p>
        </div>
      </Modal>
    </div>
  );
}
