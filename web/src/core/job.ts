// One job at a time (synthesis, blending, loading, export) with progress, cancel and status.
import { createStore } from "solid-js/store";

export interface JobState {
  busy: boolean;
  /** What is running: "Synthesizing", "Decoding video" … */
  label: string;
  done: number;
  /** null while the total is unknown. */
  total: number | null;
  startedAt: number;
  /** Last finished message, or the error. */
  message: string;
  error: boolean;
}

const [job, setJob] = createStore<JobState>({ busy: false, label: "", done: 0, total: null, startedAt: 0, message: "", error: false });
export { job };

let controller: AbortController | null = null;

export interface JobControl {
  signal: AbortSignal;
  progress: (done: number, total: number | null) => void;
  /** Change the label mid-job ("Matching", "Encoding" …). */
  stage: (label: string) => void;
}

/** Run `fn` as the current job; resolves to its message, or null if it failed or was cancelled. */
export async function runJob(label: string, fn: (c: JobControl) => Promise<string>): Promise<string | null> {
  if (job.busy) return null;
  controller = new AbortController();
  const signal = controller.signal;
  setJob({ busy: true, label, done: 0, total: null, startedAt: performance.now(), error: false });
  try {
    const message = await fn({
      signal,
      progress: (done, total) => setJob({ done, total }),
      stage: (l) => setJob({ label: l, done: 0, total: null }),
    });
    setJob({ busy: false, message, error: false });
    return message;
  } catch (e) {
    // Each core module has its own Cancelled class (they are import-free), so match the name.
    const cancelled = signal.aborted || (e instanceof Error && e.name === "Cancelled");
    const message = cancelled ? `${job.label} cancelled.` : e instanceof Error ? e.message : String(e);
    if (!cancelled) console.error(e);
    setJob({ busy: false, message, error: !cancelled });
    return null;
  } finally {
    controller = null;
  }
}

export function cancelJob(): void {
  controller?.abort();
}

export function setMessage(message: string, error = false): void {
  setJob({ message, error });
}

export function elapsed(ms: number): string {
  const s = ms / 1000;
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}
