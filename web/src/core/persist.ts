// Settings stores that survive reloads (per browser). Storage can be missing or throw (private
// windows, blocked site data), so every access is guarded and the defaults always work.
import { createEffect } from "solid-js";
import { createStore, type SetStoreFunction } from "solid-js/store";

const PREFIX = "reezsynth-";

/** Saved values that match a default's key and type; anything else is ignored. */
export function mergeSaved<T extends object>(defaults: T, saved: unknown): T {
  const out = { ...defaults };
  if (!saved || typeof saved !== "object") return out;
  for (const k of Object.keys(defaults) as (keyof T)[]) {
    const v = (saved as Record<string, unknown>)[k as string];
    if (v !== undefined && typeof v === typeof defaults[k] && !(typeof v === "number" && !Number.isFinite(v))) out[k] = v as T[keyof T];
  }
  return out;
}

function read(key: string): unknown {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** A flat store of primitives initialised from storage and saved on every change. */
export function createPersisted<T extends object>(key: string, defaults: T): [T, SetStoreFunction<T>] {
  const [state, setState] = createStore<T>(mergeSaved(defaults, read(key)));
  createEffect(() => {
    const json = JSON.stringify(state);
    try {
      localStorage.setItem(PREFIX + key, json);
    } catch {
      // Not saved; the settings still apply for this visit.
    }
  });
  return [state, setState];
}
