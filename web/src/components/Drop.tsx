// Drop files onto a region. Highlights while files are dragged over it.
import { createSignal, type JSX } from "solid-js";

/** Event handlers and an "over" flag for a drop target; spread the handlers on an element. */
export function createDrop(onFiles: (files: File[]) => void, enabled: () => boolean = () => true) {
  const [over, setOver] = createSignal(false);
  let depth = 0;
  const hasFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  const handlers = {
    onDragEnter: (e: DragEvent) => {
      if (!hasFiles(e) || !enabled()) return;
      e.preventDefault();
      depth++;
      setOver(true);
    },
    onDragOver: (e: DragEvent) => {
      if (!hasFiles(e) || !enabled()) return;
      e.preventDefault();
      e.dataTransfer!.dropEffect = "copy";
    },
    onDragLeave: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setOver(false);
    },
    onDrop: (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      // Inner drop zones handle their own files.
      e.stopPropagation();
      depth = 0;
      setOver(false);
      if (!enabled()) return;
      const files = [...(e.dataTransfer?.files ?? [])];
      if (files.length) onFiles(files);
    },
  };
  return { over, handlers };
}

export function DropZone(props: { onFiles: (files: File[]) => void; disabled?: boolean; class?: string; children: JSX.Element }) {
  const d = createDrop(
    (f) => props.onFiles(f),
    () => !props.disabled,
  );
  return (
    <div class={`rz-drop${props.class ? ` ${props.class}` : ""}`} classList={{ "rz-drop--over": d.over() }} {...d.handlers}>
      {props.children}
    </div>
  );
}

/** Keep the browser from opening files dropped outside any drop zone. */
export function guardWindowDrops(): void {
  const stop = (e: DragEvent) => {
    if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) e.preventDefault();
  };
  window.addEventListener("dragover", stop);
  window.addEventListener("drop", stop);
}
