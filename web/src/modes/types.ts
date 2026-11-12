/** What a mode puts on the transport bar's run button. */
export interface ModeAction {
  label: string;
  icon: string;
  /** Why it cannot run yet, or null when it can. */
  blocker: () => string | null;
  run: () => void;
}

export interface ModeProps {
  setAction: (a: ModeAction) => void;
}
