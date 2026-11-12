// EbSynth synthesis parameters, shared by image and video synthesis.
import type { SetStoreFunction } from "solid-js/store";
import { NumberRow, SelectRow, ToggleRow } from "./Fields";

export const SYNTH_DEFAULTS = {
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
export type SynthParams = typeof SYNTH_DEFAULTS;

export function SynthParamRows(props: { p: SynthParams; set: SetStoreFunction<SynthParams>; compact?: boolean }) {
  return (
    <>
      <NumberRow label="Patch size" desc="Odd, 3–31" value={props.p.patchSize} min={3} max={31} step={2} onChange={(v) => props.set("patchSize", v | 1)} />
      <NumberRow divider label="Uniformity" desc="Higher spreads style patches more evenly" value={props.p.uniformity} min={0} step={100} onChange={(v) => props.set("uniformity", v)} />
      <NumberRow divider label="Pyramid levels" value={props.p.pyramidLevels} min={1} max={12} onChange={(v) => props.set("pyramidLevels", v)} />
      <NumberRow divider label="Search/vote iterations" value={props.p.searchVoteIters} min={0} max={64} onChange={(v) => props.set("searchVoteIters", v)} />
      <NumberRow divider label="PatchMatch iterations" value={props.p.patchMatchIters} min={0} max={32} onChange={(v) => props.set("patchMatchIters", v)} />
      {!props.compact && (
        <>
          <NumberRow divider label="Stop threshold" desc="Stop a level early when fewer pixels change" value={props.p.stopThreshold} min={0} onChange={(v) => props.set("stopThreshold", v)} />
          <SelectRow divider label="Cost" value={props.p.costFunction} options={[{ value: "ssd", label: "SSD" }, { value: "ncc", label: "NCC" }, { value: "sad", label: "SAD" }]} onChange={(v) => props.set("costFunction", v)} />
          <SelectRow divider label="Search" value={props.p.searchMode} options={[{ value: "patchmatch", label: "PatchMatch" }, { value: "jfa", label: "Jump flooding" }, { value: "multicand", label: "Multi-candidate" }]} onChange={(v) => props.set("searchMode", v)} />
          <SelectRow divider label="Vote" value={props.p.voteMode} options={[{ value: "weighted", label: "Weighted" }, { value: "plain", label: "Plain" }]} onChange={(v) => props.set("voteMode", v)} />
          <ToggleRow divider label="Bilateral weights" checked={props.p.useBilateral} onChange={(v) => props.set("useBilateral", v)} />
          <NumberRow divider label="Seed" value={props.p.seed} min={0} onChange={(v) => props.set("seed", v)} />
        </>
      )}
    </>
  );
}
