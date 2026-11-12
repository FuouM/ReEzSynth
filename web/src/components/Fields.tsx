// Small form pieces on top of SoulWindow's settings rows and input-field class.
import { For, type JSX } from "solid-js";
import { Button, Icon, Select, SettingsRow, SettingsToggleRow } from "soulwindow/solid";

export function NumberRow(props: {
  label: JSX.Element;
  desc?: JSX.Element;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  divider?: boolean;
  disabled?: boolean;
}) {
  return (
    <SettingsRow label={props.label} desc={props.desc} divider={props.divider}>
      <input
        class="input-field rz-number"
        type="number"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        disabled={props.disabled}
        onChange={(e) => {
          const v = e.currentTarget.valueAsNumber;
          if (Number.isFinite(v)) props.onChange(v);
          else e.currentTarget.value = String(props.value);
        }}
      />
    </SettingsRow>
  );
}

export function SelectRow<T extends string>(props: {
  label: JSX.Element;
  desc?: JSX.Element;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  divider?: boolean;
  disabled?: boolean;
}) {
  return (
    <SettingsRow label={props.label} desc={props.desc} divider={props.divider}>
      <Select value={props.value} options={props.options} onChange={props.onChange} disabled={props.disabled} aria-label={typeof props.label === "string" ? props.label : undefined} />
    </SettingsRow>
  );
}

export { SettingsToggleRow as ToggleRow };

/** A button that opens a file picker. */
export function FileButton(props: {
  text: string;
  icon?: string;
  accept: string;
  multiple?: boolean;
  directory?: boolean;
  disabled?: boolean;
  title?: string;
  variant?: "default" | "primary";
  onFiles: (files: File[]) => void;
}) {
  let input!: HTMLInputElement;
  return (
    <>
      <Button icon={props.icon ? <Icon name={props.icon} /> : undefined} text={props.text} title={props.title} variant={props.variant} disabled={props.disabled} onClick={() => input.click()} />
      <input
        ref={(el) => {
          input = el;
          if (props.directory) el.setAttribute("webkitdirectory", "");
        }}
        type="file"
        hidden
        accept={props.accept}
        multiple={props.multiple}
        onChange={(e) => {
          const files = [...(e.currentTarget.files ?? [])];
          e.currentTarget.value = "";
          if (files.length) props.onFiles(files);
        }}
      />
    </>
  );
}

/** Label/value pairs under an input. */
export function Facts(props: { items: [string, JSX.Element][] }) {
  return (
    <dl class="rz-facts">
      <For each={props.items}>
        {([k, v]) => (
          <>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </>
        )}
      </For>
    </dl>
  );
}
