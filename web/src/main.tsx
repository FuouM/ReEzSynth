import "bootstrap-icons/font/bootstrap-icons.css";
import "soulwindow/css";
import "soulwindow/css/pro/daw.css";
import "./app.css";
import { render } from "solid-js/web";
import { createSoulWindow } from "soulwindow/runtime";
import { App } from "./App";

// Same options as the Vite plugin, so the boot script and the controller agree.
export const sw = createSoulWindow({ storagePrefix: "reezsynth-", density: "compact" });

render(() => <App sw={sw} />, document.getElementById("app")!);
