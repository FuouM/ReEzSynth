// Compile rust/ebsynth-web to wasm and generate its JS bindings into src/pkg.
// Needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.130:
//   rustup target add wasm32-unknown-unknown
//   cargo install wasm-bindgen-cli --version 0.2.130 --locked
// On Windows the repo-local toolchain from setup_env.ps1 (.rust/.cargo/bin) is used when present.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(web, "..");
const exe = process.platform === "win32" ? ".exe" : "";
const localBin = join(root, ".rust", ".cargo", "bin");
const env = { ...process.env };
if (existsSync(join(localBin, `cargo${exe}`))) {
  env.RUSTUP_HOME = join(root, ".rust", ".rustup");
  env.CARGO_HOME = join(root, ".rust", ".cargo");
  env.PATH = `${localBin}${process.platform === "win32" ? ";" : ":"}${env.PATH}`;
}
const release = !process.argv.includes("--debug");

function run(cmd, args) {
  console.log(`> ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", env, shell: false });
  if (r.error) {
    console.error(`could not run ${cmd}: ${r.error.message}`);
    process.exit(1);
  }
  if (r.status !== 0) process.exit(r.status ?? 1);
}

run(`cargo${exe}`, [
  "build",
  "--manifest-path", join(root, "rust", "Cargo.toml"),
  "-p", "ebsynth-web",
  "--target", "wasm32-unknown-unknown",
  ...(release ? ["--release"] : []),
]);
// Cargo honours CARGO_TARGET_DIR (relative to the working directory), so look there too.
const targetDir = env.CARGO_TARGET_DIR ? resolve(env.CARGO_TARGET_DIR) : join(root, "rust", "target");
run(`wasm-bindgen${exe}`, [
  "--target", "web",
  "--out-dir", join(web, "src", "pkg"),
  join(targetDir, "wasm32-unknown-unknown", release ? "release" : "debug", "ebsynth_web.wasm"),
]);
