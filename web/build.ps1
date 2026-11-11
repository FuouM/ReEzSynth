# Build the browser front end: compiles rust/ebsynth-web to wasm and writes web/pkg.
# Needs the local toolchain (setup_env.ps1) with wasm-bindgen-cli 0.2.130:
#   cargo install wasm-bindgen-cli --version 0.2.130 --locked
# Serve web/ from any static server, e.g. `python -m http.server -d web`, and open
# http://localhost:8000 (WebGPU needs localhost or https).
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
. "$Root\env.ps1" *> $null
$Cargo = "$Root\.rust\.cargo\bin\cargo.exe"
$Bindgen = "$Root\.rust\.cargo\bin\wasm-bindgen.exe"

& $Cargo build --manifest-path "$Root\rust\Cargo.toml" -p ebsynth-web --release --target wasm32-unknown-unknown
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Bindgen --target web --no-typescript --out-dir "$PSScriptRoot\pkg" "$Root\rust\target\wasm32-unknown-unknown\release\ebsynth_web.wasm"
exit $LASTEXITCODE
