# Installs a self-contained Rust toolchain into .\.rust (load it with . .\env.ps1)
$RootPath = $PSScriptRoot
New-Item -ItemType Directory -Force -Path "$RootPath\.rust\.rustup" | Out-Null
New-Item -ItemType Directory -Force -Path "$RootPath\.rust\.cargo" | Out-Null

$env:RUSTUP_HOME = "$RootPath\.rust\.rustup"
$env:CARGO_HOME = "$RootPath\.rust\.cargo"

$Init = "$RootPath\.rust\rustup-init.exe"
Write-Host "Downloading rustup-init.exe..."
curl.exe -L -o $Init https://win.rustup.rs/x86_64
if (-not (Test-Path $Init)) {
    Write-Error "Failed to download rustup-init.exe"
    exit 1
}

Write-Host "Installing Rust locally..."
Start-Process -FilePath $Init -ArgumentList "-y", "--no-modify-path", "--profile", "minimal", "--default-toolchain", "stable", "--target", "wasm32-unknown-unknown" -NoNewWindow -Wait
& "$RootPath\.rust\.cargo\bin\rustup.exe" component add clippy rustfmt
Write-Host "Rust installation completed."
