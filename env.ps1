# Local Rust environment for ReEzSynth (toolchain lives in .\.rust, installed by setup_env.ps1)
$RootPath = $PSScriptRoot

$env:RUSTUP_HOME = "$RootPath\.rust\.rustup"
$env:CARGO_HOME = "$RootPath\.rust\.cargo"

$BinPath = "$RootPath\.rust\.cargo\bin"
if ($env:PATH -notlike "*$BinPath*") {
    $env:PATH = "$BinPath;" + $env:PATH
}

$SccacheExe = "$BinPath\sccache.exe"
if (Test-Path $SccacheExe) {
    $env:RUSTC_WRAPPER = $SccacheExe
    $env:SCCACHE_DIR = "$RootPath\.rust\.sccache"
} else {
    $env:RUSTC_WRAPPER = ""
}

Write-Host "Local Rust environment loaded." -ForegroundColor Green
Write-Host "RUSTUP_HOME:        $env:RUSTUP_HOME"
Write-Host "CARGO_HOME:         $env:CARGO_HOME"
Write-Host "rustc version:"
& "$BinPath\rustc.exe" --version
