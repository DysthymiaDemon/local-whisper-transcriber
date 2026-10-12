param(
    [string]$Python = "py -3.12",
    [string]$Wheelhouse = ".\wheelhouse",
    [switch]$SkipInstall,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

if ($Clean) {
    Remove-Item -LiteralPath ".\build", ".\dist" -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path ".\.venv")) {
    Invoke-Expression "$Python -m venv .venv"
}

.\.venv\Scripts\Activate.ps1

# Same supply-chain rules as setup: only the exact wheels in the hash locks, nothing built from source.
$LockArgs = @(
    "--require-hashes", "--only-binary=:all:", "--no-deps",
    "-r", ".\resources\requirements.lock.txt",
    "-r", ".\resources\requirements-build.lock.txt"
)
if (-not $SkipInstall) {
    if (Test-Path $Wheelhouse) {
        python -m pip install --no-index --find-links $Wheelhouse @LockArgs
    }
    else {
        python -m pip install @LockArgs
    }
}

python -m PyInstaller --clean --noconfirm .\packaging\local_whisper_transcriber.spec

$DistDir = ".\dist\OfflineMeetingTranscriber"
New-Item -ItemType Directory -Force -Path "$DistDir\models\faster-whisper" | Out-Null
New-Item -ItemType Directory -Force -Path "$DistDir\transcripts" | Out-Null

if (-not (Test-Path "$DistDir\config.json")) {
    Copy-Item -LiteralPath ".\resources\config.template.json" -Destination "$DistDir\config.json" -Force
}
Copy-Item -LiteralPath ".\docs\offline_setup.md" -Destination "$DistDir\offline_setup.md" -Force
# Locks and the model manifest: setup and the per-start model check need them next to the executable.
Copy-Item -LiteralPath ".\resources" -Destination "$DistDir\resources" -Recurse -Force

Write-Host "Portable bundle created at $DistDir"
Write-Host "Copy model folders into $DistDir\models before moving to corporate laptop."
