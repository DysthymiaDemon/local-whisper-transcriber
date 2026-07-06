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

if (-not $SkipInstall) {
    if (Test-Path $Wheelhouse) {
        python -m pip install --no-index --find-links $Wheelhouse -r .\resources\requirements-build.txt
    }
    else {
        python -m pip install -r .\resources\requirements-build.txt
    }
}

python -m PyInstaller --clean --noconfirm .\packaging\local_whisper_transcriber.spec

$DistDir = ".\dist\OfflineMeetingTranscriber"
New-Item -ItemType Directory -Force -Path "$DistDir\models\faster-whisper" | Out-Null
New-Item -ItemType Directory -Force -Path "$DistDir\models\pyannote-pipeline" | Out-Null
New-Item -ItemType Directory -Force -Path "$DistDir\models\pyannote-embedding" | Out-Null
New-Item -ItemType Directory -Force -Path "$DistDir\transcripts" | Out-Null

Copy-Item -LiteralPath ".\packaging\Run-OfflineMeetingTranscriber.bat" -Destination "$DistDir\Run-OfflineMeetingTranscriber.bat" -Force
if (-not (Test-Path "$DistDir\config.json")) {
    Copy-Item -LiteralPath ".\resources\config.template.json" -Destination "$DistDir\config.json" -Force
}
Copy-Item -LiteralPath ".\docs\offline_setup.md" -Destination "$DistDir\offline_setup.md" -Force

Write-Host "Portable bundle created at $DistDir"
Write-Host "Copy model folders into $DistDir\models before moving to corporate laptop."
