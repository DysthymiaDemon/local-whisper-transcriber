$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $ProjectRoot "Open Offline Meeting Transcriber.pyw"
$PythonPackage = "Python.Python.3.12"

function Get-Python312Command {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            & $py.Source -3.12 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @($py.Source, "-3.12")
            }
        }
        catch {
        }
    }

    foreach ($candidate in @("python.exe", "python3.exe")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $command) {
            continue
        }
        try {
            $version = & $command.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -eq "3.12") {
                return @($command.Source)
            }
        }
        catch {
        }
    }

    return $null
}

function Install-Python312 {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "Python 3.12 not found. Installing user-scope Python 3.12 with winget..."
        & $winget.Source install --id $PythonPackage --exact --scope user --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Write-Warning "winget install failed with exit code $LASTEXITCODE."
    }

    $url = "https://www.python.org/downloads/windows/"
    Write-Host "Python 3.12 not found and automatic install unavailable."
    Write-Host "Opening Python Windows download page: $url"
    Start-Process $url
    throw "Install Python 3.12 for current user, then run this launcher again."
}

$pythonCommand = Get-Python312Command
if (-not $pythonCommand) {
    Install-Python312
    $pythonCommand = Get-Python312Command
}

if (-not $pythonCommand) {
    throw "Python 3.12 still not found after install attempt."
}

Set-Location $ProjectRoot
$pythonExe = $pythonCommand[0]
$pythonArgs = @()
if ($pythonCommand.Count -gt 1) {
    $pythonArgs = $pythonCommand[1..($pythonCommand.Count - 1)]
}
& $pythonExe @pythonArgs $Launcher
