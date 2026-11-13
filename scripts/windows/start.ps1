# Arranca aforo-vision a mano: activa .venv y lanza el pipeline.
# Uso: .\scripts\windows\start.ps1 [argumentos de src.main, p. ej. --debug]

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "ERROR: no existe .venv. Corre primero .\scripts\windows\install.ps1" -ForegroundColor Red
    exit 1
}

$others = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match 'src\.main' -and $_.CommandLine -match 'python' -and $_.ProcessId -ne $PID })
if ($others.Count -gt 0) {
    Write-Host "AVISO: ya hay otra instancia de aforo-vision corriendo (PID $($others[0].ProcessId))." -ForegroundColor Yellow
    Write-Host "Dos instancias duplican eventos y pelean por la camara. Si esta el arranque automatico, detenlo con: Stop-ScheduledTask -TaskName AforoVision"
    $answer = Read-Host "Arrancar de todos modos? (s/N)"
    if ($answer -notmatch '^[sSyY]') { exit 1 }
}

Set-Location -LiteralPath $RepoRoot
& $Python -m src.main @args
exit $LASTEXITCODE
