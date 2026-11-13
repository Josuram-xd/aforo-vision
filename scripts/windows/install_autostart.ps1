# Instala la tarea programada "AforoVision": al iniciar sesión arrancan el pipeline de visión
# y la sincronización de inscripciones web, ocultos, con reinicio automático y logs en data\logs.
# Uso (una sola vez): .\scripts\windows\install_autostart.ps1
# Se puede volver a ejecutar: reemplaza la tarea existente.

$ErrorActionPreference = 'Stop'
$TaskName = 'AforoVision'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot '.env'
$LogDir = Join-Path $RepoRoot 'data\logs'
$VisionLog = Join-Path $LogDir 'vision.log'
$EnrollmentLog = Join-Path $LogDir 'enrollment.log'

function Stop-Install([string]$Message) {
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path -LiteralPath $Python)) {
    Stop-Install "no se encontro $Python. Crea el entorno virtual en la raiz del repo (python -m venv .venv) e instala requirements.txt."
}
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Stop-Install "no se encontro $EnvFile. Crea el archivo .env con AFORO_SUPABASE_KEY y AFORO_DEVICE_KEY."
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# El supervisor corre oculto y lanza los dos procesos; cada uno escribe su salida en su log.
# Si cualquiera termina, mata al otro y sale con error: Task Scheduler lo reinicia (RestartCount).
# Las rutas se pasan como literales de PowerShell (las comillas simples se duplican).
function Quote-Literal([string]$Text) { "'" + $Text.Replace("'", "''") + "'" }
$supervisorTemplate = @'
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath __ROOT__
$vision = Start-Process -FilePath cmd.exe -WorkingDirectory __ROOT__ -WindowStyle Hidden -PassThru -ArgumentList ('/d /c ""' + __PY__ + '" -m src.main >> "' + __VLOG__ + '" 2>&1"')
$enroll = Start-Process -FilePath cmd.exe -WorkingDirectory __ROOT__ -WindowStyle Hidden -PassThru -ArgumentList ('/d /c ""' + __PY__ + '" -m enrollment.sync_web_enrollments --watch >> "' + __ELOG__ + '" 2>&1"')
while (-not $vision.HasExited -and -not $enroll.HasExited) { Start-Sleep -Seconds 5 }
foreach ($p in @($vision, $enroll)) {
    if (-not $p.HasExited) { & taskkill.exe /T /F /PID $p.Id | Out-Null }
}
exit 1
'@
$supervisor = $supervisorTemplate.
    Replace('__ROOT__', (Quote-Literal $RepoRoot)).
    Replace('__PY__', (Quote-Literal $Python)).
    Replace('__VLOG__', (Quote-Literal $VisionLog)).
    Replace('__ELOG__', (Quote-Literal $EnrollmentLog))
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($supervisor))

$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $RepoRoot `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -EncodedCommand $encoded"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description 'Aforo: pipeline de vision y sincronizacion de inscripciones al iniciar sesion.' `
        -Force | Out-Null
} catch {
    Stop-Install "no se pudo registrar la tarea: $($_.Exception.Message)"
}

Write-Host "Tarea '$TaskName' instalada. Arranca sola al iniciar sesion de $user." -ForegroundColor Green
Write-Host "Si ya estaba corriendo, cierra sesion y vuelve a entrar para aplicar los cambios."
Write-Host "Logs: $VisionLog y $EnrollmentLog"
