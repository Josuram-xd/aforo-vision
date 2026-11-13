# Elimina la tarea programada "AforoVision" creada por install_autostart.ps1.
# Uso: .\scripts\windows\uninstall_autostart.ps1
# No borra los logs de data\logs ni el entorno virtual.

$ErrorActionPreference = 'Stop'
$TaskName = 'AforoVision'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "La tarea '$TaskName' no esta instalada. Nada que hacer." -ForegroundColor Yellow
    exit 0
}

try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
} catch {
    Write-Host "ERROR: no se pudo eliminar la tarea: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "Tarea '$TaskName' eliminada. El sistema ya no la arrancara al iniciar sesion." -ForegroundColor Green
