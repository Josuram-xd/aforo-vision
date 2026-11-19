# Comandos rápidos (Windows, PowerShell, dentro de `aforo-vision`)

La tarea `AforoVision` arranca sola al **iniciar sesión** en Windows y corre la visión y la sincronización de inscripciones web.

## Instalar (una sola vez)
```powershell
.\scripts\windows\install_autostart.ps1
```
Requiere `.venv` y `.env` (con `AFORO_BACKEND_URL`, `AFORO_SUPABASE_KEY`, `AFORO_DEVICE_KEY`).

## Controlar
```powershell
Get-ScheduledTask AforoVision                                   # estado (Running = activo)
Start-ScheduledTask AforoVision                                 # iniciar
Stop-ScheduledTask AforoVision                                  # detener
Stop-ScheduledTask AforoVision; Start-ScheduledTask AforoVision # reiniciar (carga embeddings nuevos)
.\scripts\windows\uninstall_autostart.ps1                       # quitar el autoarranque
```

## Logs
```powershell
Get-Content data\logs\vision.log -Wait -Tail 20
Get-Content data\logs\enrollment.log -Wait -Tail 20
```

## A mano (detén la tarea antes), en dos terminales
```powershell
.\.venv\Scripts\python -m src.main
.\.venv\Scripts\python -m enrollment.sync_web_enrollments --watch
```

## Estudiantes
```powershell
python -m enrollment.export_roster                      # lista el roster sin biometría
python -m enrollment.remove_student --person-id <id>    # borra el embedding local de alguien
```
Si un estudiante queda en `failed`, repite "Registrar rostro" en la web (fotos de frente y con buena luz).
