# Runbook del día del piloto

Checklist para operar `aforo-vision` el día de la prueba. Todos los comandos se corren desde la raíz del repo (`aforo-vision/`). Marca cada casilla en orden; no pases al siguiente bloque si uno falla.

> Tiempos sugeridos: llegar **60 min** antes de la hora de inicio.

## 0. La noche anterior

- [ ] Laptop cargado y cargador en la maleta. Plan de energía en "Alto rendimiento" y suspensión desactivada.
- [ ] Celular con datos suficientes para el hotspot y cargado (o con cargador).
- [ ] Cable USB de la cámara, cable/adaptador de corriente de la cámara WiFi, extensión eléctrica.
- [ ] Aviso de videovigilancia impreso (`docs/aviso-videovigilancia.md`, ya completado con fecha, hora, responsable y contacto).
- [ ] Autorizaciones firmadas (`docs/consentimiento-informado.md`) guardadas **en físico**: solo se enrola a quien firmó.
- [ ] Estudiantes enrolados: revisar que `data/embeddings/` tenga solo a quienes firmaron (`python -m enrollment.export_roster` lista el roster sin biometría).
- [ ] Variables de entorno listas (ver sección 2). Nunca van en `config/pilot.yaml` ni en git.
- [ ] Proyecto de Supabase activo (no pausado) y la llave del salón (`AFORO_DEVICE_KEY`) vigente y a la mano (ver sección 5.1).
- [ ] Prueba en seco hecha: `python -m src.main --dry-run --debug` arranca sin errores.

## 1. Montaje físico (antes de que lleguen los estudiantes)

- [ ] **Aviso pegado en la puerta**, a la altura de los ojos y **antes** de la zona que ven las cámaras.
- [ ] **Hotspot del celular encendido** (no usar el WiFi institucional: suele aislar dispositivos entre sí). Anota el nombre y la clave.
- [ ] Laptop conectado al hotspot.
- [ ] **Cámara WiFi (afuera, pasillo)** encendida y conectada al mismo hotspot. Confirma su IP y que coincida con la de `AFORO_CAMERA_OUTSIDE_SOURCE` (si el hotspot le asignó otra IP, corrige la variable).
- [ ] **Cámara USB (adentro, junto al laptop)** conectada al laptop.
- [ ] Encuadre: cada cámara ve a la persona de cuerpo casi completo y la zona de los ojos es visible; sin contraluz fuerte de la puerta o ventana. Las dos cubren la misma puerta, una por cada lado.
- [ ] Cables asegurados con cinta para que nadie los jale.

## 2. Variables de entorno

PowerShell, en la misma terminal donde se va a arrancar el sistema:

```powershell
$env:AFORO_BACKEND_URL = "https://<ref-del-proyecto>.supabase.co"   # URL del proyecto de Supabase
$env:AFORO_SUPABASE_KEY = "<llave-publishable>"                     # sb_publishable_..., la misma del frontend
$env:AFORO_DEVICE_KEY = "<llave-del-salon>"                         # una por salón, se pide al admin (sección 5.1)
$env:AFORO_CAMERA_OUTSIDE_SOURCE = "rtsp://<usuario>:<clave>@<ip-camara>:554/stream1"
# La cámara de adentro (USB) usa el índice 0 de config/pilot.yaml; solo cámbialo si Windows la numera distinto.
```

- [ ] Las cinco variables definidas en esta terminal. Nunca las pegues en el chat, en capturas, en `config/pilot.yaml` ni en git (el archivo `.env` local está en `.gitignore`).

## 3. Orden de arranque

1. [ ] Hotspot encendido y laptop conectado.
2. [ ] Cámara WiFi encendida; esperar a que obtenga IP (~1 min).
3. [ ] Cámara USB conectada.
4. [ ] Arrancar con vista de depuración para verificar el encuadre:
   ```powershell
   python -m src.main --debug
   ```
5. [ ] En la consola debe aparecer, en este orden:
   - `N estudiante(s) enrolado(s)` — N debe coincidir con el roster esperado.
   - `backend OK (Supabase acepto las llaves)` — si sale `NO hay conexion con Supabase o rechazo las llaves`, ver sección 5.
   - `pipeline running`.
6. [ ] En las **dos** ventanas de depuración se ve video, con FPS y cajas sobre una persona de prueba que camine frente a cada cámara.
7. [ ] **Prueba rápida**: un compañero entra al salón por la puerta. Debe salir un evento `ENTRY` en el dashboard. Luego sale: debe salir `EXIT`.
8. [ ] Cerrar las ventanas de depuración (`q` o `Esc`) y volver a arrancar **sin** `--debug` para ahorrar CPU durante la sesión:
   ```powershell
   python -m src.main
   ```
9. [ ] Anotar la hora de inicio real para compararla luego con el conteo manual.

## 4. Durante la sesión

- [ ] Una persona hace el **conteo manual de control** (entradas y salidas, con hora) en una hoja. Es la referencia de la métrica de éxito.
- [ ] No cerrar la tapa ni dejar que el laptop se suspenda; no abrir programas pesados.
- [ ] Cada ~15 min mirar la consola: que sigan saliendo líneas y que no se repitan avisos de cámara caída.
- [ ] Anotar cualquier incidente con hora (cámara que se cayó, persona que pasó de espaldas, grupo que entró junto, etc.).

## 5. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `falta(n) AFORO_SUPABASE_KEY, AFORO_DEVICE_KEY` y se cierra | Definir las variables (sección 2) en la misma terminal y volver a arrancar. |
| `NO hay conexion con Supabase o rechazo las llaves` al arrancar | No es bloqueante: los eventos se guardan en la cola local (`data/retry_queue.sqlite`) y se envían en orden cuando se arregle. Mirar la línea anterior del log: si dice `refused with 401/403` son las llaves (ver abajo); si dice `failed (ConnectError...)` es la red: revisar el hotspot y `AFORO_BACKEND_URL`. |
| `ingest_event refused with 401/403` o `camera_heartbeat ... refused` | Llave incorrecta, revocada o ausente. Revisar `AFORO_SUPABASE_KEY` y `AFORO_DEVICE_KEY` (sin espacios ni comillas de más). Si siguen fallando, la llave del salón fue revocada: pedir una nueva (sección 5.1). Los eventos **no se pierden**: quedan en la cola y se envían al corregir la llave. |
| `ingest_event rejected event ... with 400` | El evento es inválido (campo faltante, `confidence` fuera de 0..1) o su hora está más de 5 min en el futuro (reloj del laptop mal; sincronizarlo). Ese evento se aparta en la cola como rechazado y no se reintenta; los demás siguen. |
| `ingest_event answered 429` | Se superó el límite de 120 peticiones por minuto. No hacer nada: el sistema espera y reintenta solo, en orden. |
| `ingest_event answered 5xx` / `failed (ConnectError...)` | Supabase o la red caídos. No hacer nada: reintenta con espera creciente (1 s hasta 60 s). |
| El dashboard muestra una cámara `offline` o abre la alerta `camera_offline` | Cada 30 s el sistema reporta si cada cámara está conectada. Revisar la cámara (fila "Una cámara aparece desconectada", abajo). Si las dos aparecen offline pero el video funciona, revisar las llaves o la red: los latidos no están llegando. |
| Se cae internet durante la sesión | No hacer nada: los eventos quedan en cola y se reenvían solos. Al cerrar, la consola indica cuántos quedaron en cola; reiniciar con red para vaciarla. |
| Una cámara aparece desconectada / ventana en negro | El sistema reintenta solo cada 5 s. Si no vuelve: USB → desconectar y reconectar el cable; WiFi → revisar que siga en el hotspot y su IP (puede haber cambiado; actualizar `AFORO_CAMERA_OUTSIDE_SOURCE` y reiniciar). |
| La cámara WiFi no aparece en el hotspot | Reiniciarla (desconectar corriente 10 s). Confirmar que el hotspot esté en 2.4 GHz, si la cámara no soporta 5 GHz. |
| FPS muy bajos / el video va con retraso | Cerrar otros programas, quitar `--debug` si está activo y conectar el laptop a la corriente. |
| Todos los eventos salen sin nombre | Revisar que el log inicial indique estudiantes enrolados y que haya luz suficiente en la zona de los ojos. El conteo sigue funcionando por apariencia corporal. |
| El programa se cerró o falló | Volver a ejecutar `python -m src.main` (con las variables definidas). La cola persistente conserva los eventos pendientes. Anotar la hora de la interrupción. |
| Alguien pide no ser grabado / contado o retirar su autorización | Atender de inmediato según el aviso y el consentimiento (alternativa indicada en el aviso). Anotar el caso; los datos de esa persona se eliminan luego. |

Si no se puede resolver en ~10 min: seguir con el **conteo manual** como respaldo y documentar el incidente.

## 6. Cierre

- [ ] Detener el sistema con `Ctrl+C`. Esperar el mensaje `stopped` y revisar cuántos eventos quedaron en cola.
- [ ] Si quedaron eventos en cola, reconectar a internet y volver a arrancar `python -m src.main` hasta que la cola se vacíe; luego `Ctrl+C`.
- [ ] Comparar el conteo del sistema contra el conteo manual y anotar las diferencias.
- [ ] Retirar el aviso, apagar cámaras y hotspot.
- [ ] Guardar las autorizaciones firmadas en un lugar seguro (nunca en el repo ni en la nube).
- [ ] Recordar: el video **no se grabó** y no sale del laptop; solo viajó el JSON de los eventos.

### 5.1 Pedir, rotar o revocar la llave del salón

- La **llave publishable** (`AFORO_SUPABASE_KEY`) no es secreta como una contraseña (también la usa el frontend) pero tampoco se publica; da acceso solo a lo que permiten las políticas de la base de datos.
- La **llave del salón** (`AFORO_DEVICE_KEY`) identifica este equipo y es lo que autoriza escribir eventos y latidos. Hay una por salón.
- **Pedirla**: un usuario `admin` del dashboard la genera (o la crea en `aforo-db`, task 8.2) y te la entrega por un canal privado (mensaje directo, no un grupo ni un correo reenviado). Se muestra una sola vez: guárdala en un gestor de contraseñas o en el `.env` local.
- **Revocarla** (laptop perdido, llave expuesta por error en un chat o commit): el `admin` la revoca en `aforo-db` y genera una nueva. Desde ese momento la anterior recibe 403. Después de revocar, actualiza `AFORO_DEVICE_KEY` en cada equipo y reinicia `python -m src.main`; la cola conserva lo pendiente.
- Si una llave se filtró, revócala primero y avisa después; no esperes a confirmar el uso indebido.

### 6.1 Atender una solicitud de supresión (alguien revoca su autorización)

Plazo: el que diga la autorización firmada (sección 5 de `docs/consentimiento-informado.md`). Hay que hacer **los dos pasos**: uno borra los datos de la nube y el otro la biometría de este equipo.

1. **Nube (dashboard):** un `admin` entra a **Estudiantes**, busca a la persona y pulsa **Eliminar**. Para confirmar hay que escribir su nombre. Se borran su ficha, sus inscripciones y su asistencia, y su nombre, correo y código desaparecen también de la auditoría. Sus entradas y salidas pasadas quedan sin nombre, así que el conteo de aforo no cambia. El diálogo muestra el `personId`: cópialo para el paso 2.
2. **Este equipo:** con el pipeline detenido, corre:
   ```powershell
   python -m enrollment.remove_student --person-id <personId>
   ```
   Escribe `BORRAR` para confirmar. El script borra `data/embeddings/<personId>.npy` y `.json`, y quita a la persona de `data/roster.json`. Después vuelve a arrancar `python -m src.main`: desde ese momento ya no se la reconoce, y si cruza la puerta cuenta como no identificada.
3. Si hay copias del roster fuera de `data/` (por ejemplo, `aforo-db/scripts/roster.json`), bórralas o vuelve a exportarlas con `python -m enrollment.export_roster`.
4. Confirma por escrito a la persona que sus datos se borraron, con la fecha. Guarda esa confirmación junto a las autorizaciones firmadas (nunca en el repo).
