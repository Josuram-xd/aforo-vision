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
- [ ] Backend desplegado y `GET <AFORO_BACKEND_URL>/health` responde OK desde el navegador.
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
$env:AFORO_BACKEND_URL = "https://<url-del-backend>"
$env:AFORO_BACKEND_SECRET = "<secreto-compartido>"
$env:AFORO_CAMERA_OUTSIDE_SOURCE = "rtsp://<usuario>:<clave>@<ip-camara>:554/stream1"
# La cámara de adentro (USB) usa el índice 0 de config/pilot.yaml; solo cámbialo si Windows la numera distinto.
```

- [ ] Las cuatro variables definidas en esta terminal.

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
   - `backend OK (...)` — si sale `NO hay conexion con el backend`, ver sección 5.
   - `pipeline running`.
6. [ ] En las **dos** ventanas de depuración se ve video, con FPS y cajas sobre una persona de prueba que camine frente a cada cámara.
7. [ ] **Prueba rápida**: un compañero entra al salón por la puerta. Debe salir un evento `ENTRY` en el backend/frontend. Luego sale: debe salir `EXIT`.
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
| `falta AFORO_BACKEND_SECRET` y se cierra | Definir la variable (sección 2) en la misma terminal y volver a arrancar. |
| `NO hay conexion con el backend` al arrancar | No es bloqueante: los eventos se guardan en la cola local (`data/retry_queue.sqlite`) y se envían en orden cuando vuelva la red. Revisar que el hotspot tenga datos y que `AFORO_BACKEND_URL` esté bien. |
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
