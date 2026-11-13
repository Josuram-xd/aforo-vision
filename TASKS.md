# TASKS — aforo-vision

Pipeline local de visión (laptop): 2 cámaras → detección → tracking → identidad → dirección → evento JSON → `aforo-backend`.

## Cómo usar este archivo

- Cada subtarea es **un commit**. Formato: `tipo(alcance): descripción [x.y]`
  Ejemplo: `git commit -m "feat(capture): add threaded frame grabber [2.1]"`
- En ese mismo commit cambia `[ ]` por `[x]` en este archivo, así el historial y el checklist quedan sincronizados.
- Las subtareas marcadas **(sin commit)** son verificaciones manuales: márcalas en el siguiente commit que hagas.
- Tipos: `feat` (funcionalidad), `fix` (corrección), `test`, `docs`, `chore` (configuración/estructura), `perf` (rendimiento), `refactor`.
- **Prioridad 1** = sin esto no hay piloto. **Prioridad 2** = importante para que el piloto salga bien. **Prioridad 3** = extra si sobra tiempo.
- `> Depende de:` indica que antes tienes que avanzar tareas de otro repo.

## Orden global entre los 4 repos

| Fase | aforo-db | aforo-backend | aforo-vision | aforo-frontend |
|---|---|---|---|---|
| A — Arranque (en paralelo) | 1-3 | 1-2 | 1-3 | 1-2 |
| B — Núcleo | 4 | 3-7 | 4-6 | 3-5 (con datos mock) |
| C — Integración | — | — | 7 | 6 |
| D — Ensayo y ajustes | 5 | 8-9, 11 (login) | 8-11 | 7-8, 11 (login) |
| E — Extras | 6 | 10 | 12-13 | 9-10 |

**Hito fin de septiembre:** fase A completa → modelo corriendo con la webcam del laptop (este repo, tasks 1-3) y todo lo demás desplegado aunque sea en "hola mundo".

### Versión 2 (después de la primera entrega, octubre 2026)

Retroalimentación del profe: pocas tablas, sin registro ni login, endpoints públicos, AWS sin protección contra abuso y un frontend muy básico. La v2 pasa los datos y el login a **Supabase** (PostgreSQL + Auth con Google + RLS + Realtime). Detalle en la sección "Version 2" de cada `ARCHITECTURE.md`.

| Fase | aforo-db | aforo-backend | aforo-vision | aforo-frontend |
|---|---|---|---|---|
| F — Base v2 | 7 | 12.1-12.2 | — | 12 |
| G — Login, seguridad e integración | 8 | 13 | 14 | 13 |
| H — Producto | 9 | — | 15 (evaluar visión en la nube) | 14-15 |
| I — Cierre | 10 | 12.3-12.4 | 15.3-15.6, 16 | 16 |

Las tareas v1 que la v2 reemplaza quedan marcadas con ~~tachado~~ y una nota; no se borran para conservar el historial.

---

## Prioridad 1 — Crítico

### Task 1 — Inicializar el proyecto
- [x] **1.1** `chore: init project structure` — Crear `src/` con los paquetes `capture/`, `detection/`, `tracking/`, `identity/`, `matching/`, `direction/`, `dedup/`, `events/`, más `enrollment/`, `config/`, `tests/`. Añadir `.gitignore` que excluya `data/`, `*.npy`, `roster.json`, `recordings/` y `.venv/` (nada biométrico ni video al repo).
- [x] **1.2** `chore: pin dependencies for python 3.14 cpu` — `requirements.txt` con `ultralytics`, `opencv-python`, `numpy`, `torch` (CPU), `insightface`, `onnxruntime`, `torchreid`, `httpx`, `pyyaml`, `pytest`. Antes de fijar versiones, verificar que `insightface` y `onnxruntime` tengan wheels para Python 3.14; si no, usar un venv con 3.13 y anotarlo en el README.
- [x] **1.3** `feat(config): add pilot config file and loader` — `config/pilot.yaml` (fuentes de cámara, umbrales, ventana de tiempo, URL del backend) + `src/config.py` con una función `load_config(path)` que devuelva un dataclass.
- [x] **1.4** `docs: add PRD, ARCHITECTURE and AGENTS` — Subir los documentos del proyecto a la raíz.

### Task 2 — Captura de cámaras
- [x] **2.1** `feat(capture): add threaded frame grabber for usb and rtsp` — `src/capture/frame_grabber.py`: clase `FrameGrabber(source, camera_id)` con un hilo por cámara y método `read_latest()`; la misma clase acepta índice USB (`0`) o URL RTSP.
- [x] **2.2** `feat(capture): auto-reconnect dropped streams` — Si la cámara WiFi se cae, reintentar la conexión cada N segundos sin tumbar el programa.
- [x] **2.3** `feat(debug): add local preview window with fps overlay` — Ventana `cv2.imshow` por cámara con FPS y cajas dibujadas; se activa con un flag `--debug`.
- [ ] **2.4** (sin commit) Verificar **antes de comprar** que la cámara WiFi exponga RTSP u ONVIF en red local. Presupuesto total del proyecto: ~500.000 COP. **(v2)** Si gana la opción C (task 15.2), comprobar también que el stream llegue a la VM por el túnel de Tailscale (task 15.4).

### Task 3 — Detección y tracking **[HITO SEPT]**
- [x] **3.1** `feat(detection): add yolo26n-pose detector wrapper` — `src/detection/yolo_pose.py`: función `detect(frame) -> list[Detection]` con caja, score y keypoints.
- [x] **3.2** `feat(tracking): add bipartite graph and hungarian assignment` — `src/tracking/hungarian.py`: implementación propia del grafo bipartito y el algoritmo húngaro (estructura principal del curso). Costo = 1 − IoU.
- [x] **3.3** `feat(tracking): add sort tracker with kalman filter` — `src/tracking/sort_tracker.py`: clase `SortTracker` que usa 3.2 para asociar detecciones con tracks entre frames y asigna IDs estables.
- [x] **3.4** `feat(tracking): keep per-track trajectory deque` — Cada track guarda un `deque(maxlen=N)` con sus últimos centroides y keypoints.
- [x] **3.5** `feat(tracking): handle partial occlusion with iou and keypoints` — Si dos cajas se solapan, usar los keypoints visibles para no fusionar a dos personas en un solo track.
- [x] **3.6** `test(tracking): compare hungarian against scipy on known cases` — Tests que validen 3.2 contra `scipy.optimize.linear_sum_assignment`.

### Task 4 — Identidad
- [x] **4.1** `feat(identity): add periocular crop from face landmarks` — `src/identity/periocular.py`: función `crop_periocular(frame, landmarks)` que recorta solo la región de los ojos (funciona con tapabocas).
- [x] **4.2** `feat(identity): add arcface embedding extractor` — Función `embed(crop) -> np.ndarray` usando InsightFace (ArcFace), normalizada.
- [x] **4.3** `feat(identity): add embedding lookup hash table` — Clase `IdentityIndex`: tabla hash `personId → embedding` con búsqueda por similitud coseno y umbral configurable.
- [x] **4.4** `feat(identity): add osnet body re-id fallback` — `src/identity/body_reid.py`: embedding corporal para cuando no hay rostro usable; solo sirve para contar y emparejar, nunca para poner nombre.
- [x] **4.5** `test(identity): cover lookup thresholds` — Tests con embeddings sintéticos: match, no-match y empate.

### Task 5 — Enrolamiento del curso
- [x] **5.1** `feat(enrollment): add enroll student script` — `enrollment/enroll_student.py`: captura N muestras con la webcam, promedia el embedding periocular y lo guarda en `data/embeddings/` (local, fuera de git).
- [x] **5.2** `feat(enrollment): export roster without biometrics` — Generar `roster.json` solo con `personId` (UUID v4, como exige el contrato de `aforo-backend`; `aforo-db/scripts/seed_people.py` rechaza otros formatos) y `name` (sin embeddings). Después sigue con la task 4 del repo: `aforo-db`.
- [x] **5.3** `docs: add consent form and surveillance notice templates` — Formato de consentimiento informado y aviso de videovigilancia para la puerta (Ley 1581 de 2012).

### Task 6 — Checkpoints y dirección
- [x] **6.1** `feat(matching): add cross-checkpoint bipartite matcher` — `src/matching/cross_checkpoint.py`: empareja tracks de `camera-outside` con tracks de `camera-inside` dentro de una ventana de tiempo, reutilizando el húngaro de 3.2 con costo = distancia entre embeddings.
- [x] **6.2** `feat(direction): resolve entry and exit by checkpoint order` — `src/direction/resolver.py`: afuera → adentro = `ENTRY`, adentro → afuera = `EXIT`.
- [x] **6.3** `feat(direction): add trajectory fallback for single-camera tracks` — Si solo una cámara vio a la persona, usar el deque de 3.4 para inferir la dirección (señal secundaria).
- [x] **6.4** `feat(dedup): add ttl hash set for crossings` — `src/dedup/ttl_set.py`: clase `TTLSet` que evita contar dos veces el mismo cruce.
- [x] **6.5** `test(direction): cover entry, exit, walking backwards and single camera` — Casos con secuencias sintéticas, incluido alguien que entra de espaldas (debe salir `ENTRY` igual).

### Task 7 — Eventos y envío al backend
> Depende de: Seguir con las task 1-7 del repo: `aforo-backend`

- [x] **7.1** `feat(events): add event builder matching shared contract` — `src/events/builder.py`: función `build_event(...)` con el esquema de `aforo-backend/ARCHITECTURE.md`. Mantener sincronizado con la task 3 del repo `aforo-backend`.
- [x] **7.2** `feat(events): add https uploader for post events` — `src/events/uploader.py`: clase `EventUploader` con `send(event)` sobre `httpx`.
- [x] **7.3** `feat(events): add persistent fifo retry queue` — Cola FIFO en SQLite: si falla el envío, el evento se guarda y se reintenta en orden cuando vuelva la red.
- [x] **7.4** `feat(main): wire full pipeline for two cameras` — `src/main.py`: une captura → detección → tracking → identidad → matching → dirección → dedup → envío.
- [x] **7.5** `feat(main): check backend health on startup` — Llamar `GET /health` (task 2 de `aforo-backend`) al arrancar y avisar en consola si no hay conexión.

### Task 8 — Ensayo del piloto
- [x] **8.1** `docs: add pilot day runbook` — Checklist del día: hotspot encendido, cámaras conectadas, aviso pegado en la puerta, orden de arranque, qué hacer si algo falla.
- [ ] **8.2** (sin commit) Ensayo real en la puerta comparando contra un conteo manual. **(v2)** Hacerlo después de la task 14.7, ya enviando a Supabase y mirando el dashboard v2.
- [ ] **8.3** `chore(config): tune thresholds and time window from rehearsal` — Ajustar umbral de similitud y ventana entre cámaras con los datos del ensayo.
- [ ] **8.4** (sin commit) Antes del día real, limpiar los datos del ensayo con `scripts/reset_pilot_supabase.py` del repo `aforo-db` (task 8.3; el `reset_pilot.py` de la task 4.3 era de DynamoDB y se elimina en la task 10.3).

---

## Prioridad 2 — Importante

### Task 9 — Envío autenticado
> Depende de: Seguir con la task 8 del repo: `aforo-backend`

- [ ] ~~**9.1** `feat(events): send shared secret header` — Enviar el header con el secreto compartido en cada `POST /events`; leerlo de una variable de entorno, nunca de `pilot.yaml`.~~ → reemplazada por la task 14 (llaves de Supabase).

### Task 10 — Rendimiento en CPU
- [ ] **10.1** `perf(detection): skip frames adaptively when fps drops` — Procesar 1 de cada N frames cuando los FPS bajen de un mínimo. **(v2)** Hacerla antes del benchmark de la task 15.1.
- [ ] **10.2** `perf(identity): cache embedding per track until matched` — No recalcular el embedding facial en cada frame una vez el track ya tiene identidad. **(v2)** Hacerla antes del benchmark de la task 15.1.

### Task 11 — Registro para el informe
- [ ] ~~**11.1** `feat(logging): log each decision to local csv` — Un CSV local con cada evento, método (`FACE`/`BODY_ONLY`) y confianza, sin imágenes, para las métricas del informe.~~ → innecesaria en v2: la tabla `events` de Supabase ya guarda `method` y `confidence`, y la Edge Function `export-events` (`aforo-backend` task 13.1) los descarga en CSV.

---

## Prioridad 3 — Extras

### Task 12 — Vista en vivo para la demo
> Depende de: la task 14.6 (login con el JWT de Supabase, ya hecha en `src/stream/auth.py`) y la decisión de la task 15.2 (dónde corre el pipeline).

- [ ] **12.1** `feat(stream): serve mjpeg streams on local network` — `src/stream/server.py`: `GET /stream/<cameraId>` con los frames de cada cámara, escuchando solo en la IP de la red local (configurable en `pilot.yaml`). Nunca escribe frames a disco. Exige login con `src/stream/auth.py` (roles `viewer`, `teacher` o `admin`; sin token 401, sin rol o cuenta desactivada 403) e incluye tests de las rutas. Si gana la opción C (task 15.2), el servidor queda en la VM y solo se expone por Tailscale, nunca a internet.
- [ ] ~~**12.2** `feat(stream): require cognito token and group` — Validar el token de Cognito (firma con JWKS cacheado, issuer, client, expiración) y el claim `cognito:groups`: `viewer` o `dev` para ver; sin token 401, grupo incorrecto 403. IDs del User Pool por variable de entorno.~~ → se adapta a Supabase en la task 14.6.
- [ ] **12.3** `feat(stream): add dev analysis overlay stream` — `GET /stream/<cameraId>/dev` solo para `admin` (`OVERLAY_ROLES` de `src/stream/auth.py`; el grupo `dev` de Cognito ya no existe): video con cajas, ID de track, nombre + confianza, `FACE`/`BODY_ONLY`, checkpoint, dirección resuelta y FPS (reutiliza `src/debug/preview.py`).
- [ ] ~~**12.4** `test(stream): cover auth and group checks` — Tokens firmados con una llave de prueba: válido, expirado, sin grupo, `viewer` pidiendo `/dev` (403).~~ → cubierta por `tests/test_stream_auth.py` (task 14.6); los tests de las rutas van en la 12.1.
  Después sigue con la task 17 del repo: `aforo-frontend` (video en vivo).

### Task 13 — Pruebas con video grabado
- [ ] **13.1** `test: replay recorded clips through the pipeline` — Test de integración con clips del ensayo (grabados con consentimiento, guardados solo en local y excluidos de git).

---

## Versión 2 — Supabase y dónde corre la visión

> Ver `ARCHITECTURE.md` sección 10.

### Task 14 — Envío a Supabase **(Prioridad 1)**
> Depende de: `aforo-db` tasks 7 y 8.2 (proyecto, RPCs y llave del salón).

- [x] **14.1** `feat(config): read supabase url and keys from env` — `AFORO_BACKEND_URL` pasa a ser la URL del proyecto de Supabase; nuevas `AFORO_SUPABASE_KEY` (publishable) y `AFORO_DEVICE_KEY`; se elimina `AFORO_BACKEND_SECRET`. Nunca en `pilot.yaml` ni en logs.
- [x] **14.2** `feat(events): post events to ingest event rpc` — `EventUploader` envía `{"p_event": evento}` a `/rest/v1/rpc/ingest_event` con `apikey` y `X-Device-Key`; misma cola FIFO y mismo backoff. Mapeo de códigos de `ARCHITECTURE.md` 10.1.
- [x] **14.3** `feat(events): add camera heartbeat thread` — Hilo que cada 30 s llama `camera_heartbeat` por cámara con `is_connected`; al arrancar reemplaza el `GET /health` (`backend OK`).
- [x] **14.4** `test(events): cover supabase status codes and heartbeats` — Actualizar `test_uploader`, `test_health` y `test_config` con `httpx.MockTransport`: 200, 200 duplicado, 400, 401/403, 429, 5xx, red caída; las llaves nunca aparecen en logs.
- [x] **14.5** `docs: update runbook for supabase keys` — `docs/runbook-dia-piloto.md`: nuevas variables de entorno, qué significa cada error y cómo pedir/revocar la llave del salón.
- [x] **14.6** `feat(stream): validate supabase jwt instead of cognito` — Para la task 12: validar el JWT de Supabase (JWKS del proyecto) y el rol del usuario.
- [ ] **14.7** (sin commit) Prueba de punta a punta: una persona cruza la puerta y el evento aparece en el dashboard v2 en menos de 3 s; apagar una cámara abre la alerta `camera_offline`.

### Task 15 — Sin depender del laptop **(Prioridad 2)**
- [ ] **15.1** (sin commit) **Benchmark:** medir FPS por cámara con 2 streams y reconocimiento activo (1 persona y 2 personas cruzando a la vez) en: el laptop, la VM gratis de Oracle (Ampere A1, 4 OCPU / 24 GB) y, si se consigue prestado, un mini PC o Raspberry Pi 5. Anotar FPS, CPU y latencia cruce → dashboard. Criterio: ≥5 FPS por cámara y < 3 s de latencia.
- [ ] **15.2** `docs: decide where the pipeline runs (adr-007)` — Con los datos de 15.1, elegir entre equipo en el salón (opción B) y nube gratis (opción C). Si se compra hardware, revisar el presupuesto total (~500.000 COP, regla 9).
- [ ] **15.3** `chore(deploy): add systemd service for headless run` — Arranque automático al encender, reinicio si se cae, sin ventana `--debug`; logs con `journalctl`.
- [ ] **15.4** `docs: add tailscale tunnel setup` — Solo si gana la opción C: dispositivo en el salón como *subnet router* de Tailscale para que la VM llegue a las cámaras por un túnel cifrado (WireGuard); las cámaras nunca quedan expuestas a internet.
- [ ] **15.5** `feat(identity): encrypt embeddings at rest` — Los embeddings del enrolamiento se guardan cifrados (llave en variable de entorno) en el equipo que procesa; nunca en Supabase.
- [ ] **15.6** `docs: update surveillance notice and agents rule for remote processing` — Solo si gana la opción C: actualizar `docs/aviso-videovigilancia.md` (el video viaja cifrado a un servidor propio, se procesa en memoria y no se graba) y la regla 1 de `AGENTS.md`.
- [ ] **15.7** `chore(deps): verify arm64 wheels or export models to onnx` — Si el equipo elegido es ARM64: confirmar wheels para Python 3.14; si `torch` falla, exportar YOLO a ONNX y correr todo con ONNX Runtime.

### Task 16 — Enrolamiento desde la web **(Prioridad 3, pendiente de decisión)**
- [ ] **16.1** (sin commit) Decidir con el equipo si se hace (ver `ARCHITECTURE.md` 10.5). Si no, se sigue enrolando en persona con `enroll_student.py`.

### Task 17 — Borrar la biometría de un estudiante **(Prioridad 1)**
> Los embeddings solo existen en el equipo de visión (`data/embeddings/`), así que borrar al estudiante en el dashboard (`aforo-frontend` task 19.1) no los borra.

- [x] **17.1** `feat(enrollment): add remove student script` — `python -m enrollment.remove_student --person-id <uuid>`: borra `data/embeddings/<id>.npy` y `<id>.json`, valida que el id sea un UUID (sin rutas raras) y pide confirmación. El pipeline deja de reconocerlo al reiniciar.
- [x] **17.2** `docs: add student erasure steps to runbook` — Runbook y formato de consentimiento: cómo se atiende una solicitud de supresión (dashboard + script) y en cuánto tiempo.
- [x] **17.3** `feat(events): stop sending person name to supabase` — `ingest_event` ignora `personName` (el nombre sale de `students`), así que el nombre viajaba por internet sin necesidad. Enviar solo `personId` (minimización de datos). Contrato: `personName` pasa a ser siempre `null`; avisar a `aforo-db` y `aforo-backend`.

### Task 18 — Registro de rostros desde la web **(Prioridad 1)**
> Depende de: `aforo-db` task 12 (RPCs `pending_enrollments` y `complete_enrollment`) y `aforo-frontend` task 21.

- [x] **18.1** `feat(enrollment): add script to enroll students from web photos` — `python -m enrollment.sync_web_enrollments [--watch]`: pide las fotos pendientes con la llave de dispositivo, calcula el embedding periocular en local (mínimo 3 fotos con un solo rostro), lo guarda en `data/embeddings/<studentId>.npy` y llama a `complete_enrollment`, que borra las fotos en la nube. Si fallan, marca el registro como fallido. Excepción documentada a la regla 1 de `AGENTS.md`.

### Task 19 — Cámaras conectadas desde la web **(Prioridad 1)**
> Depende de: `aforo-db` task 14 (RPC `camera_sources`) y `aforo-frontend` task 24.

- [x] **19.1** `feat(capture): read camera connections registered in the web app` — Al arrancar y en cada reconexión, `FrameGrabber` pide la IP y cuenta de cada Tapo con la llave del salón y arma `rtsp://...:554/stream1`. Orden: variable `AFORO_CAMERA_*_SOURCE` > web > `pilot.yaml`. Cambiar la IP en la web se aplica solo, sin reiniciar. Excepción: la clave de la cámara viaja cifrada desde Supabase, solo por HTTPS.

### Task 20 — Cargar `.env` y validar llaves **(Prioridad 1)**

- [x] **20.1** `feat(config): load aforo-vision .env on startup` — `main()` lee `.env` de la raíz del repo (parser propio, sin dependencias) sin pisar variables ya definidas en el entorno.
- [x] **20.2** `fix(config): reject placeholder or non-ascii keys with a clear message` — Si una llave tiene caracteres no ASCII, espacios o empieza con `<`, se falla con un mensaje en español que nombra la variable (nunca el valor) y `main()` devuelve 2 sin traceback.

### Task 21 — Arranque automático y chequeos de inicio **(Prioridad 1)**

- [x] **21.1** `feat(events): check backend with read-only camera_sources` — El chequeo de arranque ya no manda `camera_heartbeat(online=True)`: hace un POST a `rpc/camera_sources` con las dos llaves y considera OK cualquier 2xx. Así el dashboard no muestra cámaras "En línea" antes de abrir los grabbers.
- [x] **21.2** `fix(capture): mask RTSP credentials using the last @` — `_describe_source` enmascara con el último `@`, para que una contraseña con `@` no se filtre en los logs.
- [x] **21.3** `feat(scripts): add Windows autostart task for vision and enrollment sync` — `scripts/windows/install_autostart.ps1` registra la tarea programada "AforoVision" al iniciar sesión (oculta, con reinicio automático y logs en `data\logs`); `uninstall_autostart.ps1` la elimina.

### Task 22 — Foto fija bajo demanda **(Prioridad 2)**
> Depende de: `aforo-db` (RPCs `pending_snapshot_requests` y `submit_camera_snapshot`). Excepción a la regla 1 de `AGENTS.md`, aprobada por el usuario el 2026-11-12.

- [x] **22.1** `feat(capture): serve on-demand camera snapshots [22.1]` — `src/capture/snapshots.py`: cada 3 s pide las solicitudes pendientes y envía una foto JPEG (≤640 px, calidad 70) del último frame, o `null` si la cámara no está conectada o el frame tiene más de 5 s. Nunca guarda la imagen en disco ni en logs; los errores de red no detienen el hilo.
- [x] **22.2** `docs: document the on-demand snapshot privacy exception [22.2]` — Excepción en `AGENTS.md` regla 1, sección 10.6 de `ARCHITECTURE.md` y nota en `README.md` y el runbook: el aviso de videovigilancia y el consentimiento deben mencionarlo.

### Task 23 — Limitación del nodo local **(Prioridad 1)**
> Decisión del 2026-11-12 (dueño): el sistema requiere siempre un computador encendido con `aforo-vision` en la misma red local que las cámaras. Las Tapo solo entregan video dentro de su red y la institución no permite configurar el router, así que no se procesa en la nube.

- [x] **23.1** `docs: document the local-node limitation [23.1]` — Sección «Limitaciones» en `README.md`, restricción en `PRD.md` (sección 8), ADR-008 en `ARCHITECTURE.md` (con la alternativa del navegador descartada) y una línea en la lista previa de `docs/runbook-dia-piloto.md`.

### Task 24 — Detección con ONNX, sin torch **(Prioridad 2)**
> El camino con torch sigue siendo el default; ONNX es opcional hasta validarlo con la cámara real (`tools/compare_detectors.py`).

- [x] **24.1** `feat(tools): add YOLO pose ONNX export script and model [24.1]` — `python -m tools.export_yolo_onnx` genera `models/yolo26n-pose.onnx` (12 MB, versionado).
- [x] **24.2** `feat(detection): add onnxruntime pose detector without torch [24.2]` — `OnnxPoseDetector`: letterbox, onnxruntime, NMS con OpenCV, misma interfaz que `YoloPoseDetector`.
- [x] **24.3** `feat(config): add detection.backend torch|onnx with validation [24.3]` — `detection.backend` (default `torch`) y `detection.onnx_model`.
- [x] **24.4** `feat(tools): add torch vs onnx detector comparison [24.4]` — conteo, IoU, distancia de keypoints y FPS; veredicto «equivalentes» / «revisar».
- [x] **24.5** `docs: add slim requirements and install notes [24.5]` — `requirements-slim.txt`, `requirements-export.txt` y sección en el README. Entorno slim medido: ~0,5 GB frente a ~1,4 GB.

### Task 25 — Instalación en Windows sin Docker **(Prioridad 1)**
> Objetivo: que alguien sin Docker instale aforo-vision con un paso a paso corto y que el sistema diga qué falla.

- [x] **25.1** `feat(tools): add doctor diagnostic command [25.1]` — `python -m tools.doctor`: Python y arquitectura, dependencias (según `detection.backend`), modelos, `.env`, Supabase (sin internet / llave rechazada / OK), cada cámara (TCP 554 + RTSP) y espacio en disco / instancia duplicada. Nunca imprime llaves ni claves de cámara.
- [x] **25.2** `feat(scripts): add windows install and start scripts [25.2]` — `scripts/windows/install.ps1` (5 pasos, interactivo, idempotente, `-Full`) y `start.ps1`.
- [x] **25.3** `docs: add windows install guide and doctor to runbook [25.3]` — Sección «Instalación en Windows (sin Docker)» en `README.md` y el doctor como primer paso en el runbook.

### Task 27 — Menos CPU y menos calor **(Prioridad 1)**
> Problema: un solo proceso `python -m src.main` acumuló ~5400 s de CPU y el portátil se calienta. Sin cambiar conteo, dirección ni identidad. El procesamiento solo con movimiento va en otra task.

- [x] **27.1** `feat(tools): add pipeline stage benchmark [27.1]` — `python -m tools.bench_pipeline`: ms por fotograma de decodificación, preprocesado, detector (onnx/torch), tracker y embeddings, con fotogramas sintéticos.
- [x] **27.2** `perf(inference): disable onnxruntime idle thread spinning [27.2]` — `src/performance.py` crea las opciones de sesión de todos los modelos ONNX con `allow_spinning=0`. Las salidas son idénticas bit a bit; con varios modelos en el mismo proceso ArcFace pasaba de ~2000 ms a ~50 ms.
- [ ] **27.3** `feat(config): add performance.cpu_threads and low_priority [27.3]` — Sección opcional `performance:` en `pilot.yaml` con validación en español (`cpu_threads` null o entero ≥ 1, `low_priority` booleano, por defecto true).
- [ ] **27.4** `feat(main): apply thread limit and low priority at startup [27.4]` — Una sola llamada en `main()` antes de crear los modelos: limita hilos de onnxruntime, OpenCV y torch (solo con backend torch) y baja la prioridad a BELOW_NORMAL en Windows.
- [ ] **27.5** `docs: add performance and temperature section [27.5]` — «Rendimiento y temperatura» en `README.md` con tiempos medidos y recomendaciones.
