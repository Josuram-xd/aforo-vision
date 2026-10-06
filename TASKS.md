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
- [ ] **2.4** (sin commit) Verificar **antes de comprar** que la cámara WiFi exponga RTSP u ONVIF en red local. Presupuesto total del proyecto: ~500.000 COP.

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
- [ ] **8.2** (sin commit) Ensayo real en la puerta comparando contra un conteo manual.
- [ ] **8.3** `chore(config): tune thresholds and time window from rehearsal` — Ajustar umbral de similitud y ventana entre cámaras con los datos del ensayo.
- [ ] **8.4** (sin commit) Antes del día real, limpiar los datos del ensayo con la task 4.3 del repo `aforo-db`.

---

## Prioridad 2 — Importante

### Task 9 — Envío autenticado
> Depende de: Seguir con la task 8 del repo: `aforo-backend`

- [ ] **9.1** `feat(events): send shared secret header` — Enviar el header con el secreto compartido en cada `POST /events`; leerlo de una variable de entorno, nunca de `pilot.yaml`.

### Task 10 — Rendimiento en CPU
- [ ] **10.1** `perf(detection): skip frames adaptively when fps drops` — Procesar 1 de cada N frames cuando los FPS bajen de un mínimo.
- [ ] **10.2** `perf(identity): cache embedding per track until matched` — No recalcular el embedding facial en cada frame una vez el track ya tiene identidad.

### Task 11 — Registro para el informe
- [ ] **11.1** `feat(logging): log each decision to local csv` — Un CSV local con cada evento, método (`FACE`/`BODY_ONLY`) y confianza, sin imágenes, para las métricas del informe.

---

## Prioridad 3 — Extras

### Task 12 — Vista en vivo para la demo
> Depende de: Seguir con la task 11 del repo: `aforo-backend` (User Pool de Cognito).

- [ ] **12.1** `feat(stream): serve mjpeg streams on local network` — `src/stream/server.py`: `GET /stream/<cameraId>` con los frames de cada cámara, escuchando solo en la IP de la red local (configurable en `pilot.yaml`). Nunca escribe frames a disco.
- [ ] **12.2** `feat(stream): require cognito token and group` — Validar el token de Cognito (firma con JWKS cacheado, issuer, client, expiración) y el claim `cognito:groups`: `viewer` o `dev` para ver; sin token 401, grupo incorrecto 403. IDs del User Pool por variable de entorno.
- [ ] **12.3** `feat(stream): add dev analysis overlay stream` — `GET /stream/<cameraId>/dev` solo para `dev`: video con cajas, ID de track, nombre + confianza, `FACE`/`BODY_ONLY`, checkpoint, dirección resuelta y FPS (reutiliza `src/debug/preview.py`).
- [ ] **12.4** `test(stream): cover auth and group checks` — Tokens firmados con una llave de prueba: válido, expirado, sin grupo, `viewer` pidiendo `/dev` (403).
  Después sigue con la task 9 del repo: `aforo-frontend`.

### Task 13 — Pruebas con video grabado
- [ ] **13.1** `test: replay recorded clips through the pipeline` — Test de integración con clips del ensayo (grabados con consentimiento, guardados solo en local y excluidos de git).
