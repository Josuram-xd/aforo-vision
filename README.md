# aforo-vision

Pipeline de visión por computador del sistema de aforo. Corre en el equipo junto a la puerta del salón (hoy el laptop), lee **dos cámaras** (`camera-outside` y `camera-inside`), detecta y sigue a cada persona, la identifica por la región de los ojos (funciona con tapabocas) y decide si **entró** o **salió** según el orden en que pasó por las cámaras. Cada cruce se envía como un evento JSON a Supabase (`aforo-db`), que actualiza el aforo y lo muestra en el dashboard (`aforo-frontend`).

**El video nunca sale del equipo:** solo viaja el JSON del evento ya resuelto. Los embeddings biométricos se quedan en `data/` (local, fuera de git) y nunca se suben a Supabase.

Más detalle en `PRD.md`, `ARCHITECTURE.md` (sección 10 = v2) y `AGENTS.md`. El trabajo pendiente está en `TASKS.md`.

## Cómo funciona

```
cámaras ─► captura ─► detección (YOLO pose) ─► tracking (SORT + húngaro) ─► identidad (periocular ArcFace / OSNet)
        ─► emparejar afuera↔adentro ─► dirección (ENTRY/EXIT) ─► anti-duplicados ─► cola SQLite ─► Supabase (ingest_event)
```

En paralelo, un hilo envía cada 30 s el latido de cada cámara (`camera_heartbeat`) para que el dashboard sepa si está en línea. Otro hilo responde, solo cuando un admin lo pide en la web, con una foto fija JPEG por cámara (task 22, `ARCHITECTURE.md` 10.6). El aviso de videovigilancia y el consentimiento deben mencionar esta foto fija.

## Estructura

```
aforo-vision/
├── src/
│   ├── main.py               # punto de entrada: python -m src.main
│   ├── pipeline.py           # une todas las etapas para las dos cámaras
│   ├── config.py             # carga config/pilot.yaml + variables de entorno
│   ├── capture/              # frame_grabber.py: un hilo por cámara (USB o RTSP), reconexión automática
│   ├── detection/            # yolo_pose.py: personas + keypoints con YOLO26n-pose
│   ├── tracking/             # hungarian.py (algoritmo propio), sort_tracker.py (Kalman), occlusion.py
│   ├── identity/             # periocular.py, face_detector.py, arcface.py, body_reid.py (OSNet),
│   │                         # index.py (tabla hash de embeddings), roster.py, appearance.py
│   ├── matching/             # cross_checkpoint.py: empareja la misma persona entre las dos cámaras
│   ├── direction/            # resolver.py: ENTRY/EXIT por orden de checkpoint (+ trayectoria si solo una cámara la vio)
│   ├── dedup/                # ttl_set.py: evita contar dos veces el mismo cruce
│   ├── events/               # builder.py (JSON del evento), uploader.py (RPC ingest_event), retry_queue.py (cola SQLite),
│   │                         # delivery.py, heartbeat.py (latidos de cámara), health.py (chequeo al arrancar)
│   ├── stream/               # auth.py: login con JWT de Supabase para el futuro servidor de video (task 12)
│   └── debug/                # preview.py: ventana local con cajas y FPS (--debug)
├── enrollment/               # enroll_student.py (captura el embedding con la webcam), export_roster.py (roster sin biometría)
├── config/pilot.yaml         # cámaras, umbrales y ventana de tiempo (sin secretos)
├── tools/                    # export_osnet_onnx.py: descarga y exporta OSNet a ONNX (se corre una vez)
├── tests/                    # pytest: tracking, identidad, dirección, envío a Supabase, auth del stream...
├── docs/                     # runbook del día del piloto, consentimiento informado y aviso de videovigilancia
├── data/                     # (fuera de git) modelos, embeddings, roster.json y la cola de reintentos
└── PRD.md · ARCHITECTURE.md · AGENTS.md · TASKS.md
```

## Puesta en marcha

Requiere Python 3.14 y CPU (no hace falta GPU).

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Modelos (una sola vez; descargan de internet, por eso no se hace solo):

```bash
python -m src.identity.arcface --download    # InsightFace buffalo_l → data/insightface/
python -m tools.export_osnet_onnx            # OSNet → data/osnet/
```

`yolo26n-pose.pt` va en la raíz del repo.

### Instalación liviana (sin torch)

El entorno completo pesa ~1,4 GB por torch/ultralytics. El equipo que solo ejecuta el pipeline puede usar ONNX (~0,5 GB):

```bash
pip install -r requirements-slim.txt          # sin torch, torchvision ni ultralytics
```

y en `config/pilot.yaml` poner `detection.backend: onnx` (usa `models/yolo26n-pose.onnx`, ya incluido en el repo). El camino con torch sigue siendo el **default** y no cambia. Para regenerar el modelo o comparar ambos caminos (requiere torch, una sola vez):

```bash
pip install -r requirements-export.txt
python -m tools.export_yolo_onnx
python -m tools.compare_detectors --source 0      # cámara, video, imagen, carpeta o rtsp://...
```

Usa el backend `onnx` solo cuando `compare_detectors` diga «equivalentes» con tu propia cámara.

### Variables de entorno

Los secretos nunca van en `config/pilot.yaml` ni en git. En PowerShell:

```powershell
$env:AFORO_BACKEND_URL = "https://<ref>.supabase.co"          # URL del proyecto de Supabase
$env:AFORO_SUPABASE_KEY = "<publishable key>"                  # sb_publishable_...
$env:AFORO_DEVICE_KEY = "<llave del salón>"                     # la crea un admin (aforo-db, scripts/create_device_key.py)
# Opcionales: fuerzan la fuente de la cámara y PISAN lo configurado en la web (ver abajo)
# $env:AFORO_CAMERA_OUTSIDE_SOURCE = "rtsp://<usuario>:<clave>@<ip>:554/stream1"
# $env:AFORO_CAMERA_INSIDE_SOURCE = "0"                         # índice USB o URL RTSP
```

Orden de prioridad de la fuente de cada cámara: variable `AFORO_CAMERA_*_SOURCE` > configuración de la web > `config/pilot.yaml`.

### Conectar las cámaras desde la web

Para una cámara WiFi Tapo C200 ya no hace falta poner la URL RTSP a mano:

1. En la app Tapo: Configuración de la cámara → Avanzado → **Cuenta de cámara**; crea usuario y contraseña. Anota también la **IP** de la cámara.
2. En la web (admin): Cámaras (o Configuración → Cámaras) → Agregar/Conectar; captura la IP y esa cuenta de cámara.
3. Arranca `aforo-vision` en la misma red WiFi que la cámara. Descarga la configuración con la llave del salón (RPC `camera_sources`), arma `rtsp://usuario:clave@ip:554/stream1` y la cámara pasa a "En línea" con su primer latido.
4. Quita `AFORO_CAMERA_*_SOURCE` del entorno; si no, esa variable pisa lo de la web.

Si la IP cambia, edítala en la web: se aplica sin reiniciar.

### Enrolar estudiantes

Solo quienes firmaron el consentimiento (`docs/consentimiento-informado.md`):

```bash
python -m enrollment.enroll_student --name "Nombre Apellido"
python -m enrollment.export_roster          # genera data/roster.json → se carga en aforo-db con seed_students.py
```

### Ejecutar

```bash
python -m src.main --dry-run --debug        # prueba: muestra eventos en consola, no los envía
python -m src.main --debug                  # con ventana por cámara
python -m src.main                          # día del piloto
```

El checklist completo del día está en `docs/runbook-dia-piloto.md`.

### Arranque automático (Windows)

Se instala una vez con `.\scripts\windows\install_autostart.ps1` y desde entonces arranca solo al iniciar sesión (sin ventanas); los logs están en `data\logs` (`vision.log` y `enrollment.log`). Para quitarlo: `.\scripts\windows\uninstall_autostart.ps1`.

### Pruebas

```bash
python -m pytest
```

## Instalación en Windows (sin Docker)

**Requisitos:** Windows 10/11 de 64 bits, Python 3.14 (el instalador te ayuda a instalarlo), estar en el **mismo WiFi que las cámaras** y una **llave del salón** creada por un admin (más la URL y la publishable key del proyecto de Supabase).

1. Instala [Git](https://git-scm.com/download/win) y clona el repo, o descarga el ZIP desde GitHub y descomprímelo.
2. Abre PowerShell en la carpeta del repo y corre:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\install.ps1
```

   El script tiene 5 pasos (Python, entorno virtual, dependencias, `.env`, diagnóstico) y se puede repetir sin problema. Al final ofrece el arranque automático.
3. Arranca a mano cuando quieras: `.\scripts\windows\start.ps1` (añade `--debug` para ver las ventanas).
4. Si algo falla, el diagnóstico te dice qué y cómo arreglarlo: `.\.venv\Scripts\python.exe -m tools.doctor`

Los modelos (`yolo26n-pose.pt`, InsightFace, OSNet) se descargan/exportan como se explica en «Puesta en marcha»; el diagnóstico avisa si falta alguno.

### Solución de problemas (mensajes de `tools.doctor`)

| Mensaje | Qué hacer |
|---|---|
| `Python 3.14.1` / versión distinta | Instala Python 3.14.2 o superior y vuelve a correr `install.ps1` |
| `Faltan paquetes` | Vuelve a correr `install.ps1` (retoma donde quedó); revisa internet y 5 GB libres |
| `Falta el modelo ...` | Corre el comando que indica el diagnóstico junto al mensaje |
| `Falta AFORO_... en .env` / `parece un texto de ejemplo` | Pon el valor real, sin `<...>`, tildes ni espacios |
| `Sin internet o Supabase no responde` | Revisa la conexión y la URL de `AFORO_BACKEND_URL` |
| `Supabase rechazó la llave (401/403)` | La llave del salón es incorrecta o fue revocada: pide otra a un admin |
| `Cámara ...: no alcanzable` | No estás en el mismo WiFi que la cámara, o el WiFi aísla los equipos (prueba el hotspot del celular); revisa la IP en la web |
| `puerto abierto pero rechaza usuario/clave` | Revisa la «Cuenta de cámara» de la app Tapo y vuelve a escribirla en la web |
| `Hay N proceso(s) python -m src.main corriendo` | Cierra la otra instancia o detén la tarea: `Stop-ScheduledTask -TaskName AforoVision` |

## Estructuras de datos, GRASP y SOLID

Esta sección dice dónde y cómo se aplica cada punto en el código. Las rutas son relativas a la raíz del repo. Los costos marcados "(teórico)" son los estándar de la estructura: el código no los comenta.

### Estructuras aplicadas

| Estructura | Archivo y clase | Para qué sirve | Costo |
|---|---|---|---|
| Cola FIFO persistente | `src/events/retry_queue.py` (`RetryQueue`), usada por `src/events/delivery.py` (`EventDelivery`) | Cada evento se guarda en SQLite antes de enviarlo. `push` encola al final, `peek` mira el frente, `ack` (enviado) y `reject` (rechazado por el backend) lo quitan de la cola. Una columna `seq` autoincremental mantiene el orden de llegada. `EventDelivery.submit` solo hace `push`; `flush` hace `peek` → envío → `ack`/`reject`, y ante un fallo deja el evento en el frente y espera con backoff. | Encolar y leer el frente: O(log n) (teórico, índice B-tree de SQLite). |
| Conjunto hash con TTL | `src/dedup/ttl_set.py` (`TTLSet`) | Evita contar dos veces el mismo cruce: `add` devuelve `False` si la clave sigue vigente. Un duplicado no extiende el TTL. | `add` y `in`: O(1) promedio (teórico). La purga es amortizada O(1) por clave expirada: el código aprovecha que todas expiran en orden de inserción (comentario en `TTLSet.__init__`). |
| Tabla hash + búsqueda por similitud coseno | `src/identity/index.py` (`IdentityIndex`) | `dict` personId → embedding para alta y baja de estudiantes; una matriz apilada de numpy para buscar el más parecido a un embedding de consulta. `add` y `remove` invalidan la matriz; se reconstruye de forma perezosa en la primera búsqueda siguiente (O(n·d) con `np.stack`). Devuelve el mejor match si supera el umbral y no es ambiguo. | Alta y baja: O(1) promedio (teórico). Búsqueda: O(n·d) para el producto matriz-vector, más O(n log n) por el `argsort` (teórico, n = estudiantes enrolados, d = dimensión). |
| Buffer de tamaño fijo | `src/tracking/sort_tracker.py` (`Track`) | Guarda solo los últimos `trajectory_length` centroides y keypoints observados de cada persona para estimar su movimiento. Se comporta como buffer circular (`deque(maxlen=...)`): al llenarse, descarta el más viejo solo. | Agregar: O(1) (teórico). Memoria: O(trajectory_length) por track. |
| Grafo bipartito + algoritmo húngaro propio | `src/tracking/hungarian.py` (`BipartiteGraph`, `hungarian`, `associate`) y `src/matching/cross_checkpoint.py` (`CrossCheckpointMatcher`) | Asignación de mínimo costo en dos casos: (1) en el tracking, detecciones del frame actual contra tracks del frame anterior, con costo `1 - similitud`; la similitud es IoU, y en pares en disputa se mezcla con la similitud de keypoints (`occlusion.py`), así que no es 1-IoU puro. Un par con IoU menor que el umbral no se asocia. (2) En el cruce de cámaras, avistamientos de `camera-outside` (izquierda) contra `camera-inside` (derecha), con costo `1 - coseno`. Los pares fuera de la ventana de tiempo o bajo el umbral cuestan `1e6`, así el algoritmo prefiere pares válidos y luego los descarta. | O(n²·m) (comentado en `hungarian.py`: "in O(n^2 * m)"). |

### Estructuras del curso que no se aplican

No se forzaron a propósito: se usó la estructura que resolvía cada problema.

- **Pila (LIFO):** no hay un caso donde el último evento deba salir primero; todo se procesa en orden de llegada.
- **Lista simple:** las colecciones que se recorren enteras (pendientes, tracks) son listas de Python; no hace falta una lista enlazada propia.
- **Lista doble:** no hay recorridos en ambos sentidos ni borrados por referencia a un nodo.
- **Lista circular:** el buffer de trayectoria usa `deque(maxlen=...)`, que cumple el papel de buffer circular; no hace falta una lista circular propia.
- **Árbol:** no hay datos jerárquicos ni recursivos; el roster es plano (personId → embedding).
- **Cola de prioridad:** los eventos salen en orden de llegada, así que una cola FIFO basta; nada se extrae por prioridad.

### GRASP

- **Experto:** `Track` (`src/tracking/sort_tracker.py`) guarda su propia trayectoria (`centroids`, `keypoint_history`) y calcula sus keypoints predichos (`predicted_keypoints`).
- **Controlador:** `src/pipeline.py` (`Pipeline`) coordina cada frame: detección → tracking → apariencia (embeddings) → matching entre cámaras → anti-duplicados → dirección → identidad → evento. `src/main.py` solo arma las piezas y el bucle.
- **Indirección:** `EventDelivery` (`src/events/delivery.py`) separa el pipeline de la red: `submit` solo escribe en la cola y un hilo aparte envía.
- **Alta cohesión:** las carpetas de `src/` están separadas por responsabilidad, por ejemplo: `dedup/` (anti-duplicados), `direction/` (ENTRY/EXIT), `matching/` (cruce entre cámaras), `identity/` (embeddings y roster), `events/` (cola, envío, latidos), `tracking/` (seguimiento).
- **Bajo acoplamiento / inyección:** `TTLSet(ttl_seconds, clock)` recibe el reloj, así las pruebas controlan el tiempo; `EventDelivery(queue, uploader, ...)` recibe la cola y el uploader; `Pipeline(...)` recibe detectores, trackers, extractor, matcher, índice de identidad, dedup y `emit`; `AppearanceExtractor(face_detector, face_embedder, body_embedder)` recibe sus modelos.
- **Fabricación pura:** `hungarian()` (el algoritmo) en `src/tracking/hungarian.py` y `src/dedup/ttl_set.py` son genéricos, sin concepto del dominio de aforo. El mismo archivo `hungarian.py` también trae `iou`, `BipartiteGraph.from_boxes` y `associate`, que sí conocen cajas y tracks.

### SOLID

- **S (responsabilidad única), por módulo:** `RetryQueue` solo persiste la cola en SQLite, sin red; `EventUploader` (`src/events/uploader.py`) solo hace el POST del evento; `hungarian.py` solo resuelve la asignación. `Pipeline` concentra más: coordina, empareja identidad y emite el evento (`_identify_pair`, `_emit`), así que es la pieza que menos cumple este principio.
- **D (inversión de dependencias), por inyección:** las dependencias llegan por el constructor (`EventDelivery`, `Pipeline`, `TTLSet`, `AppearanceExtractor`) y `src/main.py` las ensambla. Lo que se inyecta son clases concretas, no abstracciones: el repo no define interfaces ni `Protocol`.
- **O (abierto/cerrado):** no se evidencia en este repo. No hay clases base ni puntos de extensión por herencia.
- **L (sustitución de Liskov):** no se evidencia. No hay jerarquías propias de clases; las herencias son excepciones (ConfigError, AuthError), Enum (SendStatus), NamedTuple (IdentityMatch, Detection, Annotation) y `nn.Module` de PyTorch en `tools/export_osnet_onnx.py`.
- **I (segregación de interfaces):** no se evidencia. No hay interfaces definidas; las clases dependen de los métodos concretos que usan.

El material del curso está en la carpeta «Estructura de datos» del curso (carpetas Pilas-JHC, Colas-JHC, ListasSimples-JHC, ListasDobles-JHC, ListaCircular-JHC, Arboles y List-JHC), y esta sección enlaza lo aplicado aquí con esos temas.

## Rendimiento y temperatura

Medido con `python -m tools.bench_pipeline` (fotogramas sintéticos de 1920x1080, CPU de 16 hilos lógicos, mejor de 10 repeticiones; **no** hay cámara real en estas cifras):

| Etapa (ms por fotograma o recorte) | Antes | Después |
|---|---|---|
| Decodificar 1080p (archivo MJPEG, aproxima RTSP) | 4,4 | 4,4 |
| Preprocesado (letterbox + blob) | 0,4-0,7 | igual |
| Tracker + húngaro (4 personas) | 0,1-0,2 | igual |
| YOLO pose, backend `onnx` | 22-30 | 22-26 |
| YOLO pose, backend `torch` (solo, sin otros modelos) | 24 | 24 |
| OSNet (cuerpo) | 45-65 | 4,4 |
| SCRFD (detector de rostro) | 745-800 | 45 |
| ArcFace (ojos) | 1900-2100 | 50 |

Hallazgos y opciones:

- **Los hilos que giran en vacío eran el problema.** Cada sesión de onnxruntime mantiene hilos que "giran" esperando trabajo; con 4-5 modelos en un proceso se pelean los núcleos (CPU al 100 % aunque no haya nadie en cámara y modelos 10-40 veces más lentos). Ahora todas las sesiones se crean con `allow_spinning=0` (`src/performance.py`). Las salidas son idénticas bit a bit (se comparó YOLO, OSNet y ArcFace con y sin la opción). No requiere configuración.
- **`detection.backend: onnx` es lo más recomendable.** Con `torch` en el mismo proceso, los modelos de InsightFace (SCRFD, ArcFace) y a veces YOLO siguen 10-30 veces más lentos (medido: SCRFD 370-530 ms, ArcFace 1000-1700 ms, YOLO 125-225 ms) aunque se limiten los hilos; no se pudo aislar la causa. Con `onnx` todas las etapas quedan en las cifras de la tabla. Antes de cambiar el default, valida con `python -m tools.compare_detectors` y la cámara real que los conteos coinciden.
- **`performance.cpu_threads`** (en `config/pilot.yaml`, por defecto `null` = no tocar): limita los hilos de onnxruntime, OpenCV y torch (este último solo con `backend: torch`). Recomendación: la mitad de los núcleos (p. ej. 4 con 8 núcleos) para bajar calor y ruido del ventilador a costa de algo de latencia; no se midió el efecto sobre la temperatura (sin sensor ni cámara real).
- **`performance.low_priority`** (por defecto `true`): en Windows baja el proceso a prioridad BELOW_NORMAL para que el resto del equipo siga fluido; en otros sistemas se ignora. No reduce el calor por sí solo.
- El arranque ya es liviano (`import src.main` tarda 0,13 s y no carga torch, onnxruntime ni insightface hasta que se necesitan).
- Lo que no se pudo medir sin cámara real: la lectura/decodificación RTSP H.264 de las Tapo, la temperatura y los FPS reales con personas.

## Limitaciones

**Limitación del proyecto:** el sistema requiere **siempre** un computador encendido, con `aforo-vision` corriendo, conectado a la **misma red local** que las cámaras. Las cámaras (Tapo C200) solo entregan su video dentro de su red; la red de la institución no permite configurar el router, así que no es posible alcanzarlas desde internet ni procesar el video en la nube. La web, la base de datos, los usuarios y las alertas funcionan sin ese computador, pero sin él no hay conteo ni estado de las cámaras (aparecen sin conexión).

Alternativa evaluada y descartada (2026-11-12): usar el navegador de un celular o portátil como cámara (sin `aforo-vision`). Se descartó porque obliga a dejar de usar las Tapo, cambia el modelo de dos checkpoints por una línea virtual y no permite identificar personas por rostro sin llevar biometría al navegador.

## Contrato con los otros repos

| Destino | Llamada | Qué |
|---|---|---|
| `aforo-db` (Supabase) | `POST /rest/v1/rpc/ingest_event` con `{"p_event": {...}}` y headers `apikey` + `X-Device-Key` | Un evento por cruce. Si falla, queda en la cola local y se reintenta en orden. |
| `aforo-db` (Supabase) | `POST /rest/v1/rpc/camera_heartbeat` | Estado de cada cámara cada 30 s; si una se cae, el dashboard abre la alerta `camera_offline`. |

Si cambia un campo del evento, hay que avisar a `aforo-db`, `aforo-frontend` y `aforo-backend` (`AGENTS.md`, regla 5).
