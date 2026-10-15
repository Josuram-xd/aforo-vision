# aforo-vision

Pipeline de visión por computador del sistema de aforo. Corre en el equipo junto a la puerta del salón (hoy el laptop), lee **dos cámaras** (`camera-outside` y `camera-inside`), detecta y sigue a cada persona, la identifica por la región de los ojos (funciona con tapabocas) y decide si **entró** o **salió** según el orden en que pasó por las cámaras. Cada cruce se envía como un evento JSON a Supabase (`aforo-db`), que actualiza el aforo y lo muestra en el dashboard (`aforo-frontend`).

**El video nunca sale del equipo:** solo viaja el JSON del evento ya resuelto. Los embeddings biométricos se quedan en `data/` (local, fuera de git) y nunca se suben a Supabase.

Más detalle en `PRD.md`, `ARCHITECTURE.md` (sección 10 = v2) y `AGENTS.md`. El trabajo pendiente está en `TASKS.md`.

## Cómo funciona

```
cámaras ─► captura ─► detección (YOLO pose) ─► tracking (SORT + húngaro) ─► identidad (periocular ArcFace / OSNet)
        ─► emparejar afuera↔adentro ─► dirección (ENTRY/EXIT) ─► anti-duplicados ─► cola SQLite ─► Supabase (ingest_event)
```

En paralelo, un hilo envía cada 30 s el latido de cada cámara (`camera_heartbeat`) para que el dashboard sepa si está en línea.

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

### Variables de entorno

Los secretos nunca van en `config/pilot.yaml` ni en git. En PowerShell:

```powershell
$env:AFORO_BACKEND_URL = "https://<ref>.supabase.co"          # URL del proyecto de Supabase
$env:AFORO_SUPABASE_KEY = "<publishable key>"                  # sb_publishable_...
$env:AFORO_DEVICE_KEY = "<llave del salón>"                     # la crea un admin (aforo-db, scripts/create_device_key.py)
$env:AFORO_CAMERA_OUTSIDE_SOURCE = "rtsp://<usuario>:<clave>@<ip>:554/stream1"
$env:AFORO_CAMERA_INSIDE_SOURCE = "0"                           # índice USB o URL RTSP
```

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

### Pruebas

```bash
python -m pytest
```

## Contrato con los otros repos

| Destino | Llamada | Qué |
|---|---|---|
| `aforo-db` (Supabase) | `POST /rest/v1/rpc/ingest_event` con `{"p_event": {...}}` y headers `apikey` + `X-Device-Key` | Un evento por cruce. Si falla, queda en la cola local y se reintenta en orden. |
| `aforo-db` (Supabase) | `POST /rest/v1/rpc/camera_heartbeat` | Estado de cada cámara cada 30 s; si una se cae, el dashboard abre la alerta `camera_offline`. |

Si cambia un campo del evento, hay que avisar a `aforo-db`, `aforo-frontend` y `aforo-backend` (`AGENTS.md`, regla 5).
