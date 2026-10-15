# Architecture — aforo-vision

**Repository:** `aforo-vision`
**Last updated:** October 8, 2026
**Scope:** One-day pilot, single classroom door, hybrid USB + WiFi cameras.

> **Version 2 planned (section 10).** Events go to Supabase instead of the AWS API, with a per-classroom
> device key and camera heartbeats. Where the pipeline runs (laptop, on-site box or free cloud VM) is
> under evaluation (proposed ADR-007); until that decision, sections 1-9 and ADR-001 still apply.

## 1. Purpose

`aforo-vision` is the local computer-vision pipeline. It runs entirely on Josuram's laptop during the pilot, reads frames from two cameras watching the same door from opposite sides, resolves each person crossing into an `ENTRY` or `EXIT` event with an optional identity, and pushes the resolved event (JSON only, never video) to `aforo-backend`.

This repo does **not** talk to a database and does **not** serve a UI. It is a producer of events.

## 2. Physical layout (pilot)

```
                    hallway                 classroom
                       |                        |
                (camera-outside, WiFi)   (camera-inside, USB)
                       |                        |
                       +---------[ DOOR ]-------+
                                    |
                          laptop (aforo-vision)
                     USB cable ----+---- WiFi (via phone hotspot)
```

- **camera-outside**: WiFi/IP camera, placed in the hallway outside the classroom. Connects to the laptop over a personal phone hotspot (recommended over institutional WiFi — see ADR-002).
- **camera-inside**: USB camera, plugged directly into the laptop, placed just inside the classroom.
- Both cameras face the same door from opposite sides, framing the same crossing point.

## 3. High-level pipeline

```
 camera-outside ──┐                                  ┌── camera-inside
   (WiFi/RTSP)     │                                  │      (USB)
                    ▼                                  ▼
            ┌───────────────┐                  ┌───────────────┐
            │ Frame Grabber │                  │ Frame Grabber │
            └───────┬───────┘                  └───────┬───────┘
                    ▼                                  ▼
            ┌───────────────────────────────────────────────┐
            │        Detection + Pose (YOLO26n-pose)          │
            └───────────────┬─────────────────┬──────────────┘
                            ▼                 ▼
                    ┌───────────────┐ ┌───────────────┐
                    │ Tracker (SORT)│ │ Tracker (SORT)│   (per camera)
                    └───────┬───────┘ └───────┬───────┘
                            ▼                 ▼
                    ┌───────────────────────────────┐
                    │ Identity resolution per track   │
                    │  - periocular crop → ArcFace     │
                    │  - fallback: OSNet body embedding│
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Cross-checkpoint matcher         │
                    │ (bipartite graph + Hungarian,     │
                    │  appearance similarity, time      │
                    │  window)                          │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Direction resolver                │
                    │ outside→inside = ENTRY            │
                    │ inside→outside = EXIT             │
                    │ (checkpoint order primary,        │
                    │  trajectory/pose as fallback)      │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Dedup (hash set + TTL)            │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Event builder (JSON)              │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Uploader (HTTPS → aforo-backend) │
                    │  + local FIFO retry queue         │
                    └───────────────────────────────┘
```

## 4. Components

### 4.1 Frame Grabber
- Wraps `cv2.VideoCapture`. Both camera types (USB index, RTSP/MJPEG URL) produce identical BGR numpy frames — no special-casing downstream.
- Runs each camera on its own thread to avoid blocking on frame reads.

### 4.2 Detection + Pose
- **Model**: YOLO26n-pose (Ultralytics), NMS-free, chosen for CPU inference speed (laptop has no dedicated GPU) and built-in keypoints (used for pose-based occlusion handling and the trajectory fallback signal).
- Runs independently per camera stream.

### 4.3 Tracker (per camera)
- SORT (Kalman filter + Hungarian algorithm) assigns a stable track ID to each detected person across frames, per camera.
- **Data structure**: bipartite graph (detections in frame *t* vs. tracks from frame *t-1*) solved with the Hungarian algorithm — the course's required graph-matching structure, applied here for frame-to-frame association.
- **Occlusion handling**: IoU overlap + pose keypoint visibility distinguishes two people even when one partially blocks the other, per camera, before assignment.

### 4.4 Identity resolution
- For each track, when a face is visible: crop the periocular region (via facial landmarks) and compute an embedding with a face-embedding model (ArcFace/InsightFace). This works with face masks since the mouth/nose are never required.
- Look up the embedding against the enrolled roster's embeddings (**hash table**: embedding bucket → student ID) within a similarity threshold.
- If no face is usable in a track's lifetime, compute an OSNet body-appearance embedding instead, so the person is still **counted** (method = `BODY_ONLY`), just not named.

### 4.5 Cross-checkpoint matcher
- A person seen by `camera-outside` and later by `camera-inside` (or vice versa) is the same physical crossing. This match is solved as a **bipartite graph** (outside-tracks vs. inside-tracks within a short time window) with the **Hungarian algorithm**, scored by appearance-embedding similarity (reuses the periocular/OSNet embedding from 4.4).
- Time window: configurable constant (default proposed: a few seconds — to be tuned against the door's real walking time during pilot rehearsal).

### 4.6 Direction resolver
- **Primary signal**: checkpoint order. `camera-outside` detection precedes `camera-inside` detection for the same matched identity → `ENTRY`. Reverse order → `EXIT`.
- This structurally defeats "walking backwards" as an anti-spoofing concern for direction: the trick only affects which way a body *appears* to face, not which physical checkpoint it was seen at first.
- **Fallback signal**: trajectory history (**deque** of recent centroid/pose positions per track) used only when a track is seen at just one checkpoint (e.g., the match failed) — kept as a secondary signal, not the primary one.

### 4.7 Deduplication
- A **hash set with TTL** keyed by (matched cross-checkpoint pair ID) prevents the same physical crossing from producing more than one event, even if detections flicker across frames.

### 4.8 Event builder
Produces the resolved event (see `aforo-backend/ARCHITECTURE.md` for the full shared contract):

```json
{
  "eventId": "uuid",
  "personId": "uuid | null",
  "personName": "string | null",
  "direction": "ENTRY | EXIT",
  "cameraOutsideId": "camera-outside",
  "cameraInsideId": "camera-inside",
  "confidence": 0.91,
  "method": "FACE | BODY_ONLY",
  "timestamp": "2026-09-30T14:32:00Z"
}
```

`personName` stays in the contract but is always `null` since task 17.3: Supabase (`ingest_event`) ignores it and takes the name from `students`, so sending it only moved personal data over the internet (data minimization). The name is still shown in the local log.

### 4.9 Uploader
- Sends the event via `POST /events` to `aforo-backend` over HTTPS.
- If the request fails (network down), the event is pushed onto a local **FIFO retry queue** (on-disk, e.g. SQLite or a flat file) and retried with backoff until it succeeds — no event is lost to a brief connectivity gap.

### 4.10 Local stream server (optional, demo)
- Small HTTP server on the laptop serving MJPEG per camera: `GET /stream/<cameraId>` (plain frames) and `GET /stream/<cameraId>/dev` (frames with the analysis overlay: boxes, track IDs, identity + confidence, `FACE`/`BODY_ONLY`, checkpoint and resolved direction, FPS — reuses `src/debug/preview.py`'s drawing).
- **Auth**: every request carries a Cognito access token (from `aforo-frontend`'s login). The server validates signature, issuer, audience/client and expiry against the User Pool's JWKS (cached locally) and checks the `cognito:groups` claim: `viewer` or `dev` for plain streams, `dev` only for `/dev` streams. No token → `401`, wrong group → `403`.
- Binds only to the LAN interface (the hotspot network), never exposed to the internet; frames are encoded in memory and never written to disk.

## 5. Architecture Decision Records

### ADR-001: Local processing on the laptop, not cloud/EC2
**Decision**: All video processing (detection, tracking, identity, direction) runs locally on Josuram's laptop. Only resolved JSON events cross to the cloud.
**Why**: (1) Privacy — Colombian Ley 1581 treats biometric embeddings as sensitive personal data; raw video should never leave the local device. (2) Cost — an always-on cloud GPU/CPU instance for video is unnecessary for a one-day, single-door pilot. (3) Resilience — the pipeline keeps working even if the university's internet is unstable.
**Alternative considered**: stream raw video to an EC2 instance for processing. Rejected for the reasons above; documented as a possible future direction if this becomes a permanent, multi-door installation.

### ADR-002: Personal hotspot instead of institutional WiFi for camera-outside
**Decision**: The WiFi camera connects to the laptop through Josuram's personal phone hotspot for the pilot, not the university's WiFi.
**Why**: institutional/enterprise WiFi networks commonly enable AP (access point) client isolation, which blocks devices on the same network from seeing each other — this would silently break the outside camera's connection to the laptop. A personal hotspot avoids this risk entirely, at zero cost.

### ADR-003: Checkpoint-order direction detection, not pure trajectory
**Decision**: Direction is determined primarily by which camera detects a matched identity first (outside vs. inside), not by analyzing a single camera's trajectory or body orientation.
**Why**: simpler to reason about, matches the physical reality of a door with a camera on each side, and is not fooled by someone walking backwards (a trick that only defeats orientation-based direction logic, not checkpoint-order logic). Trajectory/pose analysis is kept only as a fallback for the rare case a person is seen at just one checkpoint.

### ADR-004: Hybrid USB + WiFi cameras (not two of the same kind)
**Decision**: `camera-inside` is USB (wired to the laptop, since it sits inside the same classroom); `camera-outside` is WiFi/IP (since it sits in the hallway, too far from the laptop for a USB cable).
**Why**: matches the physical distance constraint at zero extra cost — no USB extension cables or a second laptop needed. Both feed the same `cv2.VideoCapture`-based grabber, so the rest of the pipeline treats them identically.

### ADR-005: Same detection/anti-spoofing logic on both cameras (not split roles)
**Decision**: Both cameras run the identical detection → pose → tracking → identity pipeline; neither camera has a "simpler" role.
**Why**: the professor and Josuram specifically want both cameras able to independently detect identity and resist walking-backwards/hide-behind-someone tricks, so that the cross-checkpoint match has two independently reliable observations to reconcile, rather than one strong camera and one weak one.

### ADR-006: Camera viewing stays on the local network, gated by the same login — login changes to Supabase in v2 (section 10.4)
**Decision**: Logged-in users watch the cameras through `aforo-vision`'s local stream server (4.10), not through the cloud. `aforo-backend` only authenticates users (Cognito); video never reaches AWS.
**Why**: keeps ADR-001's privacy guarantee (Ley 1581: video and identities stay on the laptop) and costs nothing; relaying video through AWS (Kinesis Video Streams, a media server on EC2) would cost money and move biometric footage to the cloud.
**Trade-off**: the camera views only work for devices on the same network as the laptop, and the page that shows them must be served over plain HTTP from that network (e.g. the frontend running on the laptop), because a browser blocks `http://` LAN images inside an HTTPS page such as the Amplify deployment.

## 6. Data structures used (course mapping — Estructuras de Datos)

| Data structure | Where it's used |
|---|---|
| Bipartite graph + Hungarian algorithm | Frame-to-frame tracking (per camera) and cross-checkpoint identity matching |
| Deque | Per-track trajectory history (fallback direction signal) |
| Hash table | Embedding → enrolled student identity lookup |
| Hash set with TTL | Deduplication of a single physical crossing |
| FIFO queue | Local retry queue for event uploads during network gaps |

## 7. Tech stack

| Layer | Choice | Version (verified Sept 2026) |
|---|---|---|
| Language | Python | 3.14.7 |
| Detection/pose | Ultralytics YOLO26n-pose | latest (Ultralytics package, Jan 2026 release) |
| Tracking | SORT / DeepSORT | latest |
| Face embeddings | InsightFace (ArcFace) | latest |
| Body re-id | OSNet (torchreid) | latest |
| Numeric/CV | OpenCV, NumPy | latest stable |
| ML runtime | PyTorch | 2.10 (confirmed Python 3.14 support) |
| HTTP client | `requests` or `httpx` | latest |
| Local retry queue | SQLite (stdlib) or flat file | — |

## 8. Repository structure

```
aforo-vision/
├── ARCHITECTURE.md
├── AGENTS.md
├── PRD.md
├── README.md
├── requirements.txt
├── config/
│   └── pilot.yaml            # camera sources, thresholds, backend URL
├── src/
│   ├── capture/
│   │   └── frame_grabber.py
│   ├── detection/
│   │   └── yolo_pose.py
│   ├── tracking/
│   │   └── sort_tracker.py
│   ├── identity/
│   │   ├── periocular.py
│   │   └── body_reid.py
│   ├── matching/
│   │   └── cross_checkpoint.py
│   ├── direction/
│   │   └── resolver.py
│   ├── dedup/
│   │   └── ttl_set.py
│   ├── events/
│   │   ├── builder.py
│   │   └── uploader.py
│   └── main.py
├── enrollment/
│   └── enroll_student.py     # captures reference embeddings for the roster
└── tests/
```

## 9. Contract with other repos

See `aforo-backend/ARCHITECTURE.md` for the full shared REST contract. `aforo-vision` is a client only: it calls `POST /events` on `aforo-backend`. It does not read from `aforo-db` or `aforo-frontend` directly.

**v2:** it calls the Supabase RPCs `ingest_event` and `camera_heartbeat` defined in `aforo-db` (section 10.1); still write-only, it never reads tables.

## 10. Version 2 (planned)

### 10.1 Event delivery to Supabase (replaces 4.9's target)

`aforo-backend`'s AWS API is retired in v2 (see `aforo-backend/ARCHITECTURE.md` section 9). The uploader keeps the same event JSON, the same on-disk FIFO retry queue and the same backoff; only the destination and the keys change:

| | v1 | v2 |
|---|---|---|
| URL | `POST <api>/events` | `POST <supabase-url>/rest/v1/rpc/ingest_event`, body `{"p_event": <event>}` |
| Keys | `X-Aforo-Secret` (`AFORO_BACKEND_SECRET`) | `apikey` (`AFORO_SUPABASE_KEY`, publishable) + `X-Device-Key` (`AFORO_DEVICE_KEY`, per classroom) |
| Sent | `201` / `200 duplicate` | `200` (`duplicate` true/false) |
| Rejected (set aside) | `400` | `400` |
| Bad keys (keep and stop) | `401`/`403` | `401`/`403` |
| Retry later | network, 5xx, 408/425/429 | same; `429` now also means the 120 req/min limit |
| Startup check | `GET /health` | one `camera_heartbeat` per camera (proves network + both keys) |

**Camera heartbeats**: a background thread calls `rpc/camera_heartbeat(code, online)` for each camera every 30 s with the grabber's `is_connected` state, so the dashboard shows camera status live and the database opens/closes `camera_offline` alerts. It never blocks frame processing.

Keys come only from environment variables (never `pilot.yaml`, never logged). `AFORO_BACKEND_URL` now holds the Supabase project URL.

### 10.2 Where the pipeline runs (under evaluation — proposed ADR-007)

The team wants the system to work **without Josuram's laptop**, ideally in a free cloud, with encrypted video and real-time results. Options:

| Option | How | Pros | Cons |
|---|---|---|---|
| A. Laptop (today, ADR-001) | as now | works today, video never leaves the room | depends on a personal laptop being on |
| B. On-site headless box | small PC / Raspberry Pi 5 next to the door, starts on boot as a service | no personal PC; video stays in the room; no internet needed for processing | hardware cost (check against the ~500.000 COP total budget, AGENTS rule 9); slower CPU than the laptop |
| C. Free cloud VM | Oracle Cloud Always Free Ampere A1 (ARM, up to 4 OCPU / 24 GB), cameras reached through a Tailscale (WireGuard) tunnel | nothing to run on the laptop; $0 | video leaves the room (encrypted); depends on internet upload from the room; ARM CPU only; needs an Oracle account (card verification) |

Notes that apply to option C:
- **An always-on device is still needed on site.** IP cameras cannot run Tailscale themselves, so a small device on the camera network must act as a Tailscale *subnet router* (for example a Raspberry Pi or a travel router with Tailscale/WireGuard). Option C therefore removes the laptop, not the on-site hardware.
- **Real time**: counting a door crossing needs about **5-8 FPS per camera**, not 30. YOLO26n-pose at reduced input size on 4 ARM cores should reach that for detection; face/periocular embeddings add cost per person, so groups entering together can lower FPS for a few seconds. **This must be measured (task 15.1) before deciding**; the decision criterion is ≥5 FPS per camera with two people crossing at once, and end-to-end latency (crossing → dashboard) under 3 s.
- **Privacy and Ley 1581**: video travels only inside the encrypted tunnel to a server the team controls, is processed in memory and never written to disk or sent to third-party services. Enrollment embeddings on the server are encrypted at rest (key from an environment variable) and never stored in Supabase. The surveillance notice (`docs/aviso-videovigilancia.md`) and AGENTS rule 1 must be updated before going live.

**Proposed ADR-007**: decide between B and C with the benchmark of task 15.1; whichever wins runs headless as a service (systemd on Linux) with automatic restart, no `--debug` window. Both keep the same code path; the only difference is the camera sources (`rtsp://` over the LAN vs. over the tunnel).

### 10.3 CPU architecture

If option C (or a Raspberry Pi) wins, the pipeline must run on **Linux ARM64**: verify wheels for `torch`, `ultralytics`, `onnxruntime`, `insightface` and `opencv-python` on ARM64 + Python 3.14 before pinning (AGENTS rule 8). ONNX Runtime for all models (YOLO exported to ONNX) is the fallback if `torch` is a problem there.

### 10.4 Camera viewing and login

The stream server of 4.10 validates **Supabase** JWTs (project JWKS) instead of Cognito, and reads the user's role from `user_roles`: `viewer`/`teacher`/`admin` see plain streams, `admin` sees the analysis overlay. LAN-only stays the rule unless the team explicitly decides otherwise for option C.

**Implementation (`src/stream/auth.py`, task 14.6)**: the token is verified locally against the project JWKS (`/auth/v1/.well-known/jwks.json`, ES256/RS256 only; issuer, audience `authenticated`, expiry and `sub` required; the anon and service tokens are rejected). The Supabase JWT carries no role claim (only `sub` and `role=authenticated`), so the app role is read with `GET /rest/v1/user_roles?select=roles(code)&user_id=eq.<sub>` using the **user's own JWT** plus the publishable key: row level security lets users read their own rows, so the stream server never holds a service-role key. If a user has several roles the highest wins (`admin` > `teacher` > `viewer`); the result is cached 60 s, which is also the maximum time a revoked role keeps working. Errors: no or invalid token 401, no role or role not allowed 403, JWKS or lookup unavailable 503. Requires `roles` to be readable by `authenticated` users (confirm in `aforo-db`).

### 10.5 Web enrollment (pending decision)

Today students are enrolled in person with `enrollment/enroll_student.py`. A web flow (the student signs up, accepts the consent text and the page captures ~10 periocular samples) is possible but sends biometric samples over the internet; it needs encrypted upload directly to the vision node (never to Supabase tables) and an updated consent form. Not planned until the team decides.
