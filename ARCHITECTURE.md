# Architecture — aforo-vision

**Repository:** `aforo-vision`
**Last updated:** September 26, 2026
**Scope:** One-day pilot, single classroom door, hybrid USB + WiFi cameras.

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

### 4.9 Uploader
- Sends the event via `POST /events` to `aforo-backend` over HTTPS.
- If the request fails (network down), the event is pushed onto a local **FIFO retry queue** (on-disk, e.g. SQLite or a flat file) and retried with backoff until it succeeds — no event is lost to a brief connectivity gap.

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
