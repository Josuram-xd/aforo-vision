"""The full pipeline for both cameras: detection -> tracking -> identity -> matching -> direction -> dedup -> events.

`Pipeline.process_frame` is called for every new frame of either camera (both run exactly the same
logic, AGENTS.md rule 3); resolved events leave through the `emit` callback. Decisions made here:

* A track becomes a *sighting* once it has enough body samples (it is "seen at that checkpoint"); the
  sighting's timestamp is that moment, which is what the cross-camera time window compares.
* Sightings wait in a pending list until the other camera produces a match. After `time_window` seconds
  no partner can match anymore, so the sighting expires; if `single_camera_fallback` is on it is then
  resolved alone from its trajectory (ADR-003: secondary signal, lower confidence).
* A person is named only when the evidence is solid: both cameras recognize the same student, or the two
  cameras matched each other by face and at least one recognizes a student, or the person was seen by a
  single camera that recognizes them. Body appearance never names anyone.
* Dedup: one event per matched pair, and one per (student, direction) within the TTL.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.dedup.ttl_set import TTLSet
from src.direction.resolver import resolve_direction, resolve_from_trajectory
from src.events.builder import BODY_ONLY, CAMERA_INSIDE_ID, CAMERA_OUTSIDE_ID, FACE, build_event
from src.matching.cross_checkpoint import CrossCheckpointMatcher, CrossingMatch, Sighting

logger = logging.getLogger(__name__)

EMBED_EVERY_FRAMES = 3  # embed a track on its 1st frame and then every Nth one (embeddings are the costly part)
READY_BODY_SAMPLES = 3  # body samples that turn a track into a sighting
SINGLE_CAMERA_CONFIDENCE = 0.4  # events decided from one camera's trajectory are the least certain ones
STATE_TTL_SECONDS = 60.0  # forget tracks that disappeared this long ago


@dataclass
class _TrackState:
    last_seen: float
    frames: int = 0
    face_sum: np.ndarray | None = None
    face_count: int = 0
    body_sum: np.ndarray | None = None
    body_count: int = 0
    sighted: bool = False

    def add(self, face: np.ndarray | None, body: np.ndarray | None) -> None:
        if face is not None:
            self.face_sum = face.copy() if self.face_sum is None else self.face_sum + face
            self.face_count += 1
        if body is not None:
            self.body_sum = body.copy() if self.body_sum is None else self.body_sum + body
            self.body_count += 1


@dataclass
class _Pending:
    sighting: Sighting
    keypoints: list = field(default_factory=list)  # snapshot of the track's keypoint history


class Pipeline:
    def __init__(
        self,
        detectors: dict,
        trackers: dict,
        extractor,
        matcher: CrossCheckpointMatcher,
        identity_index,
        names: dict[str, str],
        dedup: TTLSet,
        emit: Callable[[dict], None],
        min_direction_gap_seconds: float = 0.0,
        single_camera_fallback: bool = True,
    ):
        self.detectors, self.trackers, self.extractor = detectors, trackers, extractor
        self.matcher, self.identity_index, self.names = matcher, identity_index, names
        self.dedup, self.emit = dedup, emit
        self.min_direction_gap_seconds = min_direction_gap_seconds
        self.single_camera_fallback = single_camera_fallback
        self._states: dict[tuple[str, int], _TrackState] = {}
        self._pending: list[_Pending] = []

    def process_frame(self, camera_id: str, image: np.ndarray, timestamp: float) -> list:
        """Run one frame of one camera. Returns that camera's tracks in the frame (for the debug preview)."""
        tracks = self.trackers[camera_id].update(self.detectors[camera_id].detect(image))
        for track in tracks:
            state = self._states.setdefault((camera_id, track.id), _TrackState(timestamp))
            state.last_seen = timestamp
            state.frames += 1
            if state.sighted or (state.frames - 1) % EMBED_EVERY_FRAMES != 0:
                continue
            state.add(*self.extractor.extract(image, track.box))
            if state.body_count >= READY_BODY_SAMPLES:
                self._add_sighting(camera_id, track, state, timestamp)
        return tracks

    def expire(self, now: float) -> None:
        """Resolve or drop sightings whose partner can no longer arrive; forget long-gone tracks."""
        window = self.matcher.time_window_seconds
        still_waiting = []
        for pending in self._pending:
            if now - pending.sighting.timestamp > window:
                if self.single_camera_fallback:
                    self._resolve_single(pending)
            else:
                still_waiting.append(pending)
        self._pending = still_waiting
        self._states = {key: s for key, s in self._states.items() if now - s.last_seen <= STATE_TTL_SECONDS}

    # -- sightings and pairs --

    def _add_sighting(self, camera_id: str, track, state: _TrackState, timestamp: float) -> None:
        state.sighted = True
        sighting = Sighting(camera_id, track.id, timestamp, face=_mean(state.face_sum), body=_mean(state.body_sum))
        self._pending.append(_Pending(sighting, [np.array(kp) for kp in track.keypoint_history]))

        outside = [p for p in self._pending if p.sighting.camera_id == CAMERA_OUTSIDE_ID]
        inside = [p for p in self._pending if p.sighting.camera_id == CAMERA_INSIDE_ID]
        matches, _, _ = self.matcher.match([p.sighting for p in outside], [p.sighting for p in inside])
        for match in matches:
            # Compare by identity: sightings hold numpy arrays, whose == is not a plain bool
            matched_ids = {id(match.outside), id(match.inside)}
            pair = [p for p in self._pending if id(p.sighting) in matched_ids]
            self._pending = [p for p in self._pending if id(p.sighting) not in matched_ids]
            self._resolve_pair(match, {p.sighting.camera_id: p for p in pair})

    def _resolve_pair(self, match: CrossingMatch, pending_by_camera: dict[str, _Pending]) -> None:
        if not self.dedup.add((match.outside.track_id, match.inside.track_id)):
            return
        direction = resolve_direction(match, self.min_direction_gap_seconds)
        if direction is None:  # order not trustworthy: try the trajectory of either camera
            for camera_id, pending in pending_by_camera.items():
                direction = resolve_from_trajectory(pending.keypoints, camera_id)
                if direction:
                    break
        if direction is None:
            logger.warning("matched crossing without a decidable direction (tracks %d/%d), dropped",
                           match.outside.track_id, match.inside.track_id)
            return

        person_id, identity_similarity = self._identify_pair(match)
        confidence = match.similarity if identity_similarity is None else min(match.similarity, identity_similarity)
        method = FACE if (person_id or match.method == FACE) else BODY_ONLY
        timestamp = (match.outside.timestamp + match.inside.timestamp) / 2
        self._emit(direction, method, confidence, timestamp, person_id)

    def _resolve_single(self, pending: _Pending) -> None:
        sighting = pending.sighting
        direction = resolve_from_trajectory(pending.keypoints, sighting.camera_id)
        if direction is None:
            logger.debug("track %d on %s left no decidable direction", sighting.track_id, sighting.camera_id)
            return
        if not self.dedup.add(("single", sighting.camera_id, sighting.track_id)):
            return
        person_id, identity_similarity = self._lookup(sighting.face)
        confidence = SINGLE_CAMERA_CONFIDENCE if identity_similarity is None else min(SINGLE_CAMERA_CONFIDENCE, identity_similarity)
        self._emit(direction, FACE if person_id else BODY_ONLY, confidence, sighting.timestamp, person_id)

    # -- identity --

    def _lookup(self, face: np.ndarray | None) -> tuple[str | None, float | None]:
        if face is None:
            return None, None
        match = self.identity_index.lookup(face)
        return (match.person_id, match.similarity) if match else (None, None)

    def _identify_pair(self, match: CrossingMatch) -> tuple[str | None, float | None]:
        (out_id, out_sim), (in_id, in_sim) = self._lookup(match.outside.face), self._lookup(match.inside.face)
        if out_id and in_id:
            return (out_id, min(out_sim, in_sim)) if out_id == in_id else (None, None)  # two cameras disagree: no name
        if match.method == FACE and (out_id or in_id):  # the faces matched each other, so one recognition covers both
            return (out_id, out_sim) if out_id else (in_id, in_sim)
        return None, None  # a body-only pairing is not enough to attach a name

    def _emit(self, direction: str, method: str, confidence: float, timestamp: float, person_id: str | None) -> None:
        if person_id and not self.dedup.add(("person", person_id, direction)):
            return  # same student, same direction, moments ago: a fragment of the same crossing
        event = build_event(direction, method, max(0.0, min(confidence, 1.0)), timestamp,
                            person_id=person_id, person_name=self.names.get(person_id) if person_id else None)
        logger.info("%s %s (%s, confidence %.2f)%s", direction, event["method"], event["timestamp"], event["confidence"],
                    f" -> {event['personName']}" if event["personName"] else "")
        self.emit(event)


def _mean(total: np.ndarray | None) -> np.ndarray | None:
    if total is None:
        return None
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 0 else None
