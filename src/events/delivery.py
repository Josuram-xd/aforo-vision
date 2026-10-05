"""Delivery of events: durable queue + uploader + backoff, off the pipeline thread.

`submit(event)` only writes to the queue (fast, never touches the network), so a slow or dead network
cannot stall frame processing. A background thread sends the queue in order; after a failure it waits
with exponential backoff and tries again, so events arrive in order once the network is back.
"""

import logging
import threading
import time
from collections.abc import Callable

from src.events.retry_queue import RetryQueue
from src.events.uploader import EventUploader, SendStatus

logger = logging.getLogger(__name__)

BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0


class EventDelivery:
    def __init__(
        self,
        queue: RetryQueue,
        uploader: EventUploader,
        backoff_start_seconds: float = BACKOFF_START_SECONDS,
        backoff_max_seconds: float = BACKOFF_MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.queue = queue
        self.uploader = uploader
        self._backoff_start = backoff_start_seconds
        self._backoff_max = backoff_max_seconds
        self._clock = clock
        self._failures = 0
        self._retry_at = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.unauthorized = False  # True after a 401/403 until a send succeeds; events stay queued

    def submit(self, event: dict) -> None:
        """Queue an event for delivery. Never blocks on the network."""
        self.queue.push(event)
        self._wake.set()

    def flush(self, force: bool = False) -> int:
        """Send queued events oldest first until the queue is empty or a send fails. Returns how many were sent.

        After a failure nothing is attempted until the backoff elapses, unless `force` is set.
        """
        if not force and self._clock() < self._retry_at:
            return 0
        sent = 0
        while (event := self.queue.peek()) is not None:
            status = self.uploader.send(event)
            if status is SendStatus.SENT:
                self.queue.ack(event["eventId"])
                self._failures, self._retry_at, self.unauthorized = 0, 0.0, False
                sent += 1
            elif status is SendStatus.REJECTED:
                self.queue.reject(event["eventId"], "rejected by the backend (HTTP 4xx)")
                logger.error("event %s set aside as rejected; %d rejected so far", event["eventId"], self.queue.rejected_count())
            else:  # FAILED or UNAUTHORIZED: keep the event and its order, wait before the next try
                self.unauthorized = status is SendStatus.UNAUTHORIZED
                self._failures += 1
                delay = min(self._backoff_start * 2 ** (self._failures - 1), self._backoff_max)
                self._retry_at = self._clock() + delay
                logger.warning("%d event(s) waiting; retrying in %.0f s", len(self.queue), delay)
                break
        return sent

    def start(self) -> "EventDelivery":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="event-delivery", daemon=True)
            self._thread.start()
        return self

    def stop(self, flush_timeout_seconds: float = 5.0) -> None:
        """Stop the sender thread. Whatever is still queued stays on disk for the next run."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=flush_timeout_seconds)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.flush()
            wait = max(self._retry_at - self._clock(), 0.0) if self._retry_at else None
            self._wake.wait(timeout=wait if wait is not None else 1.0)
            self._wake.clear()
        self.flush(force=True)  # one last try on the way out

    def __enter__(self) -> "EventDelivery":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()
