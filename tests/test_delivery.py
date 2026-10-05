import threading
import time

from src.events.delivery import EventDelivery
from src.events.retry_queue import RetryQueue
from src.events.uploader import SendStatus


def _event(n):
    return {"eventId": f"00000000-0000-4000-8000-{n:012d}", "direction": "ENTRY", "personName": "Ana Pérez" if n == 1 else None}


class _Uploader:
    """Answers from a script (one status per call, the last repeats) and records what it was sent."""

    def __init__(self, *statuses):
        self.statuses = list(statuses) or [SendStatus.SENT]
        self.sent_ids = []

    def send(self, event):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status is SendStatus.SENT:
            self.sent_ids.append(event["eventId"])
        return status


class _Clock:
    now = 1000.0

    def __call__(self):
        return self.now


def _delivery(*statuses, queue=None, **kwargs):
    clock = _Clock()
    uploader = _Uploader(*statuses)
    return EventDelivery(queue if queue is not None else RetryQueue(":memory:"), uploader, clock=clock, **kwargs), uploader, clock


# --- RetryQueue ---


def test_queue_is_fifo_and_peek_does_not_remove():
    queue = RetryQueue(":memory:")
    for n in (3, 1, 2):
        queue.push(_event(n))
    assert queue.peek()["eventId"] == _event(3)["eventId"] and len(queue) == 3
    queue.ack(_event(3)["eventId"])
    assert queue.peek()["eventId"] == _event(1)["eventId"]


def test_pushing_the_same_event_id_twice_queues_it_once():
    queue = RetryQueue(":memory:")
    assert queue.push(_event(1)) is True and queue.push(_event(1)) is False and len(queue) == 1


def test_queue_survives_a_restart_in_order(tmp_path):
    path = tmp_path / "data" / "retry_queue.sqlite"  # parent folder is created
    queue = RetryQueue(path)
    queue.push(_event(1)), queue.push(_event(2))
    queue.close()
    reopened = RetryQueue(path)
    assert len(reopened) == 2 and reopened.peek() == _event(1)  # payload round-trips, accents included


def test_rejected_events_are_set_aside_not_deleted():
    queue = RetryQueue(":memory:")
    queue.push(_event(1)), queue.push(_event(2))
    queue.reject(_event(1)["eventId"], "bad")
    assert len(queue) == 1 and queue.rejected_count() == 1 and queue.peek() == _event(2)


# --- EventDelivery ---


def test_submit_only_queues_and_flush_sends_in_order():
    delivery, uploader, _ = _delivery()
    for n in (1, 2, 3):
        delivery.submit(_event(n))
    assert uploader.sent_ids == []  # submit never touches the network
    assert delivery.flush() == 3
    assert uploader.sent_ids == [_event(n)["eventId"] for n in (1, 2, 3)] and len(delivery.queue) == 0


def test_a_failure_keeps_the_event_and_the_order_of_the_rest():
    delivery, uploader, _ = _delivery(SendStatus.SENT, SendStatus.FAILED, SendStatus.SENT)
    for n in (1, 2, 3):
        delivery.submit(_event(n))
    assert delivery.flush() == 1  # event 1 sent, event 2 failed, 3 not even tried (order matters)
    assert len(delivery.queue) == 2 and delivery.queue.peek() == _event(2)
    delivery.flush(force=True)
    assert uploader.sent_ids == [_event(n)["eventId"] for n in (1, 2, 3)]


def test_backoff_doubles_up_to_the_cap_and_blocks_attempts_until_it_elapses():
    delivery, uploader, clock = _delivery(SendStatus.FAILED, backoff_start_seconds=1.0, backoff_max_seconds=4.0)
    delivery.submit(_event(1))
    waits = []
    for _ in range(5):
        delivery.flush(force=True)
        waits.append(delivery._retry_at - clock.now)
    assert waits == [1.0, 2.0, 4.0, 4.0, 4.0]

    uploader.statuses = [SendStatus.SENT]
    assert delivery.flush() == 0  # still inside the backoff window: no attempt
    clock.now += 4.0
    assert delivery.flush() == 1 and delivery._retry_at == 0.0  # success resets the backoff


def test_events_queued_during_an_outage_are_delivered_in_order_after_it():
    delivery, uploader, clock = _delivery(SendStatus.FAILED, SendStatus.SENT)
    delivery.submit(_event(1))
    delivery.flush()
    delivery.submit(_event(2))
    delivery.flush()  # backoff: nothing happens
    clock.now += 60
    assert delivery.flush() == 2
    assert uploader.sent_ids == [_event(1)["eventId"], _event(2)["eventId"]]


def test_a_rejected_event_does_not_block_the_ones_behind_it():
    delivery, uploader, _ = _delivery(SendStatus.REJECTED, SendStatus.SENT)
    delivery.submit(_event(1)), delivery.submit(_event(2))
    assert delivery.flush() == 1
    assert uploader.sent_ids == [_event(2)["eventId"]] and delivery.queue.rejected_count() == 1 and len(delivery.queue) == 0


def test_unauthorized_keeps_everything_queued_and_raises_a_flag_until_it_works():
    delivery, uploader, clock = _delivery(SendStatus.UNAUTHORIZED, SendStatus.SENT)
    delivery.submit(_event(1)), delivery.submit(_event(2))
    assert delivery.flush() == 0 and delivery.unauthorized and len(delivery.queue) == 2
    clock.now += 60
    assert delivery.flush() == 2 and not delivery.unauthorized


def test_leftovers_are_sent_by_the_next_run(tmp_path):
    path = tmp_path / "queue.sqlite"
    first, _, _ = _delivery(SendStatus.FAILED, queue=RetryQueue(path))
    first.submit(_event(1)), first.submit(_event(2))
    first.flush()
    first.queue.close()

    second, uploader, _ = _delivery(queue=RetryQueue(path))  # the app was restarted, network is back
    assert second.flush() == 2 and uploader.sent_ids == [_event(1)["eventId"], _event(2)["eventId"]]


def test_background_thread_sends_without_the_caller_flushing():
    uploader = _Uploader()
    delivered = threading.Event()
    original = uploader.send
    uploader.send = lambda event: (original(event), delivered.set())[0]
    with EventDelivery(RetryQueue(":memory:"), uploader) as delivery:
        delivery.submit(_event(1))
        assert delivered.wait(timeout=3)
    assert uploader.sent_ids == [_event(1)["eventId"]]


def test_stop_makes_a_last_attempt_and_leaves_the_rest_on_disk(tmp_path):
    path = tmp_path / "queue.sqlite"
    delivery = EventDelivery(RetryQueue(path), _Uploader(SendStatus.FAILED), backoff_start_seconds=30.0).start()
    delivery.submit(_event(1))
    time.sleep(0.2)
    delivery.stop()
    assert len(RetryQueue(path)) == 1  # nothing lost
