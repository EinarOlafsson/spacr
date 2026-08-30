"""The producer thread's two ``queue.Full`` retries, and why both are loops.

``spacrDataLoader`` runs a daemon thread that reads the dataset and feeds a
BOUNDED queue -- bounded on purpose, because an unbounded one is how a loader
reads a whole plate into memory before the trainer has touched the first
batch. A bounded queue means the producer meets a full one, and what it does
then is the entire correctness of the stream:

* ``put(..., timeout=0.1)`` and ``continue`` retries until the consumer takes
  something. Anything else -- a drop, a break, an unbounded put -- silently
  shortens the epoch, and a model trained on a truncated stream reports a
  perfectly ordinary-looking accuracy.
* the same loop guards the SENTINEL in the ``finally``. Without it a full
  queue during shutdown strands the producer while the owner is joining it,
  which is a hang rather than a wrong number.

Both loops also check ``stop_signal`` every pass, so cleanup can end them.
The comment above the finally says the sentinel is deliberately omitted during
cleanup, and that is asserted here rather than taken on trust.
"""
from __future__ import annotations

import queue
import threading

import pytest

from spacr.io import spacrDataLoader


class _FullTwiceQueue(queue.Queue):
    """A queue that refuses the first two puts of every item.

    A real full queue is timing-dependent and would make this test a race.
    Refusing deterministically asserts the same property: the producer comes
    back rather than giving up.
    """

    def __init__(self, refusals=2, **kwargs):
        super().__init__(**kwargs)
        self.refusals = int(refusals)
        self.attempts = 0

    def put(self, item, block=True, timeout=None):
        self.attempts += 1
        if self.attempts <= self.refusals:
            raise queue.Full
        return super().put(item, block=block, timeout=timeout)


def _loader():
    """A loader built without touching a dataset.

    ``__init__`` needs a DataLoader's arguments; the producer under test reads
    only ``pin_memory`` and ``_sentinel`` off self, so an object created
    without running DataLoader.__init__ is the honest minimum here -- it
    cannot accidentally exercise torch's own loader instead.
    """
    loader = spacrDataLoader.__new__(spacrDataLoader)
    loader.pin_memory = False
    loader._sentinel = object()
    loader._error = None
    return loader


def test_a_refused_batch_is_retried_until_it_is_accepted():
    """The first loop: a full queue delays a batch, it does not lose one."""
    loader = _loader()
    q = _FullTwiceQueue(refusals=2)
    stop = threading.Event()

    loader._preload_next_batches(q, iter(["first", "second"]), stop)

    delivered = []
    while not q.empty():
        delivered.append(q.get())

    assert delivered[:2] == ["first", "second"], (
        "a batch was dropped when the queue was momentarily full")
    assert delivered[-1] is loader._sentinel
    assert q.attempts > 3, "the queue never actually refused anything"


def test_a_refused_sentinel_is_retried_so_the_consumer_still_sees_the_end():
    """The second loop, in the finally.

    The consumer stops on the sentinel. A sentinel dropped because the queue
    happened to be full at that instant leaves the consumer waiting on a
    stream that has already ended.
    """
    loader = _loader()
    q = _FullTwiceQueue(refusals=1)
    stop = threading.Event()

    loader._preload_next_batches(q, iter([]), stop)

    assert q.get() is loader._sentinel


def test_a_stopped_producer_writes_no_sentinel_at_all():
    """Cleanup's case, and the reason the finally is a guarded loop.

    During cleanup the owner is joining this thread and is no longer reading.
    Retrying forever against a full queue would strand the producer and hang
    the join, so the loop ends on the stop signal -- deliberately leaving no
    sentinel, because nobody is waiting for one.
    """
    loader = _loader()
    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    stop.set()

    loader._preload_next_batches(q, iter(["never read"]), stop)

    assert q.empty(), "a stopped producer still wrote to the queue"


def test_a_dataset_that_raises_hands_the_error_to_the_consumer():
    """A decode error must not look like an empty dataset.

    That is the failure the comment in the source names: the two used to be
    indistinguishable, and "dataset is empty" trains a model on no data and
    reports success.
    """
    loader = _loader()
    q = queue.Queue()
    stop = threading.Event()

    def explodes():
        yield "one good batch"
        raise ValueError("could not collate that")

    loader._preload_next_batches(q, explodes(), stop)

    assert isinstance(loader._error, ValueError)
    assert "collate" in str(loader._error)
    # The sentinel still goes out, so a consumer waiting on the stream wakes
    # up and can re-raise rather than blocking forever on a dead producer.
    drained = []
    while not q.empty():
        drained.append(q.get())
    assert drained[-1] is loader._sentinel


def test_pinned_memory_is_applied_to_each_batch_before_it_is_queued():
    """The pin happens on the producer thread, which is the point of it.

    Pinning on the consumer would put the copy back on the thread the trainer
    is waiting on, which is the work this loader exists to move off it.
    """
    loader = _loader()
    loader.pin_memory = True
    seen = []

    def fake_pin(batch):
        seen.append(batch)
        return f"pinned:{batch}"

    loader._pin_memory_batch = fake_pin
    q = queue.Queue()

    loader._preload_next_batches(q, iter(["a", "b"]), threading.Event())

    assert seen == ["a", "b"]
    assert q.get() == "pinned:a"
    assert q.get() == "pinned:b"
