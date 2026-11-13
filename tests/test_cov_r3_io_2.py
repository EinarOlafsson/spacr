"""Narrow-condition coverage for ``spacr.io``.

Each test drives one guard a successful run never takes: a plane the
converter cannot shape, a file it must not touch, an already normalised
image, a dataset index past the end, a preloader told to stop mid-batch, a
merge whose keys match nothing.  CPU-only, offline, all output under.
"""
from __future__ import annotations

import queue
import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _png(path, value=7, size=8):
    from PIL import Image
    Image.fromarray(np.full((size, size, 3), value, np.uint8)).save(str(path))
    return str(path)


def test_combined_dataset_index_past_the_end_returns_nothing(tmp_path):
    """An out-of-range index yields ``None`` instead of raising.

    ``__getitem__`` walks its sub-datasets and falls off the end, so a
    sampler built from a stale length hands the training loop a ``None``
    sample rather than an ``IndexError`` that would name the bug.  Pinning it
    keeps that silent contract visible to whoever changes it.
    """
    from spacr.io import CombinedDataset, NoClassDataset

    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        _png(d / f"i{i}.png")
    comb = CombinedDataset([NoClassDataset(str(d), shuffle=False)],
                           shuffle=False)

    assert len(comb) == 3
    assert comb[0] is not None            # in range: a real sample
    assert comb[3] is None                # one past the end
    assert comb[99] is None


def test_unshuffled_dataset_never_touches_the_filename_order(tmp_path,
                                                             monkeypatch):
    """``shuffle=False`` must mean the disk order is the sample order.

    ``shuffle_dataset`` is public and callers invoke it directly; on a
    dataset built with ``shuffle=False`` it has to be a no-op, because the
    caller is pairing those filenames with an external list of predictions.
    A shuffle there silently re-labels every result.
    """
    import random
    import spacr.io as IO

    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(5):
        _png(d / f"i{i}.png")

    calls = []
    monkeypatch.setattr(random, "shuffle", lambda seq: calls.append(list(seq)))

    quiet = IO.NoClassDataset(str(d), shuffle=False)
    before = list(quiet.filenames)
    quiet.shuffle_dataset()
    assert quiet.filenames == before
    assert calls == []                     # random.shuffle never reached

    loud = IO.NoClassDataset(str(d), shuffle=True)
    assert len(calls) == 1                 # the shuffling dataset did reach it
    loud.shuffle_dataset()
    assert len(calls) == 2


class _StopSignal:
    """``is_set`` answers False for the first ``n`` calls, then True."""

    def __init__(self, false_calls):
        self.false_calls = false_calls
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.false_calls


def _bare_loader():
    from spacr.io import spacrDataLoader
    loader = object.__new__(spacrDataLoader)
    loader.pin_memory = False
    loader._sentinel = object()
    loader._error = None
    return loader


def test_a_batch_is_dropped_when_the_stop_arrives_before_the_queue_put():
    """A stopping preloader must not park a batch in a queue nobody drains.

    ``cleanup`` sets the stop signal and then joins; a producer that pushed
    after that point would block forever on the bounded queue and the daemon
    thread would be abandoned with the loader's memory still pinned.  So the
    signal is re-checked immediately before the put, sentinel included.
    """
    q = queue.Queue(maxsize=4)
    loader = _bare_loader()

    loader._preload_next_batches(q, iter(["a", "b"]), _StopSignal(1))
    assert q.empty(), "a stopping producer queues neither batch nor sentinel"

    q2 = queue.Queue(maxsize=4)
    loader._preload_next_batches(q2, iter(["a", "b"]), _StopSignal(10 ** 6))
    assert [q2.get_nowait(), q2.get_nowait()] == ["a", "b"]
    assert q2.get_nowait() is loader._sentinel


def test_a_preloader_that_will_not_die_is_reported_not_hidden(caplog):
    """Abandoning a live daemon thread has to leave a trace in the log.

    ``cleanup`` gives the producer five seconds and then gives up.  Without
    the message the process is left with a thread still holding pinned memory
    and a queue reference while the next ``__iter__`` starts a second
    producer beside it -- a leak that otherwise looks like nothing.
    """
    import logging
    import spacr.io as IO

    class _Undying:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    clock = iter(range(1000))
    monkey = types.SimpleNamespace(monotonic=lambda: float(next(clock)))

    loader = _bare_loader()
    loader._iteration_active = True
    loader._stop_signal = None
    loader.thread = _Undying()
    loader.batch_queue = queue.Queue()
    loader.batch_queue.put("leftover")

    original_time = IO.time
    IO.time = monkey
    try:
        with caplog.at_level(logging.ERROR, logger="spacr.io"):
            loader.cleanup()
    finally:
        IO.time = original_time

    assert loader._iteration_active is False
    assert "did not stop within five seconds" in caplog.text
    assert loader.batch_queue.empty(), "the queue is drained while joining"


def test_cardinality_errors_name_the_side_that_is_actually_duplicated():
    """The error must accuse the table that broke the contract, not both.

    ``validate='one_to_many'`` constrains only the LEFT side and
    ``'many_to_one'`` only the RIGHT.  Naming the wrong table sends the user
    to de-duplicate a file that was already fine, and a merge error that is
    not about cardinality at all must keep pandas' own message.
    """
    from spacr.io import MergeCardinalityError, _merge_with_cardinality

    left_dupes = pd.DataFrame({'k': ['a', 'a', 'b'], 'v': [0, 1, 2]})
    right_unique = pd.DataFrame({'k': ['a', 'b'], 'w': [1, 2]})

    with pytest.raises(MergeCardinalityError) as one_to_many:
        _merge_with_cardinality(left_dupes, right_unique, validate='one_to_many',
                                left_name='cell', right_name='png_list', on='k')
    assert 'cell has duplicated' in str(one_to_many.value)
    assert 'png_list has duplicated' not in str(one_to_many.value)

    with pytest.raises(MergeCardinalityError) as many_to_one:
        _merge_with_cardinality(right_unique, left_dupes, validate='many_to_one',
                                left_name='cell', right_name='png_list', on='k')
    assert 'png_list has duplicated' in str(many_to_one.value)
    assert 'cell has duplicated' not in str(many_to_one.value)

    # A MergeError that is not a cardinality violation is re-raised as-is.
    with pytest.raises(pd.errors.MergeError) as plain:
        _merge_with_cardinality(pd.DataFrame({'a': [1]}),
                                pd.DataFrame({'b': [1]}),
                                validate='one_to_one',
                                left_name='cell', right_name='png_list')
    assert not isinstance(plain.value, MergeCardinalityError)
    assert 'common columns' in str(plain.value)


_STAMP = {'measurement_ndim': 2, 'measurement_units': 'px', 'n_z': 1,
          'voxel_size_z_um': 1.0, 'voxel_size_xy_um': 0.5}


