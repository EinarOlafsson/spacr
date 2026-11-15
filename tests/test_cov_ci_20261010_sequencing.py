"""Overload spooling edges and barcode-set announcement in sequencing."""

from __future__ import annotations

import pickle

import pytest


class _Queue:
    def __init__(self):
        self.saved = []

    def put(self, item):
        self.saved.append(item)


class _FailingPool:
    """Every final attempt fails with an error naming its chunk."""

    def __init__(self):
        self.calls = []

    def apply_async(self, function, args):
        identity = args[0]["identity"]
        self.calls.append(identity)

        class Result:
            def get(self_inner):
                raise ValueError(f"final invalid chunk {identity}")
        return Result()


def test_a_chunk_deferred_twice_is_spooled_once(tmp_path):
    from spacr.sequencing import _ChunkOverloadRetries

    pool = _FailingPool()
    retry = _ChunkOverloadRetries(pool, _Queue(), None,
                                  str(tmp_path / "out.h5"))
    assert retry.defer(7, {"identity": 7}, MemoryError("first overload"))
    assert retry.defer(7, {"identity": 7}, MemoryError("second overload"))
    spooled = list(tmp_path.rglob("*.pkl"))
    assert [path.name for path in spooled] == ["7.pkl"]
    errors = list(tmp_path.rglob("*.error.txt"))
    assert len(errors) == 1 and "first overload" in errors[0].read_text(encoding="utf-8")
    assert not retry.defer(8, {"identity": 8}, ValueError("not an overload"))
    assert pool.calls == []


def test_a_chunk_that_cannot_be_spooled_aborts_the_workers(tmp_path,
                                                           monkeypatch):
    from spacr import sequencing

    aborted = []
    pool, queue = object(), _Queue()
    monkeypatch.setattr(sequencing, "_abort_chunk_workers",
                        lambda *args: aborted.append(args))
    retry = sequencing._ChunkOverloadRetries(pool, queue, "saver",
                                             str(tmp_path / "out.h5"))
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        retry.defer(1, {"unpicklable": lambda: None},
                    MemoryError("overloaded"))
    assert aborted == [(pool, queue, "saver")]
    assert retry._paths == {}


def test_every_final_failure_is_recorded_and_the_first_is_raised(
        tmp_path, monkeypatch):
    from spacr import sequencing

    aborted = []
    monkeypatch.setattr(sequencing, "_abort_chunk_workers",
                        lambda *args: aborted.append(True))
    pool, queue = _FailingPool(), _Queue()
    retry = sequencing._ChunkOverloadRetries(pool, queue, None,
                                             str(tmp_path / "out.h5"))
    for identity in (1, 2):
        assert retry.defer(identity, {"identity": identity},
                           MemoryError("primary overloaded"))
    with pytest.raises(ValueError, match="final invalid chunk 1"):
        retry.drain()
    assert pool.calls == [1, 2]
    assert queue.saved == []
    assert aborted == [True]
    finals = sorted(tmp_path.rglob("*.final-error.txt"))
    assert [path.name for path in finals] == ["1.pkl.final-error.txt",
                                              "2.pkl.final-error.txt"]
    assert "final invalid chunk 2" in finals[1].read_text(encoding="utf-8")
    assert len(list(tmp_path.rglob("*.pkl"))) == 2


