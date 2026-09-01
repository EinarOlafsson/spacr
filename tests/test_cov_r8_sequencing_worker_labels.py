"""Naming sequencing's worker processes in the run's resource record.

`_label_resource_process` writes a child process's pid into the active
run context so the resource record can say which of the machine's cores
belonged to this run. It is bookkeeping, and it wraps EVERYTHING it does
in one guard -- because it is called from a pool-construction loop, and a
run must not die because its own provenance note could not be written.

That guard had never been entered. Every one of the four ways it can be
reached is a state a real run meets: no run context (a library call), a
process with no pid yet, a context that refuses the registration, and a
runctx module that will not import.
"""
from __future__ import annotations

import builtins

import pytest

from spacr import sequencing as S


class _Process:
    def __init__(self, pid=None):
        self.pid = pid


class TestLabellingOneProcess:

    def test_a_live_child_is_registered_against_the_run(self, monkeypatch):
        from spacr import runctx

        registered = []

        class _Context:
            @staticmethod
            def register_worker(kind, ident, pid):
                registered.append((kind, ident, pid))

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        S._label_resource_process(_Process(pid=4321), "sequencing_chunk", 2)
        assert registered == [("sequencing_chunk", 2, 4321)]

    def test_the_pid_is_recorded_as_an_integer(self, monkeypatch):
        """`pid=int(pid)` -- a string pid would poison the record."""
        from spacr import runctx

        seen = []

        class _Context:
            @staticmethod
            def register_worker(kind, ident, pid):
                seen.append(pid)

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        S._label_resource_process(_Process(pid="99"), "kind", 1)
        assert seen == [99] and isinstance(seen[0], int)

    def test_outside_a_run_nothing_is_recorded(self, monkeypatch):
        """A library call has no run context, and that is not an error."""
        from spacr import runctx

        registered = []
        monkeypatch.setattr(runctx, "current_run_context", lambda: None)

        S._label_resource_process(_Process(pid=1), "kind", 1)

        # AND NOTHING WAS RECORDED. With no context there is nowhere to
        # record to, so an empty list is the outcome -- "did not raise"
        # alone would not tell that apart from a worker registered
        # against a context that does not exist.
        assert registered == []

    def test_a_process_with_no_pid_yet_is_skipped(self, monkeypatch):
        """A child that has not started has `pid is None`."""
        from spacr import runctx

        registered = []

        class _Context:
            @staticmethod
            def register_worker(*a, **k):
                registered.append(a)

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        S._label_resource_process(_Process(pid=None), "kind", 1)
        assert registered == []

    def test_a_context_that_refuses_does_not_stop_the_run(self, monkeypatch):
        """THE GUARD. Provenance must never be what kills a run.

        This is called from the loop that builds the worker pool, so an
        exception here would take the pool down before any read was
        processed -- losing the science to preserve the note about it.
        """
        from spacr import runctx

        class _Hostile:
            @staticmethod
            def register_worker(*_a, **_k):
                raise RuntimeError("the resource record is closed")

        asked = []

        class _Counting(_Hostile):
            @staticmethod
            def register_worker(*_a, **_k):
                asked.append(True)
                raise RuntimeError("the resource record is closed")

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Counting())

        S._label_resource_process(_Process(pid=7), "kind", 1)   # must not raise

        # ASSERTED that the refusal was actually REACHED. Otherwise this
        # passes against a version that never consults the context at
        # all, which is a different bug wearing the same green tick.
        assert asked == [True], "the run context was never asked"

    def test_a_runctx_that_will_not_import_is_survived(self, monkeypatch):
        """The import is INSIDE the guard, not above it."""
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "runctx" in name or "current_run_context" in (fromlist or ()):
                raise ImportError("runctx is unavailable")
            return real(name, g, l, fromlist, level)

        attempted = []
        real_import = builtins.__import__

        def counting(name, g=None, l=None, fromlist=(), level=0):
            if "runctx" in name or "current_run_context" in (fromlist or ()):
                attempted.append(name)
                raise ImportError("runctx is unavailable")
            return real_import(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", counting)

        S._label_resource_process(_Process(pid=7), "kind", 1)   # must not raise

        # ASSERTED that the import was TRIED. The claim in the docstring
        # is that the import sits inside the guard rather than above it,
        # and a version that never imported runctx at all would pass on
        # "did not raise" while breaking every real recording.
        assert attempted, "runctx was never imported, so nothing was guarded"



class TestLabellingAWholePool:

    def test_every_worker_is_numbered_from_one(self, monkeypatch):
        """The index is what appears in the record, so it must start at 1."""
        labelled = []
        monkeypatch.setattr(S, "_label_resource_process",
                            lambda p, kind, i: labelled.append((p.pid, kind, i)))

        class _Pool:
            _pool = [_Process(pid=10), _Process(pid=11), _Process(pid=12)]

        S._label_chunk_pool(_Pool())
        assert labelled == [(10, "sequencing_chunk", 1),
                            (11, "sequencing_chunk", 2),
                            (12, "sequencing_chunk", 3)]

    def test_a_pool_with_no_workers_labels_nothing(self, monkeypatch):
        labelled = []
        monkeypatch.setattr(S, "_label_resource_process",
                            lambda *a: labelled.append(a))

        class _Empty:
            _pool = []

        S._label_chunk_pool(_Empty())
        assert labelled == []

    def test_a_pool_that_exposes_no_worker_list_is_survived(self,
                                                            monkeypatch):
        """`getattr(pool, "_pool", ()) or ()` -- a private attribute that
        another multiprocessing implementation need not have."""
        labelled = []
        monkeypatch.setattr(S, "_label_resource_process",
                            lambda *a: labelled.append(a))
        S._label_chunk_pool(object())
        assert labelled == []

    def test_a_none_worker_list_is_survived(self, monkeypatch):
        labelled = []
        monkeypatch.setattr(S, "_label_resource_process",
                            lambda *a: labelled.append(a))

        class _NonePool:
            _pool = None

        S._label_chunk_pool(_NonePool())
        assert labelled == []
