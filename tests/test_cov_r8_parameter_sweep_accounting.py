"""parameter_sweep: memory checks and worker accounting, when they fail.

Two of these guard bookkeeping around a sweep, and the rule is written
into one of them: *accounting must never change whether a trial is a
result.* A sweep that lost a trial because its resource note could not be
written would have thrown away the science to preserve the note.

The third is a process-identity check: only the parent may register
children against the run.
"""
from __future__ import annotations

import pytest

from spacr import parameter_sweep as PS


class TestDecidingThatMemoryIsLow:
    """Checked BETWEEN submissions, not only at the start.

    The other things on the machine -- an editor, the spaCR GUI, another
    analysis -- grow while the sweep runs, and the sweep has to yield to
    them rather than race them.
    """

    def test_it_answers_a_bool_on_this_machine(self):
        assert isinstance(PS.memory_is_low(), bool)

    def test_an_impossible_floor_is_always_low(self):
        """A floor above the machine's total can only be crossed."""
        assert PS.memory_is_low(floor_gib=10 ** 9) is True

    def test_a_zero_floor_is_never_low_on_machine_memory_alone(self):
        assert PS.memory_is_low(floor_gib=0.0) is False

    def test_a_sampler_that_cannot_be_read_does_not_stop_the_check(
            self, monkeypatch):
        """THE UNCOVERED GUARD.

        spaCR's own footprint comes from the run context's resource
        sampler -- somebody else's object, which may be closed or of
        another shape. The machine-wide check still has to answer:
        losing it would let a sweep run the machine out of memory
        because one optional figure could not be read.
        """
        from spacr import runctx

        class _ClosedSampler:
            @property
            def _summary(self):
                raise RuntimeError("the sampler is closed")

        class _Context:
            _resource_sampler = _ClosedSampler()

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        assert PS.memory_is_low(floor_gib=0.0) is False
        assert PS.memory_is_low(floor_gib=10 ** 9) is True

    def test_spacr_own_footprint_can_trip_the_guard_on_its_own(self,
                                                               monkeypatch):
        """The live path: attributable to spaCR rather than the machine.

        With a run budget supplied, the sampler's process-tree total is
        what distinguishes "the machine is busy" from "spaCR is the
        reason" -- and the guard has to fire on the second even when the
        machine has room.
        """
        from spacr import runctx

        class _Sampler:
            _summary = {"last_tree_memory_bytes": 40 * (1024 ** 3)}

        class _Context:
            _resource_sampler = _Sampler()

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        assert PS.memory_is_low(floor_gib=0.0,
                                spacr_ceiling_gib=1.0) is True
        assert PS.memory_is_low(floor_gib=0.0,
                                spacr_ceiling_gib=1000.0) is False


class TestRegisteringChildStamps:
    """`_register_resource_workers` moves private stamps onto the run."""

    def test_the_private_columns_never_survive_into_the_row(self):
        """A public result row must not carry a private transport column."""
        row = PS._register_resource_workers({
            "status": "ok",
            "_resource_workers": [{"pid": 1}],
            "_resource_worker": {"pid": 2},
        })
        assert "_resource_workers" not in row
        assert "_resource_worker" not in row
        assert row["status"] == "ok"

    def test_stamps_are_registered_against_the_active_run(self,
                                                          monkeypatch):
        from spacr import runctx

        seen = []

        class _Context:
            @staticmethod
            def register_worker(stamp):
                seen.append(stamp)

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        PS._register_resource_workers({
            "_resource_workers": [{"pid": 1}, {"pid": 2}],
            "_resource_worker": {"pid": 3},
        })
        assert seen == [{"pid": 1}, {"pid": 2}, {"pid": 3}]

    def test_a_stamp_that_is_not_a_mapping_is_ignored(self, monkeypatch):
        from spacr import runctx

        seen = []

        class _Context:
            @staticmethod
            def register_worker(stamp):
                seen.append(stamp)

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Context())
        PS._register_resource_workers({
            "_resource_workers": [{"pid": 1}, "not a stamp", 7],
        })
        assert seen == [{"pid": 1}]

    def test_a_non_list_stamp_column_is_ignored(self):
        row = PS._register_resource_workers({"_resource_workers": "nonsense"})
        assert "_resource_workers" not in row

    def test_outside_a_run_the_row_still_comes_back_clean(self,
                                                          monkeypatch):
        from spacr import runctx

        monkeypatch.setattr(runctx, "current_run_context", lambda: None)
        row = PS._register_resource_workers({
            "status": "ok", "_resource_workers": [{"pid": 1}]})
        assert row == {"status": "ok"}

    def test_a_context_that_refuses_does_not_lose_the_trial(self,
                                                            monkeypatch):
        """THE UNCOVERED GUARD, and the comment beside it is the point.

        "Accounting must never change whether a trial is a result." A
        sweep that dropped a finished trial because its resource note
        could not be written would have thrown away the science to
        preserve the note about it.
        """
        from spacr import runctx

        class _Hostile:
            @staticmethod
            def register_worker(_stamp):
                raise RuntimeError("the resource record is closed")

        monkeypatch.setattr(runctx, "current_run_context",
                            lambda: _Hostile())
        row = PS._register_resource_workers({
            "status": "ok", "_resource_workers": [{"pid": 1}]})
        assert row["status"] == "ok", "a finished trial was lost to accounting"
        assert "_resource_workers" not in row

    def test_a_runctx_that_will_not_import_is_survived(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "runctx" in name or "current_run_context" in (fromlist or ()):
                raise ImportError("runctx is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        row = PS._register_resource_workers({"status": "ok"})
        assert row["status"] == "ok"
