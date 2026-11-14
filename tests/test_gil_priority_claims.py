"""Table-driven tests of the responsive-GUI switch-interval claims.

``spacr.qt.gil_priority.claim`` lowers ``sys.setswitchinterval`` to
``BUSY_INTERVAL`` for the first active worker, ``release`` restores the saved
interval when the last claim goes, and ``responsive_gui`` pairs the two around a
block.  An interpreter that refuses the knob (either way) must never turn into a
failed pipeline run: the refusal is logged at DEBUG and the bookkeeping stays
consistent.  Each test walks a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import logging
import sys

import pytest

from spacr.qt import gil_priority

LOGGER = "spacr.qt.gil_priority"
# A start interval distinct from both the CPython default and BUSY_INTERVAL,
# so a restore to the wrong value cannot pass by coincidence.
START = 0.0123


@pytest.fixture(autouse=True)
def a_clean_counter():
    """Every case starts at depth 0, nothing saved, interval START."""
    saved = (gil_priority._DEPTH, gil_priority._RESTORE, sys.getswitchinterval())
    # Bound before any case replaces it: a monkeypatched refusal is still in
    # place when this teardown runs.
    set_interval = sys.setswitchinterval
    gil_priority._DEPTH = 0
    gil_priority._RESTORE = None
    set_interval(START)
    yield set_interval
    gil_priority._DEPTH, gil_priority._RESTORE = saved[0], saved[1]
    set_interval(saved[2])


def _reset(set_interval):
    gil_priority._DEPTH = 0
    gil_priority._RESTORE = None
    set_interval(START)


def test_claims_and_releases_count_and_restore_the_interval_once(
        subtests, a_clean_counter):
    """Each step leaves an exact (depth, saved value, interval, active) state."""
    busy = gil_priority.BUSY_INTERVAL
    C, R = gil_priority.claim, gil_priority.release
    cases = [
        # (id, steps, expected state after each step)
        ("one claim then one release", [C, R],
         [(1, START, busy, True), (0, None, START, False)]),
        ("nested claims restore only at the last release", [C, C, R, R],
         [(1, START, busy, True), (2, START, busy, True),
          (1, START, busy, True), (0, None, START, False)]),
        ("extra releases never drive the count negative", [R, R],
         [(0, None, START, False), (0, None, START, False)]),
        ("a release past zero then a fresh claim saves afresh", [C, R, R, C],
         [(1, START, busy, True), (0, None, START, False),
          (0, None, START, False), (1, START, busy, True)]),
    ]
    for case_id, steps, expected in cases:
        with subtests.test(msg=case_id):
            _reset(a_clean_counter)
            seen = []
            for step in steps:
                step()
                seen.append((gil_priority._DEPTH, gil_priority._RESTORE,
                             sys.getswitchinterval(), gil_priority.active()))
            assert [(d, r, pytest.approx(i), a) for d, r, i, a in seen] == [
                (d, r, pytest.approx(i), a) for d, r, i, a in expected]


def test_a_refused_switch_interval_is_logged_never_raised(
        subtests, a_clean_counter, monkeypatch, caplog):
    """A refusal on the way down or up leaves nothing to restore and a DEBUG line."""

    def refusing(exc):
        def refuse(_value):
            raise exc
        return refuse

    cases = [
        # (id, refuse before claim?, exception, expected message, depth after claim)
        ("an interpreter without the knob still lets the worker run", True,
         ValueError("switch interval is not settable here"),
         "could not lower the switch interval"),
        ("a refused restore does not escape the release", False,
         RuntimeError("no"), "could not restore the switch interval"),
    ]
    for case_id, refuse_first, exc, message in cases:
        with subtests.test(msg=case_id):
            _reset(a_clean_counter)
            caplog.clear()
            with monkeypatch.context() as patch:
                if refuse_first:
                    patch.setattr(sys, "setswitchinterval", refusing(exc))
                with caplog.at_level(logging.DEBUG, logger=LOGGER):
                    gil_priority.claim()
                    assert gil_priority._DEPTH == 1
                    assert gil_priority.active() is True
                    assert gil_priority._RESTORE == (None if refuse_first
                                                     else START)
                    patch.setattr(sys, "setswitchinterval", refusing(exc))
                    gil_priority.release()           # must not raise
            assert gil_priority._DEPTH == 0
            assert gil_priority.active() is False
            assert gil_priority._RESTORE is None
            records = [(r.name, r.levelno, r.getMessage(), r.exc_info[1])
                       for r in caplog.records if r.name == LOGGER]
            assert records == [(LOGGER, logging.DEBUG, message, exc)]


def test_responsive_gui_holds_a_claim_for_the_block_and_releases_on_exit(
        subtests, a_clean_counter):
    """The context claims inside the block and restores on normal and raising exits."""
    cases = [
        ("a block that finishes", None),
        ("a block that raises", ZeroDivisionError),
    ]
    for case_id, error in cases:
        with subtests.test(msg=case_id):
            _reset(a_clean_counter)
            inside = []

            def body():
                with gil_priority.responsive_gui():
                    inside.append((gil_priority._DEPTH, gil_priority.active(),
                                   sys.getswitchinterval()))
                    if error is not None:
                        1 / 0

            if error is None:
                body()
            else:
                with pytest.raises(error):
                    body()
            assert inside == [(1, True,
                               pytest.approx(gil_priority.BUSY_INTERVAL))]
            assert (gil_priority._DEPTH, gil_priority._RESTORE,
                    gil_priority.active()) == (0, None, False)
            assert sys.getswitchinterval() == pytest.approx(START)
