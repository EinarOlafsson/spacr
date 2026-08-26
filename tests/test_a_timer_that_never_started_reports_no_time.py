"""``Timer`` used outside a ``with`` block measures nothing and says nothing.

``__exit__`` is reachable without ``__enter__`` -- ``contextlib.ExitStack``
and a hand-written ``try/finally`` both do it -- and a start time of
``None`` there would otherwise be subtracted from ``perf_counter()``.
"""
from __future__ import annotations

import logging

from spacr.logging_util import Timer


def test_exiting_a_timer_that_never_entered_leaves_no_elapsed_time(caplog):
    timer = Timer("never ran")

    with caplog.at_level(logging.DEBUG, logger="spacr.timing"):
        assert timer.__exit__(None, None, None) is None

    assert timer.elapsed_ms is None, (
        "no start means no duration, not a duration measured from zero")
    assert not [record for record in caplog.records
                if record.name == "spacr.timing"], (
        "a timer that never ran has nothing to report")


def test_a_timer_that_did_run_records_its_own_duration():
    with Timer("ran") as timer:
        pass

    assert timer.elapsed_ms is not None and timer.elapsed_ms >= 0.0
