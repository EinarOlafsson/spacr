"""``force_quit_now`` flushes everything it can and leaves regardless.

Instruction 288. It is the last thing that runs when a graceful stop has
already failed, so a log handler or a stream that will not flush must not
be what stops the process leaving -- a force quit that hangs is the
original complaint, twice.

``os._exit`` is stubbed throughout. It bypasses every cleanup path
including pytest's, so a test that let it run would take the whole
session with it.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import shutdown as S


@pytest.fixture
def no_exit(monkeypatch):
    """Record the exit code instead of leaving."""
    codes = []
    monkeypatch.setattr(S.os, "_exit", lambda code: codes.append(code))
    return codes


def test_a_handler_that_will_not_flush_does_not_stop_the_exit(no_exit,
                                                              monkeypatch):
    """THE ARM. A broken sink is exactly what a force quit is for."""
    class _Broken(logging.Handler):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def flush(self):
            self.asked += 1
            raise RuntimeError("this sink is gone")

        def emit(self, record):
            pass

    handler = _Broken()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        S.force_quit_now(3)
    finally:
        root.removeHandler(handler)

    assert handler.asked >= 1, "the handler was never flushed"
    assert no_exit == [3], (
        f"the process did not leave with the code it was given: {no_exit}")


def test_a_stream_that_will_not_flush_does_not_stop_it_either(no_exit,
                                                              monkeypatch):
    """The second loop, which flushes stdout and stderr."""
    class _Stubborn:
        def __init__(self):
            self.asked = 0

        def flush(self):
            self.asked += 1
            raise OSError("the terminal has gone")

        def write(self, _text):
            return 0

    stubborn = _Stubborn()
    monkeypatch.setattr(sys, "stdout", stubborn)

    S.force_quit_now(7)

    assert stubborn.asked >= 1, "stdout was never flushed"
    assert no_exit == [7]


def test_a_working_sink_is_flushed_rather_than_skipped(no_exit):
    """So the arms above are about the failure, not about a function
    that skips flushing altogether."""
    class _Fine(logging.Handler):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def flush(self):
            self.asked += 1

        def emit(self, record):
            pass

    handler = _Fine()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        S.force_quit_now(0)
    finally:
        root.removeHandler(handler)

    assert handler.asked >= 1
    assert no_exit == [0]


def test_it_leaves_even_with_no_handlers_at_all(no_exit, monkeypatch):
    """The empty case, which must not be mistaken for success above."""
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])

    S.force_quit_now(2)

    assert no_exit == [2]


# ---------------------------------------------------------------------------
# Two more guards of the same shape, in other screens
# ---------------------------------------------------------------------------

def test_comparing_without_a_graph_panel_declines(qtbot, monkeypatch):
    """``compare_a_measurement`` builds the graph tab, then uses it.

    If the tab could not be built there is nothing to switch to, and
    returning None is the whole behaviour -- the alternative is an
    AttributeError from a button press.
    """
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView()
    qtbot.addWidget(view)

    monkeypatch.setattr(type(view), "_all_objects",
                        lambda self: [{"a": 1}])
    monkeypatch.setattr(type(view), "picked_groups",
                        lambda self: ("g1", "g2"))
    monkeypatch.setattr(type(view), "_ensure_graph_tab",
                        lambda self: None)
    view._graph_panel = None

    assert view.compare_a_measurement() is None


def test_clearing_a_selection_with_no_table_is_survived(qtbot):
    """``set_frame`` clears the table's selection before repopulating.

    A table whose C++ half has gone raises RuntimeError, and one that was
    never built raises AttributeError. Neither is a reason to refuse the
    frame.
    """
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    view = RegressionResultsPanel()
    qtbot.addWidget(view)

    # ONLY blockSignals RAISES. Replacing the whole table with a stub
    # fails further down on setSortingEnabled, which is a different
    # method missing rather than the guard under test; shadowing the one
    # call keeps the rest of the real table in place.
    asked = []

    def _gone(_flag):
        asked.append(True)
        raise RuntimeError("Internal C++ object already deleted.")

    view.table.table.blockSignals = _gone

    import pandas as pd

    frame = pd.DataFrame({"feature": ["a", "b"],
                          "coefficient": [0.1, -0.2],
                          "p_value": [0.01, 0.5]})

    view.set_frame(frame)                # must not raise

    assert asked, "the selection was never cleared, so nothing was guarded"
