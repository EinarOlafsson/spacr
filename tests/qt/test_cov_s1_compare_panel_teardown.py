"""Cancelling a join that is not running, and closing one that is.

The join runs on a worker thread, and both ends of its life are places the
panel can be asked to do something it is not doing: Cancel pressed when
nothing is in flight, and the window closed while the runner refuses to shut
down. Neither may raise, because Qt aborts the whole process when a running
QThread is destroyed and a traceback here would leave one behind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measurement_compare_dialog as mcd
from spacr.qt.widgets.measurement_compare_dialog import (
    MeasurementCompareDialog, MeasurementComparePanel)

pytestmark = pytest.mark.qt


@pytest.fixture
def objects():
    rng = np.random.default_rng(0)
    n = 80
    return pd.DataFrame({
        "cell_area": rng.normal(10.0, 2.0, n),
        "nucleus_area": rng.normal(5.0, 1.0, n),
        "plateID": np.repeat(["p1", "p2"], n // 2),
        "rowID": rng.choice(["r1", "r2"], n),
        "columnID": rng.choice(["c1", "c2"], n),
        "fieldID": rng.choice(["f1", "f2"], n),
        "object_label": np.arange(n),
    })


def test_cancelling_when_nothing_is_joining_changes_nothing(qtbot, objects):
    """Cancel is reachable whenever the panel is open.

    Pressing it with no join in flight must not tell the runner to cancel a
    job that does not exist, and must not overwrite the note that says what
    the panel is actually showing.
    """
    panel = MeasurementComparePanel(objects, {})
    qtbot.addWidget(panel)
    before = panel.join_note.text()

    assert panel.cancel_the_join() is False
    assert panel.join_note.text() == before


def test_the_window_forwards_cancel_to_the_panel_that_owns_the_join(
        qtbot, objects):
    """The dialog reimplements nothing; it is a frame around the panel.

    A window that answered Cancel itself would leave the panel's own worker
    running behind a closed dialog.
    """
    dialog = MeasurementCompareDialog(objects, {})
    qtbot.addWidget(dialog)
    seen = []
    dialog.panel.cancel_the_join = lambda *args: seen.append(args) or "sent"

    assert dialog.cancel_the_join() == "sent"
    assert seen == [()]


def test_a_runner_that_will_not_shut_down_does_not_stop_the_close(
        qtbot, objects, monkeypatch):
    """Closing must complete even when the runner raises.

    An exception escaping ``closeEvent`` leaves the widget open with a live
    QThread under it, which is the abort this handler exists to prevent -- so
    the failure is logged and the close carries on.
    """
    panel = MeasurementComparePanel(objects, {})
    qtbot.addWidget(panel)
    panel._joining = True

    def _refuses():
        raise RuntimeError("the join runner is wedged")

    monkeypatch.setattr(panel._jobs, "shutdown", _refuses)

    panel.close()

    assert panel._joining is False
    assert panel.isVisible() is False
