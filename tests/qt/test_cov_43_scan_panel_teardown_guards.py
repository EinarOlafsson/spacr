"""The scan panel's guards against reaching a widget that is going away.

Reads land on a worker thread and path probes answer from a process-wide
object, so each path that touches the panel afterwards is guarded against
the panel's C++ half having gone. These tests make each guarded call raise
the RuntimeError a deleted widget raises and check that nothing escapes,
and pin the small accessors next to them.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measurement_scan_panel as msp  # noqa: E402

pytestmark = pytest.mark.qt


def _gone(*args, **kwargs):
    raise RuntimeError("Internal C++ object already deleted.")


@pytest.fixture
def panel(qtbot):
    widget = msp.DatabaseMergePanel(lambda: [], threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_a_pending_read_says_what_it_is():
    assert repr(msp._NotBack()) == "<reading>"


def test_a_read_that_lands_on_a_deleted_relay_still_releases_its_question():
    class _DeadRelay:
        class landed:
            emit = staticmethod(_gone)

    key = (1, ("tables", "a.db"))
    lock, reads, reading, shown = threading.Lock(), {}, {key: True}, {}
    waiting = threading.Event()
    msp._run_read(lock, reads, reading, shown, (_DeadRelay, None), key,
                  lambda: 42, waiting)
    assert waiting.is_set() and key not in reading


def test_a_step_reports_its_title_and_number(qtbot):
    step = msp.WorkflowStep(3, "Choose the columns")
    qtbot.addWidget(step)
    assert step.title() == "Choose the columns" and step.number() == 3


def test_outcomes_are_revealed_even_before_their_grip_exists(qtbot):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QWidget

    box = QWidget()
    qtbot.addWidget(box)
    box.setVisible(False)
    host = SimpleNamespace(outcomes_box=box)
    msp.WorkflowSteps._show_outcomes(host)
    assert not box.isHidden()


def test_a_landed_read_on_a_deleted_panel_is_dropped(panel, monkeypatch):
    panel._painted_pending = True
    monkeypatch.setattr(panel, "_repaint", _gone)
    panel._on_read_landed()
    assert panel._painted_pending is True


def test_a_probe_answer_for_a_deleted_panel_is_dropped(panel, monkeypatch):
    path = "/plates/one/measurements.db"
    monkeypatch.setattr(panel, "_databases",
                        [msp.AttachedDatabase(plate="one", path=path)])
    monkeypatch.setattr(panel, "_recount", _gone)
    announced = panel._announced
    panel._probe_redraw(path, False)
    assert panel._announced == announced


def test_unfollowing_twice_or_after_a_failed_disconnect_is_quiet(
        panel, monkeypatch):
    from types import SimpleNamespace

    from spacr.qt import path_probe

    redraw = panel._probe_redraw
    real = path_probe.probes
    real.answered.disconnect(redraw)
    monkeypatch.setattr(path_probe, "probes", SimpleNamespace(
        answered=SimpleNamespace(disconnect=_gone)))
    panel._unfollow_path_probes()
    assert redraw.following is False
    panel._probe_redraw = redraw
    panel._unfollow_path_probes()
    assert panel._probe_redraw is None


def test_a_count_that_moved_is_announced_once(panel, monkeypatch):
    heard = []
    panel.databases_changed.connect(heard.append)
    monkeypatch.setattr(panel, "paths", lambda: ["a.db", "b.db"])
    panel._announced = 0
    panel._recount()
    panel._recount()
    assert heard == [2]


def test_a_panel_destroyed_after_it_unfollowed_does_not_unfollow_again(
        qtbot):
    import shiboken6

    widget = msp.DatabaseMergePanel(lambda: [], threaded=False)
    redraw = widget._probe_redraw
    widget._unfollow_path_probes()
    shiboken6.delete(widget)
    assert redraw.following is False


def test_a_panel_destroyed_while_following_survives_a_failed_disconnect(
        qtbot, monkeypatch):
    from types import SimpleNamespace

    import shiboken6

    from spacr.qt import path_probe

    widget = msp.DatabaseMergePanel(lambda: [], threaded=False)
    redraw = widget._probe_redraw
    real = path_probe.probes
    monkeypatch.setattr(path_probe, "probes", SimpleNamespace(
        answered=SimpleNamespace(disconnect=_gone)))
    shiboken6.delete(widget)
    assert redraw.following is False
    real.answered.disconnect(redraw)
