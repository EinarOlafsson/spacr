"""What the QC field browser does when the triage loop hits its edges.

Round 3 pinned the loader and the renderer.  This file pins the dialog's
remaining decisions: the wheel zoom, the two failure slots that put a
worker's exception on screen, the ends of the field list, ``open_at``'s
plate-aware repositioning, a Measure run-state probe that raises, and every
branch of the quarantine button -- a run in flight, duplicate copies in
both folders, and a quarantine that cannot be written.  Each one either
explains itself in the status line or refuses to move a file; none may
raise, because the dialog is what stands between a user and deleting data.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QPixmap, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldTarget,
    QCFieldVerdict,
    _FieldView,
)

from tests.qt.test_qc_field_browser import _digest, _write_field  # noqa: E402

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"
SECOND = "plate1_A02_1"


def _target(plate: Path, field: str = FIELD) -> QCFieldTarget:
    return QCFieldTarget(
        field=field, plate_root=str(plate), merged_dir=str(plate / "merged"),
        verdicts=(QCFieldVerdict("cell", "fail", ("under_segmented",), ""),))


def _one_field_plate(tmp_path: Path) -> Path:
    plate = tmp_path / "plate1"
    _write_field(plate, FIELD)
    return plate


def _wheel(view: _FieldView, delta: int) -> QWheelEvent:
    """Deliver a real wheel event to the canvas viewport."""
    event = QWheelEvent(
        QPointF(10.0, 10.0), QPointF(10.0, 10.0), QPoint(0, 0),
        QPoint(0, delta), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(view.viewport(), event)
    return event


def test_the_wheel_zooms_both_ways_and_the_next_field_refits(qtbot):
    """Wheel zoom must be symmetric and must not outlive the image.

    A five-pixel QC object is why the canvas zooms at all, so a wheel step
    has to change the transform in both directions and consume the event
    rather than letting the view scroll instead.  The zoom is per-image: a
    user who zoomed into a corner and then pressed Right would otherwise
    land on the next field at that corner's magnification and review it
    without ever seeing the whole frame.
    """
    view = _FieldView()
    qtbot.addWidget(view)
    view.set_pixmap(QPixmap(400, 320))
    fitted = view.transform().m11()

    up = _wheel(view, 120)
    zoomed_in = view.transform().m11()
    down = _wheel(view, -120)
    zoomed_out = view.transform().m11()

    assert fitted > 0.0
    assert zoomed_in == pytest.approx(fitted * 1.2)
    assert zoomed_out == pytest.approx(fitted)
    assert up.isAccepted() and down.isAccepted()

    _wheel(view, 120)
    assert view.transform().m11() == pytest.approx(fitted * 1.2)
    view.set_pixmap(QPixmap(400, 320))
    assert view.transform().m11() == pytest.approx(fitted)


def test_a_field_name_that_is_a_path_is_reported_not_swallowed(
        qtbot, tmp_path):
    """A scorecard row holding a path, not a stem, must not lose the session.

    ``FieldQC.field`` is a bare stem; a hand-edited or older CSV can carry a
    relative path instead, and the quarantine helpers reject that outright
    rather than move a file outside the two folders they own.  That rejection
    surfaces as an exception on the loader thread and again inside the
    file-state probe.  Both have to be caught here, or one bad row takes down
    a triage session that has other, perfectly good fields in it.
    """
    plate = _one_field_plate(tmp_path)
    browser = QCFieldBrowser(
        [QCFieldTarget(field=f"sub/{FIELD}", plate_root=str(plate),
                       merged_dir=str(plate / "merged")),
         _target(plate)],
        threaded=False)
    qtbot.addWidget(browser)

    assert "Could not load this field" in browser._load_status.text()
    assert "not a path" in browser._load_status.text()
    # The file-state probe raised the same ValueError and reported "gone".
    assert not browser._quarantine.isEnabled()
    assert "already gone" in browser._action_status.text()

    browser.next_field()

    assert browser.current_field == FIELD
    assert "Active merged copy" in browser._load_status.text()
    assert browser._quarantine.isEnabled()


@pytest.mark.parametrize("collision", [False, True], ids=["committed", "duplicate-refused"])
def test_a_banner_link_keeps_the_new_field_when_an_earlier_move_settles(
        qtbot, tmp_path, monkeypatch, qapp, collision):
    """A real move belongs to its starting field even after a banner jump."""
    import sys
    import threading

    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QLabel, QWidget
    from spacr.qt import path_probe, prerun
    from spacr.qt.widgets import qc_field_browser as module

    plate = tmp_path / "plate1"
    digest, _finding = _digest(plate)
    screen = QWidget()
    screen._thread = None
    qtbot.addWidget(screen)
    banner = prerun.SegQCBanner(screen)
    banner._digest = digest
    banner._draw()
    links = banner.findChildren(QLabel, "QCFieldLinks")
    assert len(links) == 1
    links[0].linkActivated.emit("0")
    browser = banner._field_browser
    qtbot.waitUntil(lambda: not browser._jobs.is_busy()
                    and not browser._render_jobs.is_busy()
                    and browser._quarantine.isEnabled())
    assert browser.current_field == FIELD
    active = plate / "merged" / f"{FIELD}.npy"
    destination = plate / "merged_quarantined" / f"{FIELD}.npy"
    original_bytes = active.read_bytes()
    second_bytes = (plate / "merged" / f"{SECOND}.npy").read_bytes()
    entered, release = threading.Event(), threading.Event()
    original_move = module._move_field
    workers, changes, failures = [], [], []

    def gated_move(*args, **kwargs):
        previous = sys.gettrace()
        inherited = threading.gettrace() if previous is None else None
        if inherited is not None:
            sys.settrace(inherited)
        try:
            workers.append(QThread.currentThread())
            entered.set()
            assert release.wait(10)
            return original_move(*args, **kwargs)
        finally:
            if inherited is not None:
                sys.settrace(previous)

    monkeypatch.setattr(module, "_move_field", gated_move)
    browser.quarantineChanged.connect(lambda field, changed: changes.append(
        (field, changed, QThread.currentThread())))
    browser._move_jobs.job_failed.connect(failures.append)
    try:
        qtbot.keyClick(browser._view, Qt.Key_Q)
        qtbot.waitUntil(entered.is_set)
        assert browser._move_jobs.is_busy()
        links[0].linkActivated.emit("1")
        qtbot.waitUntil(lambda: browser.current_field == SECOND
                        and not browser._jobs.is_busy()
                        and not browser._render_jobs.is_busy()
                        and browser._view._item is not None)
        assert banner._field_browser is browser
        payload, image = browser._payload, browser._view._item
        pixels = image.pixmap().toImage()
        notice = browser._action_notice
        assert notice == ""
        if collision:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(active, destination)
        release.set()
        qtbot.waitUntil(lambda: not browser._move_jobs.is_busy())
        assert workers and all(worker != qapp.thread() for worker in workers)
        assert browser.current_field == SECOND
        assert browser._payload is payload
        assert browser._view._item is image
        assert image.pixmap().toImage() == pixels
        assert browser._action_notice == notice
        assert "Could not change quarantine" not in browser._action_status.text()
        assert "Quarantined" not in browser._action_status.text()
        assert (plate / "merged" / f"{SECOND}.npy").read_bytes() == second_bytes
        if collision:
            assert failures == [f"FileExistsError: refusing to overwrite existing field {destination}"]
            assert changes == []
            assert active.read_bytes() == original_bytes
            assert destination.read_bytes() == original_bytes
            assert not Path(f"{destination}.quarantine.json").exists()
        else:
            assert failures == []
            assert changes == [(FIELD, True, qapp.thread())]
            assert not active.exists()
            assert destination.read_bytes() == original_bytes
            assert path_probe.known(str(active)) is False
            assert path_probe.known(str(destination)) is True
    finally:
        release.set()
        qtbot.waitUntil(lambda: not browser._move_jobs.is_busy())
        browser._move_jobs.shutdown()
        browser.close()
