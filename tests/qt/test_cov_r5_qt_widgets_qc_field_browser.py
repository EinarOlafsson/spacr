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

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QPixmap, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldTarget,
    QCFieldVerdict,
    _FieldView,
    targets_from_digest,
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


def _key(browser: QCFieldBrowser, key: int) -> QKeyEvent:
    """Send a key press to the dialog itself, where keyPressEvent runs."""
    event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
    QApplication.sendEvent(browser, event)
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


def test_a_channel_that_outlives_its_payload_says_why_the_canvas_froze(
        qtbot, tmp_path):
    """A render that cannot run must say so instead of showing a stale frame.

    The channel picker is the only source of the rendered channel index, and
    the renderer refuses an index past the payload's last channel.  When it
    does, the canvas necessarily keeps the previous image; without the
    failure slot writing a line, the user reads that unchanged image as the
    channel they just picked.
    """
    plate = _one_field_plate(tmp_path)
    browser = QCFieldBrowser([_target(plate)], threaded=False)
    qtbot.addWidget(browser)
    first = browser._view._item
    assert first is not None

    browser._channel.setCurrentIndex(2)
    second = browser._view._item
    assert second is not first, "a real channel must repaint the canvas"

    # A selection whose payload has fewer channels than the entry claims.
    browser._channel.addItem("Channel 9: stale", 9)
    browser._channel.setCurrentIndex(browser._channel.count() - 1)

    assert "Could not render this field" in browser._load_status.text()
    assert "outside 0..2" in browser._load_status.text()
    assert browser._view._item is second, "the last good image must remain"


def test_next_stops_at_the_last_field_instead_of_wrapping(qtbot, tmp_path):
    """Holding the right arrow at the end of the list must not wrap round.

    Triage is a keyboard loop and the arrow keys auto-repeat.  Wrapping past
    the last field would silently restart the plate, and a user who had
    already quarantined the first field would meet it again with no sign
    that the list had begun a second time.
    """
    digest, _finding = _digest(tmp_path / "plate1")
    browser = QCFieldBrowser(targets_from_digest(digest), threaded=False)
    qtbot.addWidget(browser)

    browser.next_field()
    assert browser.current_field == SECOND
    assert not browser._next.isEnabled()

    browser.next_field()

    assert browser.current_field == SECOND
    assert "flagged field 2 of 2" in browser._field_title.text()


def test_open_at_repositions_only_on_an_exact_field_and_plate_match(
        qtbot, tmp_path):
    """A banner link must open the field it names, on the plate it names.

    Two plates in one digest can hold the same field stem, so the plate root
    is part of the address.  A link that matches nothing has to report that
    rather than move the dialog somewhere arbitrary -- the caller uses the
    return value to decide whether to open a second browser.
    """
    digest, _finding = _digest(tmp_path / "plate1")
    targets = targets_from_digest(digest)
    browser = QCFieldBrowser(targets, threaded=False)
    qtbot.addWidget(browser)
    assert browser.current_field == FIELD

    assert browser.open_at(SECOND) is True
    assert browser.current_field == SECOND
    assert "flagged field 2 of 2" in browser._field_title.text()

    assert browser.open_at("plate9_Z99_9") is False
    assert browser.current_field == SECOND

    assert browser.open_at(FIELD, str(tmp_path / "elsewhere")) is False
    assert browser.current_field == SECOND

    assert browser.open_at(FIELD, targets[0].plate_root) is True
    assert browser.current_field == FIELD
    assert "flagged field 1 of 2" in browser._field_title.text()


def test_a_run_state_probe_that_raises_leaves_quarantine_usable(
        qtbot, tmp_path):
    """A dead Measure screen must not lock the button it no longer owns.

    ``run_active`` reaches back into the Measure screen, which can be torn
    down while this non-modal dialog stays open; calling into it then raises.
    Treating that as "a run is in flight" would leave the user staring at a
    permanently disabled button, so an unreadable run state is read as no
    run -- while a run state that really is True still disables it.
    """
    plate = _one_field_plate(tmp_path)
    state = {"mode": "raise"}

    def run_active() -> bool:
        if state["mode"] == "raise":
            raise RuntimeError("Internal C++ object already deleted")
        return state["mode"] == "running"

    browser = QCFieldBrowser(
        [_target(plate)], threaded=False, run_active=run_active)
    qtbot.addWidget(browser)

    assert browser._quarantine.isEnabled()
    assert "reversible" in browser._action_status.text()

    state["mode"] = "running"
    # The dialog re-reads the run state on its own 400 ms timer.
    qtbot.waitUntil(lambda: not browser._quarantine.isEnabled(), timeout=5000)

    assert "Measure is running" in browser._action_status.text()


def test_an_empty_browser_keeps_disabling_the_quarantine_button(qtbot):
    """The run-state poll must not offer to quarantine a field that is absent.

    An empty digest still opens a dialog, and that dialog keeps polling.
    Every pass has to notice there is no current target and leave the button
    dead; one pass that fell through would arm a button whose handler has no
    field to act on.
    """
    browser = QCFieldBrowser([], threaded=False)
    qtbot.addWidget(browser)
    assert not browser._quarantine.isEnabled()

    browser._quarantine.setEnabled(True)
    qtbot.waitUntil(lambda: not browser._quarantine.isEnabled(), timeout=5000)

    assert "No flagged fields" in browser._field_title.text()
    assert browser.current_target is None


def test_duplicate_copies_refuse_to_move_either_of_them(qtbot, tmp_path):
    """A field in both folders is ambiguous and must not be resolved blindly.

    A crashed run, or a hand-copied backup, can leave the same field in
    ``merged`` and ``merged_quarantined``.  Quarantining would refuse to
    overwrite and restoring would clobber; either way the button cannot know
    which copy the user meant.  It says so and does nothing -- and starts
    working again the moment one copy is gone.
    """
    plate = _one_field_plate(tmp_path)
    active = plate / "merged" / f"{FIELD}.npy"
    quarantine = plate / "merged_quarantined"
    quarantine.mkdir()
    shutil.copy2(active, quarantine / f"{FIELD}.npy")

    browser = QCFieldBrowser([_target(plate)], threaded=False)
    qtbot.addWidget(browser)
    changes = []
    browser.quarantineChanged.connect(
        lambda field, moved: changes.append((field, moved)))

    assert "Resolve duplicate copies" in browser._quarantine.text()
    assert not browser._quarantine.isEnabled()
    assert "Nothing will be overwritten" in browser._action_status.text()

    browser.toggle_quarantine()

    assert changes == []
    assert active.is_file()
    assert (quarantine / f"{FIELD}.npy").is_file()

    (quarantine / f"{FIELD}.npy").unlink()
    qtbot.waitUntil(lambda: browser._quarantine.isEnabled(), timeout=5000)
    browser.toggle_quarantine()

    assert changes == [(FIELD, True)]
    assert not active.exists()
    assert (quarantine / f"{FIELD}.npy").is_file()


def test_quarantine_is_refused_outright_while_measure_runs(qtbot, tmp_path):
    """A disabled button is not enough; the Q key must be refused too.

    The keyboard shortcut reaches ``toggle_quarantine`` directly, so the
    guard has to live in the handler rather than only on the button.  Moving
    a merged array out from under a running Measure is exactly the race the
    disabled button exists to prevent.
    """
    plate = _one_field_plate(tmp_path)
    active = plate / "merged" / f"{FIELD}.npy"
    running = {"value": True}
    browser = QCFieldBrowser(
        [_target(plate)], threaded=False,
        run_active=lambda: running["value"])
    qtbot.addWidget(browser)
    changes = []
    browser.quarantineChanged.connect(
        lambda field, moved: changes.append((field, moved)))

    browser.toggle_quarantine()

    assert changes == []
    assert active.is_file()
    assert "Measure is running" in browser._action_status.text()

    running["value"] = False
    browser.toggle_quarantine()

    assert changes == [(FIELD, True)]
    assert not active.exists()
    assert (plate / "merged_quarantined" / f"{FIELD}.npy").is_file()


def test_a_quarantine_that_cannot_be_written_leaves_the_field_alone(
        qtbot, tmp_path):
    """A failed move must be reported, and must not look like it worked.

    Something else occupying the ``merged_quarantined`` name -- a stray file,
    a read-only mount -- makes the move impossible.  Reporting success would
    tell the user this field is out of the next Measure run when it is not,
    and the file itself has to still be where it was.
    """
    plate = _one_field_plate(tmp_path)
    active = plate / "merged" / f"{FIELD}.npy"
    blocker = plate / "merged_quarantined"
    blocker.write_text("not a folder", encoding="utf-8")

    browser = QCFieldBrowser([_target(plate)], threaded=False)
    qtbot.addWidget(browser)
    changes = []
    browser.quarantineChanged.connect(
        lambda field, moved: changes.append((field, moved)))

    browser.toggle_quarantine()

    assert changes == []
    assert active.is_file()
    assert blocker.is_file(), "the stray file must not have been removed"
    assert "Could not change quarantine" in browser._action_status.text()

    blocker.unlink()
    browser.toggle_quarantine()

    assert changes == [(FIELD, True)]
    assert not active.exists()
    assert "Quarantined" in browser._action_status.text()


def test_the_dialog_answers_the_triage_keys_and_passes_the_rest_on(
        qtbot, tmp_path):
    """Triage keys are handled here; everything else still belongs to Qt.

    The dialog claims Left, Right and Q even when no child has focus.  If it
    claimed the rest as well, Escape would stop closing the dialog and the
    user would have a non-modal window they could not dismiss from the
    keyboard.
    """
    digest, _finding = _digest(tmp_path / "plate1")
    browser = QCFieldBrowser(targets_from_digest(digest), threaded=False)
    qtbot.addWidget(browser)
    browser.show()

    right = _key(browser, Qt.Key_Right)
    assert browser.current_field == SECOND
    assert right.isAccepted()

    _key(browser, Qt.Key_A)
    assert browser.current_field == SECOND, \
        "an unclaimed key must not navigate"
    assert browser.isVisible()

    rejected = []
    browser.rejected.connect(lambda: rejected.append(True))
    _key(browser, Qt.Key_Escape)

    assert rejected == [True], "Escape must still reach QDialog"
