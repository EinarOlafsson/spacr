"""The calibration-gain preview: what planning refuses, what the dialog shows, and its button.

``_plan_gains`` runs production planning on a private copy of the settings and
refuses every source it cannot plan with a sentence; the dialog shows those
sentences, drops superseded scans and marks gains stale when the settings
move; the Preview button is attached once and stays disabled while an earlier
scan thread is still winding down.  One general test per unit, each walking a
case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QThread  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt.widgets import calibration_preview as preview  # noqa: E402

pytestmark = pytest.mark.qt

_LOCAL = "Choose a local source folder containing merged arrays."


class _Model:
    _defaults = {"cell_mask_dim": 4}

    def __init__(self, values):
        self.values = values

    def _valid_committed_value(self, key):
        return self.values.get(key)


def _merged_with_one_array(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "plate1_A01_1.npy").write_bytes(b"")
    return merged


def test_plan_gains_refuses_what_it_cannot_plan(tmp_path, monkeypatch, subtests):
    """Remote, empty, array-less and unreadable sources are refused; an interrupted scan returns []."""
    import spacr.measure as measure

    refusals = [
        ("remote source", {"src": ["https://host/merged"]}, None, _LOCAL),
        ("empty source", {"src": [""]}, None, _LOCAL),
        ("no arrays", {"src": "NOARRAYS", "channels": [0]}, None,
         "No eligible merged arrays were found."),
        ("failed scan", {"src": "ONEARRAY", "channels": [0]},
         lambda options, files: ({"failures": ["x"]}, {}),
         "Some merged arrays could not be read. Check the source files."),
    ]
    for case_id, extra, prepare, message in refusals:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            root = tmp_path / case_id.replace(" ", "_")
            root.mkdir()
            settings = {"intensity_calibration": True, **extra}
            if settings["src"] == "NOARRAYS":
                (root / "merged").mkdir()
                settings["src"] = str(root)
            elif settings["src"] == "ONEARRAY":
                settings["src"] = str(_merged_with_one_array(root))
            if prepare is not None:
                m.setattr(measure, "_prepare_measurement_calibration", prepare)
            with pytest.raises(ValueError) as caught:
                preview._plan_gains(settings)
            assert str(caught.value) == message

    with subtests.test(msg="interrupted scan"), monkeypatch.context() as m:
        class _Thread:
            def isInterruptionRequested(self):
                return True

        m.setattr(QThread, "currentThread", staticmethod(lambda: _Thread()))
        assert preview._plan_gains(
            {"intensity_calibration": True, "src": "/data"}) == []


def _dialog(qtbot):
    state = {"settings": {"intensity_calibration": True}}

    def getter():
        value = state["settings"]
        if isinstance(value, Exception):
            raise value
        return value

    widget = preview._CalibrationPreview(getter, threaded=False)
    qtbot.addWidget(widget)
    widget.state = state
    return widget


def test_the_dialog_reports_errors_staleness_and_drops_superseded_scans(qtbot, monkeypatch, subtests):
    """Bad settings and planning errors become status text; stale or superseded gains are never shown."""
    with subtests.test(msg="invalid settings"):
        dialog = _dialog(qtbot)
        dialog.state["settings"] = ValueError("bad well")
        dialog._start()
        assert dialog.status.text() == "Finish valid settings before previewing gains. bad well"
        assert dialog._snapshot is None

    with subtests.test(msg="planning error"), monkeypatch.context() as m:
        dialog = _dialog(qtbot)

        def fail(snapshot):
            raise ValueError("planning broke")

        m.setattr(preview, "_plan_gains", fail)
        dialog._start()
        assert dialog.status.text() == "planning broke"

    with subtests.test(msg="settings break after a scan"), monkeypatch.context() as m:
        dialog = _dialog(qtbot)
        m.setattr(preview, "_plan_gains", lambda snapshot: [])
        dialog._start()
        assert dialog._snapshot == {"intensity_calibration": True}
        dialog.state["settings"] = TypeError("gone")
        dialog._check_current()
        assert dialog.status.text() == "Settings changed. Preview the gains again."
        assert dialog._snapshot is None

    with subtests.test(msg="superseded scan"), monkeypatch.context() as m:
        dialog = _dialog(qtbot)

        def plan(snapshot):
            dialog._generation += 1
            return [{"source": "s", "calibration": {"plates": {}, "reference_plate": "p"}}]

        m.setattr(preview, "_plan_gains", plan)
        dialog._start()
        assert dialog._reports == []
        assert dialog.table.rowCount() == 0


def test_the_preview_button_is_attached_once_follows_its_field_and_waits(qtbot, monkeypatch, subtests):
    """One button per field, its position filter never eats events, and it waits for old scans."""
    with subtests.test(msg="attached once"):
        host = QWidget()
        QVBoxLayout(host)
        qtbot.addWidget(host)
        preview._attach_preview(host, _Model({}))
        preview._attach_preview(host, _Model({}))
        assert len(host.findChildren(QPushButton, "CalibrationPreviewButton")) == 1

    with subtests.test(msg="follows a line edit"):
        edit = QLineEdit()
        qtbot.addWidget(edit)
        preview._attach_preview(edit, _Model({}))
        position = edit._calibration_button_position
        for event_type in (QEvent.Resize, QEvent.Enter):
            assert position.eventFilter(edit, QEvent(event_type)) is False

    with subtests.test(msg="waits for an earlier scan"), monkeypatch.context() as m:
        thread = QThread()
        thread.start()

        class _Dialog:
            def __init__(self, getter, parent):
                self._retiring_threads = [thread]
                self._start = lambda: None

            def exec(self):
                return 0

            def deleteLater(self):
                pass

        host = QWidget()
        QVBoxLayout(host)
        qtbot.addWidget(host)
        m.setattr(preview, "_CalibrationPreview", _Dialog)
        preview._attach_preview(host, _Model({}))
        button = host.findChild(QPushButton, "CalibrationPreviewButton")
        try:
            button.click()
            assert not button.isEnabled()
            qtbot.wait(600)
            assert not button.isEnabled()
        finally:
            thread.quit()
            thread.wait()
        qtbot.waitUntil(button.isEnabled, timeout=3000)
