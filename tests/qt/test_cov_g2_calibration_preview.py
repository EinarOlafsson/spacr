"""The calibration-gain preview's refusals and its dialog bookkeeping."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QThread  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QPushButton, QWidget, QVBoxLayout  # noqa: E402

from spacr.qt.widgets import calibration_preview as preview  # noqa: E402


class _Model:
    _defaults = {"cell_mask_dim": 4}

    def __init__(self, values):
        self.values = values

    def _valid_committed_value(self, key):
        return self.values.get(key)


def test_a_preview_button_is_attached_once(qtbot):
    host = QWidget()
    QVBoxLayout(host)
    qtbot.addWidget(host)
    preview._attach_preview(host, _Model({}))
    preview._attach_preview(host, _Model({}))
    assert len(host.findChildren(QPushButton, "CalibrationPreviewButton")) == 1


def test_the_button_follows_its_line_edit(qtbot):
    edit = QLineEdit()
    qtbot.addWidget(edit)
    preview._attach_preview(edit, _Model({}))
    position = edit._calibration_button_position
    assert position.eventFilter(edit, QEvent(QEvent.Resize)) is False
    assert position.eventFilter(edit, QEvent(QEvent.Enter)) is False


def test_preview_settings_are_a_detached_copy_of_what_planning_reads():
    wells = ["A01"]
    model = _Model({"src": "/data", "intensity_calibration_wells": wells,
                    "cell_mask_dim": 4, "unrelated": 1})
    snapshot = preview._preview_settings(model)
    assert snapshot["cell_mask_dim"] == 4 and "unrelated" not in snapshot
    assert snapshot["intensity_calibration_wells"] is not wells


@pytest.mark.parametrize("settings, message", [
    ({"intensity_calibration": True, "src": ["https://host/merged"]},
     "local source folder"),
    ({"intensity_calibration": True, "src": [""]}, "local source folder"),
])
def test_remote_or_empty_sources_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        preview._plan_gains(settings)


def test_an_interrupted_scan_returns_nothing(monkeypatch):
    class _Thread:
        def isInterruptionRequested(self):
            return True

    monkeypatch.setattr(QThread, "currentThread", staticmethod(lambda: _Thread()))
    assert preview._plan_gains(
        {"intensity_calibration": True, "src": "/data"}) == []


def test_a_folder_with_no_arrays_is_refused(tmp_path):
    (tmp_path / "merged").mkdir()
    with pytest.raises(ValueError, match="No eligible merged arrays"):
        preview._plan_gains({"intensity_calibration": True, "src": str(tmp_path),
                             "channels": [0]})


def test_a_failed_scan_is_not_shown_as_a_preview(tmp_path, monkeypatch):
    import spacr.measure as measure

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "plate1_A01_1.npy").write_bytes(b"")
    monkeypatch.setattr(measure, "_prepare_measurement_calibration",
                        lambda options, files: ({"failures": ["x"]}, {}))
    with pytest.raises(ValueError, match="could not be read"):
        preview._plan_gains({"intensity_calibration": True,
                             "src": str(merged), "channels": [0]})


@pytest.fixture
def dialog(qtbot):
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


def test_invalid_settings_ask_to_be_finished(dialog):
    dialog.state["settings"] = ValueError("bad well")
    dialog._start()
    assert "bad well" in dialog.status.text()


def test_a_planning_error_is_shown_as_its_text(dialog, monkeypatch):
    def fail(snapshot):
        raise ValueError("planning broke")

    monkeypatch.setattr(preview, "_plan_gains", fail)
    dialog._start()
    assert dialog.status.text() == "planning broke"


def test_settings_that_change_or_break_make_the_gains_stale(dialog, monkeypatch):
    monkeypatch.setattr(preview, "_plan_gains", lambda snapshot: [])
    dialog._start()
    dialog.state["settings"] = TypeError("gone")
    dialog._check_current()
    assert "Settings changed" in dialog.status.text()
    assert dialog._snapshot is None


def test_a_superseded_scan_is_not_drawn(dialog, monkeypatch):
    def plan(snapshot):
        dialog._generation += 1
        return [{"source": "s", "calibration": {"plates": {}, "reference_plate": "p"}}]

    monkeypatch.setattr(preview, "_plan_gains", plan)
    dialog._start()
    assert dialog._reports == []


def test_a_closed_dialog_does_not_start_again(dialog):
    dialog.done(0)
    dialog.done(0)
    dialog._start()
    assert dialog._snapshot is None


def test_a_closed_dialog_ignores_settings_changes(dialog):
    dialog._snapshot = {"intensity_calibration": True}
    dialog._closed = True
    dialog.state["settings"] = {"changed": True}
    dialog._check_current()
    assert dialog._snapshot == {"intensity_calibration": True}


def test_the_button_waits_for_an_earlier_scan_to_end(qtbot, monkeypatch):
    from PySide6.QtCore import QThread

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
    monkeypatch.setattr(preview, "_CalibrationPreview", _Dialog)
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
