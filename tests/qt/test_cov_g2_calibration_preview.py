"""The calibration-gain preview's refusals and its dialog bookkeeping."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


from spacr.qt.widgets import calibration_preview as preview  # noqa: E402


class _Model:
    _defaults = {"cell_mask_dim": 4}

    def __init__(self, values):
        self.values = values

    def _valid_committed_value(self, key):
        return self.values.get(key)


def test_preview_settings_are_a_detached_copy_of_what_planning_reads():
    wells = ["A01"]
    model = _Model({"src": "/data", "intensity_calibration_wells": wells,
                    "cell_mask_dim": 4, "unrelated": 1})
    snapshot = preview._preview_settings(model)
    assert snapshot["cell_mask_dim"] == 4 and "unrelated" not in snapshot
    assert snapshot["intensity_calibration_wells"] is not wells


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
