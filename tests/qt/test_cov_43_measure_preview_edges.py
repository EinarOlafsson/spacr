"""Measure preview edges the coverage ratchet found untested.

A slot count that is not a number builds nothing; a confluency or wound
answer for an older request, after the button was turned off, or carrying
an error is not drawn over the current field; the analyses that are on are
redrawn when a new field or new settings arrive; an object type with no crop
switch still re-previews; and the crop dialog's organelle rows survive a
form that cannot place them.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences  # noqa: E402
from spacr.qt.widgets import measure_preview as MP  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_a_slot_count_that_is_not_a_number_builds_nothing(panel):
    built = panel._slots_built
    panel._build_slot_controls("several")
    panel._build_slot_controls(None)
    assert panel._slots_built == built


@pytest.mark.parametrize("kind", ["confluency", "wound"])
def test_stale_switched_off_or_failed_answers_are_not_drawn(panel, kind):
    button = getattr(panel, f"_{kind}_btn")
    view = getattr(panel, f"_{kind}_view")
    ready = getattr(panel, f"_on_{kind}_ready")
    token = getattr(panel, f"_{kind}_token")
    view.hide()
    ready(token + 1, {"overlay": np.zeros((4, 4, 3), np.uint8)})
    ready(token, "not a result")
    button.blockSignals(True)
    button.setChecked(False)
    button.blockSignals(False)
    ready(token, {"overlay": np.zeros((4, 4, 3), np.uint8)})
    assert view.isHidden()
    button.blockSignals(True)
    button.setChecked(True)
    button.blockSignals(False)
    ready(token, {"error": "no brightfield channel"})
    assert view.isHidden()
    assert "no brightfield channel" in panel._status.text()


def test_a_field_with_no_scratch_says_how_open_it_is(panel):
    panel._wound_btn.blockSignals(True)
    panel._wound_btn.setChecked(True)
    panel._wound_btn.blockSignals(False)
    panel._on_wound_ready(panel._wound_token, {
        "overlay": np.zeros((8, 8, 3), np.uint8), "status": "no_wound",
        "open_fraction": 0.031})
    assert "No scratch found" in panel._status.text()
    assert "3.1 %" in panel._status.text()


def test_the_analyses_that_are_on_follow_a_new_field_and_new_settings(
        panel, monkeypatch, tmp_path):
    redrawn = []
    monkeypatch.setattr(panel, "_refresh_confluency",
                        lambda: redrawn.append("confluency"))
    monkeypatch.setattr(panel, "_refresh_wound",
                        lambda: redrawn.append("wound"))
    for button in (panel._confluency_btn, panel._wound_btn):
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
    path = tmp_path / "plate1_A01_1.npy"
    np.save(path, np.zeros((32, 32, 3), np.uint16))
    panel.load_array(str(path))
    assert redrawn[-2:] == ["confluency", "wound"]
    redrawn.clear()
    panel.apply_settings({})
    assert redrawn == ["confluency", "wound"]


def test_an_object_type_without_a_crop_switch_still_repreviews(panel,
                                                              monkeypatch):
    propagated = []
    monkeypatch.setattr(panel, "_maybe_propagate",
                        lambda *a: propagated.append(True))
    panel._on_object_changed("no-such-object")
    assert propagated == [True]


def test_crop_dialog_rows_survive_a_form_that_cannot_place_them(
        qtbot, panel, caplog):
    dialog = MP.CropSettingsDialog(panel)
    qtbot.addWidget(dialog)

    class _BrokenForm:
        def getWidgetPosition(self, widget):
            raise RuntimeError("form already deleted")

    dialog._organelle_rows = [("organelle", _BrokenForm(), None)]
    with caplog.at_level(logging.DEBUG, logger=MP.LOG.name):
        dialog.refresh_organelle_slots()
    assert "could not gate the organelle rows" in caplog.text
    dialog._filter_form = None
    assert dialog._adopt_new_slot_controls() is False
