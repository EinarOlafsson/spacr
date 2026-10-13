"""Item 475: the OPS spot-detector box. Native is the default and always
usable; SpotNet is disabled with the reason when it cannot run, and its
non-commercial licence is stated on its row either way."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _combo(qtbot, ready, reason="", default="native", spotiflow=(False, "")):
    from spacr.qt.model_install import SpotDetectorCombo

    combo = SpotDetectorCombo(default=default,
                              readiness=lambda: (ready, reason),
                              spotiflow_readiness=lambda: spotiflow)
    qtbot.addWidget(combo)
    return combo


def test_native_is_first_and_chosen(qtbot):
    combo = _combo(qtbot, ready=False, reason="SpotNet is not installed")
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "native", "spotnet", "spotiflow"]
    assert combo.currentData() == "native"


def test_spotnet_without_backend_or_token_is_disabled_with_the_reason(qtbot):
    from PySide6.QtCore import Qt

    reason = "SpotNet is installed but has no DeepCell access token"
    combo = _combo(qtbot, ready=False, reason=reason, default="spotnet")
    assert not combo.model().item(1).isEnabled()
    assert combo.currentData() == "native"
    tip = combo.itemData(1, Qt.ToolTipRole)
    assert reason in tip
    assert "NON-COMMERCIAL ACADEMIC USE ONLY" in tip


def test_spotnet_that_can_run_is_chosen_and_still_states_its_licence(qtbot):
    from PySide6.QtCore import Qt

    combo = _combo(qtbot, ready=True, default="spotnet")
    assert combo.model().item(1).isEnabled()
    assert combo.currentData() == "spotnet"
    assert "NON-COMMERCIAL ACADEMIC USE ONLY" in combo.itemData(
        1, Qt.ToolTipRole)


def test_the_ops_form_uses_the_box_and_collects_native(qtbot, tmp_path,
                                                      monkeypatch):
    from PySide6.QtWidgets import QWidget

    from spacr import _segmentation_backends as backends
    from spacr.qt.model_install import SpotDetectorCombo
    from spacr.qt.screens.settings_model import SettingsWidgets

    monkeypatch.setenv(backends._ROOT_ENV, str(tmp_path / "backends"))
    holder = QWidget()
    qtbot.addWidget(holder)
    form = SettingsWidgets("ops", holder)
    form.build_sections()
    box = form._widgets["ops_spot_detector"]
    assert isinstance(box, SpotDetectorCombo)
    assert not box.model().item(1).isEnabled()
    assert form.collect()["ops_spot_detector"] == "native"


def test_spotiflow_is_disabled_with_the_reason_until_installed(qtbot):
    from PySide6.QtCore import Qt

    reason = "Spotiflow is not installed. Install it from the Model Zoo."
    combo = _combo(qtbot, ready=False, spotiflow=(False, reason))
    assert not combo.model().item(2).isEnabled()
    assert reason in combo.itemData(2, Qt.ToolTipRole)
    ready = _combo(qtbot, ready=False, spotiflow=(True, ""),
                   default="spotiflow")
    assert ready.model().item(2).isEnabled()
    assert ready.currentData() == "spotiflow"


def test_a_saved_spotiflow_choice_is_kept_for_the_run_to_judge(qtbot):
    combo = _combo(qtbot, ready=False, spotiflow=(False, "not installed"),
                   default="spotiflow")
    assert combo.currentData() == "spotiflow"


def test_the_spotiflow_row_follows_the_alpha_switch(qtbot, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    combo = _combo(qtbot, ready=False, spotiflow=(True, ""))
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    AppScreen._gate_alpha_choices("ops_spot_detector", combo)
    assert combo.view().isRowHidden(2)
    assert not combo.model().item(2).isEnabled()
    assert not combo.view().isRowHidden(0)
    combo.setCurrentText("spotiflow")
    assert combo.currentData() == "spotiflow", "a saved value still runs"
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    AppScreen._gate_alpha_choices("ops_spot_detector", combo)
    assert not combo.view().isRowHidden(2)
    assert combo.model().item(2).isEnabled()
