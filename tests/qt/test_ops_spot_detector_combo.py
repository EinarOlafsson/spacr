"""Item 475: the OPS spot-detector box. Native is the default and always
usable; SpotNet is disabled with the reason when it cannot run, and its
non-commercial licence is stated on its row either way."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _combo(qtbot, ready, reason="", default="native"):
    from spacr.qt.model_install import SpotDetectorCombo

    combo = SpotDetectorCombo(default=default,
                              readiness=lambda: (ready, reason))
    qtbot.addWidget(combo)
    return combo


def test_native_is_first_and_chosen(qtbot):
    combo = _combo(qtbot, ready=False, reason="SpotNet is not installed")
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "native", "spotnet"]
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
