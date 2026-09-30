"""Make Masks' "Use in Mask generation" button is an alpha feature.

The Enhancement card's button is registered as
``MakeMasksUseInMaskGeneration`` in ``spacr.settings.ALPHA_FEATURES``: shown
only while Preferences -> "Show alpha features" is on, and a chain written
while it is hidden still reaches Mask generation's settings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot, monkeypatch, shown):
    from spacr.qt import preferences
    from spacr.qt.screens.make_masks import MakeMasksScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: shown)
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    return made


def test_registered_under_its_item():
    from spacr.settings import ALPHA_FEATURES, _alpha_names

    assert "MakeMasksUseInMaskGeneration" in ALPHA_FEATURES[508]["widgets"]
    assert "MakeMasksUseInMaskGeneration" in _alpha_names("widgets")


def test_the_button_is_hidden_unless_alpha_features_are_shown(
        qtbot, monkeypatch):
    from spacr.qt import preferences

    for shown in (False, True):
        screen = _screen(qtbot, monkeypatch, shown)
        button = screen._btn_to_mask
        assert button.objectName() == "MakeMasksUseInMaskGeneration"
        assert button.isHidden() is not shown
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(screen)
        assert button.isHidden() is shown
        screen.close_folded()


def test_a_chain_set_while_hidden_still_reaches_mask_generation(
        qtbot, monkeypatch):
    received = []

    class Target:
        def apply_settings_dict(self, values):
            received.append(values)

    class Window:
        _screens = {"mask": Target()}

        def _on_nav_selected(self, key):
            pass

    screen = _screen(qtbot, monkeypatch, False)
    assert screen._btn_to_mask.isHidden()
    screen._enh_gamma.setValue(0.6)
    assert screen.mask_settings()["enhance_gamma"] == pytest.approx(0.6)
    monkeypatch.setattr(screen, "window", lambda: Window())
    screen._send_chain_to_mask()
    assert received[0]["enhance_gamma"] == pytest.approx(0.6)
    screen.close_folded()
