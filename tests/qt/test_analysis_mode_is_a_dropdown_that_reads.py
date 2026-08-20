"""Instruction 134 — "analasys mode should be a dropdown".

`analysis_mode` has exactly two valid values and both front ends rendered it
as a FREE-TEXT box, so a typo in it survived until the run had read the whole
database.

Three properties, and the third is the one that makes the dropdown worth
having rather than merely correct: the labels READ. `guide_permutation` is
what the settings key is called; what a user picks from should say what it is,
the way 132's model box explains what it fits — while the stored value goes on
meaning what it meant to every settings file already written.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _regression(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen, screen._settings_model


def _choose(combo, wanted: str) -> None:
    for i in range(combo.count()):
        if str(combo.itemData(i) or combo.itemText(i)).lower() == wanted:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{wanted} is not on offer")


def test_it_is_a_dropdown_with_exactly_the_two_valid_values(qtbot):
    from PySide6.QtWidgets import QComboBox

    _screen, model = _regression(qtbot)
    combo = model._widgets["analysis_mode"]

    assert isinstance(combo, QComboBox)
    stored = [combo.itemData(i) for i in range(combo.count())]
    assert stored == ["regression", "guide_permutation"]


def test_the_labels_say_what_the_modes_are(qtbot):
    _screen, model = _regression(qtbot)
    combo = model._widgets["analysis_mode"]
    shown = [combo.itemText(i) for i in range(combo.count())]

    assert any("fit every guide at once" in text for text in shown)
    assert any("test each guide on its own" in text for text in shown)
    # And the key itself is not what a user has to read.
    assert not any(text == "guide_permutation" for text in shown)


def test_the_stored_value_is_still_the_key_every_settings_file_uses(qtbot):
    _screen, model = _regression(qtbot)
    combo = model._widgets["analysis_mode"]

    _choose(combo, "guide_permutation")
    assert (model.collect() or {})["analysis_mode"] == "guide_permutation"
    _choose(combo, "regression")
    assert (model.collect() or {})["analysis_mode"] == "regression"


def test_a_settings_file_carrying_either_value_still_loads(qtbot):
    screen, model = _regression(qtbot)
    for value in ("guide_permutation", "regression"):
        screen.apply_settings_dict({"analysis_mode": value})
        assert (model.collect() or {})["analysis_mode"] == value


def test_it_is_greyed_while_inference_is_choosing_it_and_says_why(qtbot):
    """Two controls that contradict each other is what 106 forbids."""
    _screen, model = _regression(qtbot)
    combo, inference = model._widgets["analysis_mode"], model._widgets["inference"]

    _choose(inference, "nonparametric")
    assert combo.isEnabled() is False
    label = getattr(combo, "_spacr_setting_label", None)
    assert label is not None
    help_text = str(label.property("apiTooltipHtml") or label.toolTip() or "")
    assert "inference" in help_text and "nonparametric" in help_text


def test_choosing_auto_hands_the_mode_back_to_the_user(qtbot):
    _screen, model = _regression(qtbot)
    combo, inference = model._widgets["analysis_mode"], model._widgets["inference"]

    _choose(inference, "nonparametric")
    assert combo.isEnabled() is False
    _choose(inference, "auto")
    assert combo.isEnabled() is True


def test_both_front_ends_are_offered_the_same_list(qtbot):
    """One table, so Tk and Qt cannot end up offering different modes."""
    import inspect

    from spacr import settings_spec

    _screen, model = _regression(qtbot)
    combo = model._widgets["analysis_mode"]
    qt_values = [combo.itemData(i) for i in range(combo.count())]

    source = inspect.getsource(settings_spec)
    assert "'analysis_mode': ('combo'" in source
    for value in qt_values:
        assert f"'{value}'" in source
