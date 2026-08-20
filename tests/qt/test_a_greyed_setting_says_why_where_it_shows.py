"""Instruction 106, the half that was never delivered.

spaCR greys a setting the current selection does not read, and attaches the
reason. The reason was written to the EDITOR's tooltip -- and the editor is
deliberately silent on hover, because the panel shows one tooltip per setting
and it belongs on the label that names it. So every greyed control in spaCR
was disabled without saying why, and it was invisible for exactly the reason
it was written where nobody could see it.

The case that found it is 132's: `level` is greyed under
`regression_type='mixed'`, which nests guides inside genes and so answers both
levels at once. That is not a fact a user can deduce from a dead dropdown.
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


def _help(label) -> str:
    return str(label.property("apiTooltipHtml") or label.toolTip() or "")


def _choose(combo, wanted: str) -> None:
    for i in range(combo.count()):
        if str(combo.itemData(i) or combo.itemText(i)).lower() == wanted:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{wanted} is not on offer")


def test_level_is_greyed_under_mixed_and_the_reason_is_on_screen(qtbot):
    _screen, model = _regression(qtbot)
    level = model._widgets["level"]

    assert level.isEnabled() is False
    label = getattr(level, "_spacr_setting_label", None)
    assert label is not None, "the setting has no label to carry its help"
    # The REASON, not merely a disabled control: "mixed fits both levels at
    # once" is not a fact a user can deduce from a dead dropdown.
    assert "mixed" in _help(label)
    assert "nested" in _help(label).lower()


def test_the_reason_goes_away_when_the_setting_applies_again(qtbot):
    _screen, model = _regression(qtbot)
    level, kind = model._widgets["level"], model._widgets["regression_type"]
    label = level._spacr_setting_label

    _choose(kind, "ols")
    assert level.isEnabled() is True
    assert "nested" not in _help(label).lower()


def test_the_settings_own_help_survives_the_note_coming_and_going(qtbot):
    """A restore that lost a sentence would be worse than no note at all."""
    _screen, model = _regression(qtbot)
    level, kind = model._widgets["level"], model._widgets["regression_type"]
    label = level._spacr_setting_label
    # A PHRASE THE DESCRIPTION ACTUALLY OPENS WITH. This pinned "Which unit
    # results are reported at", which `level`'s description has not said for
    # some time -- so the test failed on the FIRST assertion and never
    # reached the restore it exists to check. The machinery was fine; the
    # expectation had rotted, which is the failure mode this file is about.
    own = "Select the regression fit level"

    assert own in _help(label)               # greyed, note appended
    _choose(kind, "ols")
    assert own in _help(label)               # enabled, note gone
    _choose(kind, "mixed")
    assert own in _help(label)               # greyed again
    assert "nested" in _help(label).lower()


def test_the_note_is_appended_once_however_many_times_it_is_applied(qtbot):
    _screen, model = _regression(qtbot)
    level, kind = model._widgets["level"], model._widgets["regression_type"]
    label = level._spacr_setting_label

    for _ in range(3):
        _choose(kind, "ols")
        _choose(kind, "mixed")
    assert _help(label).lower().count("nested") == 1


def test_the_label_is_greyed_with_its_field_so_the_row_reads_as_one(qtbot):
    _screen, model = _regression(qtbot)
    level, kind = model._widgets["level"], model._widgets["regression_type"]
    label = level._spacr_setting_label

    assert label.isEnabled() is False
    _choose(kind, "ols")
    assert label.isEnabled() is True
