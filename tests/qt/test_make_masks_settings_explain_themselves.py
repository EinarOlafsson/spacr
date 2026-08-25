"""Every setting on the Make Masks panel is a spaCR setting.

A spaCR setting has four things, and the panel is read back for all four
rather than trusted to have them:

* a DEFAULT, which is the value the canvas is actually working with before
  anything is touched -- not merely a number showing in a box;
* a TYPE, which is the widget that will not let a fraction into a pixel
  count or a percentage past 100;
* a TOOLTIP, on the control or on the label the layout pairs it with,
  because :func:`spacr.qt.screens.settings_model.retarget_field_tooltips`
  moves help onto the setting's NAME as the last step of ``__init__``;
* a PLACE on the panel, inside the one group the settings button hides.

The wand is the reason this file exists. Sixteen of the panel's controls
belong to it, and a knob whose help, default or wiring is missing is
worse than no knob at all: it is a number the user is invited to move
without being told what moves.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QSlider,
    QSpinBox,
)

from spacr.qt.screens.make_masks import MakeMasksScreen
from spacr.qt.screens.settings_model import _sibling_label_for

#: The control on the panel, and the key it moves in the dict the flood
#: reads. THE CLAIM IS ONE CONTROL, ONE SETTING: the test drives each
#: control and asserts the named key moved and no other key did.
WAND_CONTROLS = (
    ("_wand_salvage", "salvage_over_cap"),
    ("_wand_runaway_group", "trim_runaway"),
    ("_wand_runaway_ratio", "runaway_ratio"),
    ("_wand_runaway_warmup", "runaway_warmup"),
    ("_wand_runaway_min_base", "runaway_min_base"),
    ("_wand_runaway_confirm", "runaway_confirm"),
    ("_wand_intensity_border", "intensity_border"),
    ("_wand_intensity_steps", "intensity_steps"),
    ("_wand_gradient_taper", "gradient_taper"),
    ("_wand_gradient_sigma", "gradient_sigma"),
    ("_wand_gradient_margin", "gradient_margin"),
    ("_wand_gradient_erode", "gradient_erode"),
)

#: Widget kinds that are a setting the user types or clicks a value into.
#: A checkable QGroupBox is one of them: the "Trim a runaway flood" box is
#: the master switch for the group it draws.
_CONTROL_KINDS = (QAbstractSpinBox, QCheckBox, QComboBox, QSlider, QGroupBox)


def _controls(screen) -> list:
    """Every value-carrying control inside the settings group, once."""
    panel = screen._settings_scroll
    found: list = []
    for kind in _CONTROL_KINDS:
        for widget in panel.findChildren(kind):
            if isinstance(widget, QGroupBox) and not widget.isCheckable():
                continue                    # a heading, not a setting
            if widget not in found:
                found.append(widget)
    return found


def _name_of(screen, widget) -> str:
    for name, value in vars(screen).items():
        if value is widget:
            return name
    return f"<unnamed {type(widget).__name__}>"


def _nudge(widget) -> None:
    """Move the control off its default, whichever kind it is."""
    if isinstance(widget, (QCheckBox, QGroupBox)):
        widget.setChecked(not widget.isChecked())
        return
    step = widget.singleStep()
    value = widget.value()
    widget.setValue(value + step if value + step <= widget.maximum()
                    else value - step)


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made.close_folded()


def test_every_control_on_the_settings_panel_explains_itself(screen):
    """No setting is a bare number: each one, or its name, carries help.

    Read off the built panel rather than off the source, because the help
    does not stay where it was written -- ``retarget_field_tooltips`` moves
    a field's tooltip onto the label that names it, so a source-level
    ``setToolTip`` search would pass while every spin box on screen was
    silent.
    """
    controls = _controls(screen)
    assert len(controls) > 25, "the settings panel came up nearly empty"

    silent = []
    for widget in controls:
        label = _sibling_label_for(widget)
        helped = bool(widget.toolTip()) or bool(
            label is not None and label.toolTip())
        if not helped:
            silent.append(_name_of(screen, widget))
    assert not silent, "settings with no help anywhere: " + ", ".join(silent)


def test_every_wand_setting_is_typed_and_on_the_panel(screen):
    """The wand's sixteen controls are typed widgets inside the settings.

    Typed, so a pixel budget cannot be given a fraction and a percentage
    cannot be given 400; and inside the settings scroll, so the settings
    button takes all of them away together and the canvas gets the width.
    """
    panel = screen._settings_scroll
    wand = [w for w in _controls(screen)
            if _name_of(screen, w).startswith("_wand")]
    assert len(wand) == 16, [_name_of(screen, w) for w in wand]

    for widget in wand:
        name = _name_of(screen, widget)
        assert isinstance(
            widget, (QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox)), name
        assert panel.isAncestorOf(widget), f"{name} is outside the settings"
        if isinstance(widget, QAbstractSpinBox):
            assert widget.minimum() < widget.maximum(), name


def test_every_wand_setting_has_a_default_the_flood_is_already_using(screen):
    """A default is what the canvas holds, not what a box happens to show.

    A control wired only to its own handler shows a number the flood never
    sees until somebody touches it, which is the one state in which the
    panel and the tool disagree and nothing on screen says so.
    """
    live = screen._canvas.wand_rescue_settings()
    for attribute, key in WAND_CONTROLS:
        widget = getattr(screen, attribute)
        shown = (widget.isChecked()
                 if isinstance(widget, (QCheckBox, QGroupBox))
                 else widget.value())
        assert live[key] == pytest.approx(shown), attribute


@pytest.mark.parametrize("attribute,key", WAND_CONTROLS)
def test_a_wand_control_moves_its_own_setting_and_no_other(
        screen, attribute: str, key: str):
    """One control, one setting: the rest of the dict is untouched."""
    canvas = screen._canvas
    before = canvas.wand_rescue_settings()
    _nudge(getattr(screen, attribute))
    after = canvas.wand_rescue_settings()

    assert after[key] != before[key], f"{attribute} moved nothing"
    changed = {k for k in before if after[k] != before[k]}
    assert changed == {key}, f"{attribute} also moved {changed - {key}}"


def test_the_two_tolerance_boxes_never_both_answer(screen):
    """Only the tolerance in force is editable, so neither is ambiguous.

    Two enabled boxes on the panel would leave which one the wand uses to
    be inferred from a checkbox three rows up.
    """
    assert screen._wand_relative.isChecked()
    assert screen._wand_pct.isEnabled()
    assert not screen._wand_tol.isEnabled()

    screen._wand_relative.setChecked(False)
    assert not screen._wand_pct.isEnabled()
    assert screen._wand_tol.isEnabled()
    assert screen._canvas.wand_relative is False


def test_every_help_string_on_the_panel_has_an_exact_catalog_row(screen):
    """A caption with no exact row is translated word by word, and mangles.

    :func:`spacr.qt.i18n.tr` falls back to a term-by-term match when it has
    no row for a whole string, which turns a sentence of help into half
    English. The panel's help is long prose, so the fallback is at its
    worst here: this pins every string on the panel to a real row in every
    shipped language.
    """
    from spacr.qt.i18n import has_translation
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES

    strings = set()
    for widget in _controls(screen):
        for source in (widget.toolTip(), ):
            if source:
                strings.add(source)
        label = _sibling_label_for(widget)
        if label is not None and label.toolTip():
            strings.add(label.toolTip())
    assert len(strings) > 25

    missing = sorted(
        f"{code}: {source[:60]}…"
        for code in CATALOG_LANGUAGES
        for source in strings
        if not has_translation(source, code)
    )
    assert not missing, "captions with no exact row:\n" + "\n".join(missing)
