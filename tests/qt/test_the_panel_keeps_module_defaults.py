"""The panel must show the default the module itself declares.

``convert_settings_dict_for_gui`` replaces the caller's value whenever the
key is in its ``special_cases`` table::

    elif key in special_cases:
        variables[key] = special_cases[key]      # the supplied value is dropped

That table is ONE ROW PER KEY FOR THE WHOLE APPLICATION, so every module gets
the same answer regardless of what its own defaults factory declared.
``SettingsWidgets._widget_for`` corrects this locally, in the ``combo``
branch::

    if key in self._defaults:
        default = self._defaults[key]

Nothing pinned that correction.  The invariant used to be enforced by
``tests/test_tk_panel_keeps_module_defaults.py``, which drove the Tk repair
(``gui_core._restore_module_defaults``); both the repair and its test went
with the Tk front end, and the Qt half was left unguarded.  This file is that
guard, pointed at the front end that ships.

WHY IT MATTERS RATHER THAN BEING TIDINESS.  Every override below is a
different run, not a different caption:

    Classify    model_type              declares maxvit_t, canned resnet50
                                        -- a different network architecture
    Regression  analysis_mode           declares guide_permutation, canned
                                        regression -- a different analysis
    Mask etc.   summarize_organelles_by declares 'cell', canned None

The organelle summary row is built only when ``number_of_organelles`` says
the run has an organelle. Its named cases therefore build the panel for one
slot; that keeps testing the override without requiring an irrelevant row on
a run whose count is zero.

Opening a module and pressing Run without touching anything must run the
values the module asked for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox

from spacr.qt.app import APPS
from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.settings_spec import convert_settings_dict_for_gui


APP_KEYS = tuple(row[0] for row in APPS)


def _normal(value):
    """'[0,1,2,3]' and [0, 1, 2, 3] are one choice written two ways."""
    return str(value).replace(" ", "")


def _built(app_key, *, current=None):
    """The panel's widgets plus the spec they were built from."""
    model = SettingsWidgets(app_key, current=current)
    model.build_sections()
    return model, convert_settings_dict_for_gui(model._defaults)


def _combo_value(widget):
    """What the closed control would post.

    ``_ValueCombo`` stores the VALUE as item data and shows a caption, so
    reading the text alone answers for the label rather than for the run.
    """
    data = widget.currentData()
    return widget.currentText() if data is None else data


def _overrides(app_key):
    """(key, declared, canned) for every combo the canned table overrides."""
    model, spec = _built(app_key)
    found = []
    for key, widget in model._widgets.items():
        if not isinstance(widget, QComboBox) or key not in model._defaults:
            continue
        declared = model._defaults[key]
        canned = spec.get(key, (None, None, None))[2]
        if _normal(canned) != _normal(declared):
            found.append((key, declared, canned))
    return model, found


# ---------------------------------------------------------------------------
# the invariant, over every registered module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", APP_KEYS)
def test_every_panel_combo_shows_the_module_s_own_default(qtbot, app_key):
    """The whole point of a defaults factory is that it decides."""
    model, _ = _built(app_key)

    drift = {
        key: (model._defaults[key], _combo_value(widget))
        for key, widget in model._widgets.items()
        if isinstance(widget, QComboBox) and key in model._defaults
        and _normal(_combo_value(widget)) != _normal(model._defaults[key])
    }
    assert not drift, (
        f"the {app_key} panel disagrees with its own defaults "
        f"(declared, shown): {drift}")


def test_the_canned_table_really_does_override_something(qtbot):
    """A guard nothing exercises passes whether or not it works.

    If this fails because every canned default now agrees with every
    module's, the correction in ``_widget_for`` has become untested rather
    than unnecessary -- delete it deliberately or find the key that still
    needs it.
    """
    overridden = []
    for app_key in APP_KEYS:
        _, found = _overrides(app_key)
        overridden.extend((app_key, key) for key, _, _ in found)
    assert overridden, (
        "no module's declared default differs from the shared special_cases "
        "table, so nothing here proves the panel prefers the module's")


@pytest.mark.parametrize(("app_key", "key"), [
    ("mask", "summarize_organelles_by"),
    ("measure", "summarize_organelles_by"),
    ("external_masks", "summarize_organelles_by"),
    ("classify_merged", "model_type"),
    ("regression", "analysis_mode"),
    ("regression", "transform"),
])
def test_the_known_overrides_are_each_named(qtbot, app_key, key):
    """Named individually, so a regression says WHICH module came back."""
    current = ({"number_of_organelles": 1}
               if key == "summarize_organelles_by" else None)
    model, spec = _built(app_key, current=current)
    assert key in model._widgets, f"{app_key} no longer offers {key}"

    declared = model._defaults[key]
    canned = spec[key][2]
    assert _normal(canned) != _normal(declared), (
        f"{app_key}.{key} is no longer overridden by the canned table, so "
        f"this case no longer tests the correction; re-point or drop it")
    assert _normal(_combo_value(model._widgets[key])) == _normal(declared)


# ---------------------------------------------------------------------------
# how the default is kept, not merely which one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", APP_KEYS)
def test_a_restored_default_is_offered_rather_than_substituted(qtbot, app_key):
    """Selecting a value the list does not hold selects nothing.

    A combo whose declared default is absent from its options falls back to
    index 0 -- a value the module never asked for -- so the default has to be
    IN the list, not merely aimed at.
    """
    model, _ = _built(app_key)
    missing = []
    for key, widget in model._widgets.items():
        if not isinstance(widget, QComboBox) or key not in model._defaults:
            continue
        declared = _normal(model._defaults[key])
        offered = {_normal(widget.itemData(i)) for i in range(widget.count())}
        offered |= {_normal(widget.itemText(i)) for i in range(widget.count())}
        if declared not in offered:
            missing.append((key, model._defaults[key]))
    assert not missing, (
        f"{app_key} combos whose declared default is not among the options "
        f"they offer: {missing}")


@pytest.mark.parametrize("app_key", APP_KEYS)
def test_a_spelling_difference_does_not_add_a_duplicate_option(qtbot, app_key):
    """Restoring across whitespace would grow the list on every build.

    ``channels`` is declared ``[0, 1, 2, 3]`` and offered as ``'[0,1,2,3]'``.
    Those are one choice, and a correction that could not see that would
    insert a second entry differing only in spaces.
    """
    model, _ = _built(app_key)
    duplicated = {}
    for key, widget in model._widgets.items():
        if not isinstance(widget, QComboBox):
            continue
        seen, repeats = set(), []
        for i in range(widget.count()):
            normalised = _normal(widget.itemData(i)
                                 if widget.itemData(i) is not None
                                 else widget.itemText(i))
            if normalised in seen:
                repeats.append(normalised)
            seen.add(normalised)
        if repeats:
            duplicated[key] = repeats
    assert not duplicated, (
        f"{app_key} combos offering the same value twice: {duplicated}")


def test_a_key_the_module_never_declared_keeps_the_canned_default(qtbot):
    """The correction is "prefer the module", not "override the table".

    A key absent from the module's own defaults has no module opinion to
    prefer, and must still get the curated list's own choice rather than
    index 0.
    """
    model = SettingsWidgets("mask")
    widget = model._widget_for("combo", ["a", "b", "c"], "b", "not_a_setting")
    qtbot.addWidget(widget)
    assert "not_a_setting" not in model._defaults
    assert _combo_value(widget) == "b"
