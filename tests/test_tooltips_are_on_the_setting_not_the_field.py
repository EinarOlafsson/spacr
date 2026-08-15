"""Hover help belongs on a setting's name, never on the field.

Instruction 113. The generic decorator (`install_api_tooltips`) has done this
correctly since 2026-07-30 -- it ends with ``widget.setToolTip("")`` and the
comment "the editor itself remains quiet on hover". Hand-built screens never
went through it, and 155 editors across the Qt screens still popped help over
the field the user was about to type into.

This is a SWEEP guard, not an example test. It builds every screen it can and
asserts the rule over all of them, because the defect is not in one place --
it is a convention that 64 hand-built screens each had to remember, and the
only durable version of "remember it" is a test that fails when one does not.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QSpinBox, QTextEdit,
    QWidget,
)

pytestmark = pytest.mark.qt

#: Editors for a setting. QCheckBox and Toggle are absent on purpose: they
#: carry their own text, so they ARE the name and hovering them is correct.
EDITORS = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit,
           QTextEdit)


def _constructible_screens():
    """Every screen class that can be built with no arguments."""
    import spacr.qt.screens as pkg

    for module_info in pkgutil.iter_modules(pkg.__path__):
        try:
            module = importlib.import_module(
                f"spacr.qt.screens.{module_info.name}")
        except Exception:
            continue
        for name, cls in vars(module).items():
            if not (inspect.isclass(cls) and issubclass(cls, QWidget)):
                continue
            if cls.__module__ != module.__name__:
                continue
            yield module_info.name, name, cls


def _offenders(widget):
    """Editors carrying a tooltip that have a label naming them."""
    from spacr.qt.screens.settings_model import _sibling_label_for

    return [child for child in widget.findChildren(QWidget)
            if isinstance(child, EDITORS)
            and child.toolTip()
            and _sibling_label_for(child) is not None]


def test_no_screen_puts_hover_help_on_an_editable_field(qtbot):
    """The sweep. Zero, across every screen that builds."""
    found = []
    built = 0
    for module_name, class_name, cls in _constructible_screens():
        try:
            screen = cls()
            screen.resize(900, 700)
        except Exception:
            continue
        built += 1
        offenders = _offenders(screen)
        if offenders:
            found.append(
                f"{module_name}.{class_name}: {len(offenders)} "
                f"({', '.join(sorted({type(o).__name__ for o in offenders}))})")

    assert built > 40, (
        f"only {built} screens built -- the sweep is not covering the app, "
        "so a green result here would prove nothing")
    assert not found, (
        "hover help is on the FIELD instead of the setting's name in:\n  "
        + "\n  ".join(found)
        + "\n\nCall settings_model.retarget_field_tooltips(self) at the end "
          "of the screen's __init__.")


def test_the_helper_moves_the_tooltip_and_leaves_the_field_quiet(qtbot):
    """The mechanism itself, on a screen known to have had 14 offenders."""
    from spacr.qt.screens.power import PowerScreen
    from spacr.qt.screens.settings_model import (
        _sibling_label_for, retarget_field_tooltips)

    screen = PowerScreen()
    screen.resize(900, 700)
    # Already clean, because PowerScreen calls the helper itself now.
    assert _offenders(screen) == []

    # Put one back and prove the helper moves it.
    field = next(c for c in screen.findChildren(QWidget)
                 if isinstance(c, EDITORS) and _sibling_label_for(c))
    label = _sibling_label_for(field)
    label.setToolTip("")
    field.setToolTip("Help that belongs on the label.")
    assert _offenders(screen)

    moved = retarget_field_tooltips(screen)
    assert moved >= 1
    assert field.toolTip() == ""
    assert label.toolTip() == "Help that belongs on the label."


def test_a_disabled_reason_stays_on_the_control_it_explains(qtbot):
    """"This control does nothing because ..." explains THAT control.

    gate_settings appends exactly such a note to a field. It is not
    descriptive help and must not be swept onto the label, or it stops
    describing the thing it is about.
    """
    from spacr.qt.screens.power import PowerScreen
    from spacr.qt.screens.settings_model import (
        DISABLED_REASON_TOOLTIP, _sibling_label_for, retarget_field_tooltips)

    screen = PowerScreen()
    screen.resize(900, 700)
    field = next(c for c in screen.findChildren(QWidget)
                 if isinstance(c, EDITORS) and _sibling_label_for(c))
    field.setToolTip("NOT YET IN EFFECT: the 3D volume is not built.")
    field.setProperty(DISABLED_REASON_TOOLTIP, True)

    retarget_field_tooltips(screen)
    assert field.toolTip().startswith("NOT YET IN EFFECT")


def test_a_field_with_no_label_keeps_its_tooltip(qtbot):
    """With no separate name, the editor is the only visible identity there
    is -- taking its tooltip away would leave no help at all."""
    from spacr.qt.screens.settings_model import retarget_field_tooltips

    host = QWidget()
    host.resize(300, 80)
    lone = QComboBox(host)
    lone.setToolTip("The only help this control has.")

    retarget_field_tooltips(host)
    assert lone.toolTip() == "The only help this control has."
