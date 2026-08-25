"""Every Preferences row is explained, on its label.

"all of the settings in preferences also need tooltips and only on the
setting text not the field, like allways."

The words are what a reader points at when they want to know what
something is; a tooltip on the control is one they find only after
reaching for it. Before this, 27 rows of 110 explained themselves on the
field and 83 said nothing at all.

The check walks the FINISHED dialog rather than the source, so a row
added anywhere in it is covered -- and a row added without a tooltip
fails here rather than shipping unexplained.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout, QLabel

from spacr.qt.preferences import PREFERENCE_TIPS, PreferencesDialog


@pytest.fixture
def rows(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Every ``(label, field)`` pair the dialog builds."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)

    found = []
    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            field_item = form.itemAt(index, QFormLayout.FieldRole)
            if label_item is None or field_item is None:
                continue
            label, field = label_item.widget(), field_item.widget()
            if isinstance(label, QLabel) and field is not None:
                found.append((label, field))
    assert found, "the dialog built no rows at all"
    return found


def test_every_row_is_explained(rows):
    unexplained = [label.text() for label, _field in rows
                   if not (label.toolTip() or "").strip()]

    assert not unexplained, f"rows with no tooltip: {unexplained}"


def test_no_tooltip_is_left_on_a_setting_field(rows):
    """A tooltip on both reads as two answers, and one on the control
    alone is the thing this was asked to change.

    ACTION BUTTONS ARE EXEMPT, and the exemption is the distinction rather
    than a concession. A row whose field is a button -- Clear RAM, Clear
    GPU memory, Check disk -- is not a setting with a value: its tooltip
    says what pressing it DOES, which for these is the confirmation the
    press will ask for. Moving that onto the label and clearing the button
    leaves the user hovering the thing they are about to press and being
    told nothing, and it breaks the resource-cleanup contract that the
    button's tooltip IS `confirmation_text(action)`.
    """
    from PySide6.QtWidgets import QPushButton, QToolButton

    on_the_field = [label.text() for label, field in rows
                    if (field.toolTip() or "").strip()
                    and not isinstance(field, (QPushButton, QToolButton))]

    assert not on_the_field, f"tooltip still on the field for: {on_the_field}"


def test_the_explanations_say_something(rows):
    """A tooltip that repeats the label explains nothing."""
    empty = []
    for label, _field in rows:
        text = (label.text() or "").replace("&", "").strip().lower()
        tip = (label.toolTip() or "").strip()
        if len(tip) < 12 or tip.lower() == text:
            empty.append(label.text())

    assert not empty, f"these say nothing the label did not: {empty}"


def test_the_table_is_not_carrying_dead_keys(rows):
    """A key for a row that no longer exists is a tooltip nobody sees, and
    the next person to read the table cannot tell which."""
    shown = {(label.text() or "").replace("&", "").strip()
             for label, _field in rows}
    dead = sorted(set(PREFERENCE_TIPS) - shown)

    assert not dead, f"PREFERENCE_TIPS names rows that are not there: {dead}"
