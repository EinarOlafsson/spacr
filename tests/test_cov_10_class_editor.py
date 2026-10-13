"""The class editor when the definitions it is handed cannot be used.

The Classes setting decides what a classifier is trained on, and it arrives
from four places: a settings CSV that stored it as text, an older run that
stored a plain list, a table the user is looking at, and two fields the user
types into. The branches here are the ones where one of those hands over
something that is not a class -- a malformed entry, a value with no column, a
picker that cannot be built, an edit on a row that no longer exists. Each has
to leave the editor usable, because a class silently dropped here is a class
missing from every figure afterwards.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QTreeWidgetItem            # noqa: E402

from spacr.classify_classes import ClassDefinitionError  # noqa: E402
from spacr.qt.widgets import class_editor as CE          # noqa: E402
from spacr.qt.widgets.class_editor import (              # noqa: E402
    ClassEditorWidget)


@pytest.fixture
def editor(qtbot):
    widget = ClassEditorWidget(frame=pd.DataFrame({"condition": ["nc", "pc"]}))
    qtbot.addWidget(widget)
    return widget


def test_a_picker_that_cannot_be_built_is_not_a_broken_editor(editor,
                                                              monkeypatch):
    """The SQL picker is an extra way to name a column; typing one still
    works without it. A failure to build it must not stop the settings panel
    that owns this widget from opening."""
    from spacr.qt.widgets import column_picker

    def _explode(*args, **kwargs):
        raise RuntimeError("no database picker here")

    monkeypatch.setattr(column_picker, "attach_column_picker", _explode)
    assert editor.attach_sql_picker(lambda: "/tmp/run") is None
    assert editor.column.isEditable()


def test_an_entry_that_is_not_a_definition_is_skipped_not_shown_blank(editor):
    """A dict whose value is a bare string says nothing about which column
    selects the class. Showing it as a named row with no rule would invite
    the user to press Run on a class that selects nothing."""
    editor.set_value({"good": {"column": "condition", "value": "nc"},
                      "bad": "just a string"})
    assert [r.name for r in editor.rules()] == ["good"]


def test_a_definition_that_cannot_select_objects_is_dropped(editor):
    """A class with no name has nothing to appear as in a report, and
    ``ClassRule`` refuses it. The editor keeps the classes around it rather
    than failing to load the whole setting."""
    editor.set_value({"": {"column": "condition", "value": "nc"},
                      "keep": {"column": "condition", "value": "pc"}})
    assert [r.name for r in editor.rules()] == ["keep"]


def test_filling_from_a_column_needs_a_column_and_a_table(qtbot):
    """With no table loaded there is nothing to enumerate. The press is a
    no-op rather than an exception out of a button handler."""
    widget = ClassEditorWidget(frame=None)
    qtbot.addWidget(widget)
    widget.column.setCurrentText("condition")
    widget.populate_from_column()
    assert widget.rules() == []


def test_a_typed_value_with_no_column_says_which_field_is_empty(editor):
    """The value alone does not select anything: the editor has to say the
    column is missing rather than adding a class that matches nothing."""
    editor.column.setCurrentText("")
    editor.class_field.setText("negative")
    editor.value_field.setText("nc")
    editor.add_typed_class()
    assert editor.rules() == []
    assert "column" in editor._hint.text()


def test_a_rule_the_model_refuses_is_reported_not_raised(editor, monkeypatch):
    """The two fields are checked first, but the rule object is the authority
    on what can select objects. Its refusal has to reach the hint line
    instead of escaping a button press."""
    def _refuse(*args, **kwargs):
        raise ClassDefinitionError("this rule selects nothing")

    monkeypatch.setattr(CE, "ClassRule", _refuse)
    editor.column.setCurrentText("condition")
    editor.class_field.setText("negative")
    editor.value_field.setText("nc")
    editor.add_typed_class()
    assert editor.rules() == []
    assert "selects nothing" in editor._hint.text()


def test_removing_with_nothing_selected_removes_nothing(editor):
    """The Remove button is reachable with an empty selection; it must not
    delete whichever row happens to be first."""
    editor.set_value({"a": {"column": "condition", "value": "nc"}})
    editor.table.setCurrentItem(None)
    editor.remove_selected()
    assert [r.name for r in editor.rules()] == ["a"]


def test_an_edit_outside_the_name_column_changes_no_class(editor):
    """Only column 0 carries the class name. An edit anywhere else must not
    rename the class it sits beside."""
    editor.set_value({"a": {"column": "condition", "value": "nc"}})
    item = editor.table.topLevelItem(0)
    editor._on_item_changed(item, 1)
    assert [r.name for r in editor.rules()] == ["a"]


def test_an_edit_on_a_row_that_is_not_in_the_table_is_ignored(editor):
    """A signal can arrive for an item already taken out of the tree. Using
    its index would rename an unrelated class."""
    editor.set_value({"a": {"column": "condition", "value": "nc"}})
    stray = QTreeWidgetItem(["renamed"])
    editor._on_item_changed(stray, 0)
    assert [r.name for r in editor.rules()] == ["a"]
