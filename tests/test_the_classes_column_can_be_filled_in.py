"""The classes column can be filled in, so classes can be added.

Instruction 229's blocking bug, in the maintainer's words: "the classes
column setting cannot be filled in and therefor classes cannot be added".

THE CAUSE WAS NOT A DISABLED WIDGET. The column combo is populated from a
LOADED TABLE, and the panel is reached before one is loaded -- so the list
came back empty, and a non-editable empty QComboBox offers no way to name
anything. The setting that decides every class in the module was the one
setting that could not be filled in.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(app):
    from spacr.qt.widgets.class_editor import ClassEditorWidget

    return ClassEditorWidget()


class TestTheBugItself:

    def test_with_no_table_the_column_list_is_empty(self, editor):
        """The precondition. If this ever stops being true, so does the bug."""
        assert editor.column.count() == 0

    def test_and_a_column_can_still_be_named(self, editor):
        assert editor.column.isEditable(), (
            "an empty, non-editable combo offers no way to name a column")
        editor.column.setCurrentText("columnID")
        assert editor.column.currentText() == "columnID"

    def test_so_a_class_can_be_added(self, editor):
        editor.column.setCurrentText("columnID")
        editor.class_field.setText("pc")
        editor.value_field.setText("c3")
        editor.add_typed_class()
        assert editor.value() == {"pc": {"column": "columnID",
                                         "value": "c3"}}

    def test_and_more_than_one(self, editor):
        editor.column.setCurrentText("columnID")
        for name, value in (("pc", "c3"), ("nc", "c1")):
            editor.class_field.setText(name)
            editor.value_field.setText(value)
            editor.add_typed_class()
        assert sorted(editor.value()) == ["nc", "pc"]


class TestATypedColumnSurvives:
    """`set_frame` runs again whenever the basis changes or a table lands."""

    def test_a_typed_name_is_not_cleared_by_a_reload(self, editor):
        editor.column.setCurrentText("my_annotation")
        editor.set_frame(None)
        assert editor.column.currentText() == "my_annotation", (
            "clearing the combo threw away a column the user had named, "
            "silently -- an empty combo looks like one nobody has touched")

    def test_a_typed_name_survives_a_basis_change(self, editor):
        editor.column.setCurrentText("my_annotation")
        editor.set_basis("metadata")
        assert editor.column.currentText() == "my_annotation"

    def test_a_real_table_still_offers_its_columns(self, editor):
        frame = pd.DataFrame({"columnID": ["c1", "c3"],
                              "rowID": ["r1", "r2"]})
        editor.set_frame(frame)
        offered = [editor.column.itemText(i)
                   for i in range(editor.column.count())]
        assert "columnID" in offered and "rowID" in offered


class TestTheSqlButton:
    """Asked for in as many words: "The Classes setting Column should have a
    SQL butto"."""

    def test_the_button_attaches(self, editor):
        button = editor.attach_sql_picker(lambda: "/nonexistent")
        assert button is not None
        assert button.text() == "SQL"

    def test_classes_is_registered_for_a_column_picker(self):
        from spacr.qt.screens.app_screen import COLUMN_TABLES

        assert "classes" in COLUMN_TABLES, (
            "without this the screen never offers to attach the button")

    def test_a_bad_path_is_survivable(self, editor):
        """Not by clicking: the button opens a MODAL dialog, and a test that
        presses it hangs rather than fails. What is checkable without a user
        is that attaching against a nonexistent path returns a button at all
        rather than raising while the panel is being built."""
        assert editor.attach_sql_picker(lambda: "/nonexistent/nowhere") \
            is not None
