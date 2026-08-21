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


class TestTheClassesReplaceTheTwoSettings:
    """Instruction 229: "the Classes Column setting should also replace the
    annotation column and class metadata should be replaced by the Classes
    Class and value".

    ONE PLACE FOR EACH FACT. Nothing downstream reads both and compares
    them, so two settings naming the same column is two chances to point at
    different ones.
    """

    def test_the_column_comes_from_the_classes(self):
        from spacr.classify_classes import annotation_column_of

        assert annotation_column_of(
            {"classes": {"pc": {"column": "rowID", "value": "r2"}}}) == \
            "rowID"

    def test_the_values_come_from_the_classes(self):
        from spacr.classify_classes import class_metadata_of

        assert class_metadata_of({"classes": {
            "pc": {"column": "columnID", "value": "c3"},
            "nc": {"column": "columnID", "value": "c1"}}}) == \
            [["c3"], ["c1"]]

    def test_an_old_settings_file_is_untouched(self):
        """A file written before the Classes editor still runs."""
        from spacr.classify_classes import (annotation_column_of,
                                            class_metadata_of)

        old = {"annotation_column": "test", "class_metadata": [["c1"]]}
        assert annotation_column_of(old) == "test"
        assert class_metadata_of(old) == [["c1"]]

    def test_the_defaults_write_them_back(self):
        """Derived values are WRITTEN rather than computed at every read:
        every consumer downstream reads those two keys."""
        from spacr.settings import set_generate_training_dataset_defaults

        got = set_generate_training_dataset_defaults({"classes": {
            "pc": {"column": "rowID", "value": "r2"},
            "nc": {"column": "rowID", "value": "r1"}}})
        assert got["annotation_column"] == "rowID"
        assert got["class_metadata"] == [["r2"], ["r1"]]

    def test_no_classes_leaves_the_defaults_alone(self):
        from spacr.settings import set_generate_training_dataset_defaults

        got = set_generate_training_dataset_defaults({})
        assert got["annotation_column"] == "test"
        assert got["class_metadata"] == [["c1"], ["c2"]]

    def test_neither_is_a_control_any_more(self):
        """A box for either was a second place to say the same thing."""
        import inspect

        from spacr.qt.screens import settings_model

        source = inspect.getsource(settings_model)
        block = source.split('"Labels & Classes": [')[1].split("],")[0]
        assert '"annotation_column"' not in block
        assert '"class_metadata"' not in block

    def test_but_both_are_still_written(self):
        """Every consumer downstream is unchanged."""
        from spacr.settings import set_generate_training_dataset_defaults

        got = set_generate_training_dataset_defaults({})
        assert "annotation_column" in got and "class_metadata" in got
