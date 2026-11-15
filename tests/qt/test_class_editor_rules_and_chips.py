"""The class editor keeps its rules, chip strip and hidden table in step.

:class:`spacr.qt.widgets.class_editor.ClassEditorWidget` holds the
``classes`` setting three times over: ``rules()``, a strip of chips whose
close marks carry an index, and a hidden tree whose row order mirrors the
rules.  These tests walk the ways a value can arrive (mapping, literal text,
bare text, malformed entries), the ways a class can be added and removed, and
the strip teardown, asserting the exact rule names, chips and hint text after
each step.  A class dropped by accident here is missing from every result.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QTreeWidgetItem                         # noqa: E402

from spacr.classify_classes import ClassDefinitionError               # noqa: E402
from spacr.qt.widgets import class_editor as CE                       # noqa: E402
from spacr.qt.widgets.class_editor import ClassChip, ClassEditorWidget  # noqa: E402
from spacr.qt.widgets.sortable_table import tree_item                 # noqa: E402

pytestmark = pytest.mark.qt

NC = {"column": "condition", "value": "nc"}
PC = {"column": "condition", "value": "pc"}


@pytest.fixture
def make_editor(qtbot):
    """Editors over a two-value table, the shape the settings panel builds."""
    def make(frame="default"):
        if frame == "default":
            frame = pd.DataFrame({"condition": ["nc", "pc", "nc"]})
        widget = ClassEditorWidget(frame=frame)
        qtbot.addWidget(widget)
        return widget
    return make


def _chips(widget):
    """The chip labels in strip order."""
    layout = widget._chips_layout
    return [layout.itemAt(i).widget().name_pill.text()
            for i in range(layout.count())
            if isinstance(layout.itemAt(i).widget(), ClassChip)]


def _names(widget):
    """The rule names in order."""
    return [rule.name for rule in widget.rules()]


def test_a_stored_value_loads_only_the_classes_it_defines(make_editor,
                                                          subtests):
    """``set_value``: mappings, literal text, bare text and entries that
    cannot select objects."""
    cases = [
        ("a mapping", {"nc": NC, "pc": PC}, ["nc", "pc"],
         ["condition", "condition"]),
        ("list literal text", "['nc', 'pc']", ["nc", "pc"], ["?", "?"]),
        ("bare text", "nc", [], []),
        ("hand-typed list without brackets", "  nc,pc  ", [], []),
        ("an entry that is not a definition",
         {"good": NC, "bad": "just a string"}, ["good"], ["condition"]),
        ("a class with no name", {"": NC, "keep": PC}, ["keep"],
         ["condition"]),
    ]
    for case, value, names, columns in cases:
        with subtests.test(msg=case):
            editor = make_editor()
            editor.set_value({"nc": NC, "pc": PC})
            editor.set_value(value)
            assert _names(editor) == names
            assert list(editor.value()) == names
            assert [rule.column for rule in editor.rules()] == columns
            assert editor.table.topLevelItemCount() == len(names)
            assert _chips(editor) == names

    with subtests.test(msg="a well-formed value straight after bare text"):
        editor = make_editor()
        editor.set_value("nc,pc")
        editor.set_value({"nc": NC})
        assert _chips(editor) == ["nc"]


def test_removal_takes_exactly_the_class_pointed_at(make_editor, subtests):
    """``remove_at``, ``remove_selected`` and ``_on_item_changed`` ignore
    stale indices, non-class rows and edits outside the name column."""
    with subtests.test(msg="stale chip indices"):
        editor = make_editor()
        editor.set_value({"nc": NC, "pc": PC})
        for stale in (7, -1, 2):
            editor.remove_at(stale)
        assert _names(editor) == ["nc", "pc"]
        assert _chips(editor) == ["nc", "pc"]
        editor._chips_layout.itemAt(0).widget()._close.click()
        assert _names(editor) == _chips(editor) == ["pc"]

    with subtests.test(msg="a press left over from a longer list"):
        editor = make_editor()
        editor.set_value({"nc": NC, "pc": PC})
        editor.set_value({"nc": NC})
        editor.remove_at(1)
        assert _names(editor) == _chips(editor) == ["nc"]
        editor._chips_layout.itemAt(0).widget()._close.click()
        assert editor.rules() == [] and _chips(editor) == []

    with subtests.test(msg="a child row is not a class row"):
        editor = make_editor()
        editor.set_value({"nc": NC, "pc": PC})
        parent_row = editor.table.topLevelItem(0)
        parent_row.addChild(QTreeWidgetItem(["not a class"]))
        editor.table.setCurrentItem(parent_row.child(0))
        editor.remove_selected()
        assert _names(editor) == _chips(editor) == ["nc", "pc"]
        editor.table.setCurrentItem(editor.table.topLevelItem(1))
        editor.remove_selected()
        assert _names(editor) == ["nc"]

    with subtests.test(msg="a row no rule stands behind"):
        editor = make_editor()
        editor.set_value({"nc": NC})
        ghost = tree_item(["ghost", "x", "condition"])
        editor.table.addTopLevelItem(ghost)
        editor.table.setCurrentItem(ghost)
        editor.remove_selected()
        assert _names(editor) == ["nc"]
        assert editor.table.topLevelItemCount() == 2
        editor.table.setCurrentItem(editor.table.topLevelItem(0))
        editor.remove_selected()
        assert editor.rules() == []
        assert editor.table.topLevelItemCount() == 0
        assert _chips(editor) == []

    with subtests.test(msg="nothing selected"):
        editor = make_editor()
        editor.set_value({"a": NC})
        editor.table.setCurrentItem(None)
        editor.remove_selected()
        assert _names(editor) == ["a"]

    for case, column in [("an edit outside the name column", 1),
                         ("an edit on a row not in the table", 0)]:
        with subtests.test(msg=case):
            editor = make_editor()
            editor.set_value({"a": NC})
            item = (editor.table.topLevelItem(0) if column
                    else QTreeWidgetItem(["renamed"]))
            editor._on_item_changed(item, column)
            assert _names(editor) == ["a"]


def test_adding_a_class_says_what_it_added_or_why_not(make_editor,
                                                      monkeypatch, subtests):
    """``populate_from_column``, ``add_typed_class`` and
    ``add_random_complement`` with the exact hint each leaves."""
    with subtests.test(msg="a column's values, twice"):
        editor = make_editor()
        editor.column.setCurrentText("condition")
        editor.populate_from_column()
        assert list(editor.value()) == ["condition=nc", "condition=pc"]
        assert editor._hint.text() == "added 2 value(s) from condition"
        editor.populate_from_column()
        assert list(editor.value()) == ["condition=nc", "condition=pc"]
        assert editor._hint.text() == "condition adds nothing new"

    with subtests.test(msg="no table to fill from"):
        editor = make_editor(frame=None)
        editor.column.setCurrentText("condition")
        editor.populate_from_column()
        assert editor.rules() == []

    for case, column, refuse, hint in [
            ("a typed value with no column", "", False,
             "choose the column the value comes from"),
            ("a rule the model refuses", "condition", True,
             "this rule selects nothing"),
    ]:
        with subtests.test(msg=case):
            editor = make_editor()
            if refuse:
                def _refuse(*_args, **_kwargs):
                    raise ClassDefinitionError("this rule selects nothing")
                monkeypatch.setattr(CE, "ClassRule", _refuse)
            editor.column.setCurrentText(column)
            editor.class_field.setText("negative")
            editor.value_field.setText("nc")
            editor.add_typed_class()
            assert editor.rules() == []
            assert editor._hint.text() == hint
            monkeypatch.undo()

    with subtests.test(msg="a second random-rest class"):
        editor = make_editor()
        editor.set_value({"nc": NC})
        editor.add_random_complement()
        assert list(editor.value()) == ["nc", "rest"]
        assert editor.value()["rest"]["random_complement"] is True
        editor.add_random_complement()
        assert list(editor.value()) == ["nc", "rest"]
        assert editor._hint.text() == (
            "there is already a random-rest class; two classes both "
            "meaning 'everything else' have no boundary between them")


def test_the_chip_strip_is_rebuilt_whatever_items_it_holds(make_editor,
                                                           subtests):
    """The strip teardown removes spacing and stretches at either end, so
    the redraw shows exactly the current classes."""
    for case, start, leading_spacing, expected in [
            ("a trailing stretch", {"nc": NC}, False, ["nc", "rest"]),
            ("spacing at both ends", {"nc": NC, "pc": PC}, True,
             ["nc", "pc", "rest"]),
    ]:
        with subtests.test(msg=case):
            editor = make_editor()
            editor.set_value(start)
            if leading_spacing:
                editor._chips_layout.insertSpacing(0, 8)
            editor._chips_layout.addStretch(1)
            assert editor._chips_layout.count() == len(start) + 1 + int(
                leading_spacing)
            editor.add_random_complement()
            assert _chips(editor) == expected
            assert editor._chips_layout.count() == len(expected)
            chips = [editor._chips_layout.itemAt(i).widget()
                     for i in range(editor._chips_layout.count())]
            assert [chip.value_pill.text() for chip in chips] == (
                expected[:-1] + ["the rest, at random"])
