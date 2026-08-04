"""``V9`` ``B20`` — the lineage tree as a widget.

:mod:`tests.test_lineage` covers the tree with no Qt at all. What is left here
is what needs a widget: that the rows on screen are the parent links and not a
flattening of them, that selecting a node publishes the object it names rather
than the family around it, and that "select with contents" is the other act
and is a button rather than a surprise.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.qt.linked_selection import (register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens import lineage as lgs
from spacr.selection import Selection

F1 = ("plate1", "r1", "c1", "f1")
F2 = ("plate1", "r1", "c1", "f2")


def _rows(table, entries):
    out = []
    for (plate, row, column, field_id), label, parent in entries:
        record = {"plateID": plate, "rowID": row, "columnID": column,
                  "fieldID": field_id, "object_label": label,
                  f"{table}_area": 100.0 + label}
        if parent is not None:
            record["cell_id"] = parent
        out.append(record)
    return pd.DataFrame(out)


@pytest.fixture
def frames():
    """Field 1: cell 7 holds a nucleus and two pathogens; cell 8 holds
    nothing. Field 2: cell 7 holds one nucleus. Plus one loose nucleus."""
    return {
        "cell": _rows("cell", [(F1, 7, None), (F1, 8, None), (F2, 7, None)]),
        "nucleus": _rows("nucleus", [(F1, 1, 7), (F2, 5, 7), (F1, 9, 99)]),
        "pathogen": _rows("pathogen", [(F1, 1, 7), (F1, 2, 7)]),
    }


@pytest.fixture
def screen(qtbot, qt_theme_applied, frames):
    widget = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frames(frames)
    return widget


@pytest.fixture
def opener():
    received = []
    register_object_opener("annotate", received.append)
    try:
        yield received
    finally:
        unregister_object_opener("annotate", received.append)


def _find(screen, key):
    """The tree item carrying ``key``, anywhere in the tree."""
    stack = [screen.tree.topLevelItem(i)
             for i in range(screen.tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        if str(item.data(0, Qt.UserRole)) == key:
            return item
        stack.extend(item.child(i) for i in range(item.childCount()))
    raise AssertionError(f"{key} is not in the tree")


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------

def test_the_tree_has_one_top_level_row_per_parent_object(screen):
    assert screen.tree.topLevelItemCount() == 3
    assert [screen.tree.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(3)] == [
        "plate1_r1_c1_f1_7", "plate1_r1_c1_f1_8", "plate1_r1_c1_f2_7"]


def test_children_hang_off_the_cell_in_their_own_field(screen):
    assert _find(screen, "plate1_r1_c1_f1_7").childCount() == 3
    assert _find(screen, "plate1_r1_c1_f2_7").childCount() == 1
    assert _find(screen, "plate1_r1_c1_f1_8").childCount() == 0


def test_a_row_says_what_is_inside_it(screen):
    assert _find(screen, "plate1_r1_c1_f1_7").text(1) == "1 nucleus, 2 pathogen"
    assert _find(screen, "plate1_r1_c1_f1_8").text(1) == ""


def test_the_summary_says_how_many_parents_hold_nothing(screen):
    assert "3 cell(s) holding" in screen.summary.text()
    assert "1 of them (33%) have nothing inside" in screen.summary.text()


def test_a_loose_child_gets_its_own_list_rather_than_being_dropped(screen):
    assert screen.orphan_list.count() == 1
    item = screen.orphan_list.item(0)
    assert item.data(Qt.UserRole) == "plate1_r1_c1_f1_9"
    assert "cell 99 (missing)" in item.text()
    assert "1 unattached child" in screen.status.text()


def test_a_healthy_database_says_every_child_has_a_parent(qtbot,
                                                          qt_theme_applied,
                                                          frames):
    frames["nucleus"] = _rows("nucleus", [(F1, 1, 7)])
    widget = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frames(frames)
    assert "every child has a parent" in widget.status.text()
    assert widget.orphan_list.count() == 1        # the "(none …)" placeholder
    assert "none" in widget.orphan_list.item(0).text()


def test_tables_that_cannot_be_assembled_say_why_rather_than_drawing_nothing(
        qtbot, qt_theme_applied):
    widget = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frames({"nucleus": _rows("nucleus", [(F1, 1, 7)])})
    assert widget.tree.topLevelItemCount() == 0
    assert "no 'cell' table" in widget.summary.text()


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def test_selecting_a_node_publishes_exactly_that_object(screen):
    _find(screen, "plate1_r1_c1_f1_7").setSelected(True)
    selection = screen.link.selection
    assert list(selection.keys) == ["plate1_r1_c1_f1_7"]
    assert selection.source == "lineage"


def test_select_with_contents_publishes_the_whole_family_parent_first(screen):
    _find(screen, "plate1_r1_c1_f1_7").setSelected(True)
    keys = screen.publish_family()
    assert keys[0] == "plate1_r1_c1_f1_7"
    # Four objects, THREE keys: nucleus 1 and pathogen 1 share one, because
    # the shared key is field plus label with no table in it.
    assert len(screen.family_ids()) == 4
    assert keys == ["plate1_r1_c1_f1_7", "plate1_r1_c1_f1_1",
                    "plate1_r1_c1_f1_2"]
    assert list(screen.link.selection.keys) == keys


def test_a_family_selection_does_not_open_the_same_object_twice(screen):
    _find(screen, "plate1_r1_c1_f1_7").setSelected(True)
    _find(screen, "plate1_r1_c1_f1_2").setSelected(True)   # a pathogen inside
    keys = screen.family_keys()
    assert len(keys) == len(set(keys)) == 3
    assert len(screen.family_ids()) == len(set(screen.family_ids())) == 4


def test_the_screen_says_which_objects_the_shared_key_cannot_tell_apart(
        screen):
    note = screen.collision_note()
    assert "plate1_r1_c1_f1_1 is a nucleus and a pathogen" in note
    assert note in screen.summary.text()


def test_no_note_when_every_object_has_a_key_to_itself(qtbot,
                                                       qt_theme_applied,
                                                       frames):
    frames["pathogen"] = _rows("pathogen", [(F1, 20, 7), (F1, 21, 7)])
    widget = lgs.LineageScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frames(frames)
    assert widget.collision_note() == ""
    assert "name more than one object" not in widget.summary.text()


def test_a_selection_made_elsewhere_selects_and_reveals_the_matching_rows(
        screen):
    screen.link.set_selection(
        Selection.from_keys(["plate1_r1_c1_f1_2"], source="umap"))
    item = _find(screen, "plate1_r1_c1_f1_2")
    assert item.isSelected()
    assert item.parent().isExpanded()
    assert "1 of the 1 selected object(s)" in screen.status.text()


def test_the_resting_selection_clears_the_tree_selection(screen):
    screen.link.set_selection(
        Selection.from_keys(["plate1_r1_c1_f1_7"], source="umap"))
    screen.link.clear_selection()
    assert not screen.tree.selectedItems()


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

def test_double_clicking_a_node_opens_just_that_object(screen, opener):
    screen._on_tree_activated(_find(screen, "plate1_r1_c1_f1_1"), 0)
    assert [list(r.keys) for r in opener] == [["plate1_r1_c1_f1_1"]]
    assert opener[0].source == "lineage"


def test_opening_a_selection_opens_the_family_parents_first(screen, opener):
    _find(screen, "plate1_r1_c1_f1_7").setSelected(True)
    screen.open_selected()
    assert list(opener[0].keys)[0] == "plate1_r1_c1_f1_7"
    assert len(opener[0].keys) == 3          # four objects, three shared keys
    assert "parents first" in opener[0].reason


def test_double_clicking_a_loose_child_opens_it_and_says_what_it_is(screen,
                                                                    opener):
    screen._on_orphan_activated(screen.orphan_list.item(0))
    assert list(opener[0].keys) == ["plate1_r1_c1_f1_9"]
    assert "names no cell" in opener[0].reason


def test_opening_with_nothing_selected_says_so_rather_than_opening_all(
        screen, opener):
    assert screen.open_selected() is None
    assert opener == []
    assert "Select something" in screen.status.text()


def test_with_nothing_registered_opening_explains_rather_than_raising(screen):
    _find(screen, "plate1_r1_c1_f1_7").setSelected(True)
    assert screen.open_selected() is None
    assert "Open the Annotate screen first" in screen.status.text()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_into_the_app_registry():
    from spacr.qt.app import APPS

    lgs.register()
    assert any(row[0] == lgs.APP_KEY for row in APPS)
    assert lgs.register() is None       # idempotent
