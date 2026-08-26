"""Picking a family in the lineage tree, and what the pick reaches.

The tree exists so that "this cell and everything in it" is one act. Every
assertion here is about the identity that travels with that act: the shared
key that other views route on, the table-qualified id that separates a
nucleus 1 from a pathogen 1, and the orphan list where a child whose parent
does not exist is the only place it is ever shown.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr import lineage as lin
from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND,
                                       register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens import lineage as screen_module
from spacr.selection import Selection, as_key_index

F1 = ("plate1", "r1", "c1", "f1")


def _rows(table, entries):
    out = []
    for field, label, parent in entries:
        plate, row, column, field_id = field
        record = {"plateID": plate, "rowID": row, "columnID": column,
                  "fieldID": field_id, "object_label": label,
                  f"{table}_area": 100.0 + label}
        if parent is not None:
            record["cell_id"] = parent
        out.append(record)
    return pd.DataFrame(out)


@pytest.fixture()
def frames():
    """One cell holding a nucleus and two pathogens, plus two loose children."""
    return {
        "cell": _rows("cell", [(F1, 7, None)]),
        "nucleus": _rows("nucleus", [(F1, 1, 7), (F1, 4, 99)]),
        "pathogen": _rows("pathogen", [(F1, 1, 7), (F1, 2, 7), (F1, 5, None)]),
    }


@pytest.fixture()
def screen(qtbot):
    from spacr.qt.linked_selection import linked_selection

    linked_selection().clear_selection()
    view = screen_module.LineageScreen(threaded=False)
    qtbot.addWidget(view)
    yield view
    view.unlink_selection()
    linked_selection().clear_selection()


@pytest.fixture()
def opener():
    """Somewhere for crops to open, and the keys it was asked for."""
    seen = []

    def open_it(request):
        seen.append(list(request.keys))
        return list(request.keys)

    register_object_opener(DEFAULT_OPEN_KIND, open_it)
    yield seen
    unregister_object_opener(DEFAULT_OPEN_KIND, open_it)


def _root(screen):
    return screen.tree.topLevelItem(0)


# ---------------------------------------------------------------------------
# A tree that cannot be built
# ---------------------------------------------------------------------------

def test_frames_with_no_cell_table_leave_a_sentence_not_a_stale_tree(
        screen, frames):
    """The previous project's tree must not survive a failed rebuild.

    ``build_forest`` refuses without a root table. If the screen kept what it
    had, the user would be looking at another database's objects with a
    reassuring "1 parent object" underneath them.
    """
    screen.set_frames(frames)
    assert screen.tree.topLevelItemCount() == 1

    screen.set_frames({"nucleus": frames["nucleus"]})

    assert screen.tree.topLevelItemCount() == 0
    assert screen.orphan_list.count() == 0
    assert "cell" in screen.summary.text()
    assert screen.status.text() == "Nothing to show."
    assert screen.family_keys() == []


# ---------------------------------------------------------------------------
# The orphan list
# ---------------------------------------------------------------------------

def test_a_broken_link_and_a_missing_link_read_differently(screen, frames):
    """"Its cell is gone" and "it never named one" are two findings."""
    screen.set_frames(frames)

    texts = [screen.orphan_list.item(i).text()
             for i in range(screen.orphan_list.count())]
    assert any("nucleus 4 → cell 99 (missing)" in t for t in texts)
    assert any("pathogen 5 → no cell_id at all" in t for t in texts)
    assert "2 unattached child(ren)" in screen.status.text()


def test_an_orphan_carries_the_key_of_that_child_not_of_its_label(
        screen, frames):
    """The key names the table, so orphan nucleus 4 is not "object 4"."""
    screen.set_frames(frames)

    keys = [screen.orphan_list.item(i).data(screen_module._KEY_ROLE)
            for i in range(screen.orphan_list.count())]
    assert "plate1_r1_c1_f1_nucleus4" in keys
    assert "plate1_r1_c1_f1_pathogen5" in keys


def test_a_database_where_everything_attaches_says_so(screen, frames):
    """An empty orphan list is a statement, not a blank panel."""
    screen.set_frames({"cell": frames["cell"],
                       "nucleus": _rows("nucleus", [(F1, 1, 7)])})

    assert screen.orphan_list.count() == 1
    assert "every child names a parent" in screen.orphan_list.item(0).text()
    assert "every child has a parent" in screen.status.text()


def test_double_clicking_an_orphan_opens_that_child(screen, frames, opener):
    """The orphan list is not a read-only footnote; it opens crops too."""
    screen.set_frames(frames)
    item = next(screen.orphan_list.item(i)
                for i in range(screen.orphan_list.count())
                if "nucleus 4" in screen.orphan_list.item(i).text())

    screen._on_orphan_activated(item)

    assert opener == [["plate1_r1_c1_f1_nucleus4"]]


def test_double_clicking_the_orphan_placeholder_opens_nothing(screen, frames,
                                                              opener):
    """"(none)" is a label, and a label has no object behind it."""
    screen.set_frames({"cell": frames["cell"]})

    screen._on_orphan_activated(screen.orphan_list.item(0))

    assert opener == []


# ---------------------------------------------------------------------------
# Selecting one row, and selecting a family
# ---------------------------------------------------------------------------

def test_picking_a_cell_rings_that_cell_and_not_its_contents(screen, frames):
    """Expanding a selection behind the user leaves no way to ask for one."""
    screen.set_frames(frames)
    published = []
    screen.node_selected.connect(published.append)

    _root(screen).setSelected(True)

    assert screen.selected_keys() == ["plate1_r1_c1_f1_cell7"]
    assert published == ["plate1_r1_c1_f1_cell7"]
    assert list(screen.link.selection.keys) == ["plate1_r1_c1_f1_cell7"]


def test_clearing_the_tree_selection_publishes_nothing(screen, frames):
    """A deselect is not a selection of nothing — it must not ring anything."""
    screen.set_frames(frames)
    _root(screen).setSelected(True)
    before = list(screen.link.selection.keys)

    screen.tree.clearSelection()

    assert list(screen.link.selection.keys) == before


def test_selecting_with_contents_carries_the_whole_family(screen, frames):
    """Parents first, so the crops open in the order the tree reads."""
    screen.set_frames(frames)
    _root(screen).setSelected(True)

    keys = screen.publish_family()

    assert keys == ["plate1_r1_c1_f1_cell7",
                    "plate1_r1_c1_f1_nucleus1",
                    "plate1_r1_c1_f1_pathogen1",
                    "plate1_r1_c1_f1_pathogen2"]
    assert list(screen.link.selection.keys) == keys


def test_a_family_selected_twice_over_is_published_once(screen, frames):
    """Selecting a cell and one of its pathogens must not open it twice."""
    screen.set_frames(frames)
    root = _root(screen)
    root.setSelected(True)
    root.child(1).setSelected(True)

    keys = screen.family_keys()

    assert len(keys) == len(set(keys)) == 4


def test_publishing_a_family_with_nothing_selected_rings_nothing(screen,
                                                                 frames):
    screen.set_frames(frames)

    assert screen.publish_family() == []
    assert screen.link.selection.keys is None


def test_the_family_is_also_addressable_one_object_at_a_time(screen, frames):
    """The table-qualified id is what separates a nucleus 1 from a pathogen 1."""
    screen.set_frames(frames)
    _root(screen).setSelected(True)

    ids = screen.family_ids()

    assert len(ids) == 4
    assert len({i.rsplit(":", 1)[0] for i in ids}) >= 3
    assert any("nucleus" in i for i in ids)
    assert any("pathogen" in i for i in ids)


# ---------------------------------------------------------------------------
# Opening from the tree
# ---------------------------------------------------------------------------

def test_opening_with_nothing_selected_asks_for_a_selection(screen, frames,
                                                            opener):
    screen.set_frames(frames)

    assert screen.open_selected() is None
    assert "Select something in the tree first." in screen.status.text()
    assert opener == []


def test_opening_a_selected_cell_opens_its_family(screen, frames, opener):
    screen.set_frames(frames)
    _root(screen).setSelected(True)

    result = screen.open_selected()

    assert result == ["plate1_r1_c1_f1_cell7", "plate1_r1_c1_f1_nucleus1",
                      "plate1_r1_c1_f1_pathogen1", "plate1_r1_c1_f1_pathogen2"]
    assert opener == [result]


def test_double_clicking_a_child_opens_only_that_child(screen, frames, opener):
    """A double-click is a request for one object, not for its family."""
    screen.set_frames(frames)

    screen._on_tree_activated(_root(screen).child(0), 0)

    assert opener == [["plate1_r1_c1_f1_nucleus1"]]


def test_double_clicking_a_row_with_no_key_opens_nothing(screen, frames,
                                                         opener):
    from PySide6.QtWidgets import QTreeWidgetItem

    screen.set_frames(frames)
    screen._on_tree_activated(QTreeWidgetItem(["stray"]), 0)

    assert opener == []


# ---------------------------------------------------------------------------
# Answering another view's selection
# ---------------------------------------------------------------------------

def test_a_selection_made_elsewhere_is_revealed_here(screen, frames):
    """A pathogen picked on the plate view must be visible, not just selected.

    Its row is three levels down and collapsed; selecting it without opening
    the branch highlights something nobody can see.
    """
    screen.set_frames(frames)
    _root(screen).setExpanded(False)

    screen.on_linked_selection_changed(
        Selection(keys=as_key_index(["plate1_r1_c1_f1_pathogen2"]),
                  source="plate"))

    selected = screen.tree.selectedItems()
    assert [item.data(0, screen_module._KEY_ROLE) for item in selected] == [
        "plate1_r1_c1_f1_pathogen2"]
    assert selected[0].parent().isExpanded()
    assert "1 of the 1 selected object(s) are in this tree." in (
        screen.status.text())


def test_answering_another_view_does_not_echo_back_at_it(screen, frames):
    """Re-publishing what we were told would loop the two views together."""
    screen.set_frames(frames)
    published = []
    screen.node_selected.connect(published.append)

    screen.on_linked_selection_changed(
        Selection(keys=as_key_index(["plate1_r1_c1_f1_cell7"]),
                  source="plate"))

    assert published == []


def test_a_selection_of_objects_this_tree_does_not_hold_says_nothing(screen,
                                                                     frames):
    """Counting zero found objects would put a confusing tally on the line."""
    screen.set_frames(frames)
    screen.status.setText("untouched")

    screen.on_linked_selection_changed(
        Selection(keys=as_key_index(["plate9_r9_c9_f9_cell1"]), source="umap"))

    assert screen.tree.selectedItems() == []
    assert screen.status.text() == "untouched"


def test_a_resting_selection_clears_the_highlight(screen, frames):
    """"Nothing is selected anywhere" is not "select nothing here"."""
    screen.set_frames(frames)
    _root(screen).setSelected(True)

    screen.on_linked_selection_changed(Selection.none())

    assert screen.tree.selectedItems() == []
