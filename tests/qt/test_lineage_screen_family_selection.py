"""The lineage screen and the shared selection: building the tree, picking families, answering others.

The tree makes "this cell and everything in it" one act. These tests pin the
keys that travel with it (the shared key other views route on, the
table-qualified id), the orphan list, what the linked selection receives, and
how a selection made elsewhere is revealed without echoing back.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND, linked_selection,
                                       register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens import lineage as screen_module
from spacr.selection import Selection, as_key_index

pytestmark = pytest.mark.qt

F1 = ("plate1", "r1", "c1", "f1")
CELL = "plate1_r1_c1_f1_cell7"
FAMILY = [CELL, "plate1_r1_c1_f1_nucleus1", "plate1_r1_c1_f1_pathogen1",
          "plate1_r1_c1_f1_pathogen2"]


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
    linked_selection().clear_selection()
    view = screen_module.LineageScreen(threaded=False)
    qtbot.addWidget(view)
    yield view
    view.unlink_selection()
    linked_selection().clear_selection()


@pytest.fixture()
def opener():
    """Somewhere for crops to open, recording the keys it was asked for."""
    seen = []

    def open_it(request):
        seen.append(list(request.keys))
        return list(request.keys)

    register_object_opener(DEFAULT_OPEN_KIND, open_it)
    yield seen
    unregister_object_opener(DEFAULT_OPEN_KIND, open_it)


def _root(screen):
    return screen.tree.topLevelItem(0)


def _orphans(screen):
    return [(screen.orphan_list.item(i).text(),
             screen.orphan_list.item(i).data(screen_module._KEY_ROLE))
            for i in range(screen.orphan_list.count())]


def test_lineage_tree_build_names_parents_children_and_orphans(
        subtests, screen, frames):
    """Rows, orphan texts and keys, and the status line for each kind of frame set."""
    with subtests.test(msg="a cell, its contents and two orphans"):
        screen.set_frames(frames)
        root = _root(screen)
        assert screen.tree.topLevelItemCount() == 1
        assert root.text(0) == "cell 7"
        assert [root.child(i).text(0) for i in range(root.childCount())] == [
            "nucleus 1", "pathogen 1", "pathogen 2"]
        assert root.data(0, screen_module._KEY_ROLE) == CELL
        assert _orphans(screen) == [
            ("nucleus 4 → cell 99 (missing)", "plate1_r1_c1_f1_nucleus4"),
            ("pathogen 5 → no cell_id at all", "plate1_r1_c1_f1_pathogen5")]
        assert screen.status.text() == (
            "1 of 1 parent object(s) in the tree · 2 unattached child(ren)")
        assert screen.summary.text() == (
            "1 cell(s) holding 1 nucleus, 2 pathogen. "
            "0 of them (0%) have nothing inside.")

    with subtests.test(msg="everything attaches: the orphan list says so"):
        screen.set_frames({"cell": frames["cell"],
                           "nucleus": _rows("nucleus", [(F1, 1, 7)])})
        assert [text for text, _ in _orphans(screen)] == [
            "(none — every child names a parent that exists)"]
        assert screen.status.text() == (
            "1 of 1 parent object(s) in the tree · every child has a parent")

    with subtests.test(msg="no cell table: a sentence, not a stale tree"):
        screen.set_frames(frames)
        screen.set_frames({"nucleus": frames["nucleus"]})
        assert screen.tree.topLevelItemCount() == 0
        assert screen.orphan_list.count() == 0
        assert screen.summary.text().startswith(
            "no 'cell' table to build a lineage from; got ['nucleus'].")
        assert screen.status.text() == "Nothing to show."
        assert screen.family_keys() == []


def test_lineage_selection_publishes_the_pick_or_the_whole_family(
        subtests, screen, frames):
    """One row publishes itself; publish_family adds contents once, parents first."""
    screen.set_frames(frames)
    link = screen.link

    with subtests.test(msg="nothing selected publishes nothing"):
        assert screen.selected_keys() == []
        assert screen.family_keys() == []
        assert screen.publish_family() == []
        assert link.selection.keys is None

    with subtests.test(msg="picking a cell rings that cell only"):
        published = []
        screen.node_selected.connect(published.append)
        _root(screen).setSelected(True)
        assert screen.selected_keys() == [CELL]
        assert published == [CELL]
        assert list(link.selection.keys) == [CELL]

    with subtests.test(msg="clearing the tree selection publishes nothing"):
        screen.tree.clearSelection()
        assert list(link.selection.keys) == [CELL]
        assert screen.selected_keys() == []

    with subtests.test(msg="with contents: the whole family, in tree order"):
        _root(screen).setSelected(True)
        assert screen.publish_family() == FAMILY
        assert list(link.selection.keys) == FAMILY
        assert screen.family_ids() == [
            "cell:" + CELL, "nucleus:plate1_r1_c1_f1_nucleus1",
            "pathogen:plate1_r1_c1_f1_pathogen1",
            "pathogen:plate1_r1_c1_f1_pathogen2"]

    with subtests.test(msg="a family selected twice over is listed once"):
        _root(screen).child(1).setSelected(True)
        assert screen.family_keys() == FAMILY


def test_lineage_open_selected_opens_the_family_or_asks(subtests, screen,
                                                        frames, opener):
    """Opening needs a selection, and opens the selected cell's family."""
    screen.set_frames(frames)
    with subtests.test(msg="nothing selected asks for a selection"):
        assert screen.open_selected() is None
        assert "Select something in the tree first." in screen.status.text()
        assert opener == []

    with subtests.test(msg="a selected cell opens its family"):
        _root(screen).setSelected(True)
        assert screen.open_selected() == FAMILY
        assert opener == [FAMILY]


def test_lineage_answers_a_linked_selection_without_echoing(subtests, screen,
                                                            frames):
    """Selections from other views are revealed, never re-published, and cleared by a rest."""
    screen.set_frames(frames)
    published = []
    screen.node_selected.connect(published.append)

    with subtests.test(msg="a collapsed pathogen is selected and revealed"):
        _root(screen).setExpanded(False)
        screen.on_linked_selection_changed(
            Selection(keys=as_key_index(["plate1_r1_c1_f1_pathogen2"]),
                      source="plate"))
        selected = screen.tree.selectedItems()
        assert [item.data(0, screen_module._KEY_ROLE)
                for item in selected] == ["plate1_r1_c1_f1_pathogen2"]
        assert selected[0].parent().isExpanded() is True
        assert screen.status.text() == (
            "1 of the 1 selected object(s) are in this tree.")

    with subtests.test(msg="answering another view does not echo back"):
        screen.on_linked_selection_changed(
            Selection(keys=as_key_index([CELL]), source="plate"))
        assert [item.data(0, screen_module._KEY_ROLE)
                for item in screen.tree.selectedItems()] == [CELL]
        assert published == []

    with subtests.test(msg="objects this tree does not hold leave the status"):
        screen.status.setText("untouched")
        screen.on_linked_selection_changed(
            Selection(keys=as_key_index(["plate9_r9_c9_f9_cell1"]),
                      source="umap"))
        assert screen.tree.selectedItems() == []
        assert screen.status.text() == "untouched"

    with subtests.test(msg="a resting selection clears the highlight"):
        _root(screen).setSelected(True)
        published.clear()
        screen.on_linked_selection_changed(Selection.none())
        assert screen.tree.selectedItems() == []
        assert published == []
