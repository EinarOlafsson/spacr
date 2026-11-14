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


from spacr.qt.linked_selection import (DEFAULT_OPEN_KIND,
                                       register_object_opener,
                                       unregister_object_opener)
from spacr.qt.screens import lineage as screen_module

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


# ---------------------------------------------------------------------------
# The orphan list
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Opening from the tree
# ---------------------------------------------------------------------------


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
