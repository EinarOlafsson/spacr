"""A child table with no parent link, and the tree walk that skips its own root.

Two facts the containment tree depends on. First: an object table that is
present but carries no ``cell_id`` column contributes nothing -- it is not an
error and it is not a forest of orphans, it is a table that never claimed a
parent, and both :func:`build_forest` and :func:`orphans` have to pass over it
without inventing links. Second: ``descendants`` is what a viewer counts the
contents of a cell with, so it must exclude the cell itself.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import lineage as lin


def _table(entries, *, parent_column=None):
    """Rows for one object table. ``entries`` is ``(label, parent)``."""
    out = []
    for label, parent in entries:
        record = {"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                  "fieldID": "f1", "object_label": label}
        if parent_column is not None and parent is not None:
            record[parent_column] = parent
        out.append(record)
    return pd.DataFrame(out)


@pytest.fixture
def frames():
    """One cell holding one nucleus, plus an ``organelleb`` table with no link."""
    return {
        "cell": _table([(7, None), (8, None)]),
        "nucleus": _table([(1, 7)], parent_column="cell_id"),
        # Present, measured, and carrying no parent column of any spelling.
        "organelleb": _table([(1, None), (2, None)]),
    }


def test_a_child_table_with_no_parent_column_contributes_no_children(frames):
    """A table that never claimed a parent must not be attached by guesswork.

    ``organelleb`` here has field columns and labels and nothing else. Falling
    back to matching on the label alone would hang organelle 1 off cell 1 --
    or, worse, off every cell -- and the tree would look complete. The only
    honest answer is that the table adds nothing.
    """
    assert lin._parent_column("organelleb", frames["organelleb"]) is None

    forest = lin.build_forest(frames)
    tables = {node.table for root in forest for _d, node in root.walk()}
    assert tables == {"cell", "nucleus"}

    flat = lin.lineage_frame(forest)
    assert list(flat["key"]) == [
        "plate1_r1_c1_f1_cell7",
        "plate1_r1_c1_f1_nucleus1",
        "plate1_r1_c1_f1_cell8",
    ]


def test_a_child_table_with_no_parent_column_reports_no_orphans(frames):
    """Rows that never claimed a parent are not broken links.

    :func:`orphans` reports children whose parent link names nothing. A table
    with no link column at all has made no claim to break, so listing its rows
    would be a false alarm the size of the whole table -- and would send the
    user hunting for a segmentation disagreement that does not exist.
    """
    loose = lin.orphans(frames)
    assert loose.empty
    assert "organelleb" not in set(loose.get("table", []))

    # The detector is not merely returning empty: a real broken link shows up.
    frames["nucleus"] = _table([(1, 99)], parent_column="cell_id")
    broken = lin.orphans(frames)
    assert list(broken["table"]) == ["nucleus"]
    assert list(broken["parent_id"]) == ["99"]


def test_orphans_says_which_root_table_it_could_not_find():
    """Checking parents against a table that is not there is a caller error.

    Returning an empty frame would read as "nothing is orphaned", which is the
    opposite of the truth: nothing was checked. The message names the table
    asked for so the caller can see the typo.
    """
    with pytest.raises(lin.LineageError) as excinfo:
        lin.orphans({"nucleus": _table([(1, 7)], parent_column="cell_id")})
    assert "'cell'" in str(excinfo.value)


def test_descendants_excludes_the_node_it_was_asked_about(frames):
    """A cell's contents are what is inside it, not it and its contents.

    ``walk`` yields the node itself at depth 0 so a view can render the whole
    subtree; ``descendants`` is the count a "this cell holds N objects" label
    uses, and an off-by-one there is a number in the interface that is wrong
    for every cell in the screen.
    """
    forest = lin.build_forest(frames)
    by_key = {node.key: node for node in forest}
    cell7 = by_key["plate1_r1_c1_f1_cell7"]
    cell8 = by_key["plate1_r1_c1_f1_cell8"]

    assert [node.key for node in cell7.descendants()] == [
        "plate1_r1_c1_f1_nucleus1"]
    assert cell7 not in cell7.descendants()
    assert len(cell7.descendants()) == len(cell7.node_ids()) - 1
    assert cell8.descendants() == ()
