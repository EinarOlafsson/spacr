"""``V9`` ``B20`` — the containment tree, tested against hand-built links.

The tree is built from ``cell_id``, which is an object *label*, not a key.
Label 7 exists in every field of every plate, so the claim worth pinning
hardest is that a child is attached to the cell in ITS OWN field — a tree that
matched on the label alone would look entirely plausible and be wrong
everywhere.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import lineage as lin


def _rows(table, entries):
    """Build one object table. ``entries`` is ``(field, label, cell_id)``."""
    out = []
    for field, label, parent in entries:
        plate, row, column, field_id = field
        record = {
            "plateID": plate, "rowID": row, "columnID": column,
            "fieldID": field_id, "object_label": label,
            f"{table}_area": 100.0 + label,
        }
        if parent is not None:
            record["cell_id"] = parent
        out.append(record)
    return pd.DataFrame(out)


F1 = ("plate1", "r1", "c1", "f1")
F2 = ("plate1", "r1", "c1", "f2")


@pytest.fixture
def frames():
    """Two fields, each with a cell 7 — the trap the field key exists for.

    Field 1: cell 7 holds nucleus 1 and pathogens 1 and 2; cell 8 holds
    nothing. Field 2: cell 7 holds nucleus 5 only. Every count below is
    countable off this by eye.
    """
    return {
        "cell": _rows("cell", [(F1, 7, None), (F1, 8, None), (F2, 7, None)]),
        "nucleus": _rows("nucleus", [(F1, 1, 7), (F2, 5, 7)]),
        "pathogen": _rows("pathogen", [(F1, 1, 7), (F1, 2, 7)]),
    }


# ---------------------------------------------------------------------------
# The tree matches the hand-built parent links
# ---------------------------------------------------------------------------

def test_the_forest_matches_the_hand_built_parent_links(frames):
    forest = lin.build_forest(frames)
    flat = lin.lineage_frame(forest)

    assert list(zip(flat["key"], flat["parent_key"])) == [
        ("plate1_r1_c1_f1_7", ""),
        ("plate1_r1_c1_f1_1", "plate1_r1_c1_f1_7"),   # nucleus 1
        ("plate1_r1_c1_f1_1", "plate1_r1_c1_f1_7"),   # pathogen 1
        ("plate1_r1_c1_f1_2", "plate1_r1_c1_f1_7"),   # pathogen 2
        ("plate1_r1_c1_f1_8", ""),
        ("plate1_r1_c1_f2_7", ""),
        ("plate1_r1_c1_f2_5", "plate1_r1_c1_f2_7"),
    ]


def test_a_child_attaches_to_the_cell_in_its_own_field_not_every_field(frames):
    forest = lin.build_forest(frames)
    by_key = {node.key: node for node in forest}

    # Both fields have a cell 7. Field 1's holds three children, field 2's one.
    assert len(by_key["plate1_r1_c1_f1_7"].children) == 3
    assert len(by_key["plate1_r1_c1_f2_7"].children) == 1
    assert [child.table for child in by_key["plate1_r1_c1_f2_7"].children] \
        == ["nucleus"]


def test_a_childless_parent_is_kept_because_that_is_the_negative_control(
        frames):
    forest = lin.build_forest(frames)
    childless = [node.key for node in forest if not node.children]
    assert childless == ["plate1_r1_c1_f1_8"]


def test_children_are_grouped_by_table_then_ordered_by_label(frames):
    forest = lin.build_forest(frames)
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    assert [(c.table, c.label) for c in cell.children] == [
        ("nucleus", 1), ("pathogen", 1), ("pathogen", 2)]


def test_an_o_prefixed_parent_id_matches_the_integer_label(frames):
    # png_list stores 'o7'; the object tables store 7. Both must attach.
    frames["pathogen"]["cell_id"] = ["o7", "o7"]
    forest = lin.build_forest(frames)
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    assert len(cell.children) == 3


def test_a_sentinel_parent_id_attaches_to_nothing_rather_than_raising(frames):
    frames["pathogen"]["cell_id"] = ["onone", "omulti"]
    forest = lin.build_forest(frames)
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    assert [c.table for c in cell.children] == ["nucleus"]


def test_a_run_that_measured_no_pathogens_gives_a_forest_not_an_error(frames):
    forest = lin.build_forest({"cell": frames["cell"],
                               "nucleus": frames["nucleus"]})
    counts = {}
    for root in forest:
        for table, n in root.counts().items():
            counts[table] = counts.get(table, 0) + n
    assert counts == {"cell": 3, "nucleus": 2}


def test_no_root_table_is_an_error_because_a_tree_needs_roots(frames):
    with pytest.raises(lin.LineageError, match="no 'cell' table"):
        lin.build_forest({"nucleus": frames["nucleus"]})


def test_a_table_missing_its_identity_columns_says_which(frames):
    with pytest.raises(lin.LineageError, match="object_label"):
        lin.build_forest({"cell": pd.DataFrame({"plateID": ["p"]})})


# ---------------------------------------------------------------------------
# Walking a node
# ---------------------------------------------------------------------------

def test_a_nodes_keys_are_itself_first_then_its_contents(frames):
    cell = next(n for n in lin.build_forest(frames)
                if n.key == "plate1_r1_c1_f1_7")
    assert cell.keys()[0] == "plate1_r1_c1_f1_7"
    # Four objects but THREE keys: nucleus 1 and pathogen 1 collide, because
    # the shared key is field plus label with no table in it.
    assert cell.keys() == ("plate1_r1_c1_f1_7", "plate1_r1_c1_f1_1",
                           "plate1_r1_c1_f1_2")
    assert len(cell.node_ids()) == 4


def test_a_node_id_carries_the_table_so_it_can_address_one_object(frames):
    cell = next(n for n in lin.build_forest(frames)
                if n.key == "plate1_r1_c1_f1_7")
    assert cell.node_id == "cell:plate1_r1_c1_f1_7"
    assert set(cell.node_ids()) == {
        "cell:plate1_r1_c1_f1_7", "nucleus:plate1_r1_c1_f1_1",
        "pathogen:plate1_r1_c1_f1_1", "pathogen:plate1_r1_c1_f1_2"}


def test_the_objects_the_shared_key_cannot_tell_apart_are_reported(frames):
    forest = lin.build_forest(frames)
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    assert cell.key_collisions() == {
        "plate1_r1_c1_f1_1": ("nucleus", "pathogen")}
    assert lin.forest_key_collisions(forest) == cell.key_collisions()


def test_a_family_whose_labels_do_not_overlap_reports_no_collision(frames):
    frames["pathogen"] = _rows("pathogen", [(F1, 20, 7), (F1, 21, 7)])
    forest = lin.build_forest(frames)
    assert lin.forest_key_collisions(forest) == {}
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    assert len(cell.keys()) == len(cell.node_ids()) == 4


def test_a_node_counts_what_is_inside_it_including_itself(frames):
    cell = next(n for n in lin.build_forest(frames)
                if n.key == "plate1_r1_c1_f1_7")
    assert cell.counts() == {"cell": 1, "nucleus": 1, "pathogen": 2}


def test_a_node_describes_what_is_inside_it(frames):
    forest = lin.build_forest(frames)
    cell = next(n for n in forest if n.key == "plate1_r1_c1_f1_7")
    empty = next(n for n in forest if n.key == "plate1_r1_c1_f1_8")
    assert cell.describe() == "cell 7 · 1 nucleus, 2 pathogen"
    assert "nothing inside it" in empty.describe()


def test_asking_about_a_pathogen_returns_the_cell_around_it(frames):
    tree = lin.tree_for(frames, "plate1_r1_c1_f1_2")
    assert tree is not None
    assert tree.key == "plate1_r1_c1_f1_7"
    assert tree.find("plate1_r1_c1_f1_2").table == "pathogen"


def test_asking_about_something_that_is_not_there_returns_nothing(frames):
    assert lin.tree_for(frames, "plate9_r9_c9_f9_9") is None


# ---------------------------------------------------------------------------
# Orphans — a finding, not noise
# ---------------------------------------------------------------------------

def test_a_child_naming_a_cell_that_is_not_there_is_reported(frames):
    frames["nucleus"] = _rows("nucleus", [(F1, 1, 7), (F1, 2, 99)])
    loose = lin.orphans(frames)
    assert list(loose["object_label"]) == ["2"]
    assert list(loose["parent_id"]) == ["99"]
    assert list(loose["table"]) == ["nucleus"]


def test_a_child_with_no_parent_id_at_all_is_reported_too(frames):
    frames["nucleus"] = _rows("nucleus", [(F1, 1, 7), (F1, 2, None)])
    frames["nucleus"]["cell_id"] = [7, None]
    loose = lin.orphans(frames)
    assert list(loose["object_label"]) == ["2"]
    assert list(loose["parent_id"]) == [""]


def test_a_healthy_database_has_no_orphans(frames):
    assert lin.orphans(frames).empty


def test_an_orphan_is_left_out_of_the_tree_rather_than_attached_anywhere(
        frames):
    frames["nucleus"] = _rows("nucleus", [(F1, 2, 99)])
    forest = lin.build_forest(frames)
    assert all(not node.children or
               all(c.table != "nucleus" for c in node.children)
               for node in forest)


# ---------------------------------------------------------------------------
# The forest in words
# ---------------------------------------------------------------------------

def test_the_forest_says_how_many_parents_hold_nothing(frames):
    text = lin.describe_forest(lin.build_forest(frames))
    assert text.startswith("3 cell(s) holding 2 nucleus, 2 pathogen.")
    assert "1 of them (33%) have nothing inside" in text


def test_an_empty_forest_says_so(frames):
    assert "no lineage to show" in lin.describe_forest(())


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_the_object_tables_are_read_out_of_a_database(tmp_path, frames):
    db_path = tmp_path / "measurements.db"
    connection = sqlite3.connect(db_path)
    try:
        for table, frame in frames.items():
            frame.to_sql(table, connection, index=False)
    finally:
        connection.close()

    read = lin.read_object_tables(str(db_path))
    assert set(read) == {"cell", "nucleus", "pathogen"}
    assert len(lin.build_forest(read)) == 3


def test_a_table_the_run_never_wrote_is_simply_absent(tmp_path, frames):
    db_path = tmp_path / "measurements.db"
    connection = sqlite3.connect(db_path)
    try:
        frames["cell"].to_sql("cell", connection, index=False)
    finally:
        connection.close()
    assert set(lin.read_object_tables(str(db_path))) == {"cell"}


def test_a_missing_database_says_so(tmp_path):
    with pytest.raises(lin.LineageError, match="no measurements database"):
        lin.read_object_tables(str(tmp_path / "nope.db"))


def test_the_child_tables_come_from_the_schema_not_from_a_list_here():
    from spacr import schema

    assert set(lin.child_tables()) >= set(schema.CHILD_OBJECT_TABLES)
    assert "cell" not in lin.child_tables()
