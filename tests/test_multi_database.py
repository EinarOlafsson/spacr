"""Merging several measurement databases without pooling them.

Instruction 109.

The test that matters most is
:func:`test_a_per_source_count_survives_the_merge`. Every other failure here
is loud; pooling two plates that share a name is silent, and every per-well
number computed afterwards describes an experiment that never happened.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.multi_database import (
    SOURCE_COLUMN, MergeRefused, describe_merge, read_merged,
)


def _database(path, plate, *, rows=4, extra=None, table="cell"):
    frame = pd.DataFrame({
        "plateID": [plate] * rows,
        "rowID": [f"r{i + 1}" for i in range(rows)],
        "columnID": ["c1"] * rows,
        "area": range(rows),
    })
    if extra:
        frame[extra] = 1.0
    with sqlite3.connect(str(path)) as db:
        frame.to_sql(table, db, index=False)
    return str(path)


@pytest.fixture
def two_plates(tmp_path):
    return (_database(tmp_path / "plateA.db", "plate1"),
            _database(tmp_path / "plateB.db", "plate2", extra="perimeter"))


@pytest.fixture
def colliding(tmp_path):
    """Two databases that each contain a plate called ``plate1``."""
    return (_database(tmp_path / "runA.db", "plate1"),
            _database(tmp_path / "runB.db", "plate1"))


# --------------------------------------------------------------------------- #
#  Describe before you merge
# --------------------------------------------------------------------------- #

def test_the_plan_reports_what_the_merge_would_cost(two_plates):
    """The column set a merge produces IS the analysis about to be run.
    Finding out afterwards is finding out too late."""
    plan = describe_merge(list(two_plates), "cell")

    assert plan.total_rows == 8
    assert set(plan.common_columns) == {"plateID", "rowID", "columnID", "area"}
    assert "perimeter" in plan.partial_columns
    assert plan.partial_columns["perimeter"] == ("plateB",)
    assert not plan.has_collisions


def test_the_plan_names_the_sources_readably(two_plates):
    plan = describe_merge(list(two_plates), "cell")
    assert [s.label for s in plan.sources] == ["plateA", "plateB"]


def test_databases_sharing_a_filename_still_get_distinct_labels(tmp_path):
    """Every plate's database is often called measurements.db under a
    differently-named folder, so the stem alone is not a name."""
    first = tmp_path / "plate_one"
    second = tmp_path / "plate_two"
    first.mkdir()
    second.mkdir()
    paths = [_database(first / "measurements.db", "p1"),
             _database(second / "measurements.db", "p2")]

    labels = [s.label for s in describe_merge(paths, "cell").sources]
    assert len(set(labels)) == 2, labels
    assert any("plate_two" in label for label in labels)


# --------------------------------------------------------------------------- #
#  The merge itself
# --------------------------------------------------------------------------- #

def test_every_row_remembers_where_it_came_from(two_plates):
    """A merged UMAP whose clusters are the three plates is the most valuable
    thing this feature can show, and it cannot show it without this."""
    merged = read_merged(list(two_plates), "cell")
    assert SOURCE_COLUMN in merged.columns
    assert sorted(merged[SOURCE_COLUMN].unique()) == ["plateA", "plateB"]


def test_common_drops_and_union_keeps(two_plates):
    common = read_merged(list(two_plates), "cell", columns="common")
    assert "perimeter" not in common.columns

    union = read_merged(list(two_plates), "cell", columns="union")
    assert "perimeter" in union.columns
    # Present for the source that had it, null for the one that did not.
    assert int(union["perimeter"].isna().sum()) == 4


def test_a_per_source_count_survives_the_merge(two_plates):
    """The anti-pooling test. Rows in must equal rows out, per source."""
    paths = list(two_plates)
    before = {}
    for path in paths:
        with sqlite3.connect(path) as db:
            before[path] = int(
                db.execute("SELECT COUNT(*) FROM cell").fetchone()[0])

    merged = read_merged(paths, "cell")
    after = merged.groupby(SOURCE_COLUMN).size().to_dict()

    assert sum(before.values()) == len(merged)
    assert sorted(after.values()) == sorted(before.values())


# --------------------------------------------------------------------------- #
#  Collisions -- the part that must never be silent
# --------------------------------------------------------------------------- #

def test_a_shared_plate_id_is_refused_by_default(colliding):
    plan = describe_merge(list(colliding), "cell")
    assert plan.has_collisions
    assert "plate1" in plan.colliding_plates

    with pytest.raises(MergeRefused) as caught:
        read_merged(list(colliding), "cell", plan=plan)
    # The message has to say what it found, or the user cannot act on it.
    assert "plate1" in str(caught.value)


def test_qualifying_makes_the_plates_distinct(colliding):
    merged = read_merged(list(colliding), "cell", on_collision="qualify")
    plates = sorted(merged["plateID"].unique())
    assert len(plates) == 2, plates
    assert all("plate1" in p for p in plates)
    # Still two separate experiments, with their counts intact.
    assert merged.groupby(SOURCE_COLUMN).size().to_dict() == {
        "runA": 4, "runB": 4}


def test_a_qualified_plate_is_still_a_legal_key_token(colliding):
    """The qualifier must not introduce the key separator, or it splits the
    key into an extra component -- the bug schema.KEY_ESCAPES exists for."""
    merged = read_merged(list(colliding), "cell", on_collision="qualify")
    for plate in merged["plateID"].unique():
        assert schema_separator_absent(plate)


def schema_separator_absent(plate) -> bool:
    from spacr import schema

    return schema.KEY_SEPARATOR not in str(plate)


def test_there_is_no_option_that_pools(colliding):
    """Refusing an unknown mode matters more than it looks: 'merge' or
    'ignore' would be the obvious thing for a caller to try."""
    for mode in ("merge", "ignore", "pool", ""):
        with pytest.raises(MergeRefused):
            read_merged(list(colliding), "cell", on_collision=mode)


def test_an_unknown_column_mode_is_refused(two_plates):
    with pytest.raises(MergeRefused):
        read_merged(list(two_plates), "cell", columns="everything")


# --------------------------------------------------------------------------- #
#  Edges
# --------------------------------------------------------------------------- #

def test_no_databases_is_an_empty_frame_not_a_crash():
    assert read_merged([], "cell").empty


def test_one_database_still_gains_provenance(tmp_path):
    """Merging one is not a special case -- it is the same frame plus the
    column that says where it came from, so a caller never branches."""
    path = _database(tmp_path / "only.db", "plate1")
    merged = read_merged([path], "cell")
    assert len(merged) == 4
    assert merged[SOURCE_COLUMN].unique().tolist() == ["only"]


def test_a_row_cap_applies_per_source(two_plates):
    merged = read_merged(list(two_plates), "cell", limit_per_source=2)
    assert len(merged) == 4
    assert merged.groupby(SOURCE_COLUMN).size().to_dict() == {
        "plateA": 2, "plateB": 2}


# --------------------------------------------------------------------------- #
#  The Gate Editor, which is one of the two screens instruction 109 names
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_the_gate_editor_merges_several_databases(qtbot, two_plates):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.load_paths(list(two_plates))

    assert screen._frame is not None
    assert len(screen._frame) == 8
    assert SOURCE_COLUMN in screen._frame.columns
    # The label has to say the merge happened, and name the colour-by column.
    assert "2 databases" in screen._source.text()
    assert SOURCE_COLUMN in screen._source.text()


@pytest.mark.qt
def test_the_gate_editor_refuses_a_collision_and_says_which(qtbot, colliding):
    """Refusal must leave the previous state alone AND name the plate.

    A screen that blanks itself on a refusal has punished the user for a
    question they are allowed to ask.
    """
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.load_paths(list(colliding))

    assert screen._frame is None
    assert "plate1" in screen._source.text()
