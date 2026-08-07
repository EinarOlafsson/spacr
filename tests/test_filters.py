"""The ``filters`` table -- gates written back to the database as columns.

Qt-free by design: this is pipeline code, so it is tested without a GUI and
stays importable on a cluster.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.filters import (
    FILTERS_TABLE, FilterError, build_filters_frame, choose_anchor,
    column_name_for, ensure_filters_table, export_gate, gate_mask_over_table,
    identity_columns_of, object_tables, read_sampled, sampling_clause,
)


# ---------------------------------------------------------------------------
# Databases to test against
# ---------------------------------------------------------------------------

def _object_frame(n=6, start=1):
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(start, start + n)),
        "area": np.linspace(10.0, 60.0, n),
        "intensity": np.linspace(100.0, 600.0, n),
    })


def _make_db(tmp_path, tables, png=None, name="measurements.db"):
    path = str(tmp_path / name)
    with sqlite3.connect(path) as db:
        for table, frame in tables.items():
            frame.to_sql(table, db, index=False)
        if png is not None:
            png.to_sql("png_list", db, index=False)
    return path


@pytest.fixture
def cell_db(tmp_path):
    return _make_db(tmp_path, {"cell": _object_frame()})


# ---------------------------------------------------------------------------
# Step 1: which tables are in the database
# ---------------------------------------------------------------------------

def test_only_object_tables_that_exist_are_used(tmp_path):
    path = _make_db(tmp_path, {"cell": _object_frame(),
                               "nucleus": _object_frame()})
    assert object_tables(path) == ("cell", "nucleus")


def test_a_table_without_an_object_label_is_not_usable(tmp_path):
    """Present is not the same as usable. Discovering this at merge time
    instead would give a filter that quietly matches nothing."""
    bad = _object_frame().drop(columns=["object_label"])
    path = _make_db(tmp_path, {"cell": _object_frame(), "settings": bad})
    assert "settings" not in object_tables(path)


@pytest.mark.parametrize("only", ["cell", "nucleus", "pathogen", "organelle"])
def test_any_single_object_table_can_anchor(tmp_path, only):
    """"this mechanism should work when there are only cell, only nucleus,
    only pathogen and only organelle" -- no table has to be present."""
    path = _make_db(tmp_path, {only: _object_frame()}, name=f"{only}.db")
    assert object_tables(path) == (only,)
    assert choose_anchor(object_tables(path)) == only
    frame = build_filters_frame(path)
    assert len(frame) == 6
    assert f"in_{only}" in frame.columns


def test_cell_is_preferred_when_several_exist(tmp_path):
    path = _make_db(tmp_path, {"pathogen": _object_frame(),
                               "cell": _object_frame(),
                               "nucleus": _object_frame()})
    assert choose_anchor(object_tables(path)) == "cell"


def test_a_database_with_no_object_table_says_so(tmp_path):
    path = _make_db(tmp_path, {"settings": pd.DataFrame({"k": ["v"]})})
    with pytest.raises(FilterError, match="no object table"):
        build_filters_frame(path)


# ---------------------------------------------------------------------------
# Step 2 and 3: the identity a filter merges on
# ---------------------------------------------------------------------------

def test_the_filters_table_carries_the_metadata_and_the_object(cell_db):
    frame = build_filters_frame(cell_db)
    for column in ("plateID", "rowID", "columnID", "fieldID", "object_label"):
        assert column in frame.columns
    assert "area" not in frame.columns, (
        "identity only -- a measurement table is hundreds of columns wide")


def test_every_object_table_contributes_its_objects(tmp_path):
    """A gate drawn on nucleus measurements has to merge onto nuclei, so the
    filters table cannot be built from the anchor alone."""
    path = _make_db(tmp_path, {"cell": _object_frame(n=3, start=1),
                               "nucleus": _object_frame(n=3, start=10)})
    frame = build_filters_frame(path)
    labels = set(frame["object_label"])
    assert {1, 2, 3} <= labels, "cell objects are missing"
    assert {10, 11, 12} <= labels, "nucleus objects are missing"


def test_which_table_each_object_came_from_is_recorded(tmp_path):
    path = _make_db(tmp_path, {"cell": _object_frame(n=3, start=1),
                               "nucleus": _object_frame(n=3, start=10)})
    frame = build_filters_frame(path).set_index("object_label")
    assert frame.loc[1, "in_cell"] == 1
    assert frame.loc[1, "in_nucleus"] == 0
    assert frame.loc[10, "in_nucleus"] == 1
    assert frame.loc[10, "in_cell"] == 0


def test_crop_paths_come_across_when_png_list_exists(tmp_path):
    png = pd.DataFrame({
        "plateID": ["p1"] * 3,
        "rowID": ["A"] * 3,
        "columnID": ["1"] * 3,
        "fieldID": ["f1"] * 3,
        "cell_id": ["o1", "o2", "o3"],
        "png_path": ["/crops/1.png", "/crops/2.png", "/crops/3.png"],
    })
    path = _make_db(tmp_path, {"cell": _object_frame()}, png=png)
    frame = build_filters_frame(path).set_index("object_label")
    assert frame.loc[1, "png_path"] == "/crops/1.png"
    assert pd.isna(frame.loc[6, "png_path"]), "an uncropped object invented one"


def test_unparseable_png_ids_do_not_sink_the_build(tmp_path):
    """'omulti', 'onone', 'error' and NULL are states real crops are in."""
    png = pd.DataFrame({
        "plateID": ["p1"] * 4,
        "rowID": ["A"] * 4,
        "columnID": ["1"] * 4,
        "fieldID": ["f1"] * 4,
        "cell_id": ["o1", "omulti", "error", None],
        "png_path": ["/crops/1.png", "/m.png", "/e.png", "/n.png"],
    })
    path = _make_db(tmp_path, {"cell": _object_frame()}, png=png)
    frame = build_filters_frame(path).set_index("object_label")
    assert frame.loc[1, "png_path"] == "/crops/1.png"


def test_older_column_spellings_still_merge(tmp_path):
    """A filter that failed to merge because a column is called `row` rather
    than `rowID` would look exactly like a gate that selected nothing."""
    old = _object_frame().rename(columns={
        "plateID": "plate", "rowID": "row",
        "columnID": "column", "fieldID": "field"})
    path = _make_db(tmp_path, {"cell": old})
    found = identity_columns_of(path, "cell")
    assert found["rowID"] == "row" and found["plateID"] == "plate"
    assert len(build_filters_frame(path)) == 6


def test_a_timelapse_keeps_its_timepoint(tmp_path):
    """Without it the same object label recurs every frame and the join is
    many-to-many -- the bug already documented in io._read_and_join_tables."""
    frame = _object_frame(n=4)
    frame["timeID"] = [0, 0, 1, 1]
    frame["object_label"] = [1, 2, 1, 2]
    path = _make_db(tmp_path, {"cell": frame})
    built = build_filters_frame(path)
    assert "timeID" in built.columns
    assert len(built) == 4, "two frames of two objects collapsed into two rows"


# ---------------------------------------------------------------------------
# Writing a gate
# ---------------------------------------------------------------------------

def _gates():
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    return GateSet().add(RectGate(name="big cells", x_column="area",
                                  y_column="intensity",
                                  x_low=30.0, x_high=100.0,
                                  y_low=0.0, y_high=1e9))


def test_a_gate_becomes_a_one_zero_column_named_after_it(cell_db):
    frame = pd.read_sql_query("SELECT * FROM cell",
                              sqlite3.connect(cell_db))
    inside = frame["area"] >= 30.0
    column, marked = export_gate(cell_db, frame, inside.to_numpy(), "big cells")

    assert column == "big_cells"
    written = pd.read_sql_query(f"SELECT * FROM {FILTERS_TABLE}",
                                sqlite3.connect(cell_db))
    assert set(written[column]) == {0, 1}
    assert written[column].sum() == marked == int(inside.sum())


def test_objects_outside_the_gate_get_zero_not_null(cell_db):
    """Null is not what "outside the gate" means, and a user who gated on a
    20% sample would otherwise get a column that is null four times in five."""
    frame = pd.read_sql_query("SELECT * FROM cell", sqlite3.connect(cell_db))
    sample = frame.iloc[:3]
    export_gate(cell_db, sample, np.array([True, False, False]), "g")

    written = pd.read_sql_query(f"SELECT * FROM {FILTERS_TABLE}",
                                sqlite3.connect(cell_db))
    assert written["g"].isna().sum() == 0
    assert written["g"].sum() == 1
    assert len(written) == 6, "unsampled objects fell out of the table"


def test_re_exporting_a_gate_replaces_it(cell_db):
    frame = pd.read_sql_query("SELECT * FROM cell", sqlite3.connect(cell_db))
    export_gate(cell_db, frame, (frame["area"] >= 30.0).to_numpy(), "g")
    export_gate(cell_db, frame, (frame["area"] >= 50.0).to_numpy(), "g")

    written = pd.read_sql_query(f"SELECT * FROM {FILTERS_TABLE}",
                                sqlite3.connect(cell_db))
    assert list(written.columns).count("g") == 1, "a second g column appeared"
    assert written["g"].sum() == int((frame["area"] >= 50.0).sum())


def test_two_gates_are_two_columns(cell_db):
    frame = pd.read_sql_query("SELECT * FROM cell", sqlite3.connect(cell_db))
    export_gate(cell_db, frame, (frame["area"] >= 30.0).to_numpy(), "big")
    export_gate(cell_db, frame, (frame["intensity"] >= 400).to_numpy(), "bright")
    written = pd.read_sql_query(f"SELECT * FROM {FILTERS_TABLE}",
                                sqlite3.connect(cell_db))
    assert {"big", "bright"} <= set(written.columns)


def test_a_gate_drawn_on_a_sample_is_applied_to_every_object(cell_db):
    """The whole point of the sampling setting: gate on a fraction, export
    over everything."""
    gates = _gates()
    frame, mask = gate_mask_over_table(cell_db, "cell", gates, "big cells")
    assert len(frame) == 6, "the export read a sample instead of the table"
    column, marked = export_gate(cell_db, frame, mask, "big cells")
    assert marked == int(mask.sum()) > 0


def test_exporting_a_gate_on_a_column_the_table_lacks_says_which(cell_db):
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    gates = GateSet().add(RectGate(name="g", x_column="area", y_column="ghost",
                                   x_low=0.0, x_high=1.0,
                                   y_low=0.0, y_high=1.0))
    with pytest.raises(FilterError, match="ghost"):
        gate_mask_over_table(cell_db, "cell", gates, "g")


def test_a_mask_that_does_not_match_the_frame_is_refused(cell_db):
    frame = pd.read_sql_query("SELECT * FROM cell", sqlite3.connect(cell_db))
    with pytest.raises(FilterError, match="mask has"):
        export_gate(cell_db, frame, np.array([True, False]), "g")


def test_a_table_with_no_object_identity_cannot_be_exported(tmp_path):
    path = _make_db(tmp_path, {"cell": _object_frame()})
    with pytest.raises(FilterError, match="cannot be written back"):
        export_gate(path, pd.DataFrame({"area": [1.0]}), np.array([True]), "g")


def test_the_table_is_built_once_and_reused(cell_db):
    first = ensure_filters_table(cell_db)
    frame = pd.read_sql_query("SELECT * FROM cell", sqlite3.connect(cell_db))
    export_gate(cell_db, frame, (frame["area"] >= 30.0).to_numpy(), "g")
    second = ensure_filters_table(cell_db)
    assert "g" in second.columns, "the rebuild discarded an exported gate"
    assert len(second) == len(first)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given, expected", [
    ("big cells", "big_cells"),
    ("CD8+ / high", "CD8_high"),
    ("live", "live"),
    ("2n", "g_2n"),
])
def test_gate_names_become_usable_column_names(given, expected):
    assert column_name_for(given) == expected


def test_a_name_with_nothing_usable_in_it_is_refused():
    with pytest.raises(FilterError, match="no letters or digits"):
        column_name_for("///")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_a_full_sample_adds_no_clause():
    assert sampling_clause(1.0) == ""


def test_sampling_is_reproducible_and_roughly_the_right_size(tmp_path):
    """The same rows every time: a gate drawn on Monday has to sit on the
    same cloud on Tuesday."""
    big = pd.DataFrame({"area": np.arange(1000.0), "object_label": range(1000)})
    path = _make_db(tmp_path, {"cell": big})

    first = read_sampled(path, "cell", fraction=0.1)
    second = read_sampled(path, "cell", fraction=0.1)
    assert 50 <= len(first) <= 200, f"0.1 of 1000 gave {len(first)} rows"
    assert first.equals(second), "the same fraction gave a different sample"
    assert len(read_sampled(path, "cell", fraction=1.0)) == 1000


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_a_fraction_outside_zero_to_one_is_refused(bad):
    with pytest.raises(FilterError, match="fraction"):
        sampling_clause(bad)


def test_sampling_is_not_defeated_by_a_column_called_rowID(tmp_path):
    """Every spaCR table has a `rowID` column -- the row of the plate.

    SQLite matches column names case-insensitively, so `rowid` in a query
    means THAT column, which holds 'A'..'P'. `'A' % 5` is 0 in SQLite, so the
    obvious sampling clause is true for every row and samples nothing. The
    symptom is a sampling setting that appears to do nothing.
    """
    frame = pd.DataFrame({
        "plateID": ["p1"] * 500,
        "rowID": ["A"] * 500,
        "columnID": ["1"] * 500,
        "fieldID": ["f1"] * 500,
        "object_label": range(1, 501),
        "area": np.arange(500.0),
    })
    path = _make_db(tmp_path, {"cell": frame})
    sampled = read_sampled(path, "cell", fraction=0.2)
    assert len(sampled) < 200, (
        f"sampling returned {len(sampled)} of 500 rows -- `rowid` resolved to "
        f"the rowID column")
    assert len(sampled) > 50


def test_a_table_shadowing_every_rowid_alias_still_samples(tmp_path):
    """Slower -- read whole, sampled after -- but never silently everything."""
    frame = pd.DataFrame({
        "rowid": range(200), "oid": range(200), "_rowid_": range(200),
        "object_label": range(200), "area": np.arange(200.0),
    })
    path = _make_db(tmp_path, {"cell": frame})
    assert len(read_sampled(path, "cell", fraction=0.25)) == 50


def test_the_rowid_alias_is_chosen_per_table():
    from spacr.filters import rowid_expression

    assert rowid_expression(["area", "object_label"]) == "_rowid_"
    assert rowid_expression(["rowID", "area"]) == "_rowid_"
    assert rowid_expression(["_rowid_", "area"]) == "rowid"
    assert rowid_expression(["_rowid_", "rowid", "oid"]) is None
