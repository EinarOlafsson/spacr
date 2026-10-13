"""Build an annotation set without measuring the plate again.

Annotating means looking at single-object crops, and until now the only way to
get a set was to run Measure over a whole plate -- twenty minutes for 52 fields
on a 30-core machine, measured 2026-09-01. Every object is already described
twice once Measure has run: by the label masks in the merged arrays, and by the
coordinate columns in measurements.db. Either is enough to cut crops from.

Instruction 338.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.annotation_dataset import (PNG_LIST_COLUMNS, filter_selection,
                                      next_png_table, png_list_frame,
                                      write_png_list)


def _db(tmp_path, tables=()):
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    for name in tables:
        connection.execute(f'create table "{name}" (x)')
    connection.commit()
    connection.close()
    return path


# ---------------------------------------------------------------------------
# Naming: an existing set is never overwritten
# ---------------------------------------------------------------------------

def test_the_first_set_is_png_list(tmp_path):
    connection = sqlite3.connect(_db(tmp_path))
    assert next_png_table(connection) == "png_list"


def test_the_second_set_does_not_overwrite_the_first(tmp_path):
    """An existing set may already carry annotations, and those are hand-made
    and unrecoverable."""
    connection = sqlite3.connect(_db(tmp_path, ["png_list"]))
    assert next_png_table(connection) == "png_list_2"


def test_it_keeps_counting(tmp_path):
    connection = sqlite3.connect(
        _db(tmp_path, ["png_list", "png_list_2", "png_list_3"]))
    assert next_png_table(connection) == "png_list_4"


def test_a_gap_is_filled_rather_than_skipped(tmp_path):
    """png_list_2 deleted and png_list_3 kept: the free name is 2."""
    connection = sqlite3.connect(_db(tmp_path, ["png_list", "png_list_3"]))
    assert next_png_table(connection) == "png_list_2"


def test_unrelated_tables_do_not_take_a_name(tmp_path):
    connection = sqlite3.connect(
        _db(tmp_path, ["cell", "png_lists", "my_png_list"]))
    assert next_png_table(connection) == "png_list"


def test_a_view_takes_the_name_too(tmp_path):
    """SQLite will not let a table and a view share a name, so a view called
    png_list makes that name unusable just as a table does."""
    path = _db(tmp_path, ["cell"])
    connection = sqlite3.connect(path)
    connection.execute("create view png_list as select * from cell")
    connection.commit()
    assert next_png_table(connection) == "png_list_2"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _frame(n=3):
    selection = pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": ["c2"] * n,
        "fieldID": ["f7"] * n,
        "objectID": list(range(1, n + 1)),
    })
    paths = [f"/tmp/set/plate1_r1_c2_f7_{i}.png" for i in range(1, n + 1)]
    return png_list_frame(selection, paths)


def test_the_rows_match_measure_crops_own_schema():
    """Matching it exactly is what lets the annotation viewer open a streamed
    set without knowing how it was made."""
    assert list(_frame().columns) == list(PNG_LIST_COLUMNS)


def test_the_join_key_is_built():
    """`prcfo` is what every other measurement table joins on."""
    frame = _frame(1)
    assert frame.loc[0, "prcfo"] == "plate1_r1_c2_f7_o1"


def test_the_object_id_is_written_the_way_measure_writes_it():
    assert _frame(1).loc[0, "cell_id"] == "o1"


def test_a_new_set_starts_unannotated():
    assert _frame(2)["annotate"].isna().all()


def test_a_path_per_object_is_required():
    """One path short would name the wrong picture for every object after it,
    which nothing downstream could detect."""
    selection = pd.DataFrame({"plateID": ["p"] * 3, "objectID": [1, 2, 3]})
    with pytest.raises(ValueError, match="row for row"):
        png_list_frame(selection, ["only.png"])


def test_writing_creates_the_table_and_returns_its_name(tmp_path):
    path = _db(tmp_path)
    name = write_png_list(str(path), _frame(3))
    assert name == "png_list"
    connection = sqlite3.connect(path)
    assert connection.execute("select count(*) from png_list").fetchone()[0] == 3


def test_a_second_write_lands_beside_the_first(tmp_path):
    path = _db(tmp_path)
    first = write_png_list(str(path), _frame(2))
    second = write_png_list(str(path), _frame(3))
    assert (first, second) == ("png_list", "png_list_2")
    connection = sqlite3.connect(path)
    assert connection.execute("select count(*) from png_list").fetchone()[0] == 2
    assert connection.execute(
        "select count(*) from png_list_2").fetchone()[0] == 3


def test_a_failed_write_leaves_no_half_table(tmp_path):
    """Name and creation are one transaction, so a crash between them cannot
    leave an empty table holding a name."""
    path = _db(tmp_path)
    frame = _frame(2)
    frame["png_path"] = [object(), object()]      # unstorable
    with pytest.raises(Exception):
        write_png_list(str(path), frame)
    connection = sqlite3.connect(path)
    names = {r[0] for r in connection.execute(
        "select name from sqlite_master where type='table'")}
    assert "png_list" not in names


# ---------------------------------------------------------------------------
# Filtration -- the same population a measured set would describe
# ---------------------------------------------------------------------------

def _selection():
    return pd.DataFrame({
        "plateID": ["p"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c2", "c1", "c2"],
        "fieldID": ["f1"] * 4,
        "objectID": [1, 2, 3, 4],
        "area": [50, 500, 5000, 50000],
    })


def test_nothing_set_filters_nothing():
    assert len(filter_selection(_selection(), {})) == 4


def test_a_minimum_size_drops_the_small_ones():
    kept = filter_selection(_selection(), {"cell_min_size": 500})
    assert list(kept["objectID"]) == [2, 3, 4]


def test_a_maximum_size_drops_the_large_ones():
    kept = filter_selection(_selection(), {"cell_max_size": 5000})
    assert list(kept["objectID"]) == [1, 2, 3]


def test_zero_is_not_a_minimum():
    """The Measure panel writes 0 for an unset bound. Honouring it as a real
    minimum would filter nothing while looking filtered."""
    assert len(filter_selection(_selection(),
                                {"cell_min_size": 0, "cell_max_size": 0})) == 4


def test_wells_can_be_kept():
    kept = filter_selection(_selection(), {"wells": ["r1c1", "r2c2"]})
    assert list(kept["objectID"]) == [1, 4]


def test_wells_can_be_excluded():
    kept = filter_selection(_selection(), {"exclude_wells": ["r1c1"]})
    assert list(kept["objectID"]) == [2, 3, 4]


def test_a_cap_is_deterministic():
    """A set that differs between two runs of the same settings cannot be
    compared with anything."""
    once = filter_selection(_selection(), {"max_objects": 2})
    again = filter_selection(_selection(), {"max_objects": 2})
    assert list(once["objectID"]) == list(again["objectID"]) == [1, 2]


def test_the_input_is_not_modified():
    original = _selection()
    filter_selection(original, {"cell_min_size": 5000})
    assert len(original) == 4


def test_filtering_everything_out_is_allowed():
    assert len(filter_selection(_selection(), {"cell_min_size": 10 ** 9})) == 0


# ---------------------------------------------------------------------------
# The crop folder is named after the table it belongs to
# ---------------------------------------------------------------------------
#
# `png_list` gets `data`, `png_list_2` gets `data_2`. Two independent counters
# would drift the first time either was deleted, and then nothing on disk would
# say which folder a table describes.


def test_the_folder_carries_the_tables_suffix():
    from spacr.annotation_dataset import crops_folder_for

    assert crops_folder_for("png_list") == "data"
    assert crops_folder_for("png_list_2") == "data_2"
    assert crops_folder_for("png_list_11") == "data_11"


def test_reserving_claims_the_name(tmp_path):
    from spacr.annotation_dataset import next_png_table, reserve_png_table

    path = _db(tmp_path)
    assert reserve_png_table(str(path)) == "png_list"

    connection = sqlite3.connect(path)
    assert next_png_table(connection) == "png_list_2", (
        "the reserved name was not actually taken")


def test_two_reservations_never_collide(tmp_path):
    """Two runs started together would otherwise write a folder each and then
    fight over one table."""
    from spacr.annotation_dataset import reserve_png_table

    path = str(_db(tmp_path))
    names = [reserve_png_table(path) for _ in range(3)]
    assert names == ["png_list", "png_list_2", "png_list_3"]
    assert len(set(names)) == 3


def test_a_reserved_table_starts_empty_with_the_right_columns(tmp_path):
    from spacr.annotation_dataset import reserve_png_table

    path = _db(tmp_path)
    name = reserve_png_table(str(path))
    connection = sqlite3.connect(path)
    assert connection.execute(f'select count(*) from "{name}"').fetchone()[0] == 0
    assert [r[1] for r in connection.execute(f'PRAGMA table_info("{name}")')] \
        == list(PNG_LIST_COLUMNS)


def test_filling_a_reserved_table_does_not_recreate_it(tmp_path):
    from spacr.annotation_dataset import reserve_png_table

    path = str(_db(tmp_path))
    name = reserve_png_table(path)
    assert write_png_list(path, _frame(2), table=name) == name

    connection = sqlite3.connect(path)
    assert connection.execute(f'select count(*) from "{name}"').fetchone()[0] == 2


def test_the_generator_pairs_the_folder_with_the_table(tmp_path):
    """The claim, end to end: a second set lands in data_2 beside png_list_2."""
    import numpy as np

    from spacr.annotation_dataset import generate_annotation_dataset

    merged = tmp_path / "merged"
    merged.mkdir()
    (tmp_path / "measurements").mkdir()
    sqlite3.connect(tmp_path / "measurements" / "measurements.db").close()
    stack = np.zeros((24, 24, 4), dtype=np.uint16)
    stack[4:10, 4:10, 0] = 800
    stack[4:10, 4:10, 3] = 1
    np.save(merged / "plate1_A01_1_1.npy", stack)

    settings = {"src": str(tmp_path), "stream_source": "array",
                "object_array": "cell", "channel_arrays": [0, 1, 2]}
    first = generate_annotation_dataset(dict(settings))
    second = generate_annotation_dataset(dict(settings))

    assert (first["table"], second["table"]) == ("png_list", "png_list_2")
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data_2").is_dir(), (
        "the second set overwrote the first, or landed somewhere unrelated")
