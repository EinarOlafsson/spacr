"""Mask writes which object sits in which as its own database table.

Item 76. The maintainer, 2026-09-29: "i want masks to wirete it and it should
be its own database table!" Before this, Mask called ``write_relationships``
before Measure had made any object table, so it printed a warning and wrote
nothing. The table is now built from the merged label planes by
``spacr.filters._write_object_relationships`` and stored as
``object_relationships`` in ``measurements/measurements.db``.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from spacr.crops import MERGED_LAYOUT_SIDECAR
from spacr.filters import (
    _OBJECT_RELATIONSHIPS_COLUMNS,
    _OBJECT_RELATIONSHIPS_TABLE,
    _object_overlap_rows,
    _object_relationship_pairs,
    _write_object_relationships,
)


def _field(roles=("cell", "nucleus", "organelle", "organelleb")):
    """One 40x40 field: two intensity channels, then one plane per role.

    Cell 1 is columns 0-19, cell 2 columns 20-39. Organelle 5 and 6 both sit
    in cell 1; organelle 7 spans the border, 12 px in cell 1 and 4 in cell 2.
    Organelle 9 sits in no cell. Nucleus 3 is inside cell 1.
    """
    planes = {role: np.zeros((40, 40), np.uint16) for role in roles}
    planes["cell"][0:30, 0:20] = 1
    planes["cell"][0:30, 20:40] = 2
    planes["nucleus"][2:8, 2:8] = 3
    planes["organelle"][4:6, 4:6] = 5        # 4 px, inside nucleus 3 too
    planes["organelle"][20:23, 10:13] = 6    # 9 px
    planes["organelle"][10:14, 17:21] = 7    # 16 px: 12 in cell 1, 4 in 2
    planes["organelle"][34:36, 34:36] = 9    # 4 px, below both cells
    planes["organelleb"][25:27, 30:32] = 1   # 4 px in cell 2
    intensity = np.random.default_rng(0).random((40, 40, 2)).astype(np.float32)
    return np.concatenate(
        [intensity] + [planes[r][..., None].astype(np.float32) for r in roles],
        axis=-1), list(roles)


def _write_run(src, names=("plate1_A01_1", "plate1_B02_3")):
    """A run root with ``merged/`` and its plane layout sidecar."""
    merged = src / "merged"
    merged.mkdir(parents=True)
    stack, roles = _field()
    for name in names:
        np.save(merged / f"{name}.npy", stack)
    layout = {"version": 1, "intensity_channels": [0, 1],
              "mask_plane_order": roles,
              "mask_dims": {r: 2 + i for i, r in enumerate(roles)}}
    (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps(layout))
    return src / "measurements" / "measurements.db"


def _rows(db, where=""):
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(r) for r in connection.execute(
            f'SELECT * FROM "{_OBJECT_RELATIONSHIPS_TABLE}" {where}')]


def test_pairs_place_organelles_in_every_parent_and_never_as_parents():
    pairs = _object_relationship_pairs(
        ["cell", "nucleus", "pathogen", "organelle", "organelleb"])
    assert ("organelle", "cell") in pairs
    assert ("organelleb", "nucleus") in pairs
    assert ("organelleb", "pathogen") in pairs
    assert ("nucleus", "cell") in pairs
    assert not [p for p in pairs if p[1].startswith("organelle")]
    assert not [p for p in pairs if p[0] == p[1]]
    assert _object_relationship_pairs(["organelle"]) == []


def test_overlap_rows_count_pixels_and_state_an_orphan():
    child = np.array([[1, 1, 0, 2]])
    parent = np.array([[4, 5, 0, 0]])
    assert _object_overlap_rows(child, parent) == [
        (1, 4, 1, 2), (1, 5, 1, 2), (2, None, 0, 1)]


def _stored(tmp_path):
    db = _write_run(tmp_path)
    count = _write_object_relationships(str(tmp_path))
    return db, count


def test_a_cell_holding_two_organelles_lists_both(tmp_path):
    db, count = _stored(tmp_path)
    assert count == len(_rows(db))
    rows = _rows(db, "WHERE child_type='organelle' AND parent_type='cell' "
                     "AND file_name='plate1_A01_1'")
    by_child = {}
    for r in rows:
        by_child.setdefault(r["child_label"], {})[r["parent_label"]] = (
            r["overlap_fraction"])
    assert by_child[5] == {1: 1.0}
    assert by_child[6] == {1: 1.0}
    assert by_child[7] == {1: 0.75, 2: 0.25}
    assert by_child[9] == {None: 0.0}
    in_cell_one = sorted(c for c, parents in by_child.items()
                         if parents.get(1) == 1.0)
    assert in_cell_one == [5, 6]


def test_an_organelle_spanning_two_cells_has_one_row_per_cell(tmp_path):
    db, _ = _stored(tmp_path)
    rows = _rows(db, "WHERE child_type='organelle' AND child_label=7 AND "
                     "parent_type='cell' AND file_name='plate1_A01_1'")
    assert sorted((r["parent_label"], r["overlap_pixels"], r["child_pixels"])
                  for r in rows) == [(1, 12, 16), (2, 4, 16)]


def test_rows_carry_the_field_identity_measure_uses(tmp_path):
    db, _ = _stored(tmp_path)
    rows = _rows(db, "WHERE file_name='plate1_B02_3' LIMIT 1")
    assert rows[0]["plateID"] == "plate1"
    assert rows[0]["prcf"] == "plate1_r2_c2_f3"
    assert rows[0]["fieldID"] in ("3", "f3")


def test_second_slot_and_nucleus_parents_are_written(tmp_path):
    db, _ = _stored(tmp_path)
    slot_two = _rows(db, "WHERE child_type='organelleb' AND "
                         "parent_type='cell' AND file_name='plate1_A01_1'")
    assert [(r["child_label"], r["parent_label"]) for r in slot_two] == [
        (1, 2)]
    in_nucleus = _rows(db, "WHERE child_type='organelle' AND child_label=5 "
                           "AND parent_type='nucleus' "
                           "AND file_name='plate1_A01_1'")
    assert [r["parent_label"] for r in in_nucleus] == [3]
    nucleus = _rows(db, "WHERE child_type='nucleus' AND "
                        "file_name='plate1_A01_1'")
    assert [(r["parent_type"], r["parent_label"]) for r in nucleus] == [
        ("cell", 1)]


def test_a_rerun_replaces_the_table_rather_than_adding_to_it(tmp_path):
    db, first = _stored(tmp_path)
    before = _rows(db)
    assert _write_object_relationships(str(tmp_path)) == first
    assert _rows(db) == before
    with sqlite3.connect(db) as connection:
        tables = [r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_OBJECT_RELATIONSHIPS_TABLE,))]
    assert tables == [_OBJECT_RELATIONSHIPS_TABLE]


def test_a_rerun_after_re_masking_reflects_the_new_masks(tmp_path):
    db, _ = _stored(tmp_path)
    merged = tmp_path / "merged"
    (merged / "plate1_B02_3.npy").unlink()
    _write_object_relationships(str(tmp_path))
    assert {r["file_name"] for r in _rows(db)} == {"plate1_A01_1"}


def test_other_tables_in_the_database_are_left_alone(tmp_path):
    db = _write_run(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE cell (object_label INTEGER)")
        connection.execute("INSERT INTO cell VALUES (1), (2)")
    _write_object_relationships(str(tmp_path))
    _write_object_relationships(str(tmp_path))
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM cell").fetchone() == (2,)


def test_columns_are_stored_in_the_declared_order(tmp_path):
    db, _ = _stored(tmp_path)
    with sqlite3.connect(db) as connection:
        columns = tuple(r[1] for r in connection.execute(
            f'PRAGMA table_info("{_OBJECT_RELATIONSHIPS_TABLE}")'))
    assert columns == _OBJECT_RELATIONSHIPS_COLUMNS


def test_nothing_is_written_without_a_plane_layout(tmp_path):
    (tmp_path / "merged").mkdir()
    assert _write_object_relationships(str(tmp_path)) is None
    assert not (tmp_path / "measurements").exists()


def test_the_mask_pipeline_writes_it_inside_the_per_folder_body():
    """Mask, not Measure, owns the table; a failure only warns."""
    import inspect

    from spacr import core

    source = inspect.getsource(core.preprocess_generate_masks)
    call = source.index("_write_object_relationships(")
    assert call < source.index("cleanup_pipeline_folders(src")
    assert "except Exception" in source[call:call + 400]


@pytest.mark.parametrize("name", ["not_a_field", "plate1_A01"])
def test_an_unparsable_field_name_keeps_its_file_name(tmp_path, name):
    db = _write_run(tmp_path, names=(name,))
    _write_object_relationships(str(tmp_path))
    rows = _rows(db)
    assert rows and {r["file_name"] for r in rows} == {name}
    assert {r["prcf"] for r in rows} == {None}


# ---------------------------------------------------------------------------
# Edges the coverage ratchet found untested (dispatch 36794763761)
# ---------------------------------------------------------------------------

def test_no_layout_no_pairs_no_planes_or_no_overlap_is_an_empty_table(tmp_path):
    from spacr.filters import _object_relationships_frame

    merged = tmp_path / "merged"
    merged.mkdir()
    assert _object_relationships_frame(str(merged)).empty
    (merged / MERGED_LAYOUT_SIDECAR).write_text(
        json.dumps({"mask_dims": {"cell": 2}}))
    assert _object_relationships_frame(str(merged)).empty
    (merged / MERGED_LAYOUT_SIDECAR).write_text(
        json.dumps({"mask_dims": {"cell": 2, "nucleus": 9}}))
    np.save(merged / "plate1_A01_1.npy", np.zeros((8, 8, 3), np.float32))
    frame = _object_relationships_frame(str(merged))
    assert frame.empty
    assert list(frame.columns) == list(_OBJECT_RELATIONSHIPS_COLUMNS)


def test_write_relationships_rebuilds_the_parent_table(monkeypatch):
    """The public writer is ensure_relationships_table(rebuild=True)."""
    from spacr import filters

    calls = []
    monkeypatch.setattr(filters, "ensure_relationships_table",
                        lambda db, rebuild=False: (
                            calls.append((db, rebuild)) or "frame"))
    assert filters.write_relationships("m.db") == "frame"
    assert calls == [("m.db", True)]


def test_values_pandas_cannot_test_for_missing_are_stored_as_they_are():
    from spacr.filters import _sqlite_value

    assert _sqlite_value([1, 2]) == [1, 2]
    assert _sqlite_value(np.float32(1.5)) == 1.5 and _sqlite_value(None) is None
