"""F35 — importing foreign measurements must never replace spaCR's own tables.

``spacr.foreign.run_import`` wrote the foreign measurement frame straight
into the canonical ``cell`` / ``nucleus`` / ``pathogen`` table with
``if_exists='replace'``. Pointed at a project that already held spaCR
measurements, that dropped the table and everything in it — the feature
columns, and on a child table the ``cell_id`` link back to the parent
object.

A second way into the same loss survived that fix and is covered lower
down: ``run_import``'s mask loop rebound ``object_type``, the name
holding ``plan.join.object_type``, to the last declared mask class. An
import with cell *and* nucleus masks therefore wrote their cell
measurements into ``foreign_nucleus`` and into the canonical ``nucleus``
table — a child table with no ``cell_id`` — while ``foreign_import`` went
on recording ``canonical_table = 'cell'``.

The other direction of the same question is here too. When nothing of
anyone else's is in the destination the importer *does* fill the
canonical table, as a convenience, and that copy has to step aside the
moment spaCR measures the same objects into it — ``measure_crop``
appends, so a copy left in place makes every per-well count the sum of
two populations. :func:`spacr.foreign.release_canonical_copy` is the
hand-back: it removes only rows it has matched, in SQL, against
``foreign_<object>``, un-claims the table in the same transaction, and
builds the ``<object>_with_foreign`` view on the way out. It is what a
measure resume calls, and what ``run_import(measure=True)`` now calls
before measuring — that path used to record "I did not write the
canonical table" while leaving every imported row in it.

The release's own first attempt deleted by ``rowid``, and every test in
this file released from a ``cell`` that was 100% the importer's — where
deleting too much cannot be seen, because everything was going anyway.
On a table ``measure_crop`` had grown it took spaCR's measurements with
the copy. ``test_releasing_a_table_measure_has_grown_keeps_the_measured_rows``
is the case that was missing;
``test_the_object_tables_shadow_sqlites_own_row_identity`` and
``test_the_measured_rows_that_survive_share_a_row_key_with_the_released``
pin *why* neither ``rowid`` nor the declared row key can address one of
these rows.

Every database here is built by spaCR's own writer,
:func:`spacr.utils._merge_and_save_to_database`, which is what
``measure_crop`` calls; the foreign half is built by
:func:`spacr.foreign.run_import` itself. Nothing is hand-schema'd, because a
hand-built ``cell`` table is exactly the fixture that would have hidden
this.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr import foreign as fg
from spacr.errors import ConfigurationError

SIZE = 24


@contextlib.contextmanager
def _pre_fix_writer():
    """Write with ``measure_crop`` as it was before F34 was fixed at the writer.

    ``utils._merge_and_save_to_database`` now hands back an import's copy of
    the field it is about to write (``utils._release_imported_rows_for_field``),
    so a canonical table holding *both* writers' rows for one field can no
    longer be produced by measuring.

    That state is exactly what the release below exists for, and every database
    a spaCR release before the fix wrote can still be in it — a release that
    stops being tested on a mixed table is a release tested only where
    over-deleting is invisible, which is the gap that let the ``rowid`` bug
    through. So the writer-side release is disabled for the duration of the
    write, and nothing else is.
    """
    import spacr.utils as u

    guard = u._release_imported_rows_for_field
    u._release_imported_rows_for_field = lambda *a, **k: 0
    try:
        yield
    finally:
        u._release_imported_rows_for_field = guard


# ---------------------------------------------------------------------------
# Their data — the same shape spacr/tests/test_foreign.py uses
# ---------------------------------------------------------------------------

def _label_image():
    mask = np.zeros((SIZE, SIZE), np.uint16)
    mask[2:8, 2:8] = 1
    mask[12:20, 12:20] = 2
    return mask


class Theirs:
    def __init__(self, root):
        self.root = str(root)
        self.images = os.path.join(self.root, "their_images")
        self.cell_masks = os.path.join(self.root, "their_cell_masks")
        self.nucleus_masks = os.path.join(self.root, "their_nucleus_masks")
        self.table = os.path.join(self.root, "results.csv")

    def masks(self, *types):
        return {t: getattr(self, f"{t}_masks") for t in (types or ("cell",))}


def _write(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, array)
    return path


@pytest.fixture
def theirs(tmp_path):
    data = Theirs(tmp_path)
    rows = []
    for field in (1, 2):
        for channel in (1, 2):
            _write(os.path.join(data.images, f"fov{field:02d}_C{channel}.tif"),
                   np.full((SIZE, SIZE), field * 10 + channel, np.uint16))
        _write(os.path.join(data.cell_masks, f"fov{field:02d}_cell_mask.tif"),
               _label_image())
        _write(os.path.join(data.nucleus_masks,
                            f"fov{field:02d}_nucleus_mask.tif"), _label_image())
        for label, area in ((1, 36.0), (2, 64.0)):
            rows.append({"ImageNumber": f"fov{field:02d}_C1.tif",
                         "ObjectNumber": label,
                         "AreaShape_Area": area,
                         "Metadata_Treatment": "wt" if field == 1 else "ko"})
    pd.DataFrame(rows).to_csv(data.table, index=False)
    return data


def _plan(theirs, *types, **kwargs):
    kwargs.setdefault("um_per_px", 0.5)
    return fg.plan_import(theirs.images, theirs.masks(*types), theirs.table,
                          **kwargs)


# ---------------------------------------------------------------------------
# A real spaCR measurements database, written by spaCR's own writer
# ---------------------------------------------------------------------------

#: Three fields; the import below covers only the first two, so a replaced
#: table is visible in the row count alone (6 -> 4).
SPACR_STEMS = ("plate1_A01_1", "plate1_A01_2", "plate1_A01_3")


def _real_spacr_project(dst, stems=SPACR_STEMS):
    """Populate ``dst/measurements/measurements.db`` through spaCR's writer.

    :func:`spacr.utils._merge_and_save_to_database` is the function
    ``measure_crop`` writes every object table with: it derives
    ``plateID``/``rowID``/``columnID``/``fieldID``/``prcf`` from the stack
    file name with :func:`spacr.utils._map_wells`, keeps ``cell_id`` on a
    child table, and appends. Using it means the ``cell`` table under test
    is byte-for-byte the one a real run produces.
    """
    from spacr.utils import _merge_and_save_to_database

    dst = str(dst)
    os.makedirs(os.path.join(dst, "measurements"), exist_ok=True)
    for index, stem in enumerate(stems):
        cell_morph = pd.DataFrame({"label": [1, 2],
                                   "cell_area": [36.0 + index, 64.0 + index]})
        cell_intensity = pd.DataFrame(
            {"label": [1, 2],
             "cell_channel_0_mean_intensity": [100.0 + index, 200.0 + index]})
        _merge_and_save_to_database(cell_morph, cell_intensity, "cell",
                                    dst, stem, "spacr_run")
        nucleus_morph = pd.DataFrame(
            {"label": [1, 2],
             # The canonical child-table parent link is the numeric cell
             # object label. Field identity is carried separately by prcf.
             "cell_id": [1, 2],
             "nucleus_area": [9.0 + index, 16.0 + index]})
        nucleus_intensity = pd.DataFrame(
            {"label": [1, 2],
             "nucleus_channel_0_mean_intensity": [11.0, 12.0]})
        _merge_and_save_to_database(nucleus_morph, nucleus_intensity,
                                    "nucleus", dst, stem, "spacr_run")
    return os.path.join(dst, "measurements", "measurements.db")


def _tables(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return sorted(str(r[0]) for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"))
    finally:
        connection.close()


def _read(db_path, table):
    connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    finally:
        connection.close()


def test_the_fixture_database_really_is_spacrs(tmp_path):
    """Guard the guard: the baseline is spaCR's writer, not a hand schema."""
    db = _real_spacr_project(tmp_path / "project")
    cells = _read(db, "cell")
    nuclei = _read(db, "nucleus")
    assert len(cells) == 6 and len(nuclei) == 6
    assert list(cells.columns[:8]) == ["object_label", "plateID", "rowID",
                                       "columnID", "fieldID", "prcf",
                                       "file_name", "path_name"]
    assert list(nuclei.columns[:2]) == ["object_label", "cell_id"]
    assert sorted(cells["prcf"].unique()) == ["plate1_r1_c1_f1",
                                              "plate1_r1_c1_f2",
                                              "plate1_r1_c1_f3"]


# ---------------------------------------------------------------------------
# The reproduction
# ---------------------------------------------------------------------------

def test_importing_into_a_project_keeps_its_cell_table(theirs, tmp_path):
    """The canonical ``cell`` table survives an import, row for row."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    before = _read(db, "cell")
    assert len(before) == 6
    assert before["cell_area"].tolist() == [36.0, 64.0, 37.0, 65.0, 38.0, 66.0]

    result = fg.run_import(_plan(theirs), str(dst))

    after = _read(db, "cell")
    assert len(after) == len(before) == 6
    assert "cell_area" in after.columns
    assert after["cell_area"].tolist() == before["cell_area"].tolist()
    assert "cell_channel_0_mean_intensity" in after.columns
    assert sorted(after["prcf"].unique()) == sorted(before["prcf"].unique())
    # no foreign column has been smuggled into spaCR's own table
    assert [c for c in after.columns if c.startswith(fg.FOREIGN_PREFIX)] == []

    # …and their measurements did land, in their own table
    theirs_rows = _read(db, "foreign_cell")
    assert len(theirs_rows) == 4
    assert "foreign_areashape_area" in theirs_rows.columns
    assert result.rows["foreign_cell"] == 4
    assert "cell" not in result.rows


def test_importing_into_a_project_keeps_the_cell_id_link(theirs, tmp_path):
    """A child table's ``cell_id`` is the parent link; it must not be dropped."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    before = _read(db, "nucleus")
    assert list(before["cell_id"]) == [n for _stem in SPACR_STEMS
                                       for n in (1, 2)]

    fg.run_import(_plan(theirs, "nucleus", measurement_object="nucleus"),
                  str(dst))

    after = _read(db, "nucleus")
    assert len(after) == len(before) == 6
    assert "cell_id" in after.columns
    assert list(after["cell_id"]) == list(before["cell_id"])
    assert after["nucleus_area"].tolist() == before["nucleus_area"].tolist()
    assert len(_read(db, "foreign_nucleus")) == 4


def test_their_columns_are_reachable_beside_spacrs_through_the_view(theirs,
                                                                    tmp_path):
    """Preserved, not lost: the join on (prcf, object_label) is a view."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    fg.run_import(_plan(theirs), str(dst))

    assert "cell_with_foreign" in _tables(db)
    joined = _read(db, "cell_with_foreign")
    # 2 of the 3 fields overlap, 2 objects each
    assert len(joined) == 4
    assert "cell_area" in joined.columns
    assert "foreign_areashape_area" in joined.columns
    assert sorted(joined["prcf"].unique()) == ["plate1_r1_c1_f1",
                                               "plate1_r1_c1_f2"]
    # spaCR's number and theirs, side by side, for the same object
    row = joined[(joined["prcf"] == "plate1_r1_c1_f1")
                 & (joined["object_label"] == 2)].iloc[0]
    assert row["cell_area"] == 64.0
    assert row["foreign_areashape_area"] == 64.0


def test_a_foreign_column_on_a_spacr_name_is_aliased_not_dropped(theirs,
                                                                 tmp_path):
    """A column no query can reach is a column that was silently lost."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    maps = [m if m.source != "AreaShape_Area"
            else fg.ColumnMap(source="AreaShape_Area", target="cell_area")
            for m in fg.infer_column_map(pd.read_csv(theirs.table),
                                         image_key="ImageNumber",
                                         label_key="ObjectNumber")]
    plan = _plan(theirs, column_maps=maps, allow_spacr_targets=True)
    assert plan.ok, fg.format_plan(plan)
    fg.run_import(plan, str(dst))

    joined = _read(db, "cell_with_foreign")
    assert list(joined.columns).count("cell_area") == 1     # spaCR's
    assert "foreign_cell_area" in joined.columns            # theirs, aliased
    row = joined[(joined["prcf"] == "plate1_r1_c1_f1")
                 & (joined["object_label"] == 1)].iloc[0]
    assert row["cell_area"] == 36.0
    assert row["foreign_cell_area"] == 36.0


def test_the_summary_says_the_canonical_table_was_left_alone(theirs, tmp_path):
    dst = tmp_path / "project"
    _real_spacr_project(dst)
    result = fg.run_import(_plan(theirs), str(dst))
    text = result.summary()
    assert "cell" in text and "foreign_cell" in text
    assert "cell_with_foreign" in text


# ---------------------------------------------------------------------------
# A destination that is somebody else's experiment: refused, not merged
# ---------------------------------------------------------------------------

def test_a_destination_holding_another_experiment_is_refused(theirs, tmp_path):
    """No shared field: nothing to reconcile, so nothing is written."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst, stems=("plate9_B02_1", "plate9_B02_2"))
    before_tables = _tables(db)
    before = _read(db, "cell")

    with pytest.raises(ConfigurationError) as excinfo:
        fg.run_import(_plan(theirs), str(dst))

    message = str(excinfo.value)
    assert "cell" in message
    assert "plate9_r2_c2_f1" in message          # what is there
    assert "plate1_r1_c1_f1" in message          # what was being imported
    assert "4 row(s)" in message                 # the size of what it protects

    # nothing was written: not the database, not the project folders
    assert _tables(db) == before_tables
    assert _read(db, "cell").equals(before)
    for folder in (fg.IMAGES_DIRNAME, "stack", "merged", "masks"):
        assert not os.path.exists(os.path.join(str(dst), folder)), folder


# ---------------------------------------------------------------------------
# The importer's own tables are still its own
# ---------------------------------------------------------------------------

def test_a_fresh_import_still_fills_the_canonical_table(theirs, tmp_path):
    """Nothing to protect, so the convenience copy is written as before."""
    dst = tmp_path / "imported"
    result = fg.run_import(_plan(theirs), str(dst))
    cells = _read(result.db_path, "cell")
    assert len(cells) == 4
    assert "foreign_areashape_area" in cells.columns
    assert result.rows["cell"] == 4


def test_re_running_an_import_replaces_only_its_own_rows(theirs, tmp_path):
    """Idempotent: the importer owns that ``cell`` table and may rewrite it."""
    dst = str(tmp_path / "imported")
    first = fg.run_import(_plan(theirs), dst)
    before = {name: len(_read(first.db_path, name))
              for name in ("cell", "foreign_cell", cv.CONVERSION_TABLE,
                           fg.FOREIGN_COLUMNS_TABLE, fg.IMPORT_TABLE)}
    second = fg.run_import(_plan(theirs), dst)
    assert {name: len(_read(second.db_path, name)) for name in before} == before


def test_a_canonical_table_the_importer_wrote_and_measure_grew_is_protected(
        theirs, tmp_path):
    """Ownership is checked against the table as it is now, not as it was.

    The importer writes ``cell``, then the user runs spaCR's own measure
    over the imported project, which *appends* to the same table. The
    second import must not treat that grown table as its own to replace.
    """
    from spacr.utils import _merge_and_save_to_database

    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")
    assert len(_read(db, "cell")) == 4

    with _pre_fix_writer():
        _merge_and_save_to_database(
            pd.DataFrame({"label": [1, 2], "cell_area": [36.0, 64.0]}),
            pd.DataFrame({"label": [1, 2],
                          "cell_channel_0_mean_intensity": [1.0, 2.0]}),
            "cell", dst, "plate1_A01_1", "spacr_run")
    grown = _read(db, "cell")
    assert len(grown) == 6 and "cell_area" in grown.columns

    fg.run_import(_plan(theirs), dst)

    after = _read(db, "cell")
    assert len(after) == 6
    pd.testing.assert_series_equal(after["cell_area"], grown["cell_area"])
    assert after["cell_area"].dropna().tolist() == [36.0, 64.0]


def test_a_second_import_keeps_the_first_ones_provenance(theirs, tmp_path):
    """Two object types into one database: neither erases the other's record."""
    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")
    fg.run_import(_plan(theirs, "nucleus", measurement_object="nucleus"), dst)

    provenance = _read(db, fg.FOREIGN_COLUMNS_TABLE)
    assert {"cell", "foreign_cell", "foreign_nucleus"} <= set(
        provenance["table"])
    runs = _read(db, fg.IMPORT_TABLE)
    assert len(runs) == 2
    assert sorted(runs["object_types"]) == ["cell", "nucleus"]


def test_a_table_that_appears_mid_import_is_still_protected(theirs, tmp_path,
                                                            monkeypatch):
    """The check that drops a table has to be taken when it is dropped.

    Converting the images takes minutes, and the destination was judged
    before any of it started. A ``measure_crop`` finishing in that window
    creates the very table the import was about to write.
    """
    from spacr.utils import _merge_and_save_to_database

    dst = str(tmp_path / "imported")
    original = fg._foreign_frame
    raced = {"done": False}

    def racing(plan, stems, merged_dir):
        if not raced["done"]:                  # once, just before the write
            raced["done"] = True
            _merge_and_save_to_database(
                pd.DataFrame({"label": [1, 2], "cell_area": [36.0, 64.0]}),
                pd.DataFrame({"label": [1, 2],
                              "cell_channel_0_mean_intensity": [1.0, 2.0]}),
                "cell", dst, "plate1_A01_1", "spacr_run")
        return original(plan, stems, merged_dir)

    monkeypatch.setattr(fg, "_foreign_frame", racing)
    result = fg.run_import(_plan(theirs), dst)

    db = os.path.join(dst, "measurements", "measurements.db")
    cells = _read(db, "cell")
    assert len(cells) == 2
    assert cells["cell_area"].tolist() == [36.0, 64.0]
    assert "cell" not in result.rows
    assert len(_read(db, "foreign_cell")) == 4
    assert "cell_with_foreign" in _tables(db)


def test_the_preview_says_what_the_destination_already_holds(theirs, tmp_path,
                                                             capsys):
    """A preview that does not mention the table it will not touch is not one."""
    dst = tmp_path / "project"
    _real_spacr_project(dst)
    fg.import_project(images=theirs.images, masks=theirs.masks(),
                      measurements=theirs.table, dst=str(dst),
                      um_per_px=0.5, preview_only=True)
    printed = capsys.readouterr().out
    assert "cell" in printed and "6 row(s)" in printed
    assert "cell_with_foreign" in printed
    assert not os.path.exists(os.path.join(str(dst), fg.IMAGES_DIRNAME))


def test_the_preview_refuses_a_destination_it_cannot_write(theirs, tmp_path):
    dst = tmp_path / "project"
    _real_spacr_project(dst, stems=("plate9_B02_1",))
    with pytest.raises(ConfigurationError, match="shares no field"):
        fg.import_project(images=theirs.images, masks=theirs.masks(),
                          measurements=theirs.table, dst=str(dst),
                          um_per_px=0.5, preview_only=True)


def test_a_database_from_the_replacing_release_still_re_imports(theirs,
                                                                tmp_path):
    """Migration: a project the old importer wrote is still its own.

    The release this fixes wrote ``foreign_columns`` / ``foreign_import``
    with ``to_sql(if_exists='replace')`` and a narrower ``foreign_import``
    schema. Both are reproduced here exactly — same call, same columns —
    and the re-import has to recognise the ``cell`` table as the
    importer's, widen the provenance and supersede the run row rather than
    duplicate it.
    """
    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")

    new_columns = ["canonical_table", "canonical_table_written",
                   "canonical_table_note"]
    runs = _read(db, fg.IMPORT_TABLE)
    assert set(new_columns) <= set(runs.columns)
    connection = sqlite3.connect(db)
    try:                                       # the old release's own write
        runs.drop(columns=new_columns).to_sql(
            fg.IMPORT_TABLE, connection, if_exists="replace", index=False)
        _read(db, fg.FOREIGN_COLUMNS_TABLE).to_sql(
            fg.FOREIGN_COLUMNS_TABLE, connection, if_exists="replace",
            index=False)
    finally:
        connection.close()
    assert not set(new_columns) & set(_read(db, fg.IMPORT_TABLE).columns)

    result = fg.run_import(_plan(theirs), dst)

    assert result.rows["cell"] == 4            # still ours to rewrite
    assert len(_read(db, "cell")) == 4
    migrated = _read(db, fg.IMPORT_TABLE)
    assert len(migrated) == 1                  # superseded, not doubled
    assert migrated["canonical_table"].iloc[0] == "cell"
    assert int(migrated["canonical_table_written"].iloc[0]) == 1


def test_a_write_that_fails_half_way_leaves_the_old_table_standing(tmp_path):
    """DROP + CREATE + INSERT is not atomic; this replacement has to be.

    sqlite3 opens an implicit transaction only for DML, so pandas' DROP
    lands on its own and an insert that then fails leaves no table at all.
    """
    db = _real_spacr_project(tmp_path / "project")
    before = _read(db, "cell")
    connection = sqlite3.connect(db)
    try:
        # a value sqlite cannot bind: the insert raises after the drop would
        # have happened
        broken = pd.DataFrame({"object_label": [1], "junk": [{1, 2}]})
        with pytest.raises(Exception):
            fg._replace_table_atomically(connection, "cell", broken)
    finally:
        connection.close()
    assert "cell" in _tables(db)
    assert _read(db, "cell").equals(before)


def test_a_failed_provenance_write_keeps_the_previous_record(theirs, tmp_path,
                                                             monkeypatch):
    """The delete and the insert are one transaction, or the record is lost."""
    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")
    before = _read(db, fg.FOREIGN_COLUMNS_TABLE)
    assert len(before) > 0

    def boom(cursor, table, frame):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(fg, "_insert_rows", boom)
    with pytest.raises(RuntimeError):
        fg._write_provenance(db, _plan(theirs), dst, ["cell", "foreign_cell"])

    after = _read(db, fg.FOREIGN_COLUMNS_TABLE)
    assert len(after) == len(before)
    assert set(after["table"]) == set(before["table"])


def test_an_empty_object_table_is_not_a_conflict(theirs, tmp_path):
    """A table created and never written to has no experiment to defend."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    connection = sqlite3.connect(db)
    try:
        connection.execute("DELETE FROM cell")
        connection.execute("DELETE FROM nucleus")
        connection.commit()
    finally:
        connection.close()

    result = fg.run_import(_plan(theirs), str(dst))
    assert result.rows["cell"] == 4
    assert len(_read(db, "cell")) == 4


def test_an_existing_conversion_map_is_not_thrown_away(theirs, tmp_path):
    """The project's own image provenance survives an import into it.

    ``convert.populate_db_from_map`` replaces ``conversion_map``, which is
    right for a conversion writing its own database and wrong for an
    import landing in a project that already converted images of its own.
    """
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    # a native conversion of this project's own images, by convert.py itself
    native = tmp_path / "native" / "plateZ" / "B02"
    _write(str(native / "img_C1.tif"), np.full((SIZE, SIZE), 7, np.uint16))
    conversion = cv.convert(
        cv.plan(cv.scan(str(tmp_path / "native")), plate_naming="index"),
        str(dst / fg.IMAGES_DIRNAME))
    cv.populate_db_from_map(db, conversion.map_path)
    before = _read(db, cv.CONVERSION_TABLE)
    assert len(before) == 1
    assert before["target"].iloc[0].startswith("plate1_B02_")

    fg.run_import(_plan(theirs), str(dst))

    after = _read(db, cv.CONVERSION_TABLE)
    assert set(before["target"]) <= set(after["target"])
    assert len(after) == len(before) + 4      # theirs added, ours kept


def test_a_re_imported_field_supersedes_its_own_conversion_row(theirs,
                                                               tmp_path):
    """Merged on the output filename, so a re-run does not double it up."""
    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")
    before = _read(db, cv.CONVERSION_TABLE)
    assert len(before) == 4
    fg.run_import(_plan(theirs), dst)
    after = _read(db, cv.CONVERSION_TABLE)
    assert len(after) == 4
    assert sorted(after["target"]) == sorted(before["target"])


# ---------------------------------------------------------------------------
# More than one mask class: the write must aim at the object the table was
# joined against, not at whichever mask folder happened to be listed last
# ---------------------------------------------------------------------------

def _import_record(db_path):
    return _read(db_path, fg.IMPORT_TABLE).iloc[0]


def test_a_two_mask_import_writes_the_object_its_table_was_joined_against(
        theirs, tmp_path):
    """THE BUG. The mask loop rebound ``object_type``.

    An import declares one mask class per folder and has exactly one
    *measured* object type — ``plan.join.object_type``, the one the
    numbers were joined against and the one ``_check_destination``
    consulted. ``for object_type in plan.object_types:`` in step 3 rebound
    that name to the last mask class, so with cell **and** nucleus masks
    every later use aimed one table over.

    Measured before the fix: their *cell* areas were written to
    ``foreign_nucleus`` and to the canonical ``nucleus`` table — a child
    table with no ``cell_id`` in it — while ``foreign_import`` went on
    recording ``canonical_table = 'cell'``. No ``cell`` or ``foreign_cell``
    table existed at all.
    """
    plan = _plan(theirs, "cell", "nucleus")
    assert plan.ok, fg.format_plan(plan)
    assert plan.object_types == ["cell", "nucleus"]     # two mask classes
    assert plan.join.object_type == "cell"              # one measured object

    result = fg.run_import(plan, str(tmp_path / "imported"))
    db = result.db_path
    tables = _tables(db)

    assert "foreign_cell" in tables
    assert "foreign_nucleus" not in tables
    assert len(_read(db, "foreign_cell")) == 4
    # Their AreaShape_Area is a *cell* area. It measured their cell masks.
    assert _read(db, "foreign_cell")["foreign_areashape_area"].tolist() == [
        36.0, 64.0, 36.0, 64.0]

    # The canonical copy goes to the same object, and no child table is
    # invented for a class whose objects were never measured.
    assert result.rows == {"foreign_cell": 4, "cell": 4}
    assert "nucleus" not in tables

    # ...and the provenance describes the table that was actually written.
    record = _import_record(db)
    assert record["canonical_table"] == "cell"
    assert int(record["canonical_table_written"]) == 1
    assert record["canonical_table"] in tables


def test_a_two_mask_import_beside_a_spacr_project_joins_the_right_table(
        theirs, tmp_path):
    """The same misdirection, where it produces a plainly wrong number.

    ``_check_destination`` decided ``preserve`` for ``cell`` — that is the
    table the note is about. Before the fix the rows still went to
    ``foreign_nucleus``, and ``_write_view`` then built
    ``nucleus_with_foreign``, a view pairing spaCR's nucleus 1 with
    *their cell* 1 on ``(prcf, object_label)`` and presenting the two areas
    side by side as if they measured the same object.
    """
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    nuclei_before = _read(db, "nucleus")

    result = fg.run_import(_plan(theirs, "cell", "nucleus"), str(dst))
    tables = _tables(db)

    assert "cell_with_foreign" in tables
    assert "nucleus_with_foreign" not in tables
    assert "foreign_nucleus" not in tables

    # spaCR's own tables are untouched, cell_id included.
    after = _read(db, "nucleus")
    assert len(after) == len(nuclei_before) == 6
    assert list(after["cell_id"]) == list(nuclei_before["cell_id"])
    assert len(_read(db, "cell")) == 6

    # The note and the view agree on which table was preserved.
    assert any("cell" in note and "foreign_cell" in note
               for note in result.notes), result.notes
    joined = _read(db, "cell_with_foreign")
    row = joined[(joined["prcf"] == "plate1_r1_c1_f1")
                 & (joined["object_label"] == 2)].iloc[0]
    assert row["cell_area"] == 64.0                  # spaCR's cell
    assert row["foreign_areashape_area"] == 64.0     # their cell


def test_a_two_mask_import_still_writes_every_mask_stack(theirs, tmp_path):
    """The loop the name was borrowed from still does its own job.

    Renaming the variable must not stop the *masks* being written per
    class — both folders, both fields.
    """
    dst = tmp_path / "imported"
    fg.run_import(_plan(theirs, "cell", "nucleus"), str(dst))
    for object_type in ("cell", "nucleus"):
        folder = os.path.join(str(dst), "masks", f"{object_type}_mask_stack")
        assert sorted(os.listdir(folder)) == ["plate1_A01_1.npy",
                                              "plate1_A01_2.npy"]


# ---------------------------------------------------------------------------
# The provenance this module reads is renamed out from under it by the
# legacy-column repair every measure write runs
# ---------------------------------------------------------------------------

def test_the_column_provenance_survives_a_measure_write(theirs, tmp_path):
    """THE BUG. ``SELECT "column"`` came back with the word ``'column'``.

    ``schema.LEGACY_COLUMN_NAMES`` maps ``column`` to the plate coordinate
    ``columnID``, and ``database_schema.repair_legacy_columns`` applies it
    to *every* user table — including ``foreign_columns``, where ``column``
    holds the name of a measurement column and has nothing to do with a
    well. ``utils._merge_and_save_to_database`` runs that repair on every
    write, so a single ``measure_crop`` over an imported project renames
    this module's provenance.

    That alone would be survivable. What made it dangerous is SQLite:
    a double-quoted name matching no column is resolved as a *string
    literal*, so ``SELECT "column" FROM foreign_columns`` did not raise —
    it returned the four-letter word once per row. Every recorded name
    became ``'column'``, ``have <= recorded`` was false for a table this
    importer had written itself, and :func:`spacr.foreign._importer_owns`
    answered False with nothing anywhere saying why.

    Measured before the fix: ``_importer_owns(conn, 'cell')`` True before
    the measure write and False after it, on a ``cell`` table the measure
    write never touched.
    """
    from spacr.utils import _merge_and_save_to_database

    dst = tmp_path / "imported"
    result = fg.run_import(_plan(theirs), str(dst))
    db = result.db_path
    assert result.rows == {"foreign_cell": 4, "cell": 4}

    connection = sqlite3.connect(db)
    try:
        assert fg._provenance_name_column(connection) == "column"
        assert fg._importer_owns(connection, "cell")
    finally:
        connection.close()

    # A real measure write, into a *different* table — `cell` is untouched.
    morphology = pd.DataFrame({"label": [1, 2], "nucleus_area": [9.0, 9.0]})
    intensity = pd.DataFrame(
        {"label": [1, 2], "nucleus_channel_0_mean_intensity": [1.0, 1.0]})
    _merge_and_save_to_database(morphology, intensity, "nucleus", str(dst),
                                "plate1_A01_1", "exp", False)

    connection = sqlite3.connect(db)
    try:
        # The repair really did rename it...
        assert fg._provenance_name_column(connection) == "columnID"
        assert "column" not in _read(db, fg.FOREIGN_COLUMNS_TABLE).columns
        # ...and the answer is still the right one.
        assert fg._importer_owns(connection, "cell")
        assert fg._may_write_canonical(connection, "cell")[0] is True
        # The rows themselves are intact — only the schema name moved.
        recorded = {str(r[0]) for r in connection.execute(
            'SELECT [columnID] FROM "foreign_columns" WHERE [table] = ?',
            ("cell",))}
        assert "foreign_areashape_area" in recorded
        assert recorded != {"column"}
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Handing the canonical table back when spaCR is going to fill it itself
# ---------------------------------------------------------------------------

def _measure_into(dst, stem, table="cell", labels=(1, 2)):
    """Append one field of spaCR's own measurements through spaCR's writer."""
    from spacr.utils import _merge_and_save_to_database

    n = len(labels)
    _merge_and_save_to_database(
        pd.DataFrame({"label": list(labels),
                      f"{table}_area": [36.0 + i for i in range(n)]}),
        pd.DataFrame({"label": list(labels),
                      f"{table}_channel_0_mean_intensity":
                          [1.0 + i for i in range(n)]}),
        table, str(dst), stem, "spacr_run")


def test_the_object_tables_shadow_sqlites_own_row_identity(theirs, tmp_path):
    """``rowid`` in one of these tables is a *plate row*, not a row identity.

    Pinned on its own because it is the trap the release fell into and
    nothing else in the suite states it. Every spaCR object table
    declares a column called ``rowID``; SQLite identifiers are
    case-insensitive and a declared column always shadows the implicit
    ``rowid``, so ``SELECT rowid FROM cell`` returns ``'r1'`` once per
    row rather than 1, 2, 3 — and there is no spelling of it, quoted or
    not, that gets the row identity back. Anything in this module that
    wants to address one row must therefore address it by a predicate,
    which is what :func:`spacr.foreign.release_canonical_copy` does.
    """
    db = fg.run_import(_plan(theirs), str(tmp_path / "imported")).db_path
    connection = sqlite3.connect(db)
    try:
        assert "rowID" in {r[1] for r in
                           connection.execute('PRAGMA table_info("cell")')}
        shadowed = [r[0] for r in connection.execute('SELECT rowid FROM "cell"')]
        quoted = [r[0] for r in connection.execute('SELECT "rowid" FROM "cell"')]
        assert shadowed == ["r1"] * 4          # the plate row, four times
        assert quoted == shadowed              # quoting does not rescue it
        # The real row identity is only reachable under a name the table
        # does not declare.
        assert [r[0] for r in
                connection.execute('SELECT _rowid_ FROM "cell"')] == [1, 2, 3, 4]
    finally:
        connection.close()


def test_releasing_a_table_measure_has_grown_keeps_the_measured_rows(theirs,
                                                                     tmp_path):
    """THE BUG, and the gap that hid it: release from a *mixed* table.

    Every other release test in this file releases from a table that is
    100% the importer's, where deleting too much is invisible because
    everything was going anyway. This one releases from the table the
    feature actually exists for — an import's copy sitting beside rows
    ``measure_crop`` has since appended, for the *same field*.

    Measured before the fix, on exactly this database: the release
    issued ``DELETE FROM cell WHERE rowid IN (SELECT s.rowid …)``, both
    ``rowid``\\ s resolved to the ``rowID`` column (``'r1'`` for every
    row), and the statement removed **all six** rows — the two spaCR had
    measured along with the four it was asked to release — then reported
    six released. The measurements existed nowhere else.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    # spaCR measures field 1, which the import also covers.
    with _pre_fix_writer():                     # a pre-fix database
        _measure_into(dst, "plate1_A01_1")
    mixed = _read(db, "cell")
    assert len(mixed) == 6
    assert int(mixed["cell_area"].notna().sum()) == 2
    assert int(mixed["foreign_areashape_area"].notna().sum()) == 4

    removed = fg.release_canonical_copy(db, "cell")

    assert removed == 4                         # theirs, and only theirs
    after = _read(db, "cell")
    assert len(after) == 2
    assert after["cell_area"].tolist() == [36.0, 37.0]
    assert after["fieldID"].tolist() == ["f1", "f1"]
    assert "foreign_areashape_area" not in after.columns
    assert len(_read(db, "foreign_cell")) == 4  # theirs, unharmed
    assert "cell_with_foreign" in _tables(db)


def test_the_measured_rows_that_survive_share_a_row_key_with_the_released(
        theirs, tmp_path):
    """Why the delete is a predicate and not a key lookup.

    The obvious repair for the ``rowid`` bug is to delete by the object
    table's declared key — ``plateID``/``rowID``/``columnID``/``fieldID``
    /``object_label``. On this database that is *also* wrong, and by the
    same amount: the import's row for field 1 object 1 and the row
    ``measure_crop`` wrote for field 1 object 1 carry identical values in
    all five. They are the same object measured twice, which is the whole
    reason the copy has to be released. Measured: a keyed delete removes
    six of six rows here, exactly as the ``rowid`` one did.

    What tells them apart is the only thing that ever did — which columns
    they hold values in — so that is what the DELETE is keyed on.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    with _pre_fix_writer():                     # a pre-fix database
        _measure_into(dst, "plate1_A01_1")

    keys = _read(db, "cell")[["plateID", "rowID", "columnID", "fieldID",
                              "object_label"]]
    assert len(keys) == 6
    assert int(keys.duplicated(keep=False).sum()) == 4   # 2 imported + 2 measured

    connection = sqlite3.connect(db)
    try:
        # The keyed delete, run for real against a copy of the table, to
        # measure rather than assert the claim above.
        connection.execute('CREATE TABLE probe AS SELECT * FROM "cell"')
        where = ' AND '.join(f'"{c}" IS ?' for c in keys.columns)
        targets = keys.iloc[[0, 1, 2, 3]].itertuples(index=False, name=None)
        hit = sum(connection.execute(f'DELETE FROM probe WHERE {where}',
                                     tuple(t)).rowcount for t in targets)
        assert hit == 6                     # would have taken measure's too
    finally:
        connection.close()


def test_a_delete_that_does_not_match_the_checks_is_refused_and_rolled_back(
        theirs, tmp_path, monkeypatch):
    """The release reports what it did, or it does nothing at all.

    The checks that clear a release — "how many rows are the importer's"
    and "do all of them have a twin" — run before the write transaction
    opens, so another writer can move underneath them. If the DELETE then
    removes a different number of rows than was verified, the number this
    function returns is not what happened to the database, which is the
    exact class of failure it exists to prevent. It refuses instead, and
    the un-claim goes back with the delete: a claim removed from a table
    whose rows are still there is unrecoverable, a refusal is not.

    The interleaving is produced here rather than described: a second
    connection deletes one imported row after ``held`` has been counted.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    with _pre_fix_writer():                     # a pre-fix database
        _measure_into(dst, "plate1_A01_1")
    real_twin = fg._twin_condition

    def twin_then_meddle(connection, object_type):
        """Real answer, then somebody else writes — after ``held`` was read."""
        condition = real_twin(connection, object_type)
        other = sqlite3.connect(db)
        try:
            other.execute('DELETE FROM "cell" WHERE "fieldID" = ? '
                          'AND "foreign_areashape_area" IS NOT NULL', ("f2",))
            other.commit()
        finally:
            other.close()
        return condition

    monkeypatch.setattr(fg, "_twin_condition", twin_then_meddle)

    with pytest.raises(ConfigurationError) as raised:
        fg.release_canonical_copy(db, "cell")

    assert "did not select the rows that were verified" in str(raised.value)
    # The delete was rolled back, so the meddler's two rows are the only
    # ones missing and the table still holds both populations...
    after = _read(db, "cell")
    assert len(after) == 4
    assert int(after["cell_area"].notna().sum()) == 2
    assert int(after["foreign_areashape_area"].notna().sum()) == 2
    # ...and nothing was un-claimed, so a later release can still act.
    assert "cell" in set(_read(db, fg.FOREIGN_COLUMNS_TABLE)["table"])
    assert int(_import_record(db)["canonical_table_written"]) == 1


def test_releasing_the_copy_keeps_their_rows_and_drops_the_duplicate(theirs,
                                                                     tmp_path):
    """The copy in ``cell`` is a duplicate; the original is ``foreign_cell``.

    ``release_canonical_copy`` removes only rows it has matched, in SQL,
    against ``foreign_cell`` — same value in every shared column, NULL in
    every unshared one. Their measurements are untouched, and the view
    that reaches them is built on the way out.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    assert len(_read(db, "cell")) == 4

    removed = fg.release_canonical_copy(db, "cell")

    assert removed == 4
    assert len(_read(db, "cell")) == 0
    theirs_rows = _read(db, "foreign_cell")
    assert len(theirs_rows) == 4
    assert theirs_rows["foreign_areashape_area"].tolist() == [36.0, 64.0,
                                                              36.0, 64.0]
    assert "cell_with_foreign" in _tables(db)
    # Their measurement column goes with their rows: an empty column left
    # in `cell` would force the view to alias theirs to
    # `foreign_foreign_areashape_area`, which is a column nobody would
    # think to ask for.
    assert "foreign_areashape_area" not in _read(db, "cell").columns


def test_a_released_table_is_no_longer_claimed_by_anything(theirs, tmp_path):
    """The un-claim, which is what makes the release something to build on.

    A claim that outlives the rows it was about is a refusal nothing can
    ever lift — the dead end the backed-out resume guard reached. Both
    records move in the same transaction as the delete.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    connection = sqlite3.connect(db)
    try:
        assert fg._importer_owns(connection, "cell") is True
    finally:
        connection.close()

    fg.release_canonical_copy(db, "cell")

    assert set(_read(db, fg.FOREIGN_COLUMNS_TABLE)["table"]) == {"foreign_cell"}
    run = _import_record(db)
    assert int(run["canonical_table_written"]) == 0
    assert "released" in run["canonical_table_note"]
    connection = sqlite3.connect(db)
    try:
        assert fg._importer_owns(connection, "cell") is False
        assert fg._importer_owns(connection, "foreign_cell") is True
    finally:
        connection.close()


def test_a_dry_run_release_reports_without_writing(theirs, tmp_path):
    """"Could this be released?" is asked through the code that releases.

    A second implementation of the question is a second implementation
    that can answer differently from the one that acts.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path

    assert fg.release_canonical_copy(db, "cell", dry_run=True) == 4

    assert len(_read(db, "cell")) == 4
    assert set(_read(db, fg.FOREIGN_COLUMNS_TABLE)["table"]) == {
        "cell", "foreign_cell"}


def test_releasing_a_table_no_import_ever_wrote_does_nothing(theirs, tmp_path):
    """A project of spaCR's own is not this function's business."""
    dst = tmp_path / "project"
    db = _real_spacr_project(dst)
    before = _read(db, "cell")

    assert fg.release_canonical_copy(db, "cell") == 0
    assert fg.release_canonical_copy(db, "nucleus") == 0

    assert _read(db, "cell").equals(before)


def test_re_extracting_over_an_earlier_import_replaces_its_copy(theirs,
                                                                tmp_path,
                                                                monkeypatch):
    """THE BUG in the remedy the backed-out guard printed.

    "Re-run the import with ``measure=True``" recorded
    ``canonical_table_written = 0`` and left every imported row sitting in
    ``cell``, because ``_check_destination`` returned ``'measure'`` before
    it ever looked at the canonical table. ``measure_crop`` then appended
    to it and the table held both populations, with the provenance saying
    it held neither.

    Measured before the fix: ``cell`` 4 imported rows + spaCR's, and a
    ``foreign_import`` row claiming nothing was written.
    """
    import spacr.measure as measure

    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    db = os.path.join(dst, "measurements", "measurements.db")
    assert len(_read(db, "cell")) == 4

    def fake_measure_crop(settings):
        """Stand in for the real thing: append a spaCR-shaped cell table."""
        from spacr.utils import _merge_and_save_to_database

        for stem in ("plate1_A01_1", "plate1_A01_2"):
            _merge_and_save_to_database(
                pd.DataFrame({"label": [1, 2], "cell_area": [36.0, 64.0]}),
                pd.DataFrame({"label": [1, 2],
                              "cell_channel_0_mean_intensity": [1.0, 2.0]}),
                "cell", settings["src"], stem, "spacr_run")

    monkeypatch.setattr(measure, "measure_crop", fake_measure_crop)
    result = fg.run_import(_plan(theirs), dst, measure=True)

    cells = _read(db, "cell")
    assert len(cells) == 4                       # spaCR's four, not eight
    assert cells["cell_area"].notna().all()
    assert "foreign_areashape_area" not in cells.columns
    assert len(_read(db, "foreign_cell")) == 4   # theirs, unharmed
    assert "cell_with_foreign" in _tables(db)
    # ...and the user is told, before any of it, what will happen to it.
    assert any("copied there" in note for note in result.notes), result.notes


def test_re_extracting_into_a_fresh_destination_says_nothing_about_a_copy(
        theirs, tmp_path, monkeypatch):
    """The control: no earlier import, so no note and nothing to release."""
    import spacr.measure as measure

    monkeypatch.setattr(measure, "measure_crop", lambda settings: None)
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                           measure=True)

    assert result.notes == []
    assert "cell" not in result.rows


def test_a_database_with_no_provenance_column_at_all_is_not_ours(theirs,
                                                                 tmp_path):
    """Unreadable provenance answers "not ours", not "ours".

    ``_importer_owns`` gates a DROP-and-rewrite of the canonical table. A
    provenance table this module cannot read must therefore fall to the
    protective answer, the same as no provenance table at all — never to
    a guess that some third spelling means the same thing.
    """
    dst = tmp_path / "imported"
    db = fg.run_import(_plan(theirs), str(dst)).db_path
    connection = sqlite3.connect(db)
    try:
        connection.execute('ALTER TABLE "foreign_columns" '
                           'RENAME COLUMN "column" TO "whatever"')
        connection.commit()
        assert fg._provenance_name_column(connection) is None
        assert fg._importer_owns(connection, "cell") is False
        # ...but a table owned by its *name* still is ours.
        assert fg._importer_owns(connection, "foreign_cell") is True
    finally:
        connection.close()
