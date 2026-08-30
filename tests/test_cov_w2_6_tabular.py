"""Every door into a table opens onto the same canonical vocabulary.

These drive real files -- CSV, TSV, sniffed .txt, Parquet, Feather, Excel and
SQLite -- rather than stubbing the pandas readers, because the thing under
test is which reader a suffix picks and what the frame's columns are called
once it comes back.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr import tabular


@pytest.fixture
def legacy_frame():
    """A frame written with the legacy spellings the vocabulary folds."""
    return pd.DataFrame({
        "column_name": ["c1", "c2", "c3"],
        "row_name": ["r1", "r2", "r3"],
        "plate_name": ["pplate1", "pplate1", "pplate1"],
        "value": [1.0, 2.0, 3.0],
    })


# --------------------------------------------------------------------------
# resolve_path / table_format
# --------------------------------------------------------------------------

def test_something_that_is_not_a_path_passes_through_untouched():
    """A caller may hand an open connection to a reader; expanding it would
    turn the object into a string."""
    conn = sqlite3.connect(":memory:")
    try:
        assert tabular.resolve_path(conn) is conn
        assert tabular.resolve_path(17) == 17
    finally:
        conn.close()


def test_a_tilde_is_expanded_before_the_suffix_is_read():
    resolved = tabular.resolve_path("~/measurements.db")
    assert not resolved.startswith("~")
    assert resolved.endswith("measurements.db")


@pytest.mark.parametrize("suffix,expected", [
    (".csv", "csv"), (".tsv", "csv"), (".txt", "csv"),
    (".db", "sqlite"), (".sqlite3", "sqlite"),
    (".parquet", "parquet"), (".pq", "parquet"),
    (".feather", "feather"),
    (".xlsx", "excel"), (".xls", "excel"), (".xlsm", "excel"),
])
def test_the_suffix_alone_decides_which_reader_runs(suffix, expected):
    assert tabular.table_format(f"/tmp/whatever{suffix}") == expected


def test_an_unknown_suffix_is_refused_and_lists_the_known_ones():
    with pytest.raises(tabular.TabularFormatError) as excinfo:
        tabular.table_format("/tmp/results.h5")
    assert ".parquet" in str(excinfo.value)


# --------------------------------------------------------------------------
# delimited text
# --------------------------------------------------------------------------

def test_a_txt_of_unknown_provenance_has_its_separator_sniffed(tmp_path,
                                                               legacy_frame):
    """`.txt` implies no separator, so pandas is asked to work it out. A
    semicolon-delimited export is a real thing users hand spaCR."""
    path = tmp_path / "exported.txt"
    legacy_frame.to_csv(path, sep=";", index=False)
    frame = tabular.read_table(path, report=None)
    assert list(frame.columns) == ["columnID", "rowID", "plateID", "value"]
    assert len(frame) == 3


def test_a_caller_that_names_the_separator_keeps_it(tmp_path, legacy_frame):
    path = tmp_path / "exported.txt"
    legacy_frame.to_csv(path, sep="|", index=False)
    frame = tabular.read_table(path, report=None, sep="|")
    assert list(frame.columns) == ["columnID", "rowID", "plateID", "value"]


def test_reading_without_the_vocabulary_gives_the_header_as_written(
        tmp_path, legacy_frame):
    path = tmp_path / "raw.csv"
    legacy_frame.to_csv(path, index=False)
    frame = tabular.read_table(path, canonicalise=False, report=None)
    assert list(frame.columns) == list(legacy_frame.columns)


# --------------------------------------------------------------------------
# the binary formats
# --------------------------------------------------------------------------

def test_a_parquet_file_reads_through_the_same_vocabulary(tmp_path,
                                                          legacy_frame):
    path = tmp_path / "t.parquet"
    legacy_frame.to_parquet(path, index=False)
    frame = tabular.read_table(path, report=None)
    assert "columnID" in frame.columns and "column_name" not in frame.columns


def test_a_feather_file_reads_through_the_same_vocabulary(tmp_path,
                                                          legacy_frame):
    path = tmp_path / "t.feather"
    legacy_frame.to_feather(path)
    frame = tabular.read_table(path, report=None)
    assert "rowID" in frame.columns


def test_an_excel_workbook_reads_through_the_same_vocabulary(tmp_path,
                                                             legacy_frame):
    path = tmp_path / "t.xlsx"
    legacy_frame.to_excel(path, index=False)
    frame = tabular.read_table(path, report=None)
    assert "plateID" in frame.columns
    # The pplate1 repair applies whichever door the frame came through.
    assert set(frame["plateID"]) == {"plate1"}


def test_a_frame_written_to_excel_comes_back_canonical(tmp_path,
                                                       legacy_frame):
    path = tmp_path / "out.xlsx"
    written = tabular.write_table(legacy_frame, path)
    assert os.path.exists(written)
    assert list(pd.read_excel(written).columns)[:3] == [
        "columnID", "rowID", "plateID"]


def test_a_frame_written_to_feather_comes_back_canonical(tmp_path,
                                                         legacy_frame):
    path = tmp_path / "out.feather"
    tabular.write_table(legacy_frame, path)
    assert "columnID" in pd.read_feather(path).columns


def test_a_database_path_is_refused_by_the_file_writer(tmp_path,
                                                       legacy_frame):
    """`.db` is a table container, not a file format; the message names the
    call that does work."""
    with pytest.raises(tabular.TabularFormatError, match="write_database"):
        tabular.write_table(legacy_frame, tmp_path / "m.db")


# --------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------

@pytest.fixture
def legacy_db(tmp_path, legacy_frame):
    path = tmp_path / "measurements.db"
    conn = sqlite3.connect(path)
    try:
        legacy_frame.to_sql("cells", conn, index=False)
        pd.DataFrame({"columnID": []}).to_sql("empty_table", conn, index=False)
    finally:
        conn.close()
    return path


def test_a_database_read_through_read_table_needs_its_table_named(legacy_db):
    with pytest.raises(ValueError) as excinfo:
        tabular.read_table(legacy_db)
    assert "cells" in str(excinfo.value)


def test_read_table_reads_one_named_table_out_of_a_database(legacy_db):
    frame = tabular.read_table(legacy_db, table="cells", report=None,
                               migrate=False)
    assert list(frame.columns) == ["columnID", "rowID", "plateID", "value"]
    assert len(frame) == 3


def test_a_table_with_no_rows_still_comes_back_with_its_columns(legacy_db):
    """A table that exists but holds no rows is not the same thing as a
    table that is not there, and the columns are what the picker offers."""
    frame, = tabular.read_database(legacy_db, ["empty_table"], report=None,
                                   migrate=False)
    assert len(frame) == 0
    assert list(frame.columns) == ["columnID"]


def test_a_table_larger_than_one_chunk_is_reassembled_whole(legacy_db):
    frame, = tabular.read_database(legacy_db, "cells", report=None,
                                   migrate=False, chunksize=1)
    assert len(frame) == 3
    assert list(frame.index) == [0, 1, 2]


def test_a_limit_stops_the_read_at_that_many_rows(legacy_db):
    frame, = tabular.read_database(legacy_db, "cells", report=None,
                                   migrate=False, limit=2)
    assert len(frame) == 2


def test_a_missing_table_is_refused_by_name(legacy_db):
    with pytest.raises(ValueError, match="nowhere"):
        tabular.read_database(legacy_db, ["nowhere"], report=None,
                              migrate=False)


def test_a_table_name_that_is_not_an_identifier_is_refused(legacy_db):
    with pytest.raises(ValueError, match="Invalid table name"):
        tabular.read_database(legacy_db, [None], report=None, migrate=False)
    with pytest.raises(ValueError, match="Invalid table name"):
        tabular.read_database(legacy_db, [""], report=None, migrate=False)


def test_a_read_only_open_cannot_also_migrate(legacy_db):
    """A migration writes, so the two options are refused together rather
    than one silently winning over the user's database."""
    with pytest.raises(ValueError, match="cannot migrate"):
        tabular.read_database(legacy_db, "cells", read_only=True,
                              migrate=True, report=None)


def test_a_frame_written_to_a_database_arrives_canonical(tmp_path,
                                                         legacy_frame):
    db = tmp_path / "written.db"
    tabular.write_database(legacy_frame, db, "cells", if_exists="replace")
    conn = sqlite3.connect(db)
    try:
        names = [r[1] for r in conn.execute("PRAGMA table_info(cells)")]
    finally:
        conn.close()
    assert names == ["columnID", "rowID", "plateID", "value"]


def test_writing_verbatim_keeps_the_header_a_caller_owes_an_external_format(
        tmp_path, legacy_frame):
    db = tmp_path / "verbatim.db"
    tabular.write_database(legacy_frame, db, "cells", if_exists="replace",
                           canonicalise=False)
    conn = sqlite3.connect(db)
    try:
        names = [r[1] for r in conn.execute("PRAGMA table_info(cells)")]
    finally:
        conn.close()
    assert names[0] == "column_name"


# --------------------------------------------------------------------------
# table_columns -- what the picker offers
# --------------------------------------------------------------------------

def test_the_picker_needs_a_table_name_for_a_database(legacy_db):
    with pytest.raises(ValueError, match="table_columns needs"):
        tabular.table_columns(legacy_db)


def test_the_picker_offers_the_names_the_run_will_find(legacy_db):
    assert tabular.table_columns(legacy_db, table="cells") == (
        "columnID", "rowID", "plateID", "value")


def test_the_picker_reads_a_workbook_without_reading_its_rows(tmp_path,
                                                              legacy_frame):
    path = tmp_path / "t.xlsx"
    legacy_frame.to_excel(path, index=False)
    assert tabular.table_columns(path) == (
        "columnID", "rowID", "plateID", "value")


def test_the_picker_reads_a_parquet_header(tmp_path, legacy_frame):
    path = tmp_path / "t.parquet"
    legacy_frame.to_parquet(path, index=False)
    assert tabular.table_columns(path) == (
        "columnID", "rowID", "plateID", "value")


def test_a_collapsed_duplicate_is_not_offered_twice(tmp_path):
    """Offering both `well` and `wellID` would let a user choose a column the
    run will not find."""
    path = tmp_path / "dupes.csv"
    pd.DataFrame({"well": ["A01"], "well_name": ["A01"],
                  "value": [1]}).to_csv(path, index=False)
    names = tabular.table_columns(path)
    assert names.count("wellID") == 1
    assert "well_name" not in names


def test_listing_the_tables_does_not_rewrite_the_database(legacy_db):
    before = os.path.getmtime(legacy_db)
    assert tabular.database_tables(legacy_db) == ("cells", "empty_table")
    assert os.path.getmtime(legacy_db) == before


def test_a_csv_write_uses_the_separator_its_suffix_implies(tmp_path,
                                                           legacy_frame):
    path = tmp_path / "out.tsv"
    tabular.write_table(legacy_frame, path)
    assert "\t" in path.read_text().splitlines()[0]
    assert path.read_text().splitlines()[0].split("\t")[0] == "columnID"


def test_a_write_creates_the_directory_it_was_pointed_at(tmp_path,
                                                         legacy_frame):
    path = tmp_path / "deep" / "deeper" / "out.csv"
    written = tabular.write_table(legacy_frame, path)
    assert os.path.exists(written)


def test_a_frame_written_to_parquet_comes_back_canonical(tmp_path,
                                                         legacy_frame):
    path = tmp_path / "out.parquet"
    tabular.write_table(legacy_frame, path)
    assert list(pd.read_parquet(path).columns)[0] == "columnID"


def test_opening_with_migrate_repairs_a_legacy_measurements_schema(tmp_path):
    """A legacy database is repaired by any reader that opens it, rather
    than only by the one that remembered to call the migration."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE png_list (png_path TEXT)")
        conn.commit()
    finally:
        conn.close()
    frame, = tabular.read_database(db, "png_list", report=None, migrate=True)
    assert "png_path" in frame.columns


def test_a_read_only_open_cannot_write_to_the_users_database(legacy_db):
    """A merge reads several of the user's measurement databases at once and
    must not be able to write to any of them."""
    conn = tabular._connect(legacy_db, migrate=False, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE scribble (x INTEGER)")
    finally:
        conn.close()
    frame, = tabular.read_database(legacy_db, "cells", report=None,
                                   migrate=False, read_only=True)
    assert len(frame) == 3


def test_a_reader_that_yields_no_chunks_still_names_the_columns(
        legacy_db, monkeypatch):
    """Not every pandas yields an empty chunk for an empty result. When it
    yields none at all the frame must still carry the table's columns, or a
    picker offers nothing for a table that has columns."""
    real = pd.read_sql_query

    def _no_chunks(sql, con, *args, **kwargs):
        if kwargs.get("chunksize"):
            return iter(())
        return real(sql, con, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql_query", _no_chunks)
    frame, = tabular.read_database(legacy_db, "cells", report=None,
                                   migrate=False)
    assert len(frame) == 0
    assert list(frame.columns) == ["columnID", "rowID", "plateID", "value"]


def test_a_column_spelled_without_the_id_suffix_arrives_canonicalised(tmp_path):
    """The property several callers are written to depend on.

    ``spacr.submodules.group_cv_score`` groups on ``columnID`` and used to
    carry an ``elif 'column'`` fallback beneath it. Instruction 145 made
    ``read_table`` canonicalise, which turned that fallback into unreachable
    code -- so it was deleted, and this is what makes the deletion safe.

    If canonicalisation ever stops renaming ``column``, this fails here rather
    than as a silent empty grouping three modules away.
    """
    import pandas as pd

    from spacr.tabular import read_table

    source = tmp_path / "wells.csv"
    pd.DataFrame({"plateID": ["p1", "p1"], "rowID": ["r1", "r2"],
                  "column": ["c1", "c2"], "pred": [0.2, 0.8]}
                 ).to_csv(source, index=False)

    frame = read_table(str(source))

    assert "columnID" in frame.columns
    assert "column" not in frame.columns
    assert list(frame["columnID"]) == ["c1", "c2"]


def test_the_canonical_name_is_kept_when_it_is_already_canonical(tmp_path):
    """A file written by spaCR itself must round-trip unchanged.

    Renaming an already-canonical column would be a rename loop, and the
    assertion above cannot see the difference on its own.
    """
    import pandas as pd

    from spacr.tabular import read_table

    source = tmp_path / "wells.csv"
    pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
                  "pred": [0.5]}).to_csv(source, index=False)

    frame = read_table(str(source))

    assert list(frame.columns) == ["plateID", "rowID", "columnID", "pred"]
