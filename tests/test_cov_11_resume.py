"""A resume that cannot read something treats it as work still to do.

Every refusal in :mod:`spacr.resume` points the same way: when the module
cannot PROVE a field finished, it plans to run it again. Re-measuring a field
costs time; skipping a half-written one loses data permanently. These tests
drive the paths where the proof is unavailable -- a header that will not
parse, a dtype whose size is unknown, a table that will not answer, a
settings file that cannot be opened -- and check that each one lands on
"pending" rather than on an exception or a false "done".
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct

import numpy as np
import pytest

import spacr.resume as resume
from spacr.resume import (
    REASON_PARTIAL_DB,
    REASON_TOO_FEW_PLANES,
    REASON_UNREADABLE,
    ResumeState,
    _descr_itemsize,
    _fields_matching,
    _has_rows,
    _list_tables,
    _normalize,
    _sample_fields,
    _table_columns,
    _values_equal,
    completed_fields_in_db,
    completed_fields_in_merged,
    format_resume,
    identity_to_prcf,
    read_npy_header,
    read_recorded_settings,
    run_already_complete,
    validate_merged_field,
)

_MAGIC = b'\x93NUMPY'


# ---------------------------------------------------------------------------
# Field identity
# ---------------------------------------------------------------------------

def test_a_timelapse_identity_carries_its_timepoint():
    """Two timepoints of one field are two fields, so the key must differ.

    Dropping the timepoint would let one measured frame mark the whole
    series done.
    """
    base = {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1"}

    assert identity_to_prcf(base) == "p1_r1_c1_f1"
    assert identity_to_prcf({**base, "timeID": "t3"}) == "p1_r1_c1_f1_t3"
    assert identity_to_prcf({**base, "time_id": "t7"}) == "p1_r1_c1_f1_t7"


# ---------------------------------------------------------------------------
# Reading a dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("descr", [None, 5, (), "", "<", ">", "|O", "|V12x"])
def test_a_dtype_whose_size_is_unknown_says_so_rather_than_guessing(descr):
    """An unmeasurable dtype returns None, which the caller reads as pending.

    Guessing a size would produce an expected byte count, and a file that
    matched it by accident would be declared finished.
    """
    assert _descr_itemsize(descr) is None


def test_a_boolean_dtype_is_one_byte():
    """The two dtypes that carry no digit still have a known size."""
    assert _descr_itemsize("|?") == 1
    assert _descr_itemsize("|b") == 1


# ---------------------------------------------------------------------------
# Reading a .npy header
# ---------------------------------------------------------------------------

def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_a_file_that_stops_inside_its_header_length_is_refused(tmp_path):
    """One byte of a two-byte length field is a crash scar, not a header."""
    path = _write(tmp_path, "short_len.npy", _MAGIC + b"\x01\x00" + b"\x10")

    with pytest.raises(ValueError, match="header length"):
        read_npy_header(path)


def test_a_header_shorter_than_it_claims_is_refused(tmp_path):
    """The header says 118 bytes and the file holds four: it never finished."""
    path = _write(tmp_path, "short_header.npy",
                  _MAGIC + b"\x01\x00" + struct.pack("<H", 118) + b"{'sh")

    with pytest.raises(ValueError, match="ends inside its header"):
        read_npy_header(path)


def test_a_header_that_is_not_python_is_refused(tmp_path):
    """A header of garbage means the bytes after it are garbage too."""
    body = b"this is not a dict at all      \n"
    path = _write(tmp_path, "junk.npy",
                  _MAGIC + b"\x01\x00" + struct.pack("<H", len(body)) + body)

    with pytest.raises(ValueError, match="unparseable"):
        read_npy_header(path)


def test_a_header_that_is_not_a_shape_dict_is_refused(tmp_path):
    """A parseable header with no shape cannot say how long the file should be."""
    body = b"[1, 2, 3]                      \n"
    path = _write(tmp_path, "listheader.npy",
                  _MAGIC + b"\x01\x00" + struct.pack("<H", len(body)) + body)

    with pytest.raises(ValueError, match="not a shape dict"):
        read_npy_header(path)


# ---------------------------------------------------------------------------
# Validating a merged field
# ---------------------------------------------------------------------------

def test_a_field_that_cannot_be_opened_is_pending_not_done(tmp_path):
    """An unreadable file is never evidence that measuring finished.

    A permission problem must send the field back into the run rather than
    letting it be skipped on the strength of its name.
    """
    path = tmp_path / "unreadable.npy"
    np.save(path, np.zeros((4, 4, 3), dtype="uint16"))
    path.chmod(0o000)
    try:
        ok, reason = validate_merged_field(str(path))
    finally:
        path.chmod(0o644)

    assert ok is False
    assert reason == REASON_UNREADABLE


def test_a_field_whose_dtype_cannot_be_sized_is_pending_not_done(tmp_path):
    """Completeness is proved by byte count, and an object array has none."""
    path = tmp_path / "objects.npy"
    np.save(path, np.array([{"a": 1}, None], dtype=object), allow_pickle=True)

    ok, reason = validate_merged_field(str(path))

    assert ok is False
    assert reason == REASON_UNREADABLE


def test_a_field_whose_header_vanishes_mid_scan_is_re_run(
        tmp_path, monkeypatch):
    """A field validated and then unreadable falls back into the run.

    The folder scan reads each header twice -- once to validate, once for
    the plane count -- and another process can remove the file in between.
    The second read failing must not abort the scan; the field simply
    contributes no plane count, which puts it below its neighbours and so
    back on the list to measure.
    """
    for index in range(3):
        np.save(tmp_path / f"p1_A0{index}_1.npy",
                np.zeros((4, 4, 3), dtype="uint16"))

    real = resume.read_npy_header
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:      # the plane-count read of the first field
            raise OSError("the file went away")
        return real(path)

    monkeypatch.setattr(resume, "read_npy_header", flaky)

    reasons = {}
    done = completed_fields_in_merged(str(tmp_path), reasons=reasons)

    assert done == {"p1_A01_1", "p1_A02_1"}
    assert reasons == {"p1_A00_1": REASON_TOO_FEW_PLANES}


# ---------------------------------------------------------------------------
# A database that will not answer
# ---------------------------------------------------------------------------

@pytest.fixture()
def measured_db(tmp_path):
    """A two-table database with one field measured in both and one in one."""
    path = str(tmp_path / "measurements.db")
    conn = sqlite3.connect(path)
    for table in ("cell", "nucleus"):
        conn.execute(f'CREATE TABLE "{table}" (plateID TEXT, rowID TEXT, '
                     f'columnID TEXT, fieldID TEXT, timeID TEXT, area REAL)')
    conn.execute('INSERT INTO "cell" VALUES ("p1","r1","c1","f1","t1",1.0)')
    conn.execute('INSERT INTO "cell" VALUES ("p1","r1","c1","f2","t1",1.0)')
    conn.execute('INSERT INTO "nucleus" VALUES ("p1","r1","c1","f1","t1",2.0)')
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def closed_connection(tmp_path):
    """A connection that has been closed, so every statement on it raises."""
    path = str(tmp_path / "gone.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cell (a TEXT)")
    conn.commit()
    conn.close()
    return conn


def test_a_connection_that_will_not_answer_reports_nothing_rather_than_raising(
        closed_connection):
    """The three database probes are how a resume looks around; none may throw.

    A resume that dies inspecting a database leaves the user with no plan at
    all, where an empty answer simply means "measure everything".
    """
    assert _table_columns(closed_connection, "cell") == []
    assert _has_rows(closed_connection, "cell") is False
    assert _list_tables(closed_connection) == []


def test_provenance_that_disappears_mid_read_is_unreadable_not_absent(
        tmp_path, monkeypatch):
    """A provenance table lost between the check and the read answers None.

    None is the protective answer: it means "this database cannot say whose
    rows these are", which keeps the ownership test off the branch that
    claims every row for the measure stage and rewrites a collaborator's
    imported table.
    """
    path = str(tmp_path / "prov.db")
    writer = sqlite3.connect(path)
    writer.execute(f'CREATE TABLE "{resume.FOREIGN_COLUMNS_TABLE}" '
                   f'("table" TEXT, "column" TEXT)')
    writer.execute(f'INSERT INTO "{resume.FOREIGN_COLUMNS_TABLE}" '
                   f'VALUES ("cell", "area")')
    writer.commit()

    reader = sqlite3.connect(path)
    assert resume.importer_recorded_columns(reader, "cell") == {"area"}

    real = resume._foreign_name_column

    def answer_then_lose_the_table(conn):
        name = real(conn)
        writer.execute(f'DROP TABLE "{resume.FOREIGN_COLUMNS_TABLE}"')
        writer.commit()
        return name

    monkeypatch.setattr(resume, "_foreign_name_column",
                        answer_then_lose_the_table)
    try:
        assert resume.importer_recorded_columns(reader, "cell") is None
    finally:
        reader.close()
        writer.close()


def test_a_table_that_cannot_be_described_is_left_out_of_the_plan(
        tmp_path, measured_db):
    """A table whose columns cannot be listed is skipped, not fatal.

    A view over a dropped table answers nothing about which fields were
    measured, and the tables that CAN answer still decide the plan.
    """
    conn = sqlite3.connect(measured_db)
    conn.execute("CREATE TABLE scratch (plateID TEXT, rowID TEXT, "
                 "columnID TEXT, fieldID TEXT)")
    conn.execute("CREATE VIEW pathogen AS SELECT plateID, rowID, columnID, "
                 "fieldID FROM scratch")
    conn.commit()
    conn.execute("DROP TABLE scratch")
    conn.commit()
    conn.close()

    done = completed_fields_in_db(measured_db,
                                  tables=["cell", "nucleus", "pathogen"])

    assert done == {"p1_r1_c1_f1"}


def test_a_table_whose_rows_cannot_be_selected_does_not_sink_the_plan(
        measured_db, monkeypatch):
    """One unqueryable table costs that table's opinion, not the resume.

    The ownership clause is built from what an import recorded, and a clause
    the table cannot evaluate would otherwise abort a plan the remaining
    tables can still produce.
    """
    def clause_for(_conn, table):
        return "gone_column IS NOT NULL" if table == "nucleus" else None

    monkeypatch.setattr(resume, "measure_rows_clause", clause_for)

    done = completed_fields_in_db(measured_db, tables=["cell", "nucleus"])

    assert done == {"p1_r1_c1_f1", "p1_r1_c1_f2"}


def test_only_bookkeeping_tables_means_nothing_is_measured(measured_db):
    """`settings` and `run_status` hold no fields, so they prove nothing."""
    assert completed_fields_in_db(measured_db,
                                  tables=["settings", "run_status"]) == set()


def test_a_field_measured_in_one_table_of_two_is_partial_not_done(measured_db):
    """A field with cells and no nuclei must be cleared and re-run.

    Calling it done loses its nuclei forever; calling it pending without
    saying it is partial duplicates its cells.
    """
    partial = {}
    done = completed_fields_in_db(measured_db, tables=["cell", "nucleus"],
                                  partial=partial)

    assert done == {"p1_r1_c1_f1"}
    assert partial == {"p1_r1_c1_f2": REASON_PARTIAL_DB}


def test_a_timelapse_field_is_keyed_by_its_timepoint(measured_db):
    """With timelapse on, the timepoint column joins the key."""
    done = completed_fields_in_db(measured_db, tables=["cell", "nucleus"],
                                  timelapse=True)

    assert done == {"p1_r1_c1_f1_t1"}


def test_a_timelapse_run_over_a_table_with_no_time_column_still_keys_the_field(
        tmp_path):
    """A table predating timelapse support contributes an empty timepoint.

    Its fields still line up with the timestamped tables' keys instead of
    being dropped, which would make every field read as partial.
    """
    path = str(tmp_path / "measurements.db")
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE "cell" (plateID TEXT, rowID TEXT, '
                 'columnID TEXT, fieldID TEXT, area REAL)')
    conn.execute('INSERT INTO "cell" VALUES ("p1","r1","c1","f1",1.0)')
    conn.commit()
    conn.close()

    done = completed_fields_in_db(path, tables=["cell"], timelapse=True)

    assert done == {"p1_r1_c1_f1"}


def test_matching_fields_with_an_unusable_clause_matches_nothing(measured_db):
    """A condition the table cannot evaluate selects no fields, and no error."""
    conn = sqlite3.connect(measured_db)
    try:
        assert _fields_matching(conn, "cell", "no_such_column IS NOT NULL") \
            == set()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def test_a_long_field_list_is_sampled_and_says_how_many_were_left_out():
    """A message naming a thousand fields is a message nobody reads."""
    text = _sample_fields([f"p1_r1_c{i}_f1" for i in range(9)], limit=4)

    assert text.startswith("p1_r1_c0_f1, p1_r1_c1_f1, p1_r1_c2_f1, p1_r1_c3_f1")
    assert "(+5 more)" in text


def test_the_report_counts_the_fields_it_verified():
    """`n_done` is what the printed plan reports as already complete."""
    state = ResumeState(total=3, done=("a", "b"), pending=("c",),
                        skipped=("a", "b"))

    assert state.n_done == 2


def test_the_report_names_released_import_rows_and_where_they_went():
    """Rows taken out of a canonical table must be accounted for in the plan.

    Without this line a user sees a table shrink between runs with nothing
    saying the rows are still in the foreign copy.
    """
    text = format_resume(ResumeState(total=1, pending=("p1_r1_c1_f1",),
                                     released_rows=12))

    assert "released" in text
    assert "12 row(s)" in text
    assert "foreign_" in text


# ---------------------------------------------------------------------------
# Run stamps and recorded settings
# ---------------------------------------------------------------------------

def test_an_unreadable_run_stamp_means_go_and_check_the_files(monkeypatch):
    """"Nobody could say" must never be read as "the run finished"."""
    def refuse(_path):
        raise RuntimeError("the ledger cannot be read")

    monkeypatch.setattr(resume, "read_run_status", refuse)

    assert run_already_complete("anything.db") is False


def test_a_run_journal_folder_yields_the_settings_it_recorded(tmp_path):
    """A journal directory is one of the three shapes a resume can read."""
    (tmp_path / "settings.json").write_text(
        json.dumps({"cell_mask_dim": 4, "channels": [0, 1]}))

    recorded = read_recorded_settings(str(tmp_path))

    assert recorded["cell_mask_dim"] == 4
    assert recorded["channels"] == [0, 1]


def test_a_journal_folder_that_cannot_be_read_is_no_information(tmp_path):
    """An unreadable journal reads as "nothing recorded", not as "matches".

    Treating it as a match would let a resume continue across a settings
    change it never saw.
    """
    (tmp_path / "settings.json").write_text("{not json at all")

    assert read_recorded_settings(str(tmp_path)) == {}


def test_a_settings_csv_that_cannot_be_opened_is_no_information(tmp_path):
    """Same rule for the CSV shape `save_settings` writes."""
    path = tmp_path / "measure_crop_settings.csv"
    path.write_text("Key,Value\ncell_mask_dim,4\n")
    path.chmod(0o000)
    try:
        assert read_recorded_settings(str(path)) == {}
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("body", ["{not json", ""])
def test_a_settings_json_that_will_not_parse_is_no_information(tmp_path, body):
    """A truncated settings.json must not silently authorise a resume."""
    path = tmp_path / "settings.json"
    path.write_text(body)

    assert read_recorded_settings(str(path)) == {}


# ---------------------------------------------------------------------------
# Comparing settings without the journal
# ---------------------------------------------------------------------------

def test_normalising_a_value_falls_back_to_the_value_itself(monkeypatch):
    """Without the journal's rules, the raw value is still comparable.

    The journal is imported lazily so this module stays stdlib-only; when
    that import is unavailable the comparison degrades to identity rather
    than failing the resume.
    """
    import builtins

    real_import = builtins.__import__

    def block(name, *args, **kwargs):
        if name.endswith("run_journal"):
            raise ImportError("no run journal here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block)

    assert _normalize([0, 1, 2]) == [0, 1, 2]
    assert _values_equal([0, 1], [0, 1]) is True
    assert _values_equal([0, 1], [0, 2]) is False


def test_a_value_the_journal_chokes_on_falls_back_to_its_repr(monkeypatch):
    """An uncanonicalisable value still compares equal to an identical one."""
    from spacr import run_journal

    def refuse(_value):
        raise TypeError("this cannot be canonicalised")

    monkeypatch.setattr(run_journal, "_normalize_value", refuse)

    assert _normalize({"a": 1}) == repr({"a": 1})
