"""One reader, one writer, one key vocabulary (instruction 145).

The reported symptom, which `test_the_picker_shows_columnID_not_column_name`
reproduces: `filter_column=columnID` worked, because some path downstream
renamed on the way to the fit, while the CSV picker read the RAW header and
offered `column_name` and `column` -- the names a user must not have to know.

Everything here is about there being exactly ONE place that decides what a
column is called, and one door every read goes through to get there.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from spacr import schema, tabular


PACKAGE = Path(schema.__file__).parent


def test_the_checks_read_this_tree():
    assert "/codex/repo/spacr/" in schema.__file__, schema.__file__
    assert "/codex/repo/spacr/" in tabular.__file__, tabular.__file__


# ---------------------------------------------------------------------------
# A. the vocabulary, in one place
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", [
    "plate", "Plate", "PLATE", "plateid", "plateID", "plate_id", "Plate_ID",
    "plate_name", "plateName", "plate name", "plate.id",
])
def test_every_spelling_of_the_plate_is_one_name(spelling):
    assert schema.canonical_column_name(spelling) == schema.PLATE_KEY


@pytest.mark.parametrize("spelling,canonical", [
    ("row", "rowID"), ("ROW", "rowID"), ("Row", "rowID"),
    ("rowid", "rowID"), ("row_id", "rowID"), ("row_name", "rowID"),
    ("column", "columnID"), ("COLUMN", "columnID"), ("Column", "columnID"),
    ("col", "columnID"), ("columnid", "columnID"),
    ("column_id", "columnID"), ("column_name", "columnID"),
    ("field", "fieldID"), ("FIELD", "fieldID"), ("Field", "fieldID"),
    ("fieldid", "fieldID"), ("field_id", "fieldID"),
    ("field_name", "fieldID"),
    ("well", "wellID"), ("WELL", "wellID"), ("Well", "wellID"),
    ("wellid", "wellID"), ("well_id", "wellID"), ("well_name", "wellID"),
])
def test_the_five_keys_the_request_named(spelling, canonical):
    assert schema.canonical_column_name(spelling) == canonical


@pytest.mark.parametrize("name", [
    "c", "r", "p", "f", "w", "cell_area", "columns", "rows", "plates",
    "well_area", "row_std", "column_of_interest",
])
def test_the_vocabulary_does_not_invent_members(name):
    """A one-letter alias would rename a measurement into a plate key.

    `col` is real and spaCR wrote it. `c` is not, and a measurement column
    called `c` silently renamed to `columnID` corrupts the join it lands in
    while a name that is merely un-normalised is visible in the picker.
    """
    assert schema.canonical_column_name(name) == name


def test_a_folded_alias_cannot_mean_two_things():
    """The import-time guard, driven rather than trusted."""
    folded = {}
    for alias, canonical in schema.LEGACY_COLUMN_NAMES.items():
        key = schema.fold_column_name(alias)
        assert folded.setdefault(key, canonical) == canonical, alias


def test_a_csv_of_the_five_legacy_names_reads_back_canonical(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame({"Plate": ["plate1"], "Row": ["r1"], "Column": ["c3"],
                  "Field": ["f1"], "Well": ["A03"],
                  "cell_area": [12.5]}).to_csv(path, index=False)
    frame = tabular.read_table(path)
    assert list(frame.columns) == [
        "plateID", "rowID", "columnID", "fieldID", "wellID", "cell_area"]


# ---------------------------------------------------------------------------
# B. the collision rule
# ---------------------------------------------------------------------------

def test_agreeing_columns_collapse_with_a_print_and_no_warning(capsys):
    frame = pd.DataFrame({"well": ["A01", "A02"], "wellID": ["A01", "A02"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = schema.canonicalise_frame(frame, report=print)
    assert list(out.columns) == ["wellID"]
    printed = capsys.readouterr().out
    assert "well" in printed and "wellID" in printed
    assert "do not agree" not in printed


def test_disagreeing_columns_warn_and_name_the_columns_and_the_row_count():
    frame = pd.DataFrame({"well": ["A01", "A02", "A03", "A04"],
                          "wellID": ["A01", "B02", "C03", "A04"]})
    with pytest.warns(UserWarning) as record:
        out = schema.canonicalise_frame(frame, report=None)
    message = str(record[0].message)
    assert list(out.columns) == ["wellID"]
    assert "well" in message and "wellID" in message
    assert "2 of 4 rows differ" in message
    assert "wellID is being used" in message


def test_a_dtype_difference_is_not_a_disagreement():
    """`1`, `1.0` and `' 1 '` are the same well.

    A naive `.equals()` warns on every file that stored one copy as text and
    the other as a number, which teaches the user to ignore the warning that
    matters.
    """
    frame = pd.DataFrame({"columnID": [1, 2, 3],
                          "column_name": [" 1 ", "2.0", "03"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = schema.canonicalise_frame(frame, report=None)
    assert list(out.columns) == ["columnID"]
    assert list(out["columnID"]) == [1, 2, 3]


def test_the_canonical_spelling_wins_when_one_is_present():
    frame = pd.DataFrame({"well_name": ["A01"], "wellID": ["A01"],
                          "well": ["A01"]})
    _, collisions = schema.resolve_metadata_collisions(frame, report=None)
    assert collisions[0].chosen == "wellID"
    assert set(collisions[0].dropped) == {"well_name", "well"}


def test_when_none_is_canonical_one_is_chosen_silently_and_renamed(capsys):
    frame = pd.DataFrame({"well": ["A01"], "well_name": ["A01"]})
    out = schema.canonicalise_frame(frame, report=print)
    assert list(out.columns) == ["wellID"]
    assert "well" in capsys.readouterr().out


def test_only_metadata_keys_collapse_never_a_measurement():
    """Two feature spellings keep both columns; a measurement is data."""
    frame = pd.DataFrame({"cell_periphery_25_percentile": [1.0],
                          "cell_periphery_percentile_25": [2.0]})
    out = schema.canonicalise_frame(frame, report=None)
    assert len(out.columns) == 2


def test_two_columns_literally_named_rowID_are_repaired_not_raised():
    """pandas allows duplicate labels; `to_sql` refuses them."""
    frame = pd.DataFrame([[1, 1]], columns=["rowID", "rowID"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = schema.canonicalise_frame(frame, report=None)
    assert list(out.columns) == ["rowID"]


def test_a_collision_is_recorded_on_the_frame_as_well_as_reported():
    frame = pd.DataFrame({"well": ["A01"], "wellID": ["A01"]})
    out = schema.canonicalise_frame(frame, report=None)
    recorded = out.attrs["column_collisions"]
    assert len(recorded) == 1 and recorded[0].canonical == "wellID"
    assert recorded[0].agreed


# ---------------------------------------------------------------------------
# C. one reader, one writer
# ---------------------------------------------------------------------------

def test_a_tilde_path_is_expanded_by_the_reader(tmp_path, monkeypatch):
    """GitHub issue #108, held at the funnel rather than at ~99 call sites."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1))
    (tmp_path / "sub").mkdir()
    pd.DataFrame({"Plate": ["plate1"]}).to_csv(tmp_path / "sub/x.csv",
                                               index=False)
    frame = tabular.read_table("~/sub/x.csv", report=None)
    assert list(frame.columns) == ["plateID"]


def test_an_environment_variable_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_TEST_ROOT", str(tmp_path))
    pd.DataFrame({"Row": ["r1"]}).to_csv(tmp_path / "y.csv", index=False)
    frame = tabular.read_table("$SPACR_TEST_ROOT/y.csv", report=None)
    assert list(frame.columns) == ["rowID"]


def test_the_schema_migration_still_runs_on_open(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame({"plateID": ["plate1"], "rowID": ["r1"],
                      "columnID": ["c1"], "fieldID": ["f1"],
                      "prcf": ["plate1_r1_c1_f1"],
                      "object_label": [1]}).to_sql("cell", conn, index=False)
    from spacr import database_schema
    calls = []
    original = database_schema.ensure_database_schema

    def spy(path, **kwargs):
        calls.append(str(path))
        return original(path, **kwargs)

    database_schema.ensure_database_schema = spy
    try:
        tabular.read_database(db, ["cell"], report=None)
    finally:
        database_schema.ensure_database_schema = original
    assert calls and calls[0].endswith("measurements.db")


def test_a_read_only_read_cannot_migrate():
    with pytest.raises(ValueError, match="read_only"):
        tabular._connect("/tmp/nope.db", migrate=True, read_only=True)


def test_a_database_read_without_a_table_says_which_tables_there_are(tmp_path):
    db = tmp_path / "m.db"
    tabular.write_database(pd.DataFrame({"a": [1]}), db, "cell",
                           if_exists="replace")
    with pytest.raises(ValueError, match="cell"):
        tabular.read_table(db)


def test_a_missing_table_names_itself(tmp_path):
    db = tmp_path / "m.db"
    tabular.write_database(pd.DataFrame({"a": [1]}), db, "cell",
                           if_exists="replace")
    with pytest.raises(ValueError, match="nucleus"):
        tabular.read_database(db, ["nucleus"], migrate=False)


def test_an_unknown_suffix_is_refused_by_name(tmp_path):
    with pytest.raises(tabular.TabularFormatError, match=r"\.h5"):
        tabular.read_table(tmp_path / "x.h5")


# pyarrow 24 deprecates the writer pandas' `to_feather` calls; that is the
# installed stack's business, not this module's.
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".parquet", ".feather"])
def test_a_round_trip_keeps_the_canonical_names(tmp_path, suffix):
    frame = pd.DataFrame({"Plate": ["plate1"], "Column": [3],
                          "cell_area": [1.5]})
    path = tmp_path / f"t{suffix}"
    tabular.write_table(frame, path)
    back = tabular.read_table(path, report=None)
    assert list(back.columns) == ["plateID", "columnID", "cell_area"]


def test_the_writer_writes_canonical_names(tmp_path):
    """The deliberate half of the trade, pinned so it cannot drift back."""
    path = tmp_path / "out.csv"
    tabular.write_table(pd.DataFrame({"column_name": [1]}), path)
    assert path.read_text().splitlines()[0] == "columnID"


def test_the_writer_can_be_told_to_leave_a_foreign_header_alone(tmp_path):
    path = tmp_path / "out.csv"
    tabular.write_table(pd.DataFrame({"column_name": [1]}), path,
                        canonicalise=False)
    assert path.read_text().splitlines()[0] == "column_name"


def test_write_table_refuses_a_database_path_and_says_what_to_use(tmp_path):
    with pytest.raises(tabular.TabularFormatError, match="write_database"):
        tabular.write_table(pd.DataFrame({"a": [1]}), tmp_path / "m.db")


def test_the_reader_repairs_a_doubled_plate_prefix(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame({"plate": ["pplate1"],
                  "prc": ["pplate1_r1_c1"]}).to_csv(path, index=False)
    frame = tabular.read_table(path, report=None)
    assert list(frame["plateID"]) == ["plate1"]
    assert list(frame["prc"]) == ["plate1_r1_c1"]


# ---------------------------------------------------------------------------
# D. what the user sees
# ---------------------------------------------------------------------------

def test_the_picker_shows_columnID_not_column_name(tmp_path):
    """THE REPORTED BUG. The picker read the raw header; the run did not."""
    path = tmp_path / "scores.csv"
    pd.DataFrame({"column_name": [3], "column": [3],
                  "gene": ["g1"]}).to_csv(path, index=False)
    offered = tabular.table_columns(path)
    assert "columnID" in offered
    assert "column_name" not in offered and "column" not in offered


def test_the_picker_never_offers_a_column_the_run_will_not_find(tmp_path):
    """A collapsed duplicate is listed ONCE, not as a second selectable entry."""
    path = tmp_path / "scores.csv"
    pd.DataFrame({"well": ["A01"], "wellID": ["A01"]}).to_csv(path,
                                                              index=False)
    offered = tabular.table_columns(path)
    assert offered.count("wellID") == 1
    assert set(offered) <= set(tabular.read_table(path, report=None).columns)


def test_the_sql_column_list_is_canonical_too(tmp_path):
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame({"column_name": [3]}).to_sql("scores", conn, index=False)
    assert tabular.table_columns(db, table="scores") == ("columnID",)
    assert tabular.database_tables(db) == ("scores",)


# ---------------------------------------------------------------------------
# The ratchet, and the "no second implementation" guards
# ---------------------------------------------------------------------------

#: Measured 2026-08-18 after the first batch of readers moved onto the funnel:
#: 96. The ceiling is 100 rather than 96 so a concurrent worker adding a
#: reader does not turn this red on somebody else's change -- but it is a
#: RATCHET: it may only ever be lowered, never raised, and it is already
#: below the 101 the instruction set as the bar.
DIRECT_READER_CEILING = 100

_DIRECT_READER = re.compile(r"pd\.read_csv\(|pd\.read_sql")


def _direct_reader_sites():
    sites = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "tabular.py":
            continue          # the funnel is allowed to be the funnel
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if _DIRECT_READER.search(line):
                sites.append(f"{path.relative_to(PACKAGE.parent)}:{number}")
    return sites


def test_the_number_of_direct_readers_only_falls():
    sites = _direct_reader_sites()
    assert len(sites) <= DIRECT_READER_CEILING, (
        f"{len(sites)} direct pd.read_csv / pd.read_sql call sites, ceiling "
        f"{DIRECT_READER_CEILING}. Move a reader onto spacr.tabular and "
        f"LOWER the ceiling; it is never raised.\n" + "\n".join(sites))


def test_none_of_the_owned_modules_reads_a_table_directly():
    """The batch that has moved, held so it cannot move back."""
    owned = ("schema.py", "hits.py", "ml.py", "multi_database.py",
             "plate_measurements.py", "utils.py")
    offenders = [site for site in _direct_reader_sites()
                 if os.path.basename(site.split(":")[0]) in owned]
    assert offenders == [], offenders


def test_there_is_not_a_second_plate_id_normaliser():
    """`multi_database` re-exports rather than redefining."""
    from spacr import multi_database
    assert multi_database.canonical_plate_id is schema.canonical_plate_id
    assert multi_database.PLATE_BEARING_COLUMNS is schema.PLATE_BEARING_COLUMNS
    source = (PACKAGE / "multi_database.py").read_text(encoding="utf-8")
    assert 'text.startswith("pp")' not in source
    assert "startswith('pp')" not in source


def test_the_frame_normalisers_all_answer_the_same():
    """`utils.correct_metadata` and the schema funnel cannot disagree."""
    from spacr.utils import correct_metadata
    raw = {"plate": ["pplate1"], "Row": ["r1"], "column_name": [3],
           "Field": ["f1"], "cell_area": [2.0]}
    through_utils = correct_metadata(pd.DataFrame(raw))
    through_schema = schema.canonicalise_frame(pd.DataFrame(raw), report=None)
    pd.testing.assert_frame_equal(through_utils, through_schema)


def test_tabular_imports_without_pandas_of_the_heavy_kind():
    """The picker must be able to import the reader without paying for torch."""
    code = (
        "import sys, spacr.tabular as t;"
        "assert '/codex/repo/spacr/' in t.__file__, t.__file__;"
        "heavy = [m for m in ('torch', 'matplotlib', 'cellpose', 'skimage')"
        " if m in sys.modules];"
        "print(heavy)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
