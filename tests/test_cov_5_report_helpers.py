"""What the report does when the folder it is describing misbehaves.

The report is read by somebody who does not have spaCR, so it has exactly one
job when a piece of the plate folder is unreadable: say so and carry on. A
half-written CSV, a figure that will not decode, a run journal that has gone
missing — each of them must cost that one item and nothing else. These tests
make each failure actually happen and assert what survives it.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spacr import report as R


# ---------------------------------------------------------------------------
# Formatting values that came off disk and may be anything at all
# ---------------------------------------------------------------------------

def test_a_missing_value_escapes_to_nothing_not_to_the_word_none():
    """"None" in a settings column reads as a value the run actually used."""
    assert R._esc(None) == ""
    assert R._esc("A01<script>") == "A01&lt;script&gt;"


def test_a_size_that_is_not_a_number_renders_as_a_dash():
    """A stat that failed leaves no size; the column still has to line up."""
    assert R._fmt_bytes(None) == "-"
    assert R._fmt_bytes("not a size") == "-"
    assert R._fmt_bytes(0) == "0 B"
    assert R._fmt_bytes(1536) == "1.5 KB"
    # A plate folder really does reach terabytes, and the top unit is the last.
    assert R._fmt_bytes(7 * 1024 ** 4) == "7.0 TB"
    assert R._fmt_bytes(4551 * 1024 ** 4) == "4551.0 TB"


def test_an_epoch_timestamp_is_read_when_it_is_not_an_iso_string():
    """Manifests carry both spellings; neither may print as a raw number."""
    text = R._fmt_time("1700000000")
    assert text == datetime.fromtimestamp(1700000000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC")


def test_a_timestamp_that_is_neither_is_shown_verbatim():
    """Better the unparseable string than a fabricated date or a crash."""
    assert R._fmt_time("some time last Tuesday") == "some time last Tuesday"
    assert R._fmt_time(None) == "-"


def test_an_elapsed_time_is_read_at_the_scale_it_happened_at():
    assert R._fmt_elapsed("nonsense") == "-"
    assert R._fmt_elapsed(12.34) == "12.3 s"
    assert R._fmt_elapsed(200) == "3 m 20 s"
    assert R._fmt_elapsed(3600 * 2 + 60 * 5) == "2 h 5 m"


def test_a_path_with_a_null_byte_is_not_the_same_place_as_anything():
    """``realpath`` raises on it, and a raise here would lose the report."""
    assert R._same_path("/tmp/a\x00b", "/tmp") is False
    assert R._same_path("/tmp", "/tmp/") is True


# ---------------------------------------------------------------------------
# Walking the plate folder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reading a CSV that is still being written
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The data model
# ---------------------------------------------------------------------------

def test_asking_a_skipped_figure_for_its_bytes_says_why_it_has_none():
    """The reason is the whole value of the failure; it must reach the caller."""
    figure = R.Figure(path=Path("/plate/results/volcano.png"),
                      reason="could not decode (UnidentifiedImageError)")

    assert figure.embedded is False
    with pytest.raises(ValueError, match="could not decode"):
        figure.data_uri()


def test_an_embedded_figure_renders_as_a_self_contained_uri():
    figure = R.Figure(path=Path("x.png"), mime="image/png", data=b"\x89PNG")

    assert figure.embedded is True
    assert figure.data_uri().startswith("data:image/png;base64,")


def test_a_missing_section_is_the_only_kind_that_was_not_found():
    """"Not run" and "ran and failed" are different facts about a section."""
    assert R.Section(title="t", status=R.STATUS_MISSING).found is False
    assert R.Section(title="t", status=R.STATUS_PROBLEM).found is True
    assert R.Section(title="t").found is True
    assert "plugins use their registered contribution" in (
        R.Section.__doc__ or "")


# ---------------------------------------------------------------------------
# Matching a journalled run to this plate folder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The run journal, when it is not there
# ---------------------------------------------------------------------------

def _break_import(monkeypatch, dotted: str):
    """Make importing ``dotted`` fail the way a broken install does."""
    monkeypatch.setitem(sys.modules, dotted, None)



def test_run_status_needs_spacr_errors_and_says_so_without_it(tmp_path,
                                                              monkeypatch):
    _break_import(monkeypatch, "spacr.errors")

    stamps, problems = R._read_stamps([tmp_path / "measurements.db"])

    assert stamps == []
    assert problems == ["spacr.errors unavailable (ModuleNotFoundError)"]



# ---------------------------------------------------------------------------
# Figures that will not embed
# ---------------------------------------------------------------------------


def test_a_figure_outside_the_plate_folder_is_captioned_by_its_name(tmp_path):
    """``relative_to`` raises for it, and a raise would lose the figure."""
    assert R._figure_title(Path("/elsewhere/volcano.png"),
                           tmp_path) == "volcano.png"
    assert R._figure_title(tmp_path / "results" / "v.png",
                           tmp_path) == "results/v.png"


def test_a_figure_whose_file_vanishes_is_still_listed_with_no_size(
        tmp_path, monkeypatch):
    """The stat races the run that is still writing; the caption survives it."""
    Image = pytest.importorskip("PIL.Image")
    results = tmp_path / "results"
    results.mkdir()
    path = results / "plot.png"
    Image.new("RGB", (10, 10), "white").save(path)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self == path:
            raise OSError(2, "No such file or directory")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    section = R._collect_figures(tmp_path, {"raster": [path]}, 10, 800)

    assert [f.n_bytes for f in section.figures] == [0]
    assert "Embedded 1 of 1 figure(s)" in section.body_html


def test_a_figure_that_will_not_embed_is_counted_as_omitted(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"")

    section = R._collect_figures(tmp_path, {"raster": [path]}, 10, 800)

    assert section.figures == []
    assert "Embedded 0 of 1 figure(s) found; 1 omitted." in section.body_html


# ---------------------------------------------------------------------------
# The statistics section, over files it cannot stat or read
# ---------------------------------------------------------------------------

def test_a_database_that_cannot_be_opened_contributes_no_tables(tmp_path,
                                                                monkeypatch):
    from spacr import database_concurrency

    def refuse(_path, **_kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(database_concurrency, "connect", refuse)

    assert R._sqlite_table_counts(tmp_path / "measurements.db") == []


def test_the_statistics_index_survives_files_it_cannot_stat_or_parse(
        tmp_path, monkeypatch):
    """One damaged result CSV costs its own row, not the section."""
    results = tmp_path / "results"
    results.mkdir()
    good = results / "good.csv"
    good.write_text("gene,score\nA,1\n", encoding="utf-8")
    unreadable = results / "locked.csv"
    unreadable.write_text("gene,score\nB,2\n", encoding="utf-8")
    db = tmp_path / "measurements.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE cell (id INTEGER)")
    connection.execute("INSERT INTO cell VALUES (1)")
    connection.commit()
    connection.close()

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self in (good, db):
            raise OSError(2, "No such file or directory")
        return real_stat(self, *args, **kwargs)

    real_open = R.open if hasattr(R, "open") else open

    def fake_open(file, *args, **kwargs):
        if Path(file) == unreadable:
            raise PermissionError(13, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(R, "open", fake_open, raising=False)

    section = R._collect_statistics(
        tmp_path, {"result_csv": [good, unreadable], "databases": [db]},
        max_rows=5, max_files=1)

    assert any("locked.csv unreadable (PermissionError)" in note
               for note in section.notes)
    assert [row[0] for row in section.table.rows] == [
        "measurements.db", "results/good.csv"]
    assert "0 B" in [row[3] for row in section.table.rows]


def test_further_result_tables_are_listed_but_not_previewed(tmp_path):
    """A report that previewed forty CSVs would be unreadable and enormous."""
    results = tmp_path / "results"
    results.mkdir()
    paths = []
    for i in range(3):
        path = results / f"table{i}.csv"
        path.write_text("gene,score\nA,1\n", encoding="utf-8")
        paths.append(path)

    section = R._collect_statistics(tmp_path, {"result_csv": paths},
                                    max_rows=5, max_files=1)

    assert "2 further result CSV(s) are listed above but not previewed." in (
        section.body_html)
    assert len(section.table.rows) == 3
