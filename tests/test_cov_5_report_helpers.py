"""What the report does when the folder it is describing misbehaves.

The report is read by somebody who does not have spaCR, so it has exactly one
job when a piece of the plate folder is unreadable: say so and carry on. A
half-written CSV, a figure that will not decode, a run journal that has gone
missing — each of them must cost that one item and nothing else. These tests
make each failure actually happen and assert what survives it.
"""
from __future__ import annotations

import os
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

class _RaisingEntry:
    """A directory entry for a file that vanished between listing and stat."""

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def is_dir(self, follow_symlinks=True):
        raise OSError(2, "No such file or directory")

    def is_file(self, follow_symlinks=True):
        raise OSError(2, "No such file or directory")

    def stat(self, follow_symlinks=True):
        raise OSError(2, "No such file or directory")


def _patch_scandir(monkeypatch, target: Path, entries):
    """Make ``os.scandir`` answer with ``entries`` for one directory only."""
    real = os.scandir

    def fake(path=".", *args, **kwargs):
        if Path(path) == target:
            return iter(entries)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", fake)


def test_a_folder_that_is_not_there_is_walked_as_empty(tmp_path):
    files, truncated = R._iter_dir_files(tmp_path / "no_such_folder")
    assert files == [] and truncated is False


def test_a_subfolder_that_cannot_be_listed_costs_only_that_subfolder(
        tmp_path, monkeypatch):
    """A permission error deep in the tree must not empty the whole inventory."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.csv").write_text("x\n")
    locked = tmp_path / "locked"
    locked.mkdir()

    real = os.scandir

    def fake(path=".", *args, **kwargs):
        if Path(path) == locked:
            raise PermissionError(13, "Permission denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", fake)

    files, truncated = R._iter_dir_files(tmp_path)

    assert [f.name for f in files] == ["a.csv"]
    assert truncated is False


def test_the_walk_stops_at_its_budget_and_says_it_did(tmp_path):
    """A truncated inventory that did not say so would read as a short run."""
    for i in range(5):
        (tmp_path / f"f{i}.csv").write_text("x\n")

    files, truncated = R._iter_dir_files(tmp_path, budget=2)

    assert len(files) == 2 and truncated is True


def test_the_raw_image_folders_are_never_descended_into(tmp_path):
    """``orig`` holds the plate's images; walking it costs minutes for nothing."""
    (tmp_path / "orig").mkdir()
    (tmp_path / "orig" / "image.tif").write_bytes(b"0")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.csv").write_text("x\n")

    files, _truncated = R._iter_dir_files(tmp_path)
    assert [f.name for f in files] == ["a.csv"]

    shallow, _ = R._iter_dir_files(tmp_path, recurse=False)
    assert shallow == []


def test_a_file_that_vanishes_mid_walk_is_skipped(tmp_path, monkeypatch):
    """A live run writes into this folder while the report reads it."""
    _patch_scandir(monkeypatch, tmp_path,
                   [_RaisingEntry("gone.csv", str(tmp_path / "gone.csv"))])

    files, truncated = R._iter_dir_files(tmp_path)

    assert files == [] and truncated is False


def test_sizing_a_folder_that_is_not_there_reports_nothing(tmp_path):
    assert R._dir_stats(tmp_path / "absent") == (0, 0, False)


def test_sizing_descends_into_the_bulk_folders_the_inventory_skips(tmp_path):
    """The size of the plate is the whole plate, images included."""
    (tmp_path / "orig").mkdir()
    (tmp_path / "orig" / "image.tif").write_bytes(b"0" * 100)
    (tmp_path / "a.csv").write_text("xy")

    n_files, total, truncated = R._dir_stats(tmp_path)

    assert n_files == 2 and total == 102 and truncated is False


def test_sizing_stops_at_its_budget(tmp_path):
    for i in range(4):
        (tmp_path / f"f{i}.bin").write_bytes(b"0")

    n_files, _total, truncated = R._dir_stats(tmp_path, budget=2)

    assert n_files == 2 and truncated is True


def test_a_file_that_vanishes_before_it_is_sized_is_skipped(tmp_path,
                                                            monkeypatch):
    _patch_scandir(monkeypatch, tmp_path,
                   [_RaisingEntry("gone.bin", str(tmp_path / "gone.bin"))])

    assert R._dir_stats(tmp_path) == (0, 0, False)


# ---------------------------------------------------------------------------
# Reading a CSV that is still being written
# ---------------------------------------------------------------------------

def test_a_half_flushed_row_ends_the_scan_without_ending_the_report(tmp_path):
    """A NUL byte is the boundary of what was actually written."""
    path = tmp_path / "results.csv"
    path.write_text("gene,score\nA,1\nB,2\nC\x00,3\nD,4\n", encoding="utf-8")

    columns, rows, n_total = R._read_csv_head(path, max_rows=10)

    assert columns == ["gene", "score"]
    assert rows == [["A", "1"], ["B", "2"]]
    assert n_total == 2


def test_a_huge_csv_stops_counting_once_it_has_its_preview(tmp_path):
    """Counting every row of a gigabyte of measurements is not the job."""
    path = tmp_path / "big.csv"
    path.write_text("gene,score\n" + "".join(f"gene{i},{i}\n" for i in range(50)),
                    encoding="utf-8")

    columns, rows, n_total = R._read_csv_head(path, max_rows=2, max_bytes=20)

    assert columns == ["gene", "score"]
    assert len(rows) == 2
    assert 2 < n_total < 50


def test_a_csv_the_module_itself_rejects_keeps_the_rows_read_so_far(tmp_path):
    """``csv`` raises on a cell past its field-size limit; that is not fatal."""
    import csv as _csv

    path = tmp_path / "damaged.csv"
    path.write_text("gene,score\nA,1\n" + "B," + "x" * 200 + "\n",
                    encoding="utf-8")
    limit = _csv.field_size_limit()
    _csv.field_size_limit(50)
    try:
        columns, rows, n_total = R._read_csv_head(path, max_rows=10)
    finally:
        _csv.field_size_limit(limit)

    assert columns == ["gene", "score"]
    assert rows == [["A", "1"]]
    assert n_total == 1


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


# ---------------------------------------------------------------------------
# Matching a journalled run to this plate folder
# ---------------------------------------------------------------------------

def test_settings_that_are_not_a_mapping_match_no_folder():
    assert R._settings_point_at(None, Path("/plate")) is False
    assert R._settings_point_at([], Path("/plate")) is False


def test_a_run_over_several_plates_matches_each_of_them(tmp_path):
    """Apps that take a list of plates must still find their own report."""
    plate = tmp_path / "plate1"
    plate.mkdir()

    assert R._settings_point_at({"src": [str(plate), "/other"]}, plate) is True
    assert R._settings_point_at({"src": (str(plate),)}, plate) is True


def test_a_plate_list_that_went_through_a_csv_is_read_back(tmp_path):
    """A settings CSV stores a list as its repr; the match must survive that."""
    plate = tmp_path / "plate1"
    plate.mkdir()

    assert R._settings_point_at({"src": f"['{plate}', '/other']"}, plate) is True


def test_a_broken_plate_list_is_treated_as_one_path_not_dropped(tmp_path):
    """An unparseable value is still compared, so a real match is not lost."""
    plate = tmp_path / "plate1"
    plate.mkdir()

    assert R._settings_point_at({"src": "[unclosed"}, plate) is False
    assert R._settings_point_at({"src": f"[{plate}"}, plate) is False


# ---------------------------------------------------------------------------
# The run journal, when it is not there
# ---------------------------------------------------------------------------

def _break_import(monkeypatch, dotted: str):
    """Make importing ``dotted`` fail the way a broken install does."""
    monkeypatch.setitem(sys.modules, dotted, None)


def test_a_report_is_still_written_when_the_journal_module_will_not_import(
        tmp_path, monkeypatch):
    """Provenance is one section; losing it must not lose the report."""
    _break_import(monkeypatch, "spacr.run_journal")

    records, problems = R._load_journal_runs(tmp_path, None, True, 5)

    assert records == []
    assert problems == ["run journal unavailable (ModuleNotFoundError)"]


def test_an_unreadable_journal_is_reported_as_a_problem(tmp_path, monkeypatch):
    from spacr import run_journal as journal

    def refuse(_limit):
        raise OSError("journal directory is not readable")

    monkeypatch.setattr(journal, "recent_runs", refuse)

    records, problems = R._load_journal_runs(tmp_path, None, True, 5)

    assert records == []
    assert problems == ["could not read the run journal (OSError)"]


def test_a_run_folder_that_does_not_exist_is_named(tmp_path):
    records, problems = R._load_journal_runs(
        tmp_path, [tmp_path / "no_such_run"], False, 5)

    assert records == []
    assert problems == [f"no such run folder: {tmp_path / 'no_such_run'}"]


def test_a_run_with_a_corrupt_manifest_still_appears(tmp_path):
    """The folder is evidence the run happened, whatever its manifest says."""
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    records, problems = R._load_journal_runs(tmp_path, [run_dir], False, 5)

    assert problems == []
    assert len(records) == 1
    assert records[0]["manifest"] == {}
    assert records[0]["app_key"] == "?" and records[0]["status"] == "?"


def test_a_manifest_that_is_a_list_is_not_taken_as_a_manifest(tmp_path):
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("[1, 2]", encoding="utf-8")

    records, _problems = R._load_journal_runs(tmp_path, [run_dir], False, 5)

    assert records[0]["manifest"] == {}


def test_settings_that_will_not_load_leave_the_run_listed(tmp_path,
                                                          monkeypatch):
    from spacr import run_journal as journal

    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"app_key": "measure"}',
                                           encoding="utf-8")

    def refuse(_path):
        raise ValueError("settings.csv is not a settings file")

    monkeypatch.setattr(journal, "load_run_settings", refuse)

    records, _problems = R._load_journal_runs(tmp_path, [run_dir], False, 5)

    assert records[0]["settings"] == {}
    assert records[0]["app_key"] == "measure"


def test_run_status_needs_spacr_errors_and_says_so_without_it(tmp_path,
                                                              monkeypatch):
    _break_import(monkeypatch, "spacr.errors")

    stamps, problems = R._read_stamps([tmp_path / "measurements.db"])

    assert stamps == []
    assert problems == ["spacr.errors unavailable (ModuleNotFoundError)"]


def test_segmentation_scorecards_need_seg_qc_and_say_so_without_it(
        tmp_path, monkeypatch):
    _break_import(monkeypatch, "spacr.seg_qc")

    field_qcs, error = R._field_qcs_from_csv(tmp_path / "segmentation_qc_cell.csv")

    assert field_qcs == []
    assert error == "spacr.seg_qc unavailable (ModuleNotFoundError)"


# ---------------------------------------------------------------------------
# Figures that will not embed
# ---------------------------------------------------------------------------

def test_a_figure_that_cannot_be_read_names_the_failure(tmp_path):
    data, mime, reason = R._embed_figure(tmp_path / "gone.png", 800)

    assert data is None and mime == ""
    assert reason.startswith("unreadable (")


def test_a_zero_byte_figure_is_not_embedded_as_a_blank(tmp_path):
    """A plot whose write was interrupted must not appear as an empty box."""
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    assert R._embed_figure(path, 800) == (None, "", "empty file")


def test_a_figure_that_is_not_an_image_names_the_decode_failure(tmp_path):
    pytest.importorskip("PIL")
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 4)

    data, mime, reason = R._embed_figure(path, 800)

    assert data is None and mime == ""
    assert reason.startswith("could not decode (")


def test_without_pillow_the_figure_is_embedded_as_it_lies_on_disk(tmp_path,
                                                                  monkeypatch):
    """No Pillow means no downscaling, not a report with no figures in it."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    png = tmp_path / "plot.png"
    png.write_bytes(b"\x89PNG-not-really")
    jpg = tmp_path / "plot.jpg"
    jpg.write_bytes(b"\xff\xd8-not-really")

    assert R._embed_figure(png, 800) == (b"\x89PNG-not-really", "image/png", "")
    assert R._embed_figure(jpg, 800) == (b"\xff\xd8-not-really", "image/jpeg", "")


def test_an_oversized_figure_is_downscaled_before_it_is_embedded(tmp_path):
    """A page of full-resolution montages is a report nobody can open."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "big.png"
    Image.new("RGB", (400, 200), "white").save(path)
    on_disk = path.stat().st_size

    data, mime, reason = R._embed_figure(path, 100)

    assert reason == "" and mime == "image/png"
    with Image.open(__import__("io").BytesIO(data)) as shrunk:
        assert max(shrunk.size) == 100
    assert len(data) < on_disk + 1


def test_a_palette_figure_is_converted_before_it_is_embedded(tmp_path):
    """A mode PNG saves without complaint only after a convert."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "paletted.png"
    Image.new("RGB", (300, 300), "white").convert("P").save(path)

    data, mime, reason = R._embed_figure(path, 100)

    assert reason == "" and mime == "image/png" and data


def test_a_jpeg_small_enough_to_keep_is_still_re_encoded_as_png(tmp_path):
    """Only a small PNG passes through untouched; the rest are normalised."""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "small.jpg"
    Image.new("RGB", (40, 30), "white").save(path)

    data, mime, reason = R._embed_figure(path, 800)

    assert reason == "" and mime == "image/png"
    assert data.startswith(b"\x89PNG")


def test_a_figure_outside_the_plate_folder_is_captioned_by_its_name(tmp_path):
    """``relative_to`` raises for it, and a raise would lose the figure."""
    assert R._figure_title(Path("/elsewhere/volcano.png"),
                           tmp_path) == "volcano.png"
    assert R._figure_title(tmp_path / "results" / "v.png",
                           tmp_path) == os.path.join("results", "v.png")


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
        os.path.join("measurements.db"), os.path.join("results", "good.csv")]
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
