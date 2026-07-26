"""The report generator's edges — :mod:`spacr.report`, part two.

:mod:`tests.test_report` pins the behaviour a healthy plate folder gets.
This file pins what happens when the folder is *not* healthy, because that
is the situation the module was written for: a run that half-finished, a
CSV that was truncated mid-write, a figure that is zero bytes, a database
whose pages are corrupt, a journal that cannot be read, a dependency that
is not installed.

The governing rule is the one in the module docstring — *a missing section
is stated, never omitted* — extended one step: **damage is stated, never
fatal**. A report that raises because one of forty result CSVs has a NUL
byte in it tells the collaborator nothing at all. Every test below drives a
real damaged artifact through the real function and asserts on the value it
produced, not on the absence of an exception.

Two real defects were found writing it and are fixed in ``spacr/report.py``:

* a CSV damaged past its first row (embedded NUL, or a cell longer than
  :func:`csv.field_size_limit`) made :mod:`csv` raise straight out of
  :func:`collect_report` — one corrupt file, no report;
* a ``src`` that is a symlink loop made :mod:`pathlib` raise ``RuntimeError``
  out of :func:`collect_report`, which documents that it never raises for
  bad input.

Both have a test here that fails against the old code.
"""
from __future__ import annotations

import csv
import errno
import io
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import pytest

import spacr.report as R
from spacr.report import (
    Figure,
    Report,
    Section,
    Table,
    build_report,
    collect_report,
    pdf_page_count,
    render_html,
    render_text,
    write_pdf,
)


# ---------------------------------------------------------------------------
# Builders (kept local so this file stands on its own)
# ---------------------------------------------------------------------------

QC_HEADER = ["field", "object_type", "n_objects", "severity", "flags",
             "border_fraction", "count_ratio", "foreground_fraction",
             "median_diameter", "outlier_fraction", "note"]

LAYOUT_HEADER = ["plateID", "well", "rowID", "columnID", "row_index",
                 "column_index", "n", "value", "ring", "is_edge"]


def _write_csv(path: Path, header, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _write_png(path: Path, size=(24, 18), color=(40, 90, 160)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def _qc_row(field, severity="ok", object_type="cell", n_objects=100,
            border="0.42"):
    flags = "high_border_fraction" if severity != "ok" else ""
    return [field, object_type, n_objects, severity, flags, border,
            1.0, 0.21, 22.5, 0.03, f"{severity} field"]


def _write_db(src: Path, stamps=(), annotation=True) -> Path:
    path = src / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE cell (prc TEXT, cell_area REAL)")
        conn.executemany("INSERT INTO cell VALUES (?, ?)",
                         [(f"plate1_A01_{i}", 100.0 + i) for i in range(12)])
        if annotation:
            conn.execute("CREATE TABLE png_list (png_path TEXT, prc TEXT, "
                         "annotation INTEGER)")
            conn.executemany("INSERT INTO png_list VALUES (?, ?, ?)",
                             [(f"/crops/{i}.png", f"plate1_A01_{i}", i % 2)
                              for i in range(10)])
        else:
            conn.execute("CREATE TABLE png_list (png_path TEXT, prc TEXT)")
            conn.executemany("INSERT INTO png_list VALUES (?, ?)",
                             [(f"/crops/{i}.png", f"plate1_A01_{i}")
                              for i in range(10)])
        if stamps:
            conn.execute(
                "CREATE TABLE run_status (run_id TEXT, name TEXT, status TEXT, "
                "n_attempted INTEGER, n_succeeded INTEGER, n_failed INTEGER, "
                "failure_rate REAL, started_utc TEXT, stamped_utc TEXT, "
                "failures_json TEXT, summary TEXT)")
            conn.executemany(
                "INSERT INTO run_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                stamps)
        conn.commit()
    finally:
        conn.close()
    return path


def _stamp(name="measure_crop", status="complete", attempted=100, failed=0):
    return ("run-1", name, status, attempted, attempted - failed, failed,
            failed / attempted if attempted else 0.0,
            "2026-07-01T10:00:00+00:00", "2026-07-01T10:20:00+00:00",
            json.dumps([]), "")


def _write_run_dir(root: Path, name: str, settings, app_key="mask",
                   status="success", start="2026-07-01T10:00:00+00:00",
                   manifest_text=None) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        (run_dir / "settings.json").write_text(json.dumps(settings, default=str))
    if manifest_text is None:
        manifest_text = json.dumps({
            "app_key": app_key, "start_utc": start,
            "elapsed_s": 1200.0, "status": status,
            "env": {"spacr": "1.3.6", "python": "3.11.9"},
        })
    (run_dir / "manifest.json").write_text(manifest_text)
    return run_dir


def _chmod_back(path: Path, mode=0o755):
    """Restore a mode so pytest can clean the tmp tree up afterwards."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


# ===========================================================================
# Formatting helpers
# ===========================================================================

def test_escaping_turns_every_dangerous_character_into_an_entity():
    assert R._esc(None) == ""
    assert R._esc(0) == "0"
    assert R._esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"


@pytest.mark.parametrize("value,expected", [
    (0, "0 B"),
    (999, "999 B"),
    (1024, "1.0 KB"),
    (1536, "1.5 KB"),
    (5 * 1024 ** 2, "5.0 MB"),
    (3 * 1024 ** 3, "3.0 GB"),
    (2 * 1024 ** 4, "2.0 TB"),
    (1024 ** 5, "1024.0 TB"),
    (None, "-"),
    ("not a number", "-"),
])
def test_byte_counts_render_in_the_unit_a_reader_expects(value, expected):
    assert R._fmt_bytes(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("2026-07-01T10:00:00+00:00", "2026-07-01 10:00:00 UTC"),
    ("2026-07-01T10:00:00Z", "2026-07-01 10:00:00 UTC"),
    ("1700000000.5", "2023-11-14 22:13:20 UTC"),
    ("halfway through tuesday", "halfway through tuesday"),
    (None, "-"),
    ("", "-"),
])
def test_timestamps_render_as_utc_and_unparseable_ones_survive_verbatim(
        value, expected):
    assert R._fmt_time(value) == expected


@pytest.mark.parametrize("value,expected", [
    (None, "-"),
    ("about an hour", "-"),
    (12.5, "12.5 s"),
    (89.9, "89.9 s"),
    (90, "1 m 30 s"),
    (1200, "20 m 0 s"),
    (5400, "1 h 30 m"),
    ("30", "30.0 s"),
])
def test_elapsed_times_switch_unit_at_ninety(value, expected):
    assert R._fmt_elapsed(value) == expected


def test_same_path_sees_through_a_symlink_and_survives_a_null_byte(tmp_path):
    real = tmp_path / "plate"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(str(real), str(link))
    assert R._same_path(link, real) is True
    assert R._same_path(tmp_path / "other", real) is False
    # A settings value carrying a NUL cannot be realpath'd; that is a
    # non-match, not a crash halfway through building the report.
    assert R._same_path("plate\x00name", real) is False


def test_a_runs_src_setting_is_matched_in_every_shape_it_is_stored_in(tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    other = tmp_path / "elsewhere"

    assert R._settings_point_at({"src": str(src)}, src) is True
    assert R._settings_point_at({"src": [str(other), str(src)]}, src) is True
    assert R._settings_point_at({"src": (str(src),)}, src) is True
    # The shape a list of plates takes after a CSV round-trip.
    assert R._settings_point_at({"src": f"['{other}', '{src}']"}, src) is True

    assert R._settings_point_at({"src": str(other)}, src) is False
    assert R._settings_point_at({"src": f"[{src}"}, src) is False   # unparseable
    assert R._settings_point_at({"src": None}, src) is False
    assert R._settings_point_at({}, src) is False
    assert R._settings_point_at(["not", "a", "dict"], src) is False


# ===========================================================================
# Bounded directory walks
# ===========================================================================

def test_walking_a_path_that_is_not_a_directory_yields_nothing(tmp_path):
    loose = tmp_path / "file.txt"
    loose.write_text("x")
    assert R._iter_dir_files(loose) == ([], False)
    assert R._iter_dir_files(tmp_path / "missing") == ([], False)


def test_the_walk_descends_subfolders_but_never_into_bulk_pixel_folders(tmp_path):
    root = tmp_path / "results"
    (root / "deep" / "deeper").mkdir(parents=True)
    (root / "deep" / "deeper" / "kept.csv").write_text("a\n")
    (root / "top.csv").write_text("a\n")
    (root / "orig").mkdir()                     # a BULK_DIR
    (root / "orig" / "raw_0001.tif").write_bytes(b"\x00")

    files, truncated = R._iter_dir_files(root)
    names = sorted(p.name for p in files)
    assert names == ["kept.csv", "top.csv"], (
        "the walk either skipped a nested result or read the raw images")
    assert truncated is False

    flat, _ = R._iter_dir_files(root, recurse=False)
    assert [p.name for p in flat] == ["top.csv"]


def test_the_walk_stops_at_its_budget_and_says_that_it_did(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for i in range(6):
        (root / f"f{i}.csv").write_text("a\n")
    files, truncated = R._iter_dir_files(root, budget=4)
    assert len(files) == 4
    assert truncated is True


def test_a_folder_the_user_cannot_open_is_skipped_not_fatal(tmp_path):
    root = tmp_path / "results"
    locked = root / "locked"
    locked.mkdir(parents=True)
    (locked / "hidden.csv").write_text("a\n")
    (root / "readable.csv").write_text("a\n")
    os.chmod(locked, 0o000)
    try:
        files, truncated = R._iter_dir_files(root)
        assert [p.name for p in files] == ["readable.csv"]
        assert truncated is False
        assert R._dir_stats(locked) == (0, 0, False)
    finally:
        _chmod_back(locked)


class _VanishedEntry:
    """A directory entry whose file was unlinked between readdir and stat.

    ``os.scandir`` hands back names copied out of the directory block; a
    crop the pipeline deletes a microsecond later is still in that list,
    and every question asked about it raises ENOENT. The walk has to step
    over it, not abandon the folder.
    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path

    def _gone(self):
        return FileNotFoundError(errno.ENOENT, "vanished mid-walk", self.path)

    def is_dir(self, follow_symlinks=True):
        raise self._gone()

    def is_file(self, follow_symlinks=True):
        raise self._gone()

    def stat(self, follow_symlinks=True):
        raise self._gone()


def test_a_file_deleted_mid_walk_is_stepped_over(tmp_path, monkeypatch):
    root = tmp_path / "results"
    root.mkdir()
    for i in range(3):
        (root / f"f{i}.csv").write_text("a\n")

    real_scandir = os.scandir

    def fake_scandir(path="."):
        entries = list(real_scandir(path))
        if Path(path) == root:
            entries.append(_VanishedEntry("gone.csv", str(root / "gone.csv")))
        return entries

    monkeypatch.setattr(os, "scandir", fake_scandir)
    files, truncated = R._iter_dir_files(root)
    assert sorted(p.name for p in files) == ["f0.csv", "f1.csv", "f2.csv"]
    assert truncated is False


def test_dir_stats_counts_bytes_and_stops_at_its_budget(tmp_path):
    root = tmp_path / "plate"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.bin").write_bytes(b"x" * 100)
    (root / "b.bin").write_bytes(b"y" * 40)

    assert R._dir_stats(root) == (2, 140, False)
    n_files, _total, truncated = R._dir_stats(root, budget=1)
    assert (n_files, truncated) == (1, True)


def test_a_file_that_cannot_be_stat_ed_is_counted_but_adds_no_bytes(tmp_path):
    # r-- on a folder lists its names but forbids resolving them, so
    # scandir succeeds and every entry.stat() raises EACCES.
    root = tmp_path / "unsearchable"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 100)
    os.chmod(root, 0o400)
    try:
        n_files, total, truncated = R._dir_stats(root)
        assert (n_files, total, truncated) == (1, 0, False)
    finally:
        _chmod_back(root)


def test_the_file_inventory_lists_loose_top_level_files(tmp_path):
    src = tmp_path / "plate"
    (src / "qc").mkdir(parents=True)
    (src / "qc" / "card.csv").write_bytes(b"x" * 10)
    (src / "notes.txt").write_bytes(b"y" * 25)
    (src / "log.txt").write_bytes(b"z" * 5)

    rows, truncated = R._file_inventory(src)
    assert truncated is False
    assert ["qc/", "1", "10 B"] in rows
    assert ["(files at the top level)", "2", "30 B"] in rows


def test_the_file_inventory_of_an_unreadable_folder_is_empty_not_fatal(tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    (src / "a.txt").write_text("x")
    os.chmod(src, 0o000)
    try:
        assert R._file_inventory(src) == ([], False)
    finally:
        _chmod_back(src)


def test_loose_files_that_cannot_be_stat_ed_are_still_counted(tmp_path):
    src = tmp_path / "plate"
    (src / "qc").mkdir(parents=True)
    (src / "a.txt").write_bytes(b"x" * 9)
    (src / "b.txt").write_bytes(b"y" * 9)
    os.chmod(src, 0o400)
    try:
        rows, truncated = R._file_inventory(src)
        assert ["(files at the top level)", "2", "0 B"] in rows
        # The subfolder is listed even though it could not be descended.
        assert ["qc/", "0", "0 B"] in rows
        assert truncated is False
    finally:
        _chmod_back(src)


def test_a_folder_with_more_files_than_the_budget_says_the_counts_are_bounds(
        tmp_path):
    """The report must not read a million crops to tell you the run failed."""
    src = tmp_path / "plate_huge"
    tiles = src / "results" / "tiles"
    tiles.mkdir(parents=True)
    for i in range(R.WALK_BUDGET + 1):
        os.close(os.open(str(tiles / f"t{i:05d}.txt"),
                         os.O_CREAT | os.O_WRONLY, 0o644))

    report = collect_report(src, run_dirs=[])
    assert any(str(R.WALK_BUDGET) in note for note in report.sections[0].notes), (
        "the truncated scan was not disclosed on the run-status section")
    appendix_notes = " ".join(report.section("appendix").notes)
    assert "lower bounds" in appendix_notes
    inventory = {row[0]: int(row[1]) for row in report.section("appendix").table.rows}
    assert inventory["results/"] == R.WALK_BUDGET


# ===========================================================================
# CSV head reading
# ===========================================================================

def test_reading_a_csv_head_stops_once_it_has_read_its_byte_budget(tmp_path):
    path = _write_csv(tmp_path / "big.csv", ["a", "b"],
                      [[i, "x" * 40] for i in range(50)])
    columns, rows, n_total = R._read_csv_head(path, max_rows=1, max_bytes=1)
    assert columns == ["a", "b"]
    assert rows == [["0", "x" * 40]]
    assert n_total == 2, "the scan did not stop at the byte budget"


def test_a_csv_damaged_past_its_first_rows_still_yields_what_parsed(tmp_path):
    """BUG (fixed): csv.Error escaped and took the whole report with it."""
    path = tmp_path / "hits.csv"
    path.write_bytes(b"gene,score\ng1,1\ng2,2\ng3,\x003\ng4,4\n")
    columns, rows, n_total = R._read_csv_head(path, max_rows=10)
    assert columns == ["gene", "score"]
    assert rows == [["g1", "1"], ["g2", "2"]]
    assert n_total == 2


def test_a_cell_longer_than_the_csv_field_limit_is_not_fatal(tmp_path):
    path = tmp_path / "wide.csv"
    huge = "z" * (csv.field_size_limit() + 10)
    path.write_text(f"gene,blob\ng1,ok\ng2,{huge}\n", encoding="utf-8")
    columns, rows, n_total = R._read_csv_head(path, max_rows=10)
    assert columns == ["gene", "blob"]
    assert rows == [["g1", "ok"]]
    assert n_total == 1


def test_a_corrupt_result_csv_does_not_take_the_report_down(tmp_path):
    """End-to-end version of the same defect."""
    src = tmp_path / "plate_corrupt_csv"
    (src / "results").mkdir(parents=True)
    (src / "results" / "hits.csv").write_bytes(
        b"gene,score\ng1,1\ng2,2\ng3,\x003\ng4,4\n")

    report = collect_report(src, run_dirs=[])
    section = report.section("statistics")
    assert section.status != R.STATUS_MISSING
    assert section.table.rows[0][0] == "results/hits.csv"
    assert section.table.rows[0][2] == "2", "the rows that parsed were lost"
    assert "g1" in render_html(report)


# ===========================================================================
# Data model
# ===========================================================================

def test_a_figure_that_was_not_embedded_refuses_to_hand_out_a_data_uri():
    figure = Figure(path=Path("/plate/results/x.png"), reason="empty file")
    assert figure.embedded is False
    with pytest.raises(ValueError, match="empty file"):
        figure.data_uri()


def test_a_section_knows_whether_its_evidence_exists():
    assert Section(title="t", status=R.STATUS_OK).found is True
    assert Section(title="t", status=R.STATUS_PROBLEM).found is True
    assert Section(title="t", status=R.STATUS_MISSING).found is False


def test_a_report_partitions_its_sections_into_found_and_missing():
    report = Report(
        src=Path("/plate"),
        sections=[Section(title="a", key="figures", status=R.STATUS_MISSING),
                  Section(title="b", key="statistics", status=R.STATUS_OK),
                  Section(title="c", key="appendix", status=R.STATUS_PROBLEM)])
    assert report.found_sections == ["statistics", "appendix"]
    assert report.missing_sections == ["figures"]


# ===========================================================================
# The run journal
# ===========================================================================

def test_the_journal_scan_keeps_exactly_the_runs_whose_src_is_this_folder(
        tmp_path, monkeypatch):
    home = tmp_path / "home"
    runs = home / ".spacr" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    other = tmp_path / "elsewhere"
    other.mkdir()

    plain = _write_run_dir(runs, "2026-07-01_100000_aaaa__mask",
                           {"src": str(src), "cell_diameter": 30},
                           start="2026-07-01T10:00:00+00:00")
    as_list = _write_run_dir(runs, "2026-07-02_100000_bbbb__measure",
                             {"src": [str(other), str(src)]},
                             app_key="measure",
                             start="2026-07-02T10:00:00+00:00")
    as_repr = _write_run_dir(runs, "2026-07-03_100000_cccc__mask",
                             {"src": f"['{src}']"},
                             start="2026-07-03T10:00:00+00:00")
    # ... and four that must not be kept.
    _write_run_dir(runs, "2026-06-01_100000_dddd__mask", {"src": str(other)})
    _write_run_dir(runs, "2026-06-02_100000_eeee__mask", {"src": f"[{src}"})
    _write_run_dir(runs, "2026-06-03_100000_ffff__mask", ["not", "a", "dict"])
    _write_run_dir(runs, "2026-06-04_100000_gggg__mask", None)  # no settings

    report = collect_report(src)          # no run_dirs: the journal is scanned
    provenance = report.section("provenance")
    assert provenance.status == R.STATUS_OK
    assert [row[0] for row in provenance.table.rows] == [
        as_repr.name, as_list.name, plain.name], "newest first, matches only"
    assert provenance.table.rows[1][1] == "measure"


def test_a_journal_that_cannot_be_read_is_a_note_not_a_crash(
        tmp_path, monkeypatch):
    not_a_home = tmp_path / "home_is_a_file"
    not_a_home.write_text("x")
    monkeypatch.setenv("HOME", str(not_a_home))

    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])

    report = collect_report(src)
    provenance = report.section("provenance")
    assert provenance.status == R.STATUS_MISSING
    assert any("could not read the run journal" in n for n in provenance.notes)
    assert "could not read the run journal" in render_text(report)


def test_a_run_folder_whose_manifest_is_unreadable_is_still_listed(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", "2026-07-01_100000_aaaa__mask",
                             {"src": str(src), "channels": [0, 1]},
                             manifest_text="{ this is not json")

    report = collect_report(src, run_dirs=[run_dir])
    provenance = report.section("provenance")
    assert provenance.table.rows == [
        [run_dir.name, "?", "?", "-", "-", "-", "2"]], (
        "an unreadable manifest must not erase the run from the report")
    # The settings themselves were readable and are still shown.
    settings = report.section("settings")
    assert ["channels", "[0, 1]"] in settings.table.rows


def test_a_run_folder_with_no_settings_at_all_is_still_listed(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", "2026-07-01_100000_aaaa__mask",
                             None)

    report = collect_report(src, run_dirs=[run_dir])
    provenance = report.section("provenance")
    assert provenance.table.rows[0][0] == run_dir.name
    assert provenance.table.rows[0][6] == "0", "settings appeared from nowhere"
    assert report.section("settings").status == R.STATUS_MISSING


def test_a_broken_install_degrades_the_report_instead_of_killing_it(tmp_path):
    """Both halves of provenance survive their own module going missing."""
    import spacr

    src = tmp_path / "plate"
    db = _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", "2026-07-01_100000_aaaa__mask",
                             {"src": str(src)})

    with pytest.MonkeyPatch.context() as patch:
        patch.delattr(spacr, "run_journal", raising=False)
        patch.setitem(sys.modules, "spacr.run_journal", None)
        records, problems = R._load_journal_runs(src, [run_dir], True, 10)
    assert records == []
    assert problems == ["run journal unavailable (ModuleNotFoundError)"]

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spacr.errors", None)
        stamps, stamp_problems = R._read_stamps([db])
    assert stamps == []
    assert stamp_problems == ["spacr.errors unavailable (ModuleNotFoundError)"]


def test_a_corrupt_run_status_sidecar_is_noted_not_fatal(tmp_path):
    src = tmp_path / "plate_bad_sidecar"
    (src / "results").mkdir(parents=True)
    (src / "results" / "wells.csv").write_text("well,value\nA01,1\n")
    (src / "results" / "wells.run_status.json").write_text("{ not json")

    report = collect_report(src, run_dirs=[])
    first = report.sections[0]
    assert report.status == "unknown"
    assert any("wells.run_status.json: unreadable run status" in n
               for n in first.notes)
    assert "unreadable run status" in render_html(report)


# ===========================================================================
# Segmentation QC
# ===========================================================================

def test_a_recorded_verdict_json_is_shown_beside_the_scorecard(tmp_path):
    src = tmp_path / "plate_verdict"
    _write_csv(src / "qc" / "segmentation_qc_cell.csv", QC_HEADER,
               [_qc_row(f"plate1_A{i:02d}_1") for i in range(1, 9)])
    (src / "qc" / "segmentation_qc_cell.json").write_text(json.dumps(
        {"object_type": "cell", "verdict": "warn", "n_fields": 8}))
    (src / "qc" / "no_object_type.json").write_text(json.dumps({"note": "hi"}))
    (src / "qc" / "half_written.json").write_text("{ oops")

    report = collect_report(src, run_dirs=[])
    section = report.section("segmentation_qc")
    assert "Recorded verdicts: cell = warn" in section.body_html
    assert "Every scored field is clean." in section.body_html
    assert section.status == R.STATUS_OK
    assert "Every scored field is clean." in render_text(report)


def test_a_scorecard_with_unparseable_numbers_reads_as_missing_not_zero(
        tmp_path):
    card = _write_csv(tmp_path / "qc" / "segmentation_qc_cell.csv", QC_HEADER,
                      [_qc_row("f1", n_objects="many", border="n/a"),
                       _qc_row("f2", n_objects=7, border="")])
    field_qcs, error = R._field_qcs_from_csv(card)
    assert error is None
    assert [q.field for q in field_qcs] == ["f1", "f2"]
    assert field_qcs[0].n_objects == 0, "'many' silently became a real count"
    assert math.isnan(field_qcs[0].metrics["border_fraction"])
    assert math.isnan(field_qcs[1].metrics["border_fraction"])
    assert field_qcs[1].n_objects == 7
    assert field_qcs[1].metrics["count_ratio"] == 1.0


def test_a_scorecard_that_cannot_be_opened_is_reported_as_such(tmp_path):
    src = tmp_path / "plate"
    card = _write_csv(src / "qc" / "segmentation_qc_cell.csv", QC_HEADER,
                      [_qc_row("f1")])
    _write_csv(src / "qc" / "segmentation_qc_nucleus.csv", QC_HEADER, [])
    os.chmod(card, 0o000)
    try:
        report = collect_report(src, run_dirs=[])
        section = report.section("segmentation_qc")
        assert section.status == R.STATUS_PROBLEM
        assert any("segmentation_qc_cell.csv unreadable" in n
                   for n in section.notes)
        assert any("segmentation_qc_nucleus.csv is empty" in n
                   for n in section.notes)
        assert "none could be read" in section.body_html
        assert section.table is None
    finally:
        _chmod_back(card, 0o644)


def test_a_scorecard_damaged_mid_file_is_reported_not_summarised(tmp_path):
    """A plate verdict from half a scorecard is a different verdict."""
    src = tmp_path / "plate"
    (src / "qc").mkdir(parents=True)
    (src / "qc" / "segmentation_qc_cell.csv").write_bytes(
        b"field,object_type,n_objects,severity,flags,note\n"
        b"f1,cell,10,ok,,fine\n"
        b"f2,cell,10,\x00fail,,broken\n")

    report = collect_report(src, run_dirs=[])
    section = report.section("segmentation_qc")
    assert section.status == R.STATUS_PROBLEM
    assert any("is not readable as CSV" in n for n in section.notes)
    assert section.table is None


def test_scorecards_are_useless_without_seg_qc_and_the_report_says_so(tmp_path):
    src = tmp_path / "plate"
    _write_csv(src / "qc" / "segmentation_qc_cell.csv", QC_HEADER,
               [_qc_row("f1")])
    artifacts = R._find_artifacts(src)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spacr.seg_qc", None)
        section = R._collect_segmentation_qc(src, artifacts, 25)
        # The same unavailability, seen one level down.
        field_qcs, error = R._field_qcs_from_csv(artifacts["qc_csv"][0])
    assert section.status == R.STATUS_PROBLEM
    assert "spacr.seg_qc could not be imported" in section.body_html
    assert section.text_lines == [
        "Segmentation QC: scorecards found but unreadable."]
    assert field_qcs == []
    assert error == "spacr.seg_qc unavailable (ModuleNotFoundError)"


# ===========================================================================
# Plate QC
# ===========================================================================

def test_a_layout_export_that_cannot_be_opened_is_skipped(tmp_path):
    src = tmp_path / "plate"
    good = _write_csv(src / "qc" / "plate_wells.csv", LAYOUT_HEADER,
                      [["p", f"A{i:02d}", "r", "c", 0, i, 20, 1.5, 0, i < 2]
                       for i in range(6)])
    locked = _write_csv(src / "qc" / "plate_other.csv", LAYOUT_HEADER,
                        [["p", "A01", "r", "c", 0, 0, 20, 1.5, 0, True]])
    os.chmod(locked, 0o000)
    try:
        report = collect_report(src, run_dirs=[])
        section = report.section("plate_qc")
        assert section.status == R.STATUS_OK
        assert section.table.rows == [[good.name, "6", "2",
                                       R._fmt_bytes(good.stat().st_size)]]
    finally:
        _chmod_back(locked, 0o644)


def test_a_layout_export_damaged_mid_file_reports_the_wells_that_parsed(tmp_path):
    src = tmp_path / "plate"
    (src / "qc").mkdir(parents=True)
    (src / "qc" / "plate_wells.csv").write_bytes(
        b"plateID,well,ring,is_edge\n"
        b"p,A01,0,True\n"
        b"p,A02,0,True\n"
        b"p,A03,1,False\n"
        b"p,A0\x004,1,False\n")

    report = collect_report(src, run_dirs=[])
    section = report.section("plate_qc")
    assert section.status == R.STATUS_OK
    assert section.table.rows[0][1] == "3", "wells parsed before the damage"
    assert section.table.rows[0][2] == "2", "edge wells parsed before the damage"
    assert "A03" in section.body_html


# ===========================================================================
# Figures
# ===========================================================================

def test_a_zero_byte_and_a_shredded_figure_are_named_with_their_reason(tmp_path):
    src = tmp_path / "plate_bad_figures"
    _write_png(src / "results" / "good.png")
    (src / "results" / "empty.png").write_bytes(b"")
    (src / "results" / "shredded.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"?" * 40)

    report = collect_report(src, run_dirs=[])
    section = report.section("figures")
    assert report.n_figures_found == 3
    assert report.n_figures_embedded == 1
    reasons = {name: why for name, why in section.table.rows}
    assert reasons["results/empty.png"] == "empty file"
    assert reasons["results/shredded.png"].startswith("could not decode")

    html = render_html(report)
    assert html.count("data:image/png;base64,") == 1
    assert "empty file" in html


def test_without_pillow_the_bytes_on_disk_are_embedded_unchanged(tmp_path):
    src = tmp_path / "plate_no_pil"
    png = _write_png(src / "results" / "a.png")
    from PIL import Image
    Image.new("RGB", (30, 20), (200, 30, 30)).save(src / "results" / "b.jpg")
    jpg = src / "results" / "b.jpg"

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "PIL", None)
        report = collect_report(src, run_dirs=[])

    section = report.section("figures")
    by_name = {f.path.name: f for f in section.figures}
    assert by_name["a.png"].data == png.read_bytes()
    assert by_name["a.png"].mime == "image/png"
    assert by_name["b.jpg"].data == jpg.read_bytes()
    assert by_name["b.jpg"].mime == "image/jpeg"
    assert "data:image/jpeg;base64," in render_html(report)


def test_an_oversized_figure_is_downscaled_before_it_is_embedded(tmp_path):
    from PIL import Image
    big = _write_png(tmp_path / "big.png", size=(300, 100))
    data, mime, reason = R._embed_figure(big, max_px=50)
    assert (mime, reason) == ("image/png", "")
    with Image.open(io.BytesIO(data)) as shrunk:
        assert shrunk.size == (50, 16)
    assert len(data) < big.stat().st_size


def test_a_palette_image_is_converted_before_it_is_embedded(tmp_path):
    from PIL import Image
    path = tmp_path / "a.gif"
    Image.new("P", (10, 10)).save(path)
    data, mime, reason = R._embed_figure(path, max_px=1400)
    assert (mime, reason) == ("image/png", "")
    with Image.open(io.BytesIO(data)) as converted:
        assert converted.format == "PNG", "a GIF was embedded as a GIF"
        assert converted.mode == "RGB"
        assert converted.size == (10, 10)


def test_a_figure_that_cannot_be_read_at_all_says_why(tmp_path):
    data, mime, reason = R._embed_figure(tmp_path, max_px=1400)   # a directory
    assert data is None and mime == ""
    assert reason == "unreadable (IsADirectoryError)"


def test_a_figure_outside_the_run_folder_is_captioned_by_name(tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    assert R._figure_title(src / "results" / "a.png", src) == "results/a.png"
    assert R._figure_title(tmp_path / "elsewhere" / "b.png", src) == "b.png"


class _StatlessPath(type(Path())):
    """A real path whose file was deleted after its bytes were read."""

    def stat(self, **kwargs):
        raise OSError(errno.ENOENT, "vanished after read", str(self))


def test_a_figure_deleted_after_it_was_read_is_still_embedded(tmp_path):
    src = tmp_path / "plate"
    png = _write_png(src / "results" / "a.png")
    section = R._collect_figures(src, {"raster": [_StatlessPath(str(png))]},
                                 10, 1400)
    assert len(section.figures) == 1
    assert section.figures[0].n_bytes == 0, "a size appeared from nowhere"
    assert section.figures[0].data == png.read_bytes()
    assert "Embedded 1 of 1 figure(s) found; 0 omitted." in section.body_html


# ===========================================================================
# Statistics
# ===========================================================================

def test_a_database_with_a_corrupt_page_still_reports_its_readable_tables(
        tmp_path):
    src = tmp_path / "plate_corrupt_db"
    path = src / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE cell (a TEXT, b INTEGER)")
    conn.executemany("INSERT INTO cell VALUES (?,?)",
                     [("x" * 300, i) for i in range(200)])
    conn.execute("CREATE TABLE png_list (c TEXT)")
    conn.executemany("INSERT INTO png_list VALUES (?)", [("hi",)] * 5)
    conn.commit()
    root_pages = dict(conn.execute("SELECT name, rootpage FROM sqlite_master"))
    conn.close()
    assert root_pages["cell"] == 2, "the test corrupts the wrong page"

    blob = bytearray(path.read_bytes())
    blob[4096:8192] = b"\xee" * 4096          # shred cell's root page
    path.write_bytes(bytes(blob))

    assert R._sqlite_table_counts(path) == [("cell", -1), ("png_list", 5)]

    report = collect_report(src, run_dirs=[])
    section = report.section("statistics")
    assert section.status != R.STATUS_MISSING
    assert ["png_list", "5"] in _body_table_rows(section)
    assert ["cell", "-1"] in _body_table_rows(section)


def _body_table_rows(section):
    """Every ``<td>`` row of the tables rendered inline in a section body."""
    import re
    rows = []
    for row in re.findall(r"<tr>(.*?)</tr>", section.body_html):
        cells = re.findall(r"<td>(.*?)</td>", row)
        if cells:
            rows.append(cells)
    return rows


def test_counting_the_tables_of_a_database_that_is_not_there_is_empty(tmp_path):
    assert R._sqlite_table_counts(tmp_path / "nope.db") == []


def test_a_result_csv_that_cannot_be_opened_is_noted_and_left_out(tmp_path):
    src = tmp_path / "plate"
    good = _write_csv(src / "results" / "good.csv", ["gene", "coef"],
                      [["g1", 1.0]])
    locked = _write_csv(src / "results" / "locked.csv", ["gene"], [["g2"]])
    os.chmod(locked, 0o000)
    try:
        report = collect_report(src, run_dirs=[])
        section = report.section("statistics")
        assert any("locked.csv unreadable (PermissionError)" in n
                   for n in section.notes)
        assert [row[0] for row in section.table.rows] == ["results/good.csv"]
        assert section.table.rows[0] == ["results/good.csv", "2 columns", "1",
                                         R._fmt_bytes(good.stat().st_size)]
        assert "locked.csv" not in section.body_html
    finally:
        _chmod_back(locked, 0o644)


def test_a_result_csv_with_no_rows_is_indexed_but_not_previewed(tmp_path):
    src = tmp_path / "plate"
    _write_csv(src / "results" / "a_hits.csv", ["gene", "coef"], [["g1", 1.0]])
    _write_csv(src / "results" / "b_empty.csv", ["gene", "coef"], [])

    report = collect_report(src, run_dirs=[])
    section = report.section("statistics")
    assert [row[0] for row in section.table.rows] == [
        "results/a_hits.csv", "results/b_empty.csv"]
    assert section.table.rows[1][2] == "0"
    assert "1 further result CSV(s) are listed above but not previewed" in \
        section.body_html


def test_a_database_and_a_csv_deleted_after_indexing_still_appear(tmp_path):
    src = tmp_path / "plate"
    db = _write_db(src, stamps=[_stamp()])
    csv_path = _write_csv(src / "results" / "hits.csv", ["gene"], [["g1"]])
    section = R._collect_statistics(
        src, {"databases": [_StatlessPath(str(db))],
              "result_csv": [_StatlessPath(str(csv_path))]}, 25)
    index = {row[0]: row for row in section.table.rows}
    assert index["measurements/measurements.db"][3] == "0 B"
    assert index["results/hits.csv"][2] == "1"
    assert index["results/hits.csv"][3] == "0 B"


# ===========================================================================
# Settings
# ===========================================================================

def test_a_setting_recorded_as_null_renders_as_an_empty_value(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", "2026-07-01_100000_aaaa__mask",
                             {"src": str(src), "cell_diameter": None,
                              "note": "x" * 400})

    report = collect_report(src, run_dirs=[run_dir], include_plan=False)
    rows = {row[0]: row[1] for row in report.section("settings").table.rows}
    assert rows["cell_diameter"] == ""
    assert R._render_setting(None) == ""
    assert len(rows["note"]) == 160 and rows["note"].endswith("…")


def test_a_settings_csv_that_cannot_be_opened_is_noted(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    good = _write_csv(src / "settings" / "a_mask.csv", ["Key", "Value"],
                      [["cell_diameter", "30"]])
    locked = _write_csv(src / "settings" / "b_measure.csv", ["Key", "Value"],
                        [["src", str(src)]])
    os.chmod(locked, 0o000)
    try:
        report = collect_report(src, run_dirs=[])
        section = report.section("settings")
        assert any("b_measure.csv unreadable (PermissionError)" in n
                   for n in section.notes)
        assert good.name in section.body_html
        assert locked.name not in section.body_html
        assert "cell_diameter" in section.body_html
    finally:
        _chmod_back(locked, 0o644)


def test_the_settings_plan_is_omitted_when_validate_is_unavailable(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", "2026-07-01_100000_aaaa__mask",
                             {"src": str(src), "cell_diameter": 30})
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spacr.validate", None)
        assert R._describe_plan_safe({"src": str(src)}, "mask") == ""
        report = collect_report(src, run_dirs=[run_dir], include_plan=True)
    section = report.section("settings")
    assert section.status == R.STATUS_OK
    assert "How spaCR reads these settings" not in section.body_html
    assert ["cell_diameter", "30"] in section.table.rows


# ===========================================================================
# Appendix
# ===========================================================================

def test_a_database_with_no_annotation_column_gets_no_annotation_block(tmp_path):
    src = tmp_path / "plate"
    db = _write_db(src, stamps=[_stamp()], annotation=False)
    assert R._annotation_summary(db) == ([], None, "")
    report = collect_report(src, run_dirs=[])
    section = report.section("appendix")
    assert "<h3>Annotations</h3>" not in section.body_html
    assert "<h3>Measured features</h3>" in section.body_html


def test_the_appendix_says_when_it_could_not_load_a_describer(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    artifacts = R._find_artifacts(src)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spacr.agreement", None)
        patch.setitem(sys.modules, "spacr.feature_dict", None)
        section = R._collect_appendix(src, artifacts)
    assert "spacr.agreement unavailable (ModuleNotFoundError)" in section.notes
    assert "spacr.feature_dict unavailable (ModuleNotFoundError)" in section.notes
    # The file inventory does not depend on either, so the section stands.
    assert section.status != R.STATUS_MISSING
    assert {row[0] for row in section.table.rows} == {"measurements/"}


def test_a_feature_dictionary_of_an_unexpected_shape_is_reported(tmp_path):
    """The guard against a feature_dict / pandas schema that has moved on."""
    import pandas as pd
    import spacr.feature_dict as fd

    src = tmp_path / "plate"
    db = _write_db(src, stamps=[_stamp()])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fd, "describe_database",
                      lambda *a, **k: pd.DataFrame({"col": ["cell_area"]}))
        families, rows, n_total, error = R._feature_dictionary(db, 10)
    assert (families, rows, n_total) == ([], [], 0)
    assert error == "feature dictionary unreadable (KeyError)"


# ===========================================================================
# collect_report
# ===========================================================================

def test_a_src_that_is_a_symlink_loop_reports_instead_of_raising(tmp_path):
    """BUG (fixed): pathlib raised RuntimeError straight out of collect_report."""
    loop = tmp_path / "loop"
    os.symlink(str(tmp_path / "other"), str(loop))
    os.symlink(str(loop), str(tmp_path / "other"))

    report = collect_report(loop, run_dirs=[])
    assert report.status == "empty"
    assert "does not exist" in report.sections[0].body_html
    assert [s.key for s in report.sections] == list(R.SECTION_KEYS)
    assert "does not exist" in render_html(report)


def test_a_report_written_without_spacr_version_says_unknown(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spacr.version", None)
        report = collect_report(src, run_dirs=[])
    assert report.spacr_version == "unknown"
    provenance = report.section("provenance")
    assert "Versions," not in provenance.body_html, (
        "versions were invented for a machine that could not be inspected")
    assert "spaCR unknown" in render_html(report)


# ===========================================================================
# Rendering
# ===========================================================================

def test_an_empty_table_renders_to_nothing_at_all():
    assert R._table_html(None) == ""
    assert R._table_html(Table()) == ""
    assert R._table_text(Table()) == []


def test_a_table_that_hid_rows_says_so_in_both_renderings():
    table = Table(columns=["gene", "coef"], rows=[["g1", "0.1"]],
                  caption="hits", n_total_rows=9)
    html = R._table_html(table)
    assert "8 further row(s) not shown." in html
    assert "<caption>hits</caption>" in html
    text = R._table_text(table)
    assert text[0] == "  hits"
    assert text[-1] == "  … and 8 further row(s) not shown."


def test_a_figure_that_is_only_listed_is_not_rendered_as_an_image():
    section = Section(title="Key figures", key="figures", figures=[
        Figure(path=Path("/plate/results/a.png"), title="results/a.png",
               reason="empty file")])
    assert R._figures_html(section) == "<div class='figgrid'></div>"


def test_section_notes_reach_the_plain_text_rendering(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[tmp_path / "no_such_run"])
    text = render_text(report)
    assert "  ! no such run folder:" in text


# ===========================================================================
# PDF
# ===========================================================================

def test_write_pdf_creates_the_folder_it_is_pointed_at(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    out = write_pdf(report, tmp_path / "deep" / "nested" / "r.pdf")
    assert out.is_file()
    assert out.read_bytes().startswith(b"%PDF-")


def test_a_figure_the_pdf_cannot_rasterise_gets_a_page_saying_so(tmp_path):
    """Without Pillow a JPEG is embedded raw, and matplotlib cannot draw it."""
    src = tmp_path / "plate_jpeg"
    (src / "results").mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (40, 30), (200, 30, 30)).save(src / "results" / "shot.jpg")

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "PIL", None)
        report = collect_report(src, run_dirs=[])

    assert report.n_figures_embedded == 1
    assert report.section("figures").figures[0].mime == "image/jpeg"

    kinds = [kind for kind, _payload in R._pdf_page_specs(report)]
    assert kinds.count("figure") == 1, (
        "the undrawable figure lost its page instead of gaining a caption")
    assert kinds[-1] == "figure"

    out = write_pdf(report, tmp_path / "jpeg.pdf")
    blob = out.read_bytes()
    assert _pdf_pages_on_disk(blob) == pdf_page_count(report)
    assert blob.count(b"/Subtype /Image") == 0, (
        "matplotlib drew a JPEG it was told to read as PNG")


def _pdf_pages_on_disk(blob: bytes) -> int:
    import re
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", blob)]
    assert counts, "no /Count in the PDF page tree"
    return max(counts)


# ===========================================================================
# build_report
# ===========================================================================

def test_build_report_both_beside_a_named_html_file_keeps_the_name(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    written = build_report(src, tmp_path / "out" / "plate1.html", fmt="both",
                           run_dirs=[])
    assert [p.name for p in written] == ["plate1.html", "plate1.pdf"]
    assert all(p.is_file() for p in written)
    assert all(p.parent == tmp_path / "out" for p in written)


def test_build_report_html_into_a_pdf_named_target_corrects_the_suffix(tmp_path):
    src = tmp_path / "plate"
    _write_db(src, stamps=[_stamp()])
    written = build_report(src, tmp_path / "r.pdf", fmt="html", run_dirs=[])
    assert [p.name for p in written] == ["r.html"]
    assert written[0].read_text(encoding="utf-8").startswith("<!doctype html>")
