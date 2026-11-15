"""How ``spacr.report`` reads a plate folder that misbehaves while it reads it.

The report is read by somebody who does not have spaCR, so a piece of the plate
folder that cannot be read -- a half-written CSV, a figure that will not
decode, a run journal that has gone missing, a plugin chapter whose builder is
wrong -- must cost that one item, be named, and leave the rest of the report.
One general test per unit, each walking a case table with ``subtests`` and
comparing exact values: the folder walk and sizing, the CSV preview, the
journal run matching and loading, the segmentation scorecard reader, figure
embedding, the failure-record line and the plugin chapters of
:func:`spacr.report.collect_report`.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

import pytest

from spacr import plugins
from spacr import report as R

MODULE = __name__


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

class _VanishingEntry:
    """A directory entry for a file that vanished between listing and stat."""

    def __init__(self, path: Path):
        self.name = path.name
        self.path = str(path)

    def is_dir(self, follow_symlinks=True):
        raise OSError(2, "No such file or directory")

    is_file = stat = is_dir


def _scandir_answering(monkeypatch, answers):
    """Make ``os.scandir`` answer ``answers[path]`` (a list or an exception)."""
    real = os.scandir

    def fake(path=".", *args, **kwargs):
        answer = answers.get(Path(path))
        if isinstance(answer, BaseException):
            raise answer
        if answer is not None:
            return iter(answer)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", fake)


def _write(path: Path, data) -> Path:
    """Write ``data`` (str or bytes) to ``path``, creating its parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def _names(paths):
    """Base names of ``paths`` in order."""
    return [p.name for p in paths]


# ---------------------------------------------------------------------------
# _iter_dir_files and _dir_stats
# ---------------------------------------------------------------------------

def test_iter_dir_files_walks_bounded_and_skips_what_it_cannot_read(
        tmp_path, monkeypatch, subtests):
    """Bulk folders are skipped, the budget truncates, unreadable entries cost only themselves."""
    _write(tmp_path / "orig" / "image.tif", b"0")
    _write(tmp_path / "results" / "a.csv", "x\n")
    _write(tmp_path / "results" / "deep" / "b.csv", "x\n")
    _write(tmp_path / "top.csv", "x\n")
    (tmp_path / "locked").mkdir()
    flat = tmp_path / "flat"
    for i in range(5):
        _write(flat / f"f{i}.csv", "x\n")

    cases = [
        ("missing folder", tmp_path / "absent", {}, [], False),
        ("orig never entered, breadth first", tmp_path, {},
         ["top.csv", "f0.csv", "f1.csv", "f2.csv", "f3.csv", "f4.csv",
          "a.csv", "b.csv"], False),
        ("no recursion", tmp_path, {"recurse": False}, ["top.csv"], False),
        ("budget reached", flat, {"budget": 2}, ["f0.csv", "f1.csv"], True),
        ("budget exactly the file count", flat, {"budget": 5},
         ["f0.csv", "f1.csv", "f2.csv", "f3.csv", "f4.csv"], False),
    ]
    _scandir_answering(monkeypatch, {
        tmp_path / "locked": PermissionError(13, "Permission denied")})
    for case_id, root, kwargs, names, truncated in cases:
        with subtests.test(msg=case_id):
            files, was_truncated = R._iter_dir_files(root, **kwargs)
            assert (_names(files), was_truncated) == (names, truncated)

    with subtests.test(msg="file vanishing mid walk"):
        _scandir_answering(monkeypatch, {
            flat: [_VanishingEntry(flat / "gone.csv")]})
        assert R._iter_dir_files(flat) == ([], False)


def test_dir_stats_sizes_everything_including_bulk_folders(
        tmp_path, monkeypatch, subtests):
    """Sizing descends into ``orig``, honours the budget and skips vanished files."""
    _write(tmp_path / "plate" / "orig" / "image.tif", b"0" * 100)
    _write(tmp_path / "plate" / "a.csv", "xy")
    (tmp_path / "plate" / "locked").mkdir()
    for i in range(4):
        _write(tmp_path / "four" / f"f{i}.bin", b"0" * (i + 1))
    _scandir_answering(monkeypatch, {
        tmp_path / "plate" / "locked": PermissionError(13, "Permission denied")})

    cases = [
        ("missing folder", tmp_path / "absent", {}, (0, 0, False)),
        ("bulk folder included, locked one skipped", tmp_path / "plate", {},
         (2, 102, False)),
        ("all four files", tmp_path / "four", {}, (4, 10, False)),
        ("budget reached", tmp_path / "four", {"budget": 4}, (4, 10, False)),
        ("budget below the count", tmp_path / "four", {"budget": 0},
         (0, 0, True)),
    ]
    for case_id, root, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert R._dir_stats(root, **kwargs) == expected

    with subtests.test(msg="budget truncates part way"):
        n_files, total, truncated = R._dir_stats(tmp_path / "four", budget=2)
        assert (n_files, truncated) == (2, True)
        assert total in {a + b for a in range(1, 5) for b in range(1, 5) if a != b}

    with subtests.test(msg="file vanishing before it is sized"):
        gone = tmp_path / "gone"
        gone.mkdir()
        _scandir_answering(monkeypatch, {gone: [_VanishingEntry(gone / "g.bin")]})
        assert R._dir_stats(gone) == (0, 0, False)


# ---------------------------------------------------------------------------
# _read_csv_head
# ---------------------------------------------------------------------------

def test_read_csv_head_keeps_the_rows_read_before_any_damage(tmp_path, subtests):
    """NUL bytes, the byte budget and an oversized field each end the scan, never the report."""
    many = "gene,score\n" + "".join(f"gene{i},{i}\n" for i in range(50))
    cases = [
        ("ordinary file", "gene,score\nA,1\nB,2\n", {"max_rows": 10},
         (["gene", "score"], [["A", "1"], ["B", "2"]], 2)),
        ("preview shorter than the file", "gene,score\nA,1\nB,2\nC,3\n",
         {"max_rows": 1}, (["gene", "score"], [["A", "1"]], 3)),
        ("half-flushed row", "gene,score\nA,1\nB,2\nC\x00,3\nD,4\n",
         {"max_rows": 10}, (["gene", "score"], [["A", "1"], ["B", "2"]], 2)),
        ("byte budget stops the count", many, {"max_rows": 2, "max_bytes": 20},
         (["gene", "score"], [["gene0", "0"], ["gene1", "1"]], 3)),
        ("header only", "gene,score\n", {"max_rows": 5},
         (["gene", "score"], [], 0)),
        ("empty file", "", {"max_rows": 5}, ([], [], 0)),
    ]
    for case_id, text, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            path = _write(tmp_path / f"{len(case_id)}_{case_id[:4]}.csv", text)
            assert R._read_csv_head(path, **kwargs) == expected

    with subtests.test(msg="field past the csv module's size limit"):
        path = _write(tmp_path / "damaged.csv",
                      "gene,score\nA,1\nB," + "x" * 200 + "\n")
        limit = csv.field_size_limit()
        csv.field_size_limit(50)
        try:
            got = R._read_csv_head(path, max_rows=10)
        finally:
            csv.field_size_limit(limit)
        assert got == (["gene", "score"], [["A", "1"]], 1)


# ---------------------------------------------------------------------------
# _settings_point_at and _load_journal_runs
# ---------------------------------------------------------------------------

def test_settings_point_at_matches_every_spelling_of_the_plate(tmp_path,
                                                               subtests):
    """Strings, lists, tuples and CSV-repr lists match; other shapes do not."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    cases = [
        ("not a mapping: None", None, False),
        ("not a mapping: list", [], False),
        ("no src key", {}, False),
        ("empty src", {"src": ""}, False),
        ("same string", {"src": str(plate)}, True),
        ("trailing slash", {"src": str(plate) + "/"}, True),
        ("other folder", {"src": "/other"}, False),
        ("list of plates", {"src": [str(plate), "/other"]}, True),
        ("tuple of plates", {"src": (str(plate),)}, True),
        ("set of plates", {"src": {str(plate)}}, True),
        ("list that went through a CSV", {"src": repr([str(plate), "/other"])},
         True),
        ("repr of a list without the plate", {"src": repr(["/other"])}, False),
        ("unclosed list text", {"src": "[unclosed"}, False),
        ("unclosed list naming the plate", {"src": f"[{plate}"}, False),
        ("bracketed literal that is not a list", {"src": "[1][0]"}, False),
        ("path with a NUL byte", {"src": f"{plate}\x00"}, False),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert R._settings_point_at(settings, plate) is expected


def _record(run_dir, manifest=None, settings=None):
    """The record ``_load_journal_runs`` builds for one run folder."""
    manifest = manifest or {}
    return {"dir": Path(run_dir), "manifest": manifest,
            "settings": settings or {},
            "app_key": manifest.get("app_key", "?"),
            "status": manifest.get("status", "?"),
            "start_utc": manifest.get("start_utc", ""),
            "elapsed_s": manifest.get("elapsed_s"),
            "env": manifest.get("env") if isinstance(manifest.get("env"), dict)
            else {}}


def test_load_journal_runs_lists_every_run_it_can_and_names_the_rest(
        tmp_path, monkeypatch, subtests):
    """Broken journals, folders and manifests cost one record or one problem line."""
    from spacr import run_journal as journal

    def run(name, manifest_text):
        run_dir = tmp_path / name
        run_dir.mkdir()
        if manifest_text is not None:
            _write(run_dir / "manifest.json", manifest_text)
        return run_dir

    corrupt = run("run_corrupt", "{not json")
    listed = run("run_list", "[1, 2]")
    bare = run("run_bare", None)
    good = run("run_good", '{"app_key": "measure", "status": "ok", '
                           '"start_utc": "2026-01-02", "elapsed_s": 4.5, '
                           '"env": {"python": "3.12"}}')
    old = run("run_old", '{"app_key": "mask", "start_utc": "2025-01-01", '
                         '"env": "not a dict"}')

    cases = [
        ("missing run folder", [tmp_path / "no_such_run"], [],
         [f"no such run folder: {tmp_path / 'no_such_run'}"]),
        ("corrupt manifest still listed", [corrupt], [_record(corrupt)], []),
        ("manifest that is a list", [listed], [_record(listed)], []),
        ("no manifest at all", [bare], [_record(bare)], []),
        ("newest first, env kept only as a dict", [old, good],
         [_record(good, {"app_key": "measure", "status": "ok",
                         "start_utc": "2026-01-02", "elapsed_s": 4.5,
                         "env": {"python": "3.12"}}),
          _record(old, {"app_key": "mask", "start_utc": "2025-01-01",
                        "env": "not a dict"})], []),
        ("no run folders and no journal search", None, [], []),
    ]
    for case_id, run_dirs, records, problems in cases:
        with subtests.test(msg=case_id):
            assert R._load_journal_runs(tmp_path, run_dirs, False, 5) == (
                records, problems)

    with subtests.test(msg="settings that will not load"):
        with monkeypatch.context() as m:
            def refuse(_path):
                raise ValueError("settings.csv is not a settings file")

            m.setattr(journal, "load_run_settings", refuse)
            record = R._load_journal_runs(tmp_path, [good], False, 5)[0][0]
            assert (record["settings"], record["app_key"]) == ({}, "measure")

    with subtests.test(msg="journal search keeps only runs of this plate"):
        with monkeypatch.context() as m:
            m.setattr(journal, "recent_runs",
                      lambda limit: [{"dir": good}, {"dir": old}][:limit])
            m.setattr(journal, "load_run_settings",
                      lambda path: {"src": str(tmp_path)} if path == good
                      else {"src": "/elsewhere"})
            records, problems = R._load_journal_runs(tmp_path, None, True, 5)
            assert ([r["dir"] for r in records], problems) == ([good], [])
            assert records[0]["settings"] == {"src": str(tmp_path)}

    with subtests.test(msg="journal that cannot be read"):
        with monkeypatch.context() as m:
            def unreadable(_limit):
                raise OSError("journal directory is not readable")

            m.setattr(journal, "recent_runs", unreadable)
            assert R._load_journal_runs(tmp_path, None, True, 5) == (
                [], ["could not read the run journal (OSError)"])

    with subtests.test(msg="journal module that will not import"):
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "spacr.run_journal", None)
            assert R._load_journal_runs(tmp_path, [good], True, 5) == (
                [], ["run journal unavailable (ModuleNotFoundError)"])


# ---------------------------------------------------------------------------
# _field_qcs_from_csv
# ---------------------------------------------------------------------------

def test_field_qcs_from_csv_rebuilds_scorecards_or_says_why_not(
        tmp_path, monkeypatch, subtests):
    """A readable scorecard becomes FieldQC rows; every failure is a named error."""
    from spacr.seg_qc import FieldQC

    header = "field,object_type,n_objects,severity,flags,note,area"
    good = _write(tmp_path / "segmentation_qc_cell.csv",
                  f"{header}\nf1,cell,3,warn,under;border,low,12.5\n"
                  "f2,cell,bad,,,,n/a\n")
    rows, error = R._field_qcs_from_csv(good)
    with subtests.test(msg="readable scorecard"):
        assert error is None
        assert [(q.field, q.object_type, q.n_objects, q.flags, q.severity,
                 q.note) for q in rows] == [
            ("f1", "cell", 3, ["under", "border"], "warn", "low"),
            ("f2", "cell", 0, [], "ok", "")]
        assert rows[0].metrics == {"area": 12.5}
        assert list(rows[1].metrics) == ["area"]
        assert rows[1].metrics["area"] != rows[1].metrics["area"]  # NaN
        assert all(isinstance(q, FieldQC) for q in rows)

    nul = _write(tmp_path / "segmentation_qc_nul.csv",
                 f"{header}\nf1\x00,cell,3,ok,,,1\n")
    huge = _write(tmp_path / "segmentation_qc_huge.csv",
                  f"{header}\n" + "x" * (csv.field_size_limit() + 1000)
                  + ",cell,3,ok,,,1\n")
    cases = [
        ("embedded NUL byte", nul,
         ([], "segmentation_qc_nul.csv is not readable as CSV "
              "(embedded NUL byte)")),
        ("field past the csv limit", huge,
         ([], "segmentation_qc_huge.csv is not readable as CSV "
              f"(field larger than field limit ({csv.field_size_limit()}))")),
        ("missing file", tmp_path / "segmentation_qc_gone.csv",
         ([], "segmentation_qc_gone.csv unreadable (FileNotFoundError)")),
    ]
    for case_id, path, expected in cases:
        with subtests.test(msg=case_id):
            assert R._field_qcs_from_csv(path) == expected

    with subtests.test(msg="file that cannot be opened"):
        with monkeypatch.context() as m:
            def refuse(*_args, **_kwargs):
                raise PermissionError("nope")

            m.setattr("builtins.open", refuse)
            assert R._field_qcs_from_csv(good) == (
                [], "segmentation_qc_cell.csv unreadable (PermissionError)")

    with subtests.test(msg="seg_qc module that will not import"):
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "spacr.seg_qc", None)
            assert R._field_qcs_from_csv(good) == (
                [], "spacr.seg_qc unavailable (ModuleNotFoundError)")


# ---------------------------------------------------------------------------
# _embed_figure
# ---------------------------------------------------------------------------

def test_embed_figure_returns_png_bytes_or_the_reason_it_cannot(
        tmp_path, monkeypatch, subtests):
    """Unreadable, empty and undecodable figures are named; the rest become PNG."""
    Image = pytest.importorskip("PIL.Image")
    _write(tmp_path / "empty.png", b"")
    _write(tmp_path / "broken.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 4)
    _write(tmp_path / "text.png", b"this is not an image")
    (tmp_path / "folder.png").mkdir()
    Image.new("RGB", (40, 30), "white").save(tmp_path / "small.png")
    Image.new("RGB", (400, 200), "white").save(tmp_path / "big.png")
    Image.new("RGB", (300, 300), "white").convert("P").save(tmp_path / "pal.png")
    Image.new("RGB", (40, 30), "white").save(tmp_path / "small.jpg")
    Image.new("CMYK", (40, 30)).save(tmp_path / "cmyk.jpg")

    refusals = [
        ("missing file", "gone.png", (None, "", "unreadable (FileNotFoundError)")),
        ("a folder", "folder.png", (None, "", "unreadable (IsADirectoryError)")),
        ("zero bytes", "empty.png", (None, "", "empty file")),
        ("PNG signature over garbage", "broken.png",
         (None, "", "could not decode (OSError)")),
        ("not an image at all", "text.png",
         (None, "", "could not decode (UnidentifiedImageError)")),
    ]
    for case_id, name, expected in refusals:
        with subtests.test(msg=case_id):
            assert R._embed_figure(tmp_path / name, 800) == expected

    with subtests.test(msg="small PNG passes through byte for byte"):
        raw = (tmp_path / "small.png").read_bytes()
        assert R._embed_figure(tmp_path / "small.png", 800) == (
            raw, "image/png", "")

    # name, max_px, expected size and mode of the embedded PNG
    re_encoded = [
        ("oversized PNG is downscaled", "big.png", 100, (100, 50), "RGB"),
        ("palette PNG is converted", "pal.png", 100, (100, 100), "RGB"),
        ("small JPEG is still re-encoded", "small.jpg", 800, (40, 30), "RGB"),
        ("CMYK JPEG is converted", "cmyk.jpg", 800, (40, 30), "RGB"),
    ]
    for case_id, name, max_px, size, mode in re_encoded:
        with subtests.test(msg=case_id):
            data, mime, reason = R._embed_figure(tmp_path / name, max_px)
            assert (mime, reason, data[:8]) == ("image/png", "",
                                                b"\x89PNG\r\n\x1a\n")
            with Image.open(io.BytesIO(data)) as embedded:
                assert (embedded.size, embedded.mode) == (size, mode)

    with subtests.test(msg="without Pillow the bytes on disk are embedded"):
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "PIL", None)
            png = _write(tmp_path / "plot.png", b"\x89PNG-not-really")
            jpg = _write(tmp_path / "plot.jpg", b"\xff\xd8-not-really")
            assert R._embed_figure(png, 800) == (
                b"\x89PNG-not-really", "image/png", "")
            assert R._embed_figure(jpg, 800) == (
                b"\xff\xd8-not-really", "image/jpeg", "")


# ---------------------------------------------------------------------------
# _failure_record_text
# ---------------------------------------------------------------------------

class _Opaque:
    """An object whose repr must never be called."""

    def __repr__(self):
        raise RuntimeError("this repr is not safe to call")


def test_failure_record_text_is_one_stable_bounded_line(subtests):
    """Strings pass, other shapes become sorted JSON, the unserialisable is named by type."""
    from decimal import Decimal

    recursive: dict = {}
    recursive["self"] = recursive
    cases = [
        ("ordinary record", {"unit": "plate1", "tried": 3}, {},
         '{"tried": 3, "unit": "plate1"}'),
        ("nested list sorted", ["legacy", {"z": 1, "a": 2}], {},
         '["legacy", {"a": 2, "z": 1}]'),
        ("legacy string", "OSError: truncated <file>", {},
         "OSError: truncated <file>"),
        ("whitespace collapsed", "Traceback:\n    line 1\n\tline 2", {},
         "Traceback: line 1 line 2"),
        ("non-ASCII kept", {"well": "Å01"}, {}, '{"well": "Å01"}'),
        ("empty string", "", {}, "(empty failure record)"),
        ("blank string", "   ", {}, "(empty failure record)"),
        ("None is JSON null", None, {}, "null"),
        ("unknown object named by type", {"held": _Opaque()}, {},
         '{"held": "<_Opaque>"}'),
        ("number JSON cannot hold", {"scale": Decimal("NaN")}, {},
         '{"scale": "<Decimal>"}'),
        ("recursive record", recursive, {}, "<dict>"),
        ("default width", "x" * 600, {}, "x" * 499 + "…"),
        ("narrow width", "x" * 5_000, {"width": 100}, "x" * 99 + "…"),
        ("exactly the width", "x" * 100, {"width": 100}, "x" * 100),
    ]
    for case_id, value, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert R._failure_record_text(value, **kwargs) == expected


# ---------------------------------------------------------------------------
# collect_report: plugin chapters
# ---------------------------------------------------------------------------

def good_section(_context):
    """A plugin builder that returns a valid section with a blank key."""
    return R.Section(title="Extra", key="", text_lines=["fine"])


def untitled_section(_context):
    """A plugin builder that returns a section with no title of its own."""
    return R.Section(title="", key="extra_chapter")


def wrong_key_section(_context):
    """A plugin builder that returns a section under someone else's key."""
    return R.Section(title="Extra", key="statistics")


def not_a_section(_context):
    """A plugin builder that returns the wrong type entirely."""
    return {"title": "Extra"}


def raising_section(_context):
    """A plugin builder that fails outright."""
    raise RuntimeError("the builder fell over")


NOT_CALLABLE = "this is a string, not a builder"

CORE_KEYS = ["run_status", "provenance", "segmentation_qc", "plate_qc",
             "figures", "statistics", "settings", "appendix"]


def _contribution(key, builder, after="statistics"):
    """A report-section contribution pointing at a builder in this module."""
    return plugins.ReportSectionContribution(
        key=key, title="Extra chapter", builder=f"{MODULE}:{builder}",
        after=after)


def test_collect_report_inserts_plugin_chapters_or_a_visible_problem(
        tmp_path, monkeypatch, subtests):
    """Working chapters land where they asked; broken ones become a named problem chapter."""
    recorded = []
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda *args, **kw: recorded.append(args))

    def collect(*contributions):
        recorded.clear()
        monkeypatch.setattr(plugins, "report_sections",
                            lambda: tuple(("demo", c) for c in contributions))
        return R.collect_report(tmp_path, search_journal=False)

    ref = f"'{MODULE}:"
    failures = [
        ("duplicate core key", "statistics", "good_section",
         "ValueError: report section key 'statistics' already exists"),
        ("builder that is not callable", "extra_chapter", "NOT_CALLABLE",
         f"TypeError: {ref}NOT_CALLABLE' is not callable"),
        ("builder returning a dict", "extra_chapter", "not_a_section",
         f"TypeError: {ref}not_a_section' returned dict, expected Section"),
        ("builder returning another key", "extra_chapter", "wrong_key_section",
         "ValueError: builder returned key 'statistics'; expected "
         "'extra_chapter'"),
        ("builder that raises", "extra_chapter", "raising_section",
         "RuntimeError: the builder fell over"),
    ]
    for case_id, key, builder, line in failures:
        with subtests.test(msg=case_id):
            result = collect(_contribution(key, builder))
            keys = [s.key for s in result.sections]
            assert keys == CORE_KEYS[:6] + [key] + CORE_KEYS[6:]
            section = result.sections[6]
            assert (section.title, section.status, section.text_lines) == (
                "Extra chapter", R.STATUS_PROBLEM,
                [f"Plugin report section failed: {line}"])
            assert section.body_html.startswith(
                "<p>This plugin report section could not be generated. <code>")
            assert [args[:2] for args in recorded] == [
                ("demo", f"Report section {key!r} failed")]

    with subtests.test(msg="working chapter after statistics"):
        result = collect(_contribution("extra_chapter", "good_section"))
        keys = [s.key for s in result.sections]
        assert keys == CORE_KEYS[:6] + ["extra_chapter"] + CORE_KEYS[6:]
        section = result.sections[6]
        assert (section.title, section.status, section.text_lines) == (
            "Extra", R.Section(title="x").status, ["fine"])
        assert recorded == []

    with subtests.test(msg="untitled chapter takes the contribution title"):
        result = collect(_contribution("extra_chapter", "untitled_section",
                                       after="no_such_section"))
        assert [s.key for s in result.sections] == CORE_KEYS + ["extra_chapter"]
        assert result.sections[-1].title == "Extra chapter"

    with subtests.test(msg="registry that cannot be read"):
        def explode():
            raise RuntimeError("the plugin registry is corrupt")

        monkeypatch.setattr(plugins, "report_sections", explode)
        result = R.collect_report(tmp_path, search_journal=False)
        assert [s.key for s in result.sections] == CORE_KEYS
