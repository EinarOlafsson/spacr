"""The Report generator — :mod:`spacr.report`.

Everything here runs against a *real* temporary run folder built the way a
finished spaCR plate looks: a ``measurements/measurements.db`` carrying a
``run_status`` stamp, a ``qc/segmentation_qc_cell.csv`` scorecard, a
``results`` tree of CSVs and PNGs, a ``settings`` copy, and a run-journal
folder with a manifest.

The suite pins the properties the module lives or dies by:

* **a missing section is stated, not omitted** — a folder with no QC card
  still has a "Segmentation QC" section, and it says *not run*;
* **failure is at the top** — a partial run's status is in the first
  section, not in an appendix;
* **the HTML is self-contained** — no ``http(s)://``, no ``<link>``, no
  ``<script>``, and every ``src=`` is a ``data:`` URI. Asserted, not
  assumed: a report that only renders on the machine that made it defeats
  the entire point;
* **user data is escaped** — a well called ``A01<script>`` renders as text;
* **truncation is stated** — 50 figures with a cap of 20 embeds 20 and says
  30 were omitted, so a short figure list never reads as "that was all";
* **nothing is recomputed** — the plate verdict is whatever
  :func:`spacr.seg_qc.summarize_qc` says about the stored scorecard.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from pathlib import Path

import pytest

from spacr.report import (
    DEFAULT_MAX_FIGURES,
    SECTION_KEYS,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_PROBLEM,
    Report,
    Section,
    Table,
    build_report,
    collect_report,
    pdf_page_count,
    render_html,
    render_text,
    write_html,
    write_pdf,
)


# ---------------------------------------------------------------------------
# Builders
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


def _write_png(path: Path, size=(24, 18)) -> Path:
    """A real PNG — the report decodes and re-encodes what it embeds."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 90, 160)).save(path)
    return path


def _write_qc_card(src: Path, object_type="cell", n_fields=12,
                   field_names=None) -> Path:
    rows = []
    for i in range(n_fields):
        severity = "fail" if i < 2 else ("warn" if i < 4 else "ok")
        flags = "high_border_fraction" if severity != "ok" else ""
        name = field_names[i] if field_names else f"plate1_A{i + 1:02d}_1"
        rows.append([name, object_type, 100 + i, severity, flags,
                     0.42, 1.0, 0.21, 22.5, 0.03,
                     f"{severity} because of the border fraction"])
    return _write_csv(src / "qc" / f"segmentation_qc_{object_type}.csv",
                      QC_HEADER, rows)


def _write_db(src: Path, stamps=()) -> Path:
    """A measurements.db with a cell table, png_list, and optional stamps."""
    path = src / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE cell (prc TEXT, cell_area REAL, "
                     "cell_channel_1_mean_intensity REAL)")
        conn.executemany("INSERT INTO cell VALUES (?, ?, ?)",
                         [(f"plate1_A01_{i}", 100.0 + i, 5.0) for i in range(30)])
        conn.execute("CREATE TABLE png_list (png_path TEXT, prc TEXT, "
                     "annotation INTEGER, test_annotation INTEGER)")
        conn.executemany("INSERT INTO png_list VALUES (?, ?, ?, ?)",
                         [(f"/crops/{i}.png", f"plate1_A01_{i}", 1, 1)
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


def _stamp(name="measure_crop", status="complete", attempted=100, failed=0,
           failures=None, summary=""):
    return ("run-1", name, status, attempted, attempted - failed, failed,
            failed / attempted if attempted else 0.0,
            "2026-07-01T10:00:00+00:00", "2026-07-01T10:20:00+00:00",
            json.dumps(failures or []), summary)


def _write_run_dir(root: Path, src: Path, app_key="mask", status="success",
                   settings=None, name="2026-07-01_100000_abcd1234__mask",
                   manifest_extra=None) -> Path:
    """A run-journal folder the way :func:`spacr.run_journal.open_run` writes one.

    ``manifest_extra`` merges into the manifest, so a fingerprint test can
    pin those fields without every other test's manifest gaining them.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"src": str(src), "channels": [0, 1, 2], "cell_diameter": 30,
               "magnification": 20, "plot": True}
    payload.update(settings or {})
    (run_dir / "settings.json").write_text(json.dumps(payload, default=str))
    manifest = {
        "app_key": app_key,
        "start_utc": "2026-07-01T10:00:00+00:00",
        "end_utc": "2026-07-01T10:20:00+00:00",
        "elapsed_s": 1200.0,
        "status": status,
        "env": {"spacr": "1.3.6", "python": "3.11.9", "torch": "2.4.0",
                "cellpose": "4.0.1", "platform": "Linux-test"},
        "model_hashes": {"cellpose_cyto": "cyto3.pth:0123456789abcdef"},
        "n_settings": len(payload),
        "traceback": ("Traceback (most recent call last):\n"
                      "ValueError: the plate ran out of nuclei"
                      if status == "failed" else None),
    }
    manifest.update(manifest_extra or {})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, default=str))
    return run_dir


@pytest.fixture
def full_run(tmp_path):
    """A plate folder with every section's evidence present."""
    src = tmp_path / "plate1"
    _write_qc_card(src)
    _write_csv(src / "qc" / "plate_wells.csv", LAYOUT_HEADER,
               [["plate1", f"A{i + 1:02d}", "r01", f"c{i + 1:02d}", 0, i, 20,
                 1.5 + i, 0, i < 2] for i in range(8)])
    _write_csv(src / "results" / "results.csv", ["gene", "coef", "p_value"],
               [[f"gene{i}", 0.1 * i, 0.001 * i] for i in range(40)])
    _write_csv(src / "settings" / "gen_mask_settings.csv", ["Key", "Value"],
               [["src", str(src)], ["channels", "[0, 1, 2]"],
                ["cell_diameter", "30"]])
    _write_db(src, stamps=[_stamp(summary="every field processed")])
    for i in range(3):
        _write_png(src / "results" / f"figure_{i}.png")
    (src / "figure").mkdir(parents=True, exist_ok=True)
    (src / "figure" / "plate1_merged_all_channels.pdf").write_bytes(b"%PDF-1.4\n")
    run_dir = _write_run_dir(tmp_path / "runs", src)
    return {"src": src, "run_dir": run_dir, "runs_root": tmp_path / "runs"}


# ---------------------------------------------------------------------------
# Structure: every section, always, in reading order
# ---------------------------------------------------------------------------

def test_every_section_is_present_and_in_reading_order(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    assert [s.key for s in report.sections] == list(SECTION_KEYS)
    assert report.sections[0].key == "run_status", (
        "run status must be first — failure is not an appendix item")


def test_a_complete_run_folder_leaves_nothing_unavailable(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    assert report.missing_sections == [], (
        f"sections went missing on a complete folder: {report.missing_sections}")
    assert report.status == "complete"


def test_sections_carry_their_own_titles_and_keys(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    for section in report.sections:
        assert isinstance(section, Section)
        assert section.title.strip()
        assert section.key in SECTION_KEYS


# ---------------------------------------------------------------------------
# A missing section is stated, never omitted
# ---------------------------------------------------------------------------

def test_a_folder_with_no_qc_card_still_has_a_qc_section_saying_not_run(tmp_path):
    src = tmp_path / "plate_no_qc"
    _write_db(src, stamps=[_stamp()])
    _write_csv(src / "results" / "results.csv", ["gene", "coef"], [["g1", 1.0]])

    report = collect_report(src, run_dirs=[])
    section = report.section("segmentation_qc")

    assert section is not None, "the QC heading vanished with the QC data"
    assert section.status == STATUS_MISSING
    assert "not run" in section.body_html.lower()
    html = render_html(report)
    assert "Segmentation QC" in html
    assert "not run" in html.lower()
    # And the distinction the whole rule exists for.
    assert "not the same as clean" in html.lower()


def test_missing_sections_are_named_in_the_html_banner(tmp_path):
    src = tmp_path / "plate_thin"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    html = render_html(report)
    assert report.missing_sections
    assert "Not available for this run" in html
    for key in report.missing_sections:
        assert f"id='{key}'" in html, f"{key} has no section in the document"


def test_every_section_appears_in_the_html_even_when_missing(tmp_path):
    report = collect_report(tmp_path / "empty_plate", run_dirs=[])
    html = render_html(report)
    for key in SECTION_KEYS:
        assert f"id='{key}'" in html


# ---------------------------------------------------------------------------
# Failure is surfaced, not buried
# ---------------------------------------------------------------------------

def test_a_partial_run_is_reported_in_the_first_section(tmp_path):
    src = tmp_path / "plate_partial"
    _write_db(src, stamps=[_stamp(
        status="partial", attempted=100, failed=7,
        failures=[{"item": "A01_f3", "stage": "crop", "error": "ValueError: bad"}],
        summary="7 of 100 fields failed")])

    report = collect_report(src, run_dirs=[])
    assert report.status == "partial"
    assert report.has_failures

    first = report.sections[0]
    blob = (first.body_html + " ".join(first.text_lines)).lower()
    assert "partial" in blob
    assert "7" in first.body_html
    assert "A01_f3" in first.body_html, "the failing item was not named"

    html = render_html(report)
    # It has to be at the top, not merely somewhere.
    assert html.index("PARTIAL") < html.index("id='appendix'")


def test_a_failed_journal_run_is_reported_in_the_first_section(tmp_path):
    src = tmp_path / "plate_failed"
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", src, status="failed")

    report = collect_report(src, run_dirs=[run_dir])
    assert report.status == "failed"
    assert "ran out of nuclei" in report.sections[0].body_html


def test_an_unstamped_folder_says_completeness_is_unknown(tmp_path):
    src = tmp_path / "plate_unstamped"
    _write_qc_card(src)
    _write_db(src)                       # no run_status table at all

    report = collect_report(src, run_dirs=[])
    assert report.status == "unknown"
    body = report.sections[0].body_html.lower()
    assert "no run-status stamp" in body
    assert "not evidence of success" in body


def test_a_complete_stamp_reads_as_complete(tmp_path):
    src = tmp_path / "plate_ok"
    _write_db(src, stamps=[_stamp(failed=0)])
    report = collect_report(src, run_dirs=[])
    assert report.status == "complete"
    assert not report.has_failures


def test_a_json_sidecar_stamp_is_read_too(tmp_path):
    src = tmp_path / "plate_sidecar"
    (src / "results").mkdir(parents=True)
    (src / "results" / "wells.csv").write_text("well,value\nA01,1\n")
    (src / "results" / "wells.run_status.json").write_text(json.dumps({
        "run_id": "r2", "name": "convert_to_yokogawa", "status": "partial",
        "n_attempted": 40, "n_succeeded": 38, "n_failed": 2,
        "failure_rate": 0.05, "started_utc": "", "stamped_utc": "",
        "failures": [{"item": "img_07.nd2", "stage": "convert",
                      "error": "OSError: truncated"}],
        "summary": "2 of 40 images failed to convert",
    }))
    report = collect_report(src, run_dirs=[])
    assert report.status == "partial"
    assert "img_07.nd2" in report.sections[0].body_html


# ---------------------------------------------------------------------------
# Self-containment — the entire point of the feature
# ---------------------------------------------------------------------------

def test_html_makes_no_external_request(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    html = render_html(report)

    assert "http://" not in html
    assert "https://" not in html
    assert "<link" not in html.lower(), "an external stylesheet crept in"
    assert "<script" not in html.lower(), "the report must carry no JavaScript"
    assert "@import" not in html
    assert "url(" not in html, "a CSS url() would fetch from outside the file"
    assert "<iframe" not in html.lower()

    for value in re.findall(r'src="([^"]*)"', html):
        assert value.startswith("data:"), f"external image reference: {value[:60]}"
    for value in re.findall(r'href="([^"]*)"', html):
        assert value.startswith("#"), f"external link: {value[:60]}"


def test_figures_are_embedded_as_base64_data_uris(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    html = render_html(report)
    assert report.n_figures_embedded == 3
    assert html.count("data:image/png;base64,") == 3


def test_the_written_html_file_is_one_file_with_no_siblings(full_run, tmp_path):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    out_dir = tmp_path / "out"
    written = write_html(report, out_dir / "report.html")
    assert written.is_file()
    assert sorted(p.name for p in out_dir.iterdir()) == ["report.html"]
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_generating_a_report_writes_nothing_into_the_run_folder(full_run, tmp_path):
    src = full_run["src"]
    before = sorted(str(p) for p in src.rglob("*"))
    build_report(src, tmp_path / "out", fmt="html",
                 run_dirs=[full_run["run_dir"]])
    assert sorted(str(p) for p in src.rglob("*")) == before


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

SCRIPTY_WELL = 'A01<script>alert("xss")</script>&<b>'


def test_a_well_name_containing_markup_is_escaped_everywhere(tmp_path):
    src = tmp_path / "plate_xss"
    names = [SCRIPTY_WELL] + [f"plate1_A{i:02d}_1" for i in range(1, 6)]
    _write_qc_card(src, n_fields=6, field_names=names)
    _write_csv(src / "results" / "hits.csv", ["well", "score"],
               [[SCRIPTY_WELL, 1.0]])
    _write_csv(src / "settings" / "gen_mask_settings.csv", ["Key", "Value"],
               [["dodgy", SCRIPTY_WELL]])
    _write_db(src, stamps=[_stamp(
        status="partial", failed=1, summary=f"{SCRIPTY_WELL} failed",
        failures=[{"item": SCRIPTY_WELL, "stage": "crop", "error": "boom"}])])

    report = collect_report(src, run_dirs=[])
    html = render_html(report)

    assert "<script>" not in html
    assert "alert(" not in html or "&lt;script&gt;" in html
    assert "&lt;script&gt;" in html, "the well name never reached the page"
    assert "&amp;" in html
    # The escaped form must appear in more than one place — QC, statistics
    # and the failure list all carry it.
    assert html.count("&lt;script&gt;") >= 2


def test_escaping_survives_a_title_from_user_data(tmp_path):
    src = tmp_path / "plate_title"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, title="<b>my plate</b>", run_dirs=[])
    html = render_html(report)
    assert "<b>my plate</b>" not in html
    assert "&lt;b&gt;my plate&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# Figure cap — a truncated list must never read as "that was all"
# ---------------------------------------------------------------------------

def test_fifty_figures_with_a_cap_of_twenty_embeds_twenty_and_says_thirty_omitted(tmp_path):
    src = tmp_path / "plate_many_figures"
    for i in range(50):
        _write_png(src / "results" / f"figure_{i:03d}.png")

    report = collect_report(src, max_figures=20, run_dirs=[])
    figures = report.section("figures")

    assert report.n_figures_found == 50
    assert report.n_figures_embedded == 20
    assert len(figures.figures) == 20

    html = render_html(report)
    assert html.count("data:image/png;base64,") == 20
    assert "30 omitted" in html, "the omitted count is not stated"
    assert "20 of 50" in html
    # Every omitted figure is still named, so none looks unproduced.
    assert figures.table is not None
    assert figures.table.n_total_rows == 30


def test_the_default_cap_is_the_documented_one(tmp_path):
    src = tmp_path / "plate_default_cap"
    for i in range(DEFAULT_MAX_FIGURES + 5):
        _write_png(src / "results" / f"figure_{i:03d}.png")
    report = collect_report(src, run_dirs=[])
    assert report.n_figures_embedded == DEFAULT_MAX_FIGURES
    assert "5 omitted" in render_html(report)


def test_a_cap_of_zero_embeds_nothing_but_still_lists_everything(tmp_path):
    src = tmp_path / "plate_cap0"
    for i in range(4):
        _write_png(src / "results" / f"figure_{i}.png")
    report = collect_report(src, max_figures=0, run_dirs=[])
    assert report.n_figures_embedded == 0
    assert report.n_figures_found == 4
    html = render_html(report)
    assert "4 omitted" in html
    assert "data:image/png;base64," not in html


def test_pdf_only_figures_are_listed_not_silently_dropped(tmp_path):
    src = tmp_path / "plate_vector"
    (src / "figure").mkdir(parents=True)
    for i in range(3):
        (src / "figure" / f"plot_{i}.pdf").write_bytes(b"%PDF-1.4\n")
    report = collect_report(src, run_dirs=[])
    figures = report.section("figures")
    assert report.n_figures_found == 3
    assert report.n_figures_embedded == 0
    assert figures.status != STATUS_MISSING
    html = render_html(report)
    assert "vector" in html.lower()
    assert "plot_0.pdf" in html


def test_a_folder_with_no_figures_says_so(tmp_path):
    src = tmp_path / "plate_no_figures"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    figures = report.section("figures")
    assert figures.status == STATUS_MISSING
    assert "No figures were found" in figures.body_html


# ---------------------------------------------------------------------------
# Nothing is recomputed
# ---------------------------------------------------------------------------

def test_the_plate_verdict_is_the_one_seg_qc_reached(tmp_path):
    from spacr.seg_qc import FieldQC, summarize_qc

    src = tmp_path / "plate_qc_verdict"
    _write_qc_card(src, n_fields=12)
    report = collect_report(src, run_dirs=[])
    section = report.section("segmentation_qc")

    # Reconstruct the same verdict directly from the stored card.
    with open(src / "qc" / "segmentation_qc_cell.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    expected = summarize_qc([
        FieldQC(field=r["field"], object_type=r["object_type"],
                n_objects=int(r["n_objects"]), severity=r["severity"],
                flags=[f for f in r["flags"].split(";") if f])
        for r in rows])

    assert section.table is not None
    row = section.table.rows[0]
    assert row[1] == str(expected["n_fields"])
    assert row[4] == str(expected["n_fail"])
    assert row[5] == expected["verdict"].upper()
    assert expected["message"] in " ".join(row)


def test_a_failing_plate_marks_the_qc_section_as_needing_attention(tmp_path):
    src = tmp_path / "plate_qc_fail"
    _write_qc_card(src, n_fields=12)          # 2 of 12 fail = 17 % > 10 %
    report = collect_report(src, run_dirs=[])
    assert report.section("segmentation_qc").status == STATUS_PROBLEM


def test_plate_qc_says_not_run_when_no_layout_was_exported(tmp_path):
    src = tmp_path / "plate_no_layout"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    section = report.section("plate_qc")
    assert section.status == STATUS_MISSING
    assert "not run" in section.body_html.lower()
    # And it says why it did not just compute one.
    assert "p-value" in section.body_html


def test_plate_qc_reports_an_exported_layout(tmp_path):
    src = tmp_path / "plate_layout"
    _write_csv(src / "qc" / "plate_wells.csv", LAYOUT_HEADER,
               [["plate1", f"A{i + 1:02d}", "r01", f"c{i + 1:02d}", 0, i, 20,
                 1.5 + i, 0, i < 3] for i in range(8)])
    report = collect_report(src, run_dirs=[])
    section = report.section("plate_qc")
    assert section.status == STATUS_OK
    assert section.table is not None
    assert section.table.rows[0][1] == "8"      # wells
    assert section.table.rows[0][2] == "3"      # edge wells


def test_a_qc_folder_csv_that_is_not_a_layout_is_not_mistaken_for_one(tmp_path):
    src = tmp_path / "plate_odd_qc"
    _write_csv(src / "qc" / "notes.csv", ["a", "b"], [[1, 2]])
    report = collect_report(src, run_dirs=[])
    assert report.section("plate_qc").status == STATUS_MISSING


# ---------------------------------------------------------------------------
# Provenance, settings, statistics, appendix
# ---------------------------------------------------------------------------

def test_provenance_reports_the_journalled_run_and_its_versions(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    section = report.section("provenance")
    assert section.status == STATUS_OK
    assert section.table is not None
    assert section.table.rows[0][1] == "mask"
    assert "1.3.6" in section.body_html          # spacr version from the manifest
    assert "cellpose" in section.body_html
    assert "cyto3.pth" in section.body_html      # model fingerprint


def test_provenance_without_a_journal_entry_says_whose_versions_these_are(tmp_path):
    src = tmp_path / "plate_no_journal"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    section = report.section("provenance")
    assert section.status == STATUS_MISSING
    assert "No journalled run was found" in section.body_html
    assert "not the machine that produced the data" in section.body_html


# -- fingerprints ----------------------------------------------------------
#
# The three outcomes must be told apart. An absent fingerprint that reads as
# an absent difference is a false assurance, which is worse than no
# fingerprint at all -- so each branch is pinned by its own test.

_HASHED_MANIFEST = {
    "schema_version": 2,
    "input_hashing": "on",
    "settings_sha256": "aa" * 32,
    "input_tree_sha256": "bb" * 32,
    "output_tree_sha256": "cc" * 32,
    "performance": {"input_files": 4812, "input_bytes": 61_200_000_000,
                    "output_files": 319, "output_bytes": 2_100_000_000},
}


def _provenance_for(tmp_path, manifest_extra, name="plate_fp"):
    src = tmp_path / name
    _write_db(src, stamps=[_stamp()])
    run_dir = _write_run_dir(tmp_path / "runs", src,
                             manifest_extra=manifest_extra)
    report = collect_report(src, run_dirs=[run_dir])
    return report.section("provenance")


def test_provenance_carries_the_input_and_output_digests(tmp_path):
    section = _provenance_for(tmp_path, _HASHED_MANIFEST)
    for digest in ("aa" * 32, "bb" * 32, "cc" * 32):
        assert digest in section.body_html
        assert any(digest in line for line in section.text_lines)
    # The counts come with them, or a digest is a number with no scale.
    assert "4,812" in section.body_html
    assert "319" in section.body_html


def test_skipped_hashing_says_so_rather_than_showing_nothing(tmp_path):
    manifest = dict(_HASHED_MANIFEST, input_hashing="skipped")
    section = _provenance_for(tmp_path, manifest, name="plate_skipped")
    assert "input hashing was off" in section.body_html
    # The tree digests must NOT appear -- they are stale or absent, and
    # printing them would assert a verification that did not happen.
    assert "bb" * 32 not in section.body_html
    assert "cc" * 32 not in section.body_html
    # The settings digest is written either way, so it survives.
    assert "aa" * 32 in section.body_html


def test_a_manifest_older_than_the_field_is_not_reported_as_a_choice(tmp_path):
    """Schema 1 predates input digests: nothing was declined."""
    section = _provenance_for(tmp_path, {"schema_version": 1,
                                         "settings_sha256": "dd" * 32},
                              name="plate_old")
    assert "journalled before spaCR recorded input digests" in section.body_html
    assert "hashing was off" not in section.body_html


def test_two_runs_keep_their_own_digests_and_are_never_merged(tmp_path):
    src = tmp_path / "plate_two"
    _write_db(src, stamps=[_stamp()])
    first = _write_run_dir(tmp_path / "runs", src, manifest_extra=dict(
        _HASHED_MANIFEST, input_tree_sha256="11" * 32), name="run_one")
    second = _write_run_dir(tmp_path / "runs", src, manifest_extra=dict(
        _HASHED_MANIFEST, input_tree_sha256="22" * 32), name="run_two")
    section = collect_report(src, run_dirs=[first, second]).section("provenance")
    assert "11" * 32 in section.body_html
    assert "22" * 32 in section.body_html
    assert "run_one" in section.body_html and "run_two" in section.body_html


def test_no_journalled_run_renders_no_fingerprint_block(tmp_path):
    src = tmp_path / "plate_none"
    _write_db(src, stamps=[_stamp()])
    section = collect_report(src, run_dirs=[]).section("provenance")
    assert "Fingerprints" not in section.body_html


def test_settings_section_carries_every_recorded_setting(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]],
                            include_plan=False)
    section = report.section("settings")
    assert section.status == STATUS_OK
    keys = {row[0] for row in section.table.rows}
    assert {"src", "channels", "cell_diameter", "magnification"} <= keys
    # The on-disk copy is shown too.
    assert "gen_mask_settings.csv" in section.body_html


def test_a_folder_with_no_settings_says_it_is_not_reproducible(tmp_path):
    src = tmp_path / "plate_no_settings"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    section = report.section("settings")
    assert section.status == STATUS_MISSING
    assert "not reproducible" in section.body_html


def test_statistics_lists_the_result_csv_and_the_database_tables(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    section = report.section("statistics")
    assert section.status == STATUS_OK
    names = {row[0] for row in section.table.rows}
    assert any(n.endswith("results.csv") for n in names)
    assert any(n.endswith("measurements.db") for n in names)
    assert "cell" in section.body_html and "png_list" in section.body_html
    assert "nothing on this page was recomputed" in section.body_html.lower()


def test_statistics_previews_only_the_first_rows_and_says_so(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]],
                            max_table_rows=5)
    html = render_html(report)
    assert "first 5 of 40 row(s)" in html
    assert "35 further row(s) not shown" in html


def test_appendix_lists_annotation_columns_without_scoring_them(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    section = report.section("appendix")
    assert "annotation" in section.body_html
    assert "kappa" in section.body_html.lower() or "&kappa;" in section.body_html
    assert "Annotator Agreement" in section.body_html


def test_appendix_carries_a_file_inventory(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    section = report.section("appendix")
    folders = {row[0] for row in section.table.rows}
    assert {"qc/", "results/", "settings/", "measurements/"} <= folders


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_an_empty_folder_produces_a_report_that_says_it_is_empty(tmp_path):
    src = tmp_path / "nothing_here"
    src.mkdir()
    report = collect_report(src, run_dirs=[])
    assert report.status == "empty"
    html = render_html(report)
    assert "No spaCR output was found" in html
    assert len(report.sections) == len(SECTION_KEYS)


def test_a_nonexistent_folder_does_not_raise(tmp_path):
    report = collect_report(tmp_path / "does" / "not" / "exist", run_dirs=[])
    assert report.status == "empty"
    assert "does not exist" in report.sections[0].body_html
    html = render_html(report)
    assert "does not exist" in html


def test_a_corrupt_database_is_reported_not_fatal(tmp_path):
    src = tmp_path / "plate_corrupt"
    (src / "measurements").mkdir(parents=True)
    (src / "measurements" / "measurements.db").write_bytes(b"not a database")
    report = collect_report(src, run_dirs=[])
    html = render_html(report)
    assert "measurements.db" in html


def test_a_run_dir_that_does_not_exist_is_noted_not_fatal(tmp_path):
    src = tmp_path / "plate_bad_run"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[tmp_path / "no_such_run"])
    section = report.section("provenance")
    assert section.status == STATUS_MISSING
    assert any("no such run folder" in n for n in section.notes)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _pdf_page_count_on_disk(path: Path) -> int:
    """Page count read out of the PDF itself, not out of our own bookkeeping."""
    blob = path.read_bytes()
    assert blob.startswith(b"%PDF-"), "not a PDF"
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", blob)]
    assert counts, "no /Count in the PDF page tree"
    return max(counts)


def test_pdf_opens_and_has_the_page_count_the_composer_planned(full_run, tmp_path):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    out = write_pdf(report, tmp_path / "report.pdf")
    assert out.is_file()
    expected = pdf_page_count(report)
    assert expected >= 2
    assert _pdf_page_count_on_disk(out) == expected


def test_the_pdf_gains_one_page_per_embedded_figure(tmp_path):
    src = tmp_path / "plate_pdf_pages"
    _write_db(src, stamps=[_stamp()])
    bare = collect_report(src, run_dirs=[])
    n_bare = pdf_page_count(bare)

    for i in range(3):
        _write_png(src / "results" / f"figure_{i}.png")
    with_figures = collect_report(src, run_dirs=[])
    assert with_figures.n_figures_embedded == 3

    out = write_pdf(with_figures, tmp_path / "with.pdf")
    assert _pdf_page_count_on_disk(out) == pdf_page_count(with_figures)
    assert pdf_page_count(with_figures) >= n_bare + 3


def test_the_pdf_figure_pages_really_contain_the_image(tmp_path):
    """The figure pages must draw, not fall through to the failure text.

    ``import matplotlib`` does not bind ``matplotlib.image``; reaching for
    it that way raised AttributeError, and the per-page guard turned every
    figure into "[… could not be drawn]" without a word in the log. The
    page count looked right, so nothing else here caught it.
    """
    src = tmp_path / "plate_pdf_images"
    for i in range(2):
        _write_png(src / "results" / f"figure_{i}.png", size=(60, 40))
    report = collect_report(src, run_dirs=[])
    assert report.n_figures_embedded == 2

    out = write_pdf(report, tmp_path / "images.pdf")
    blob = out.read_bytes()
    assert blob.count(b"/Subtype /Image") == 2, "figure pages carry no image"
    assert b"could not be drawn" not in blob

    # Control: a report with no figures embeds no image XObject, so the
    # assertion above is measuring the figures and not the page furniture.
    bare = collect_report(tmp_path / "void_control", run_dirs=[])
    control = write_pdf(bare, tmp_path / "control.pdf")
    assert control.read_bytes().count(b"/Subtype /Image") == 0


def test_a_pdf_of_an_empty_report_still_has_a_page(tmp_path):
    report = collect_report(tmp_path / "void", run_dirs=[])
    out = write_pdf(report, tmp_path / "void.pdf")
    assert _pdf_page_count_on_disk(out) >= 1


def test_the_pdf_transcribes_the_missing_section_statements(tmp_path):
    src = tmp_path / "plate_text"
    _write_db(src, stamps=[_stamp()])
    report = collect_report(src, run_dirs=[])
    text = render_text(report)
    assert "NOT AVAILABLE" in text
    assert "Segmentation QC: NOT RUN" in text
    for key in SECTION_KEYS:
        from spacr.report import SECTION_TITLES
        assert SECTION_TITLES[key] in text


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def test_build_report_writes_html_by_default(full_run, tmp_path):
    written = build_report(full_run["src"], tmp_path / "r.html",
                           run_dirs=[full_run["run_dir"]])
    assert [p.name for p in written] == ["r.html"]
    assert written[0].is_file()


def test_build_report_both_writes_two_files_into_a_folder(full_run, tmp_path):
    out_dir = tmp_path / "reports"
    written = build_report(full_run["src"], out_dir, fmt="both",
                           run_dirs=[full_run["run_dir"]])
    assert len(written) == 2
    assert [p.suffix for p in written] == [".html", ".pdf"]
    assert all(p.is_file() for p in written)
    assert all(p.parent == out_dir for p in written)
    assert written[0].stem.startswith("spacr_report_plate1_")


def test_build_report_pdf_only(full_run, tmp_path):
    written = build_report(full_run["src"], tmp_path / "r.pdf", fmt="pdf",
                           run_dirs=[full_run["run_dir"]])
    assert [p.suffix for p in written] == [".pdf"]
    assert _pdf_page_count_on_disk(written[0]) >= 1


def test_build_report_rejects_an_unknown_format(full_run, tmp_path):
    with pytest.raises(ValueError, match="fmt must be"):
        build_report(full_run["src"], tmp_path / "r.docx", fmt="docx")


def test_build_report_creates_missing_parent_directories(full_run, tmp_path):
    target = tmp_path / "deep" / "nested" / "r.html"
    written = build_report(full_run["src"], target,
                           run_dirs=[full_run["run_dir"]])
    assert written[0].is_file()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def test_table_reports_how_many_rows_it_hid():
    table = Table(columns=["a"], rows=[["1"], ["2"]], n_total_rows=10)
    assert table.n_omitted == 8


def test_table_defaults_n_total_rows_to_what_it_holds():
    table = Table(columns=["a"], rows=[["1"], ["2"]])
    assert table.n_total_rows == 2
    assert table.n_omitted == 0


def test_report_section_lookup_returns_none_for_an_unknown_key(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    assert report.section("no_such_section") is None
    assert isinstance(report, Report)


def test_render_text_covers_every_section(full_run):
    report = collect_report(full_run["src"], run_dirs=[full_run["run_dir"]])
    text = render_text(report)
    from spacr.report import SECTION_TITLES
    for key in SECTION_KEYS:
        assert SECTION_TITLES[key] in text
    assert str(report.src) in text
