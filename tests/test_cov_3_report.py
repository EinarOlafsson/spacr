"""The plate report degrades in public rather than failing or inventing.

Every branch exercised here is one where the report has been handed something
it cannot read -- a manifest whose schema version is not a number, a scorecard
too damaged for the csv module, a plugin whose builder is wrong -- and where
the two wrong answers are a traceback out of `collect_report` and a report
that quietly omits the chapter. The contract is that the document always
builds and always says what it could not read.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spacr import plugins, report


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_a_file_under_two_scanned_roots_is_listed_once(tmp_path, monkeypatch):
    """The roots the scan walks can overlap -- a folder named in more than
    one list is walked twice -- and a database counted twice would be
    reported as two databases in the same plate."""
    qc = tmp_path / "qc"
    qc.mkdir()
    (qc / "plate.db").write_bytes(b"")
    # Make `qc` reachable as a result directory as well as by its own name,
    # so the same folder is scanned under two roots.
    monkeypatch.setattr(report, "RESULT_DIRS", ("qc",))

    found = report._find_artifacts(tmp_path)

    assert [p.name for p in found["databases"]] == ["plate.db"]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _run(tmp_path, manifest):
    return {"dir": tmp_path / "run_1", "manifest": manifest}


def test_a_schema_version_that_is_not_a_number_reads_as_the_oldest(tmp_path):
    """A hand-edited manifest must not take the provenance chapter down. An
    unparseable schema is treated as pre-fingerprint, which is the statement
    that makes no claim about the data."""
    _html, text = report._fingerprints(
        [_run(tmp_path, {"schema_version": "two",
                         "settings_sha256": "abc123"})])

    joined = "\n".join(text)
    assert "abc123" in joined
    assert "journalled before spaCR recorded input digests" in joined


def test_a_digest_the_manifest_never_recorded_is_left_out_not_blank(tmp_path):
    """An absent input digest must not be rendered as an empty one: a blank
    fingerprint reads as 'nothing changed', which is a false assurance."""
    _html, text = report._fingerprints(
        [_run(tmp_path, {"schema_version": 2,
                         "output_tree_sha256": "0" * 64,
                         "performance": {"output_files": 3,
                                         "output_bytes": 2048}})])

    joined = "\n".join(text)
    assert "outputs" in joined
    assert "3 file(s)" in joined
    assert "inputs" not in joined


# ---------------------------------------------------------------------------
# Damaged CSVs
# ---------------------------------------------------------------------------

def _oversized_field_csv(path: Path, header: str) -> Path:
    """A CSV the csv module refuses: one field past its field-size limit."""
    huge = "x" * (csv.field_size_limit() + 1000)
    path.write_text(f"{header}\n{huge},cell,3,ok,,\n", encoding="utf-8")
    return path


def test_a_scorecard_too_damaged_to_parse_is_reported_not_summarised(tmp_path):
    """Half a scorecard gives a different plate verdict from the whole one,
    so the reader is told it is unreadable rather than shown a verdict
    derived from the rows that happened to parse."""
    path = _oversized_field_csv(tmp_path / "segmentation_qc_cell.csv",
                                "field,object_type,n_objects,severity,flags,note")

    rows, error = report._field_qcs_from_csv(path)

    assert rows == []
    assert error is not None
    assert "not readable as CSV" in error
    assert path.name in error


def test_a_scorecard_that_cannot_be_opened_says_so(tmp_path, monkeypatch):
    """An unreadable file is a different failure from a malformed one, and
    the message names which."""
    path = tmp_path / "segmentation_qc_cell.csv"
    path.write_text("field,object_type,n_objects\na,cell,3\n", encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise PermissionError("nope")

    monkeypatch.setattr("builtins.open", refuse)
    rows, error = report._field_qcs_from_csv(path)

    assert rows == []
    assert "unreadable (PermissionError)" in error


def test_a_layout_export_damaged_past_its_header_still_reports_its_wells(
        tmp_path):
    """The header parsed, so the file IS a plate layout; the rows counted
    before the damage are real. A parse failure at row 900 must not delete
    the whole plate-QC chapter."""
    qc = tmp_path / "qc"
    qc.mkdir()
    path = _oversized_field_csv(qc / "plate_layout.csv", "ring,is_edge,well")
    artifacts = {"layout_csv": [path]}

    section = report._collect_plate_qc(tmp_path, artifacts, max_rows=5)

    assert section.key == "plate_qc"
    assert section.status != report.STATUS_MISSING
    assert any(path.name in line for line in section.text_lines)
    assert any("on the outer ring" in line for line in section.text_lines)


# ---------------------------------------------------------------------------
# Plugin chapters
# ---------------------------------------------------------------------------

def good_section(_context):
    """A plugin builder that returns a valid section."""
    return report.Section(title="Extra", key="", text_lines=["fine"])


def wrong_key_section(_context):
    """A plugin builder that returns a section under someone else's key."""
    return report.Section(title="Extra", key="statistics")


def not_a_section(_context):
    """A plugin builder that returns the wrong type entirely."""
    return {"title": "Extra"}


NOT_CALLABLE = "this is a string, not a builder"


@pytest.fixture()
def plugin_report(monkeypatch):
    """Install report-section contributions and capture their diagnostics."""
    recorded = []
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda *args, **kw: recorded.append(args))

    def install(*contributions):
        monkeypatch.setattr(plugins, "report_sections",
                            lambda: tuple(("demo", c) for c in contributions))
        return recorded

    return install


def _contribution(key, builder, title="Extra chapter"):
    return plugins.ReportSectionContribution(
        key=key, title=title,
        builder=f"tests.test_cov_3_report:{builder}")


@pytest.mark.parametrize("key,builder,expected", [
    ("statistics", "good_section", "already exists"),
    ("extra_chapter", "NOT_CALLABLE", "is not callable"),
    ("extra_chapter", "not_a_section", "expected Section"),
    ("extra_chapter", "wrong_key_section", "expected"),
])
def test_a_broken_plugin_chapter_becomes_a_visible_problem(
        tmp_path, plugin_report, key, builder, expected):
    """A plugin that cannot produce its chapter must leave a chapter saying
    so. Dropping it silently makes the report look complete while a whole
    section of it is missing."""
    recorded = plugin_report(_contribution(key, builder))

    result = report.collect_report(tmp_path, search_journal=False)

    # A duplicate key leaves the core chapter in place and appends the
    # problem chapter after it, so the last one carries the complaint.
    section = [s for s in result.sections if s.key == key][-1]
    assert section.status == report.STATUS_PROBLEM
    assert "could not be generated" in section.body_html
    assert expected in section.body_html + " ".join(section.text_lines)
    assert recorded, "the failure was not recorded as a plugin diagnostic"


def test_a_working_plugin_chapter_is_inserted_where_it_asked(
        tmp_path, plugin_report):
    """The contrast that makes the problem chapters meaningful, and the
    check that a blank builder key is filled in from the contribution."""
    plugin_report(_contribution("extra_chapter", "good_section"))

    result = report.collect_report(tmp_path, search_journal=False)

    keys = [s.key for s in result.sections]
    assert keys[keys.index("statistics") + 1] == "extra_chapter"
    section = result.sections[keys.index("extra_chapter")]
    assert section.title == "Extra"
    assert section.status != report.STATUS_PROBLEM


def test_a_plugin_registry_that_cannot_be_read_leaves_the_core_report(
        tmp_path, monkeypatch):
    """Plugin discovery itself can fail. The eight core chapters are the
    report; they must survive a registry that raises."""
    def explode():
        raise RuntimeError("the plugin registry is corrupt")

    monkeypatch.setattr(plugins, "report_sections", explode)

    result = report.collect_report(tmp_path, search_journal=False)

    assert [s.key for s in result.sections] == [
        "run_status", "provenance", "segmentation_qc", "plate_qc",
        "figures", "statistics", "settings", "appendix"]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def test_a_report_that_plans_no_pages_still_writes_a_pdf(tmp_path,
                                                         monkeypatch):
    """A PDF with no pages is not a file any reader can open, so a plan that
    came back empty gets one page saying the report was empty."""
    monkeypatch.setattr(report, "_pdf_page_specs", lambda _report: [])
    document = report.Report(title="Nothing", src=tmp_path)

    out = report.write_pdf(document, tmp_path / "empty.pdf")

    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
    assert out.stat().st_size > 500, "the single page was not written"
