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


from spacr import report


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
