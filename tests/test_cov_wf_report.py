"""The report's fall-through paths: what it does when a piece is shaped oddly.

Every branch exercised here is one where the report finds something it can
read but not use in the usual way -- a failure record written as a bare
string by an older tool, a measurements database with no measurements in it
yet, an annotation table that is really a view, a symlink whose target is
gone, a table with rows but no header, a figure that could not be
rasterised, a caller who asked for PDF only. The report is read by somebody
who does not have spaCR and cannot re-run anything, so each of these has to
degrade to "state it and carry on" rather than to a crash, a blank page or
a silently dropped section.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from spacr import report as R


# ---------------------------------------------------------------------------
# Run status: a failure record that is not a dict
# ---------------------------------------------------------------------------

def _sidecar(src: Path, failures) -> Path:
    """Write a results CSV and the run-status sidecar that stamps it."""
    results = src / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "wells.csv").write_text("well,value\nA01,1\n")
    sidecar = results / "wells.run_status.json"
    sidecar.write_text(json.dumps({
        "run_id": "r1", "name": "measure", "status": "partial",
        "n_attempted": 3, "n_succeeded": 1, "n_failed": 2,
        "failure_rate": 0.67, "started_utc": "", "stamped_utc": "",
        "failures": failures,
        "summary": "2 of 3 wells failed",
    }))
    return sidecar


def test_a_failure_written_as_a_string_does_not_break_the_verdict(tmp_path):
    """A sidecar can be hand-written, or written by an older spaCR, with its
    failures as plain strings instead of the {item, stage, error} records
    this version writes. The per-failure bullets are built by reaching into
    those records, so a string among them must be stepped over rather than
    indexed -- and, far more importantly, the plate must still read PARTIAL.
    A folder whose failures could not be itemised reading as "complete" is
    the exact way a broken run gets forwarded as a finished one.
    """
    src = tmp_path / "plate_legacy"
    sidecar = _sidecar(src, [
        "OSError: truncated file",
        {"item": "B02.tif", "stage": "measure", "error": "ValueError: nan"},
    ])

    section, status, detail = R._collect_run_status(src, {"sidecars": [sidecar]}, [])
    body = section.body_html

    assert status == "partial"
    assert detail == R._STATUS_LABELS["partial"]
    assert section.status == R.STATUS_PROBLEM
    # The count comes off the stamp, not off the itemised list, so both
    # failures are still declared even though only one can be itemised.
    assert "2 of 3 item(s) failed" in body
    assert "2 of 3 wells failed" in body
    # The dict failure is itemised; the string one contributes no bullet.
    assert body.count("class='sub'") == 1
    assert "B02.tif" in body and "ValueError: nan" in body
    assert "OSError: truncated file" not in body


# ---------------------------------------------------------------------------
# Appendix: a database that describes nothing, and annotations in a view
# ---------------------------------------------------------------------------

def _measurements_db(path: Path) -> Path:
    """A database with one real measurement table."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (object_label INTEGER, cell_area REAL)")
    con.execute("INSERT INTO cell VALUES (1, 12.5), (2, 30.0)")
    con.commit()
    con.close()
    return path


def test_an_empty_database_does_not_swallow_the_rest_of_the_appendix(tmp_path):
    """A run that dies before Measure writes a table leaves a measurements.db
    that exists and holds nothing. The feature dictionary then has no
    families to show and -- because nothing went wrong -- no error to report
    either. That silent case must fall through to the annotation scan and
    the file inventory instead of ending the appendix: a reader looking at
    an empty appendix cannot tell "no features" from "the appendix broke".
    """
    empty_dir = tmp_path / "empty_plate"
    empty_dir.mkdir()
    empty_db = empty_dir / "measurements.db"
    sqlite3.connect(empty_db).close()

    families, rows, n_total, error = R._feature_dictionary(empty_db, 10)
    assert (families, rows, n_total, error) == ([], [], 0, "")

    section = R._collect_appendix(empty_dir, {"databases": [empty_db]})
    assert section.notes == []
    assert "Measured features" not in section.body_html
    # The inventory still ran, so the appendix is not "nothing found".
    assert section.status == R.STATUS_OK
    assert section.table is not None
    assert section.table.caption == "File inventory"
    assert any("top level" in row[0] for row in section.table.rows)

    # And the same appendix on a database that does hold measurements shows
    # the heading the empty one had to omit.
    full_dir = tmp_path / "full_plate"
    full_dir.mkdir()
    full = R._collect_appendix(full_dir, {
        "databases": [_measurements_db(full_dir / "measurements.db")]})
    assert "Measured features" in full.body_html


def _annotated_db(path: Path, as_view: bool) -> Path:
    """A database whose annotations live in a png_list table -- or view."""
    con = sqlite3.connect(path)
    if as_view:
        con.execute("CREATE TABLE crops (png_path TEXT, test_ann INTEGER)")
        con.execute("INSERT INTO crops VALUES ('a.png', 1), ('b.png', 0)")
        con.execute("CREATE VIEW png_list AS SELECT * FROM crops")
    else:
        con.execute("CREATE TABLE png_list (png_path TEXT, test_ann INTEGER)")
        con.execute("INSERT INTO png_list VALUES ('a.png', 1), ('b.png', 0)")
    con.commit()
    con.close()
    return path


def test_annotations_in_a_view_are_listed_without_inventing_a_row_count(tmp_path):
    """``png_list`` is sometimes a view over the physical crop table -- a
    curated subset, or a rename kept for compatibility. The annotation
    columns are still real and must be listed, but the row count is taken
    from the database's *tables*, so the view has none to give. The report
    must then say which columns exist and stay silent about how many rows
    were annotated, rather than printing a count of 0 or None: "0 annotated
    rows" would tell a reader the annotation pass produced nothing.
    """
    view_db = _annotated_db(tmp_path / "view.db", as_view=True)

    counts = R._sqlite_table_counts(view_db)
    assert dict(counts) == {"crops": 2}, "the view must not appear as a table"
    columns, n_annotated, error = R._annotation_summary(view_db)
    assert columns == ["test_ann"]
    assert n_annotated is None
    assert error == ""

    view_dir = tmp_path / "view_plate"
    view_dir.mkdir()
    section = R._collect_appendix(view_dir, {"databases": [view_db]})
    assert "<code>test_ann</code>" in section.body_html
    assert "annotation columns: test_ann" in "\n".join(section.text_lines)
    assert "annotated " not in section.body_html

    # The same database with png_list as a real table does report the count,
    # so the silence above is the view's doing and not a broken lookup.
    table_db = _annotated_db(tmp_path / "table.db", as_view=False)
    assert R._annotation_summary(table_db) == (["test_ann"], 2, "")
    table_dir = tmp_path / "table_plate"
    table_dir.mkdir()
    listed = R._collect_appendix(table_dir, {"databases": [table_db]})
    assert "2 annotated " in listed.body_html


# ---------------------------------------------------------------------------
# File inventory: an entry that is neither a directory nor a file
# ---------------------------------------------------------------------------

def test_a_dangling_symlink_is_counted_as_nothing_not_as_a_file(tmp_path):
    """Plate folders are full of symlinks into shared image stores, and a
    link whose target has been moved is neither a directory nor a file. It
    must not be counted as a loose file -- its size cannot be read, so
    counting it would either raise inside the inventory (losing the whole
    appendix) or add a phantom 0-byte entry to the folder sizes a reader
    uses to decide what to archive.
    """
    src = tmp_path / "plate_links"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.bin").write_bytes(b"x" * 10)
    (src / "real.txt").write_text("hello")
    os.symlink(str(src / "gone.tif"), str(src / "dangling.tif"))
    assert not (src / "dangling.tif").exists(), "the link must really dangle"

    rows, truncated = R._file_inventory(src)

    assert truncated is False
    assert rows == [
        ["sub/", "1", "10 B"],
        ["(files at the top level)", "1", "5 B"],
    ], "the dangling link must not be counted among the loose files"


# ---------------------------------------------------------------------------
# Table rendering: no header row, no caption
# ---------------------------------------------------------------------------

def test_a_table_with_no_header_still_renders_its_rows():
    """Some tables the report builds are key/value pairs with no meaningful
    column names -- a settings dump, a version list. Dropping the whole
    table because there is no header would delete content a reader needs;
    the rows have to render, with no empty ``<thead>`` faking a header that
    was never there.
    """
    headerless = R._table_html(R.Table(rows=[["seed", "42"], ["gpu", "0"]],
                                       caption="Settings"))

    assert "<thead>" not in headerless
    assert "<caption>Settings</caption>" in headerless
    assert headerless.count("<tr>") == 2
    assert "<td>seed</td><td>42</td>" in headerless
    # A table that does have columns proves the header path is what was
    # skipped, not table rendering as a whole.
    with_header = R._table_html(R.Table(columns=["key", "value"],
                                        rows=[["seed", "42"]]))
    assert "<thead><tr><th>key</th><th>value</th></tr></thead>" in with_header


def test_a_caption_free_table_starts_at_its_header_in_the_pdf_text():
    """The PDF is a monospace transcription, not a rendering of the HTML, so
    the text renderer builds its own layout. A table with no caption must
    begin at the header row: emitting a blank caption line would push every
    table down a line and, on a page boundary, split a table's header from
    its body.
    """
    plain = R._table_text(R.Table(columns=["x", "y"], rows=[["1", "2"]]))

    assert plain[0].split() == ["x", "y"]
    assert plain[1].strip().startswith("-")
    assert plain[-1].split() == ["1", "2"]
    captioned = R._table_text(R.Table(columns=["x", "y"], rows=[["1", "2"]],
                                      caption="Wells"))
    assert captioned[0].strip() == "Wells"
    assert captioned[1:] == plain


# ---------------------------------------------------------------------------
# PDF: a figure that was found but never embedded
# ---------------------------------------------------------------------------

def test_a_figure_that_could_not_be_embedded_costs_no_pdf_page():
    """spaCR writes most of its figures as PDF, which this module cannot
    rasterise, so a collected report routinely carries figures with no
    bytes. Each one is listed in the HTML by name and reason, but the PDF
    composer has nothing to draw -- and a page reserved for it would come
    out blank. A blank page in the middle of a report reads as a rendering
    failure, so the page count must follow the embedded figures only.
    """
    drawn = R.Figure(path=Path("volcano.png"), title="volcano", data=b"\x89PNG...")
    listed = R.Figure(path=Path("plate.pdf"), title="plate map",
                      reason="vector figure; not embedded")
    report = R.Report(src=Path("plate"), sections=[
        R.Section(title="Key figures", key="figures",
                  figures=[listed, drawn], text_lines=["Key figures"])])

    specs = R._pdf_page_specs(report)

    assert [kind for kind, _ in specs] == ["text", "figure"]
    assert specs[1][1] is drawn
    assert listed.embedded is False and drawn.embedded is True
    assert R.pdf_page_count(report) == 2
    # One more embedded figure is one more page; the un-embedded one still
    # is not, so the count tracks bytes and not figure records.
    report.sections[0].figures.append(
        R.Figure(path=Path("hits.png"), title="hits", data=b"\x89PNG!"))
    assert R.pdf_page_count(report) == 3


# ---------------------------------------------------------------------------
# build_report: pdf only, into a folder
# ---------------------------------------------------------------------------

def test_asking_a_folder_for_a_pdf_writes_only_the_pdf(tmp_path):
    """``build_report(src, folder, fmt='pdf')`` is what the GUI's Report tool
    calls when the user picks PDF and a destination directory. The names are
    generated from the plate and the clock, so nothing tells the caller what
    was written except the returned list -- and an HTML file written anyway
    would sit in the folder as a second, differently-formatted copy of the
    same report for the user to send by mistake.
    """
    src = tmp_path / "plate1"
    (src / "results").mkdir(parents=True)
    (src / "results" / "wells.csv").write_text("well,value\nA01,1\n")

    out = tmp_path / "reports"
    written = R.build_report(src, out, fmt="pdf", run_dirs=[],
                             search_journal=False, include_plan=False)

    assert len(written) == 1
    assert written[0].suffix == ".pdf"
    assert written[0].parent == out
    assert written[0].name.startswith("spacr_report_plate1_")
    assert written[0].read_bytes().startswith(b"%PDF")
    assert [p.suffix for p in sorted(out.iterdir())] == [".pdf"]

    # The same call with fmt='both' does put an HTML beside it, so the
    # missing file above is the format choice being honoured.
    both = R.build_report(src, out, fmt="both", run_dirs=[],
                          search_journal=False, include_plan=False)
    assert [p.suffix for p in both] == [".html", ".pdf"]
    assert "<html" in both[0].read_text(encoding="utf-8").lower()
    assert sorted({p.suffix for p in out.iterdir()}) == [".html", ".pdf"]
