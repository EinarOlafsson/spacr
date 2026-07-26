"""Report — the Tools screen that turns a run folder into one shareable file.

Everything runs offscreen against a *real* temporary plate folder: a
``measurements/measurements.db`` with a ``run_status`` stamp, a ``qc``
scorecard, a ``results`` PNG and a ``settings`` copy — the same shapes
``tests/test_report.py`` builds, just smaller.

The suite pins the four properties this panel lives or dies by:

* it **states what is missing** — the section list shows every section, with
  the unavailable ones greyed and labelled, *before* anything is generated;
* it **writes where it is told** and never into the run folder;
* it is **threaded** — collection base64-encodes figures and must not run on
  the GUI thread, and no QThread is left alive afterwards;
* it is **modal-free** — a bad folder, an empty box and a premature Open all
  land in the inline status label. A ``QMessageBox`` would hang a headless
  run forever, so this file makes one impossible to reintroduce.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3

import pytest

from spacr import report as rep
from spacr.qt.screens.report import FIGURE_CAP_RANGE, FORMATS, ReportScreen
from spacr.qt.theme import PALETTE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite on
    a QMessageBox; this fixture makes that failure mode impossible to
    reintroduce here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture(autouse=True)
def _isolated_run_journal(tmp_path, monkeypatch):
    """Point the run journal at a temp folder.

    Collection searches ``~/.spacr/runs`` for the run that made a folder.
    A test must not read the developer's real journal — it is slow and it
    is not reproducible.
    """
    root = tmp_path / "journal"
    root.mkdir(parents=True, exist_ok=True)
    from spacr import run_journal
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _write_plate(root, name="plate1", partial=False, n_figures=1):
    """A small but complete-looking plate folder."""
    from PIL import Image

    src = root / name
    (src / "qc").mkdir(parents=True, exist_ok=True)
    (src / "results").mkdir(parents=True, exist_ok=True)
    (src / "settings").mkdir(parents=True, exist_ok=True)
    (src / "measurements").mkdir(parents=True, exist_ok=True)

    with open(src / "qc" / "segmentation_qc_cell.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "object_type", "n_objects", "severity",
                         "flags", "border_fraction", "note"])
        for i in range(6):
            writer.writerow([f"plate1_A{i + 1:02d}_1", "cell", 100 + i, "ok",
                             "", 0.05, "clean"])

    with open(src / "settings" / "gen_mask_settings.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerow(["src", str(src)])
        writer.writerow(["channels", "[0, 1, 2]"])

    with open(src / "results" / "results.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", "coef"])
        writer.writerow(["gene1", 0.5])

    for i in range(n_figures):
        Image.new("RGB", (24, 18), (40, 90, 160)).save(
            src / "results" / f"figure_{i}.png")

    db = src / "measurements" / "measurements.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE cell (prc TEXT, cell_area REAL)")
        conn.executemany("INSERT INTO cell VALUES (?, ?)",
                         [(f"plate1_A01_{i}", 100.0 + i) for i in range(5)])
        conn.execute("CREATE TABLE run_status (run_id TEXT, name TEXT, "
                     "status TEXT, n_attempted INTEGER, n_succeeded INTEGER, "
                     "n_failed INTEGER, failure_rate REAL, started_utc TEXT, "
                     "stamped_utc TEXT, failures_json TEXT, summary TEXT)")
        conn.execute(
            "INSERT INTO run_status VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "measure_crop", "partial" if partial else "complete",
             100, 93 if partial else 100, 7 if partial else 0,
             0.07 if partial else 0.0, "", "",
             json.dumps([{"item": "A01_f3", "stage": "crop",
                          "error": "ValueError: bad"}]) if partial else "[]",
             "7 of 100 fields failed" if partial else "all fields processed"))
        conn.commit()
    finally:
        conn.close()
    return src


@pytest.fixture
def plate(tmp_path):
    """A clean plate folder."""
    return _write_plate(tmp_path / "runs")


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — collection runs inline so assertions are exact."""
    widget = ReportScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _items(widget):
    """(text, is_greyed) for every row of the section list."""
    out = []
    for i in range(widget._sections.count()):
        item = widget._sections.item(i)
        colour = item.foreground().color().name().lower()
        out.append((item.text(), colour == PALETTE["fg_dim"].lower()))
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_the_screen_builds_offscreen(screen):
    assert screen.report is None
    assert screen.written == []
    assert screen.last_error == ""
    assert "run folder" in screen.status_text().lower()


def test_the_format_choices_are_the_ones_build_report_understands(screen):
    keys = [screen._format.itemData(i) for i in range(screen._format.count())]
    assert keys == [key for _label, key in FORMATS]
    assert keys == ["html", "pdf", "both"]
    assert screen.output_format() == "html"


def test_the_figure_cap_defaults_to_the_module_default(screen):
    assert screen.figure_cap() == rep.DEFAULT_MAX_FIGURES
    assert screen._figure_cap.minimum() == FIGURE_CAP_RANGE[0]
    assert screen._figure_cap.maximum() == FIGURE_CAP_RANGE[1]


def test_generate_is_disabled_until_a_folder_is_named(screen):
    assert not screen._btn_generate.isEnabled()
    assert not screen._btn_open.isEnabled()
    screen.set_source("/tmp")
    assert screen._btn_generate.isEnabled()


# ---------------------------------------------------------------------------
# Scanning: found and missing sections
# ---------------------------------------------------------------------------

def test_scanning_lists_every_section_found_and_missing(screen, plate):
    screen.set_source(str(plate))
    assert screen.scan() is True

    rows = _items(screen)
    assert len(rows) == len(rep.SECTION_KEYS), (
        "a section disappeared from the list instead of being greyed")
    assert set(screen.found_sections()) | set(screen.missing_sections()) == \
        set(rep.SECTION_KEYS)
    assert "segmentation_qc" in screen.found_sections()
    assert "statistics" in screen.found_sections()
    assert "settings" in screen.found_sections()
    # No journal entry and no exported layout in this folder.
    assert "provenance" in screen.missing_sections()
    assert "plate_qc" in screen.missing_sections()


def test_missing_sections_are_greyed_and_labelled(screen, plate):
    screen.set_source(str(plate))
    screen.scan()
    greyed = [text for text, grey in _items(screen) if grey]
    assert greyed, "nothing was greyed even though sections are missing"
    assert all(t.endswith("— not available") for t in greyed)
    assert any("Plate QC" in t for t in greyed)
    # A found section is not greyed.
    found = [text for text, grey in _items(screen) if not grey]
    assert any("Segmentation QC" in t for t in found)


def test_the_verdict_line_reports_the_run_status(screen, plate):
    screen.set_source(str(plate))
    screen.scan()
    assert "Complete" in screen._verdict.text()


def test_a_partial_run_shows_a_partial_verdict(screen, tmp_path):
    src = _write_plate(tmp_path / "runs2", name="plate_partial", partial=True)
    screen.set_source(str(src))
    screen.scan()
    assert "PARTIAL" in screen._verdict.text()
    assert screen.report.status == "partial"
    attention = [t for t, _grey in _items(screen) if "needs attention" in t]
    assert attention, "the failing section is not flagged in the list"


def test_scanning_emits_folder_scanned(screen, plate, qtbot):
    screen.set_source(str(plate))
    with qtbot.waitSignal(screen.folder_scanned, timeout=1000) as blocker:
        screen.scan()
    assert os.path.realpath(blocker.args[0]) == os.path.realpath(str(plate))


def test_scanning_proposes_an_output_path_outside_the_run_folder(screen, plate):
    screen.set_source(str(plate))
    screen.scan()
    proposed = screen._out_edit.text()
    assert proposed.endswith(".html")
    assert not proposed.startswith(str(plate)), (
        "the default output would litter the dataset")


def test_the_figure_cap_is_honoured_by_the_scan(screen, tmp_path):
    src = _write_plate(tmp_path / "runs3", name="plate_figs", n_figures=5)
    screen.set_source(str(src))
    screen._figure_cap.setValue(2)
    screen.scan()
    assert screen.report.n_figures_found == 5
    assert screen.report.n_figures_embedded == 2


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------

def test_generating_writes_html_to_the_chosen_path(screen, plate, tmp_path):
    out = tmp_path / "out" / "plate1_report.html"
    screen.set_source(str(plate))
    screen.set_output(str(out))
    assert screen.generate() is True
    assert screen.written == [str(out)]
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "http://" not in text and "https://" not in text


def test_generating_emits_report_written(screen, plate, tmp_path, qtbot):
    screen.set_source(str(plate))
    screen.set_output(str(tmp_path / "r.html"))
    with qtbot.waitSignal(screen.report_written, timeout=1000) as blocker:
        screen.generate()
    assert blocker.args[0] == screen.written


def test_generating_both_writes_two_files(screen, plate, tmp_path):
    out_dir = tmp_path / "both"
    screen.set_source(str(plate))
    screen.set_output(str(out_dir))
    screen.set_format("both")
    assert screen.generate() is True
    assert len(screen.written) == 2
    assert sorted(os.path.splitext(p)[1] for p in screen.written) == \
        [".html", ".pdf"]
    assert all(os.path.isfile(p) for p in screen.written)


def test_generating_pdf_only(screen, plate, tmp_path):
    out = tmp_path / "r.pdf"
    screen.set_source(str(plate))
    screen.set_output(str(out))
    screen.set_format("pdf")
    screen.generate()
    assert screen.written == [str(out)]
    assert out.read_bytes().startswith(b"%PDF-")


def test_generating_writes_nothing_into_the_run_folder(screen, plate, tmp_path):
    before = sorted(str(p) for p in plate.rglob("*"))
    screen.set_source(str(plate))
    screen.set_output(str(tmp_path / "elsewhere" / "r.html"))
    screen.generate()
    assert sorted(str(p) for p in plate.rglob("*")) == before


def test_generating_without_an_output_path_falls_back_to_a_default(
        screen, plate, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path / "home"), 1))
    screen.set_source(str(plate))
    screen.set_output("")
    assert screen.generate() is True
    assert screen.written
    assert os.path.isfile(screen.written[0])


def test_open_hands_the_report_to_the_desktop(screen, plate, tmp_path,
                                              monkeypatch):
    opened = []
    from PySide6.QtGui import QDesktopServices
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url.toLocalFile())))
    out = tmp_path / "r.html"
    screen.set_source(str(plate))
    screen.set_output(str(out))
    screen.generate()
    assert screen.open_output() is True
    assert opened == [str(out)]


# ---------------------------------------------------------------------------
# Errors — inline, never modal
# ---------------------------------------------------------------------------

def test_a_folder_that_is_not_a_folder_is_reported_inline(screen, tmp_path):
    screen.set_source(str(tmp_path / "no_such_plate"))
    assert screen.scan() is False
    assert "Not a folder" in screen.last_error
    assert "Not a folder" in screen.status_text()
    assert screen.report is None


def test_a_file_given_as_a_run_folder_is_reported_inline(screen, tmp_path):
    target = tmp_path / "a_file.txt"
    target.write_text("not a folder")
    screen.set_source(str(target))
    assert screen.scan() is False
    assert "Not a folder" in screen.last_error


def test_an_empty_source_box_is_reported_inline(screen):
    screen.set_source("")
    assert screen.scan() is False
    assert "No run folder" in screen.last_error
    assert screen.generate() is False
    assert "No run folder" in screen.last_error


def test_generating_from_a_bad_folder_is_reported_inline(screen, tmp_path):
    screen.set_source(str(tmp_path / "gone"))
    screen.set_output(str(tmp_path / "r.html"))
    assert screen.generate() is False
    assert "Not a folder" in screen.last_error
    assert screen.written == []


def test_open_before_generate_is_reported_inline(screen):
    assert screen.open_output() is False
    assert "Nothing has been generated" in screen.last_error


def test_open_after_the_file_is_deleted_is_reported_inline(screen, plate,
                                                           tmp_path):
    out = tmp_path / "r.html"
    screen.set_source(str(plate))
    screen.set_output(str(out))
    screen.generate()
    out.unlink()
    assert screen.open_output() is False
    assert "no longer there" in screen.last_error


def test_a_scan_that_raises_lands_in_the_status_label(screen, plate,
                                                      monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("collection exploded")

    monkeypatch.setattr(rep, "collect_report", _boom)
    screen.set_source(str(plate))
    assert screen.scan() is False
    assert "collection exploded" in screen.last_error


def test_an_unwritable_output_lands_in_the_status_label(screen, plate,
                                                        monkeypatch):
    def _boom(*_a, **_k):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(rep, "build_report", _boom)
    screen.set_source(str(plate))
    screen.set_output("/definitely/not/writable/r.html")
    assert screen.generate() is False
    assert "read-only file system" in screen.last_error
    assert screen.written == []


# ---------------------------------------------------------------------------
# Threading — the default path
# ---------------------------------------------------------------------------

def test_a_threaded_scan_settles_and_leaves_no_thread_running(qtbot,
                                                              qt_theme_applied,
                                                              plate):
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_source(str(plate))
    with qtbot.waitSignal(widget.job_finished, timeout=60000) as blocker:
        assert widget.scan() is True
    assert blocker.args[0] is True
    assert widget.report is not None
    assert widget.found_sections()
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=20000)
    assert not widget.is_busy()


def test_a_threaded_generate_writes_the_file(qtbot, qt_theme_applied, plate,
                                             tmp_path):
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)
    out = tmp_path / "threaded.html"
    widget.set_source(str(plate))
    widget.set_output(str(out))
    with qtbot.waitSignal(widget.job_finished, timeout=60000) as blocker:
        assert widget.generate() is True
    assert blocker.args[0] is True
    assert out.is_file()
    assert widget.written == [str(out)]
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=20000)


def test_a_second_request_while_busy_is_refused_inline(qtbot, qt_theme_applied,
                                                       plate):
    widget = ReportScreen(threaded=True)
    qtbot.addWidget(widget)
    widget.set_source(str(plate))
    widget.scan()
    if widget.is_busy():
        assert widget.scan() is False
        assert "previous request" in widget.last_error
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=60000)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_report_registration_is_complete_if_it_is_registered():
    """Guard against a half-applied registration.

    ``spacr/qt/app.py`` and ``spacr/qt/screens/app_screen.py`` are owned by
    another change, so this file cannot register the screen itself. What it
    *can* do is refuse to let the screen be listed in the launcher without a
    Tools section, a title and a real intro — the three places every other
    Tools app has to appear in.
    """
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "report"), None)
    if entry is None:
        pytest.skip("the report screen is not registered in APPS yet")
    key, name, desc, section = entry
    assert name == "Report"
    assert desc and desc.strip()
    assert section == "Tools", f"report filed under {section!r}, want Tools"

    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES.get("report", "").strip() == "Report"
    intro = APP_INTROS.get("report", "")
    assert len(intro.split()) >= 10, f"intro is too thin: {intro!r}"
    assert intro.endswith(".")

    from spacr.qt import app as qt_app
    icon = qt_app._ICON_OVERRIDES.get("report")
    if icon:
        here = os.path.dirname(os.path.abspath(qt_app.__file__))
        path = os.path.normpath(os.path.join(
            here, "..", "resources", "icons", icon))
        assert os.path.isfile(path), f"missing icon file: {path}"


def test_the_screen_class_is_what_a_launcher_would_build(qtbot,
                                                         qt_theme_applied):
    """The constructor a MainWindow dispatch branch would call."""
    widget = ReportScreen()
    qtbot.addWidget(widget)
    assert widget.windowTitle() == "" or isinstance(widget.windowTitle(), str)
    assert hasattr(widget, "scan") and hasattr(widget, "generate")
