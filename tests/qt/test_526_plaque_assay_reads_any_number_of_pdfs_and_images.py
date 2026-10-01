"""Item 526: Plaque Assay reads any number of PDFs and images.

Real ``QDropEvent`` objects with local file URLs are sent at the real Plaque
Assay screen, offscreen. Only the figure reader itself is faked: it writes one
small real PNG and a ``paper.json`` into the paper's folder, the way
:func:`spacr.plaque_papers.fetch_paper_to_folder` does, so everything after it
-- the batch, the status line, the note, ``src``, the preview's listing -- is
the real code. A folder with ten PDFs and unrelated files is read whole with
no refusal; a ``src`` naming such a folder is too; one PDF that cannot be read
is named while the others are read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from spacr import plaque_papers as PP
from spacr.qt import dnd as dnd_mod
from spacr.qt import dnd_handlers as dh
from spacr.qt.widgets import plaque_preview as ppv


def _png(path: Path, value: int = 90) -> Path:
    """A small real PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((24, 24, 3), value, dtype=np.uint8))
    return path


def _pdf(path: Path) -> Path:
    """A PDF-shaped file; only the faked reader sees it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


def _folder_of_papers(folder: Path, n: int = 10) -> list:
    """``n`` PDFs and three files Plaque Assay has no use for."""
    pdfs = [_pdf(folder / f"plate_{index:02d}.pdf") for index in range(1, n + 1)]
    (folder / "readme.txt").write_text("notes")
    (folder / "results.xlsx").write_bytes(b"PK\x03\x04")
    (folder / "methods.docx").write_bytes(b"PK\x03\x04")
    return pdfs


def _mime(paths) -> QMimeData:
    """Mime data with a local file URL per path."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _wait(condition, timeout_s: float = 30.0) -> None:
    """Process events until ``condition()`` holds."""
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert condition(), "timed out"


def _drop(screen, paths) -> None:
    """Replay drag enter, move and drop on ``screen``; wait for the scan."""
    enter_mime, move_mime, drop_mime = _mime(paths), _mime(paths), _mime(paths)
    QApplication.sendEvent(screen, QDragEnterEvent(
        QPoint(5, 5), Qt.CopyAction, enter_mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(screen, QDragMoveEvent(
        QPoint(5, 5), Qt.CopyAction, move_mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(screen, QDropEvent(
        QPointF(5, 5), Qt.CopyAction, drop_mime, Qt.LeftButton, Qt.NoModifier))
    scanner = getattr(getattr(screen, "_dnd_screen", screen), "_dnd_scanner",
                      None)
    runner = getattr(scanner, "_runner", None)
    if runner is not None:
        seen = []
        runner.job_finished.connect(lambda *_: seen.append(True))
        _wait(lambda: bool(seen))


def _src(screen) -> str:
    """The form's ``src``."""
    return screen._settings_model._widgets["src"].text()


@pytest.fixture
def said(monkeypatch):
    """Every refusal the drop reports, and every console line it writes."""
    reports = []
    lines = []
    real_log = dh._log

    def record(screen, path, reason, suggestion, **kwargs):
        reports.append((str(path), reason))

    def log(screen, msg):
        lines.append(msg)
        real_log(screen, msg)

    monkeypatch.setattr(dnd_mod, "_report_drop_problem", record)
    monkeypatch.setattr(dh, "_log", log)
    return {"refusals": reports, "lines": lines}


@pytest.fixture
def reader(monkeypatch):
    """The figure reader, faked: one figure and a paper.json per PDF."""
    calls = []
    failing = set()

    def fetch(reference, dest):
        calls.append((Path(reference).name, Path(dest)))
        if Path(reference).name in failing:
            raise RuntimeError("the PDF has no pages")
        dest = Path(dest)
        _png(dest / "figure_1.png", value=10 + len(calls))
        (dest / PP.PAPER_FILE).write_text(json.dumps({"key": dest.name}))
        return {"folder": str(dest), "paper": dest.name, "figures": 1,
                "with_legend": 0, "licence": "CC BY", "text_layer": 0}

    monkeypatch.setattr(PP, "fetch_paper_to_folder", fetch)
    monkeypatch.setattr(PP, "reader_problem", lambda pdf=False: ("", ""))
    fetch.calls = calls
    fetch.failing = failing
    return fetch


@pytest.fixture
def figure_screen(qtbot, qt_theme_applied, monkeypatch):
    """A real Plaque Assay screen in Figure mode, never asked a question."""
    from spacr.qt.screens.app_screen import AppScreen

    asked = []
    monkeypatch.setattr(ppv, "ask_input_mode",
                        lambda parent, question, name, mode, **kw:
                        asked.append(question) or mode)
    made = AppScreen("analyze_plaques")
    qtbot.addWidget(made)
    ppv.choose_plaque_mode(made, ppv.FIGURE_MODE)
    panel = made._live_preview
    statuses = []
    real_status = panel.set_preview_status

    def status(text, *args, **kwargs):
        statuses.append(str(text))
        return real_status(text, *args, **kwargs)

    monkeypatch.setattr(panel, "set_preview_status", status)
    made.statuses = statuses
    made.asked = asked
    return made


def _read_all(screen, count: int) -> None:
    """Wait for the batch, then for the preview to list ``count`` figures."""
    panel = screen._live_preview
    _wait(lambda: getattr(panel, "_paper_batch", None) is None
          and not panel._paper_jobs.is_busy())
    if count:
        _wait(lambda: panel._picker.count() == count)


def test_a_folder_of_ten_pdfs_and_other_files_is_read_whole(
        figure_screen, tmp_path, said, reader):
    folder = tmp_path / "plates"
    pdfs = _folder_of_papers(folder)
    _drop(figure_screen, [folder])
    _read_all(figure_screen, 10)
    assert said["refusals"] == []
    assert [name for name, _dest in reader.calls] == [p.name for p in pdfs]
    for pdf in pdfs:
        assert (folder / pdf.stem / "figure_1.png").is_file(), pdf
    assert _src(figure_screen) == str(folder)
    panel = figure_screen._live_preview
    labels = [panel._picker.itemText(i) for i in range(panel._picker.count())]
    assert labels[0] == "plate_01/figure_1.png"
    assert labels[-1] == "plate_10/figure_1.png"
    left_aside = [line for line in said["lines"] if "neither images" in line]
    assert len(left_aside) == 1 and "3 file(s)" in left_aside[0]
    assert any("Reading paper 1 of 10: plate_01.pdf" in s
               for s in figure_screen.statuses)
    assert any("Reading paper 10 of 10: plate_10.pdf" in s
               for s in figure_screen.statuses)
    assert "Read 10 of 10 papers" in panel._paper_note.text()
    assert figure_screen.asked == []


def test_a_src_naming_a_folder_of_pdfs_reads_them_all(
        figure_screen, tmp_path, said, reader):
    folder = tmp_path / "from_src"
    pdfs = _folder_of_papers(folder)
    figure_screen.show()
    ppv.check_the_src(figure_screen, str(folder))
    _read_all(figure_screen, 10)
    assert sorted(name for name, _dest in reader.calls) == [p.name for p in pdfs]
    assert _src(figure_screen) == str(folder)
    assert said["refusals"] == []
    ppv.check_the_src(figure_screen, str(folder))
    _read_all(figure_screen, 10)
    assert len(reader.calls) == 10, "a folder already read is not read again"


def test_one_pdf_that_fails_is_named_and_the_others_are_read(
        figure_screen, tmp_path, said, reader):
    folder = tmp_path / "plates"
    _folder_of_papers(folder, n=5)
    reader.failing.add("plate_03.pdf")
    _drop(figure_screen, [folder])
    _read_all(figure_screen, 4)
    assert len(reader.calls) == 5
    assert said["refusals"] == []
    panel = figure_screen._live_preview
    note = panel._paper_note.text()
    assert "Read 4 of 5 papers" in note
    assert "Could not read plate_03.pdf: the PDF has no pages" in note
    assert any("Could not read 1 of the papers: plate_03.pdf." in s
               for s in figure_screen.statuses)
    assert not (folder / "plate_03" / "figure_1.png").exists()
    assert (folder / "plate_04" / "figure_1.png").is_file()


def test_loose_pdfs_dropped_with_other_files_are_read_and_the_rest_ignored(
        figure_screen, tmp_path, said, reader):
    first = _pdf(tmp_path / "a" / "one.pdf")
    second = _pdf(tmp_path / "b" / "one.pdf")
    third = _pdf(tmp_path / "a" / "two.pdf")
    note = tmp_path / "a" / "notes.txt"
    note.write_text("x")
    _drop(figure_screen, [first, note, second, third])
    _read_all(figure_screen, 3)
    assert said["refusals"] == []
    assert sorted(p.name for p in (tmp_path / "a").iterdir() if p.is_dir()) == [
        "b_one", "one", "two"]
    assert [line for line in said["lines"] if "neither images" in line] == [
        "[drop] 1 file(s) that are neither images nor PDFs were left aside.\n"]


def test_any_number_of_images_with_other_files(qtbot, qt_theme_applied,
                                               tmp_path, said):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    folder = tmp_path / "wells"
    for index in range(12):
        _png(folder / f"well_{index:02d}.png", value=20 + index)
    (folder / "plate_map.txt").write_text("x")
    (folder / "run.log").write_text("x")
    _drop(screen, [folder])
    assert said["refusals"] == []
    assert _src(screen) == str(folder)
    assert [line for line in said["lines"] if "neither images" in line] == [
        "[drop] 2 file(s) that are neither images nor PDFs were left aside.\n"]


def test_a_file_that_is_neither_is_still_refused_alone(
        figure_screen, tmp_path, said):
    note = tmp_path / "notes.txt"
    note.write_text("x")
    _drop(figure_screen, [note])
    assert len(said["refusals"]) == 1
    assert "Plaque Assay reads" in said["refusals"][0][1]


def test_figure_mode_reads_a_folder_of_papers_paper_by_paper(tmp_path):
    folder = tmp_path / "plates"
    for name in ("b_paper", "a_paper"):
        _png(folder / name / "figure_1.png")
        (folder / name / PP.PAPER_FILE).write_text("{}")
    _png(folder / "loose.png")
    (folder / "not_a_paper").mkdir()
    _png(folder / "not_a_paper" / "x.png")
    assert PP.figure_folders(folder) == [folder, folder / "a_paper",
                                         folder / "b_paper"]
    assert PP.figure_folders(folder / "a_paper") == [folder / "a_paper"]
    assert [p.relative_to(folder).as_posix()
            for p in ppv.images_in(folder, papers=True)] == [
        "loose.png", "a_paper/figure_1.png", "b_paper/figure_1.png"]
    assert [p.name for p in ppv.images_in(folder)] == ["loose.png"]
    pdfs, images, papers = dh.plaque_inputs([folder])
    assert papers == [folder]


def test_the_figure_run_measures_every_paper_into_one_database(
        tmp_path, monkeypatch):
    from spacr import submodules

    folder = tmp_path / "plates"
    for name in ("p1", "p2", "p3"):
        _png(folder / name / "figure_1.png")
        (folder / name / PP.PAPER_FILE).write_text("{}")
    seen = []

    def measure(src, dst, **kwargs):
        seen.append((Path(src).name, Path(dst)))
        return {"papers": 1, "figures": 2, "regions": 3, "plaques": 4,
                "with_ruler": 3, "database": str(Path(dst) / "db"),
                "run_id": len(seen), "awaiting_approval": 0}

    monkeypatch.setattr(PP, "measure_figure_folder", measure)
    summary = submodules._analyze_plaque_figures(
        {"src": str(folder)}, "model.pt")
    assert [name for name, _dst in seen] == ["p1", "p2", "p3"]
    assert {dst for _name, dst in seen} == {folder / "plaque_figures"}
    assert summary["papers"] == 3 and summary["plaques"] == 12
    assert summary["run_id"] == 1
