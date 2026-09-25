"""Item 517: Make Masks and Plaque Assay take dropped files as well as folders.

Real ``QDropEvent`` objects carrying real ``QMimeData`` with local file URLs
are sent at the real screens, offscreen, with small real TIFF and PNG files
written in ``tmp_path``. A folder, one file, several files and a mix of files
and folders are dropped on each screen and the tests look at what the screen
then has open: Make Masks' queue, Plaque Assay's ``src``.

Plaque Assay used to be given Make Masks' drop policy, so a folder it could
not use was refused with "Make Masks needs a folder of images ...". It has its
own now, and no sentence it can show names Make Masks.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QUrl, Qt
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from spacr.qt import dnd as dnd_mod
from spacr.qt import dnd_handlers as dh


def _tif(path: Path, value: int = 7) -> Path:
    """Write a small real 16-bit TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((32, 32), value, dtype=np.uint16))
    return path


def _png(path: Path, value: int = 90) -> Path:
    """Write a small real 8-bit RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((32, 32, 3), value, dtype=np.uint8))
    return path


def _mime(paths) -> QMimeData:
    """Mime data with a local file URL per path."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _settle(widget, timeout_s: float = 20.0) -> None:
    """Wait for the drop's classification to come back from the scanner."""
    scanner = getattr(getattr(widget, "_dnd_screen", widget), "_dnd_scanner",
                      None)
    runner = getattr(scanner, "_runner", None)
    if runner is None:
        return
    seen = []
    runner.job_finished.connect(lambda *_: seen.append(True))
    deadline = time.monotonic() + timeout_s
    while not seen and time.monotonic() < deadline:
        QApplication.processEvents()
    assert seen, "the drop classification never came back"


def _drop(widget, paths) -> QDropEvent:
    """Replay drag enter, move and drop on ``widget``, then settle.

    Each event gets its own mime data, held in a local for the event's
    lifetime: an event does not own its mime data, and a temporary would be
    collected before Qt reads it.
    """
    enter_mime, move_mime, drop_mime = _mime(paths), _mime(paths), _mime(paths)
    enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, enter_mime,
                            Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    move = QDragMoveEvent(QPoint(5, 5), Qt.CopyAction, move_mime,
                          Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, move)
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, drop_mime,
                       Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    _settle(widget)
    return event


class _Box:
    """Stands in for QMessageBox and records what would have been shown."""

    calls: list = []

    @classmethod
    def information(cls, parent, title, text):
        """Record an information box."""
        cls.calls.append(text)


@pytest.fixture
def said(monkeypatch):
    """Every refusal the drop reports, as ``(reason, suggestion)``."""
    reports = []
    real = dnd_mod._report_drop_problem

    def record(screen, path, reason, suggestion, **kwargs):
        reports.append((reason, suggestion))
        return real(screen, path, reason, suggestion, **kwargs)

    _Box.calls = []
    monkeypatch.setattr(dnd_mod, "_report_drop_problem", record)
    monkeypatch.setattr(dnd_mod, "QMessageBox", _Box)
    return reports


@pytest.fixture
def masks_screen(qtbot, qt_theme_applied):
    """A real Make Masks screen with its drop zone installed."""
    from spacr.qt.screens import make_masks as mm

    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    try:
        made._magnifier.close()
        made._canvas.close_enhancer()
    except Exception:
        pass


@pytest.fixture
def plaque_screen(qtbot, qt_theme_applied):
    """A real Plaque Assay screen with its drop zone installed."""
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen("analyze_plaques")
    qtbot.addWidget(made)
    return made


def _src(screen) -> str:
    """The Plaque Assay form's ``src`` as text."""
    widget = screen._settings_model._widgets.get("src")
    for reader in ("get_value", "text"):
        read = getattr(widget, reader, None)
        if callable(read):
            return str(read())
    raise AssertionError("no readable src control")


def _queue(screen):
    """Make Masks' queue as absolute paths, in queue order."""
    folders = screen._field_folders or [screen._folder] * len(
        screen._image_files)
    return [os.path.join(folder, name)
            for folder, name in zip(folders, screen._image_files)]


def test_the_screens_have_their_own_handlers():
    assert type(dh.get_handler("analyze_plaques")) is dh.PlaqueDropHandler
    assert type(dh.get_handler("make_masks")) is dh.MakeMasksDropHandler
    assert type(dh.get_handler("train_cellpose")) is \
        dh.CellposeFolderDropHandler


def test_make_masks_takes_a_dropped_folder(masks_screen, tmp_path, said):
    a = _tif(tmp_path / "plate" / "a.tif")
    b = _png(tmp_path / "plate" / "b.png")
    _drop(masks_screen, [tmp_path / "plate"])
    assert _queue(masks_screen) == [str(a), str(b)]
    assert masks_screen._field_folders is None
    assert said == []


def test_make_masks_takes_one_dropped_file(masks_screen, tmp_path, said):
    _tif(tmp_path / "plate" / "a.tif")
    b = _tif(tmp_path / "plate" / "b.tif")
    _drop(masks_screen, [b])
    assert _queue(masks_screen) == [str(b)]
    assert masks_screen._canvas.image is not None
    assert said == []


def test_make_masks_queues_several_files_in_drop_order(masks_screen,
                                                       tmp_path, said):
    a = _tif(tmp_path / "plate" / "a.tif")
    b = _png(tmp_path / "plate" / "b.png")
    c = _tif(tmp_path / "plate" / "c.tif")
    _drop(masks_screen, [c, a, b])
    assert _queue(masks_screen) == [str(c), str(a), str(b)]
    assert masks_screen._folder == str(tmp_path / "plate")
    assert said == []


def test_make_masks_queues_a_mix_across_folders(masks_screen, tmp_path,
                                                said):
    x = _tif(tmp_path / "one" / "x.tif")
    y = _tif(tmp_path / "one" / "y.tif")
    z = _png(tmp_path / "two" / "z.png")
    _drop(masks_screen, [z, tmp_path / "one"])
    assert _queue(masks_screen) == [str(z), str(x), str(y)]
    assert masks_screen._folder == str(tmp_path / "two")
    masks_screen._on_next()
    assert masks_screen._folder == str(tmp_path / "one")
    assert masks_screen._canvas.image is not None
    from spacr.qt import mask_engine as engine
    assert engine.mask_save_path(masks_screen._folder, "x.tif") == \
        str(tmp_path / "one" / "masks" / "x.tif")
    assert said == []


def test_make_masks_still_refuses_what_it_cannot_open(masks_screen,
                                                      tmp_path, said):
    note = tmp_path / "notes.txt"
    note.write_text("not an image")
    _drop(masks_screen, [note])
    assert masks_screen._image_files == []
    assert said and "Make Masks" in said[0][0]


def test_plaque_takes_a_dropped_folder(plaque_screen, tmp_path, said):
    _png(tmp_path / "plaques" / "p1.png")
    _drop(plaque_screen, [tmp_path / "plaques"])
    assert _src(plaque_screen) == str(tmp_path / "plaques")
    assert said == []


def test_plaque_takes_one_dropped_file_without_copying(plaque_screen,
                                                       tmp_path, said):
    _png(tmp_path / "plaques" / "p1.png")
    p2 = _png(tmp_path / "plaques" / "p2.png", value=10)
    _drop(plaque_screen, [p2])
    src = Path(_src(plaque_screen))
    assert src.parent == tmp_path / "plaques"
    assert src.name.startswith("plaque_selection_")
    listed = sorted(src.iterdir())
    assert [p.name for p in listed] == ["p2.png"]
    assert listed[0].resolve() == p2.resolve()
    assert listed[0].is_symlink() or os.path.samefile(listed[0], p2)
    assert said == []


def test_plaque_takes_several_files(plaque_screen, tmp_path, said):
    p1 = _png(tmp_path / "plaques" / "p1.png")
    p2 = _tif(tmp_path / "plaques" / "p2.tif")
    _png(tmp_path / "plaques" / "p3.png")
    _drop(plaque_screen, [p2, p1])
    src = Path(_src(plaque_screen))
    assert sorted(p.name for p in src.iterdir()) == ["p1.png", "p2.tif"]
    assert said == []


def test_plaque_files_that_are_the_whole_folder_use_the_folder(
        plaque_screen, tmp_path, said):
    p1 = _png(tmp_path / "plaques" / "p1.png")
    p2 = _png(tmp_path / "plaques" / "p2.png")
    _drop(plaque_screen, [p2, p1])
    assert _src(plaque_screen) == str(tmp_path / "plaques")
    assert not any(p.name.startswith("plaque_selection_")
                   for p in (tmp_path / "plaques").iterdir())


def test_plaque_takes_a_mix_of_files_and_folders(plaque_screen, tmp_path,
                                                 said):
    _png(tmp_path / "a" / "p1.png")
    _png(tmp_path / "a" / "p2.png")
    same_name = _png(tmp_path / "b" / "p1.png", value=3)
    _drop(plaque_screen, [tmp_path / "a", same_name])
    src = Path(_src(plaque_screen))
    names = sorted(p.name for p in src.iterdir())
    assert names == ["b_p1.png", "p1.png", "p2.png"]
    assert (src / "b_p1.png").resolve() == same_name.resolve()
    assert said == []


def test_plaque_refusals_never_mention_make_masks(plaque_screen, tmp_path,
                                                  said, monkeypatch):
    from spacr.qt.widgets import plaque_preview as ppv

    asked = []
    monkeypatch.setattr(ppv, "ask_input_mode",
                        lambda parent, question, name, mode, **kw:
                        asked.append(question) or mode)
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    (empty / "readme.txt").write_text("x")
    _drop(plaque_screen, [empty])
    _drop(plaque_screen, [empty / "readme.txt"])
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    _drop(plaque_screen, [pdf])
    assert asked == [ppv.TO_FIGURE], (
        "item 518: a PDF in Plaque mode asks to switch instead of refusing")
    assert len(said) == 2
    for reason, suggestion in said:
        assert "Plaque Assay" in reason
        assert "Make Masks" not in reason + suggestion
    for text in _Box.calls:
        assert "Make Masks" not in text
    handler = dh.PlaqueDropHandler()
    assert "Make Masks" not in handler.error_message(empty)


def test_plaque_figure_mode_reads_a_dropped_pdf(plaque_screen, tmp_path,
                                                said, monkeypatch):
    from spacr.qt.widgets.plaque_preview import FIGURE_MODE

    panel = plaque_screen._live_preview
    monkeypatch.setattr(panel, "mode", lambda: FIGURE_MODE)
    fetched = []
    monkeypatch.setattr(panel, "fetch_paper",
                        lambda ref, parent: fetched.append((ref, parent))
                        or True)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    _drop(plaque_screen, [pdf])
    assert fetched == [(str(pdf), str(tmp_path))]
    assert said == []


def test_the_cellpose_screens_still_take_only_a_folder(tmp_path):
    image = _tif(tmp_path / "a.tif")
    handler = dh.CellposeFolderDropHandler()
    assert handler.can_accept(tmp_path) is True
    assert handler.can_accept(image) is False
    assert handler.accepts_multiple() is False
