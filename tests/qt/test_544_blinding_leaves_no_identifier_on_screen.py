"""Blind scoring (544): no identifier anywhere on a blinded screen.

The first tests of 544 read the labels a user sees. These read EVERY string
the two blinded screens hold -- label text, tooltips, window titles, status
tips, accessible names, placeholder text, combo items and item-view cells,
on hidden widgets as well as shown ones except the consoles, which blinding
hides -- and look for the plate, the condition in the source folder's name,
a well, a file name or the folder itself.

They also cover the routes the audit found open: a failure message that
quotes a path (status line and warning box), Train and Generate (which hand
the source to another screen), the source/folder pickers (which would open
in the plate folder) and Make Masks' ROI export (files named after the
field). Unblinding gives every one of them back.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr import run_journal as rj  # noqa: E402


@pytest.fixture
def journal(tmp_path, monkeypatch):
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    return runs


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)


@pytest.fixture
def plate(tmp_path) -> Path:
    """A plate whose folder names the condition and whose crops name the well."""
    src = tmp_path / "plate7_drugA"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir()
    rng = np.random.default_rng(1)
    paths = []
    for row in range(1, 3):
        for col in range(1, 4):
            path = src / "data" / f"plate7_r{row}_c{col}_f1_o0.png"
            Image.fromarray(rng.integers(0, 255, (12, 12, 3),
                                         dtype=np.uint8)).save(path)
            paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    return src


def _leak(tmp_path) -> "re.Pattern":
    """What no blinded screen may show: plate, condition, well, file, folder."""
    return re.compile(r"plate7|drugA|r\d_c\d|\w\.(?:png|tif)\b|"
                      + re.escape(str(tmp_path)))


def _every_string(root, skip=()):
    """Every user-facing string held by ``root`` and its children.

    :param root: the screen.
    :param skip: widgets whose subtree is not read (the hidden consoles).
    :returns: ``[(widget class, object name, property, text)]``.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (QAbstractItemView, QComboBox,
                                   QPlainTextEdit, QTextEdit, QWidget)

    def skipped(widget):
        """Whether ``widget`` sits inside one of the skipped subtrees."""
        while widget is not None:
            if widget in skip:
                return True
            widget = widget.parentWidget()
        return False

    found = []
    for widget in [root] + root.findChildren(QWidget):
        if skipped(widget):
            continue
        texts = []
        for name in ("text", "toolTip", "windowTitle", "statusTip",
                     "whatsThis", "accessibleName", "accessibleDescription",
                     "placeholderText", "title", "windowFilePath"):
            getter = getattr(widget, name, None)
            if callable(getter):
                try:
                    value = getter()
                except TypeError:
                    continue
                if isinstance(value, str):
                    texts.append((name, value))
        if isinstance(widget, (QTextEdit, QPlainTextEdit)):
            texts.append(("plain", widget.toPlainText()))
        if isinstance(widget, QComboBox):
            texts += [("item", widget.itemText(i)) for i in range(widget.count())]
            texts += [("tip", str(widget.itemData(i, Qt.ToolTipRole) or ""))
                      for i in range(widget.count())]
        if isinstance(widget, QAbstractItemView) and widget.model() is not None:
            model = widget.model()
            for r in range(min(model.rowCount(), 200)):
                for c in range(min(model.columnCount(), 20)):
                    for role in (Qt.DisplayRole, Qt.ToolTipRole):
                        value = model.data(model.index(r, c), role)
                        if isinstance(value, str):
                            texts.append(("cell", value))
        found += [(type(widget).__name__, widget.objectName(), k, v)
                  for k, v in texts if v]
    return found


def _leaks(root, pattern, skip=()):
    """The strings of :func:`_every_string` that match ``pattern``."""
    return [entry for entry in _every_string(root, skip)
            if pattern.search(entry[3])]


def test_scrub_replaces_paths_names_stems_and_the_source_folder():
    from spacr.qt.screens.annotate import _blind_lookup, _blind_scrub

    src = "/data/plate7_drugA"
    a = f"{src}/data/plate7_r1_c1_f1_o0.png"
    b = f"{src}/data/plate7_r1_c2_f1_o0.png"
    lookup = _blind_lookup({a: "B0002", b: "B0001"}, src)
    folders = [src, f"{src}/data"]
    text = (f"Could not read {a}: bad header. Also plate7_r1_c2_f1_o0.png, "
            f"(plate7_r1_c1_f1_o0) in plate7_drugA and {src}/other.db. "
            f"The data were fine.")
    out = _blind_scrub(text, lookup, folders)
    assert "plate7" not in out and "drugA" not in out
    assert "Could not read B0002: bad header." in out
    assert "B0001," in out and "(B0002)" in out
    assert "The data were fine." in out
    assert _blind_scrub("", lookup, folders) == ""


def test_scrub_does_not_give_two_crops_that_share_a_name_one_code():
    from spacr.qt.screens.annotate import _blind_lookup, _blind_scrub

    lookup = _blind_lookup({"/p1/a.png": "B0001", "/p2/a.png": "B0002"}, "")
    out = _blind_scrub("a.png failed", lookup, [])
    assert "B000" not in out and "a.png" not in out


@pytest.fixture
def annotate(qtbot, qt_theme_applied, plate, journal, alpha):
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.image_size = (32, 32)
    widget.resize(900, 700)
    widget.show()
    widget._open_source(str(plate))
    qtbot.waitUntil(lambda: bool(widget._page_paths), timeout=10000)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def test_blinded_annotate_holds_no_identifier_in_any_string(
        annotate, plate, tmp_path, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    screen = annotate
    pattern = _leak(tmp_path)
    assert _leaks(screen, pattern, skip=(screen._console_wrap,))
    screen._btn_blind.setChecked(True)
    qtbot.waitUntil(lambda: bool(screen._page_paths), timeout=10000)
    assert screen._console_wrap.isHidden()
    assert _leaks(screen, pattern, skip=(screen._console_wrap,)) == []

    crop = screen._page_paths[0][0]
    code = screen._blind["codes"][crop]
    screen._status_label.setText(f"Save failed — could not write {crop}")
    assert screen._status_label.text() == f"Save failed — could not write {code}"
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda parent, title, text, *a: shown.append(text))
    screen._on_retrain_failed(f"no features for {os.path.basename(crop)} "
                              f"in {plate / 'measurements'}")
    assert shown and not pattern.search(shown[0]) and code in shown[0]
    assert not pattern.search(screen._status_label.text())

    assert not screen._btn_train.isEnabled()
    assert not screen._btn_generate.isEnabled()
    assert screen._starting_folder() == os.path.expanduser("~")

    assert screen._end_blind(ask=lambda: True)
    assert screen._btn_train.isEnabled() and screen._btn_generate.isEnabled()
    screen._status_label.setText(f"could not write {crop}")
    assert crop in screen._status_label.text()
    assert screen._starting_folder() != os.path.expanduser("~") or \
        str(plate) == os.path.expanduser("~")
    assert str(plate) in screen._src_label.text()


@pytest.fixture
def masks(qtbot, qt_theme_applied, tmp_path, journal, alpha):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    folder = tmp_path / "plate7_drugA_images"
    folder.mkdir()
    rng = np.random.default_rng(2)
    names = [f"plate7_r{r}_c{c}_f1.tif" for r in (1, 2) for c in (1, 2, 3)]
    for name in names:
        imageio.imwrite(folder / name,
                        rng.integers(0, 65535, (32, 32), dtype=np.uint16))
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    widget._open_folder(str(folder))
    return widget, folder


def test_blinded_make_masks_holds_no_identifier_and_locks_the_roi_menu(
        masks, tmp_path, qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    screen, folder = masks
    pattern = _leak(tmp_path)
    qtbot.waitUntil(lambda: "(1/6)" in screen._status_label.text(),
                    timeout=10000)
    assert screen._btn_rois.isEnabled()
    assert _leaks(screen, pattern, skip=(screen._console_section,))
    screen._btn_blind.setChecked(True)
    qtbot.waitUntil(lambda: "(1/6)" in screen._status_label.text()
                    and "plate7" not in screen._status_label.text(),
                    timeout=10000)
    screen._on_next()
    qtbot.wait(50)
    assert screen._console_section.isHidden()
    assert _leaks(screen, pattern, skip=(screen._console_section,)) == []
    assert not screen._btn_rois.isEnabled()

    started = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda parent, caption, start, *a: started.append(start)
                        or "")
    screen._on_pick_folder()
    assert started == [os.path.expanduser("~")]

    assert screen._end_blind(ask=lambda: True)
    assert screen._btn_rois.isEnabled()
    screen._on_pick_folder()
    assert started[-1] == str(folder)
