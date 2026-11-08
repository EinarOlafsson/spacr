"""Annotate's pair review dialog and count column keep their edge-case behaviour.

The pair review covers its folder pickers, plane parsing, unreadable sets,
failed verdict writes and the keys it leaves to Qt. The count column covers a
tile painted with its count, stepping and typing with nothing to annotate, and
values that are not numbers.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tifffile                                                    # noqa: E402
from PySide6.QtCore import Qt                                      # noqa: E402
from PySide6.QtGui import QPixmap                                  # noqa: E402
from PySide6.QtTest import QTest                                   # noqa: E402
from PySide6.QtWidgets import QFileDialog                          # noqa: E402

from spacr import _merge_pairs as pairs                            # noqa: E402
from spacr.qt.screens import annotate as annotate_mod              # noqa: E402

N_CROPS = 4


def _field() -> np.ndarray:
    mask = np.zeros((160, 160), dtype=np.uint16)
    mask[30:70, 30:50] = 1
    mask[30:70, 50:70] = 2
    mask[100:130, 90:100] = 3
    mask[100:130, 100:104] = 4
    return mask


@pytest.fixture
def stacks(tmp_path):
    folder = tmp_path / "masks"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for stem in ("P1_A01_1", "P1_B02_4"):
        image = rng.integers(100, 4000, size=(3, 160, 160)).astype(np.uint16)
        tifffile.imwrite(folder / f"{stem}.tif",
                         np.concatenate([image, _field()[None]], 0),
                         photometric="minisblack")
    return folder


def _dialog(qtbot, stacks, tmp_path):
    dialog = annotate_mod._PairReviewDialog()
    qtbot.addWidget(dialog)
    dialog._masks.setText(str(stacks))
    dialog._images.setText(str(stacks))
    dialog._mask_plane.setValue(-1)
    dialog._object.setText("cell")
    dialog._target.setText(str(tmp_path / "set"))
    return dialog


def test_a_browse_button_fills_its_box_with_the_chosen_folder_and_keeps_it_when_dismissed(
        qtbot, stacks, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, stacks, tmp_path)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: '/picked/folder'))
    dialog._browse_into(dialog._images, 'Images')
    assert dialog._images.text() == '/picked/folder'

    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: ''))
    dialog._browse_into(dialog._images, 'Images')
    assert dialog._images.text() == '/picked/folder'


def test_the_open_button_opens_the_chosen_set_and_ignores_a_dismissed_dialog(
        qtbot, stacks, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()
    again = annotate_mod._PairReviewDialog()
    qtbot.addWidget(again)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: ''))
    again._on_open()
    assert again._pairs == []

    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        staticmethod(lambda *a, **k: str(tmp_path / 'set')))
    again._on_open()
    assert len(again._pairs) == 4


def test_the_planes_box_skips_blank_entries_and_defaults_to_the_first_plane(
        qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog._image_planes.setText('1, ,2,')
    assert dialog._planes() == [1, 2]
    dialog._image_planes.setText(' , ')
    assert dialog._planes() == [0]


def test_building_with_a_plane_that_is_not_a_number_reports_why_and_builds_nothing(
        qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog._image_planes.setText('x')
    assert dialog.build_set() == 0
    assert dialog._status.text().startswith('No set was built')
    assert dialog._pairs == []


def test_a_folder_that_holds_no_readable_set_reports_it_and_disables_the_buttons(
        qtbot, stacks, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()

    def broken(folder):
        raise OSError('database is locked')

    monkeypatch.setattr(pairs, '_read_set', broken)
    assert dialog.open_set(str(tmp_path / 'set')) == 0
    assert 'database is locked' in dialog._status.text()
    assert dialog._pairs == []
    assert not dialog._next.isEnabled()


def test_with_no_pairs_the_caption_says_so_and_a_verdict_writes_nothing(qtbot):
    dialog = annotate_mod._PairReviewDialog()
    qtbot.addWidget(dialog)
    dialog.show_pair(3)
    assert dialog._caption.text() == 'No pairs to review.'
    assert dialog.current_pair() is None
    assert dialog.record(1) is False


def test_a_verdict_that_cannot_be_saved_is_reported_and_the_pair_stays_unjudged(
        qtbot, stacks, tmp_path, monkeypatch):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()

    def refuse(*args):
        raise OSError('disk full')

    monkeypatch.setattr(pairs, '_write_verdict', refuse)
    assert dialog.record(2) is False
    assert 'disk full' in dialog._status.text()
    assert dialog.current_pair()[pairs.MERGE_COLUMN] is None
    assert dialog._index == 0


def test_a_key_the_review_does_not_use_is_left_to_qt_and_space_moves_on(
        qtbot, stacks, tmp_path):
    dialog = _dialog(qtbot, stacks, tmp_path)
    dialog.build_set()
    dialog.show()
    QTest.keyClick(dialog, Qt.Key_Space)
    assert dialog._index == 1
    QTest.keyClick(dialog, Qt.Key_Q)
    assert dialog._index == 1


def test_the_pair_review_opens_on_the_source_when_it_is_a_set_and_empty_otherwise(
        qtbot, stacks, tmp_path):
    builder = _dialog(qtbot, stacks, tmp_path)
    builder.build_set()
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.src = str(tmp_path / 'set')
    dialog = screen._on_pair_review()
    qtbot.addWidget(dialog)
    assert len(dialog._pairs) == 4

    screen._settings.src = str(tmp_path / 'nothing')
    empty = screen._on_pair_review()
    qtbot.addWidget(empty)
    assert empty._pairs == []


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir(parents=True)
    rng = np.random.default_rng(5)
    paths = []
    for i in range(N_CROPS):
        p = src / "data" / f"crop_{i}.png"
        Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(p)
        paths.append((str(p),))
    con = sqlite3.connect(src / "measurements" / "measurements.db", timeout=30)
    try:
        con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        con.executemany('INSERT INTO "png_list" (png_path) VALUES (?)', paths)
        con.commit()
    finally:
        con.close()
    return src


def _screen(qtbot):
    screen = annotate_mod.AnnotateScreen()
    qtbot.addWidget(screen)
    screen._settings.grid_rows = 1
    screen._settings.grid_cols = N_CROPS
    screen._settings.image_size = (24, 24)
    screen._settings.annotation_column = "pv_count"
    screen._settings.annotation_counts = True
    screen._compute_grid_dims = lambda: None
    screen._rebuild_grid()
    return screen


def _open(qtbot, src: Path):
    screen = _screen(qtbot)
    screen._open_source(str(src))
    qtbot.waitUntil(lambda: len(screen._page_paths) == N_CROPS, timeout=5000)
    return screen


def test_stepping_and_typing_a_count_with_nothing_open_asks_for_a_source(qtbot):
    screen = _screen(qtbot)
    assert screen.handle_key("+") is True
    assert "open a source first" in screen._kbd_hint.text()
    screen._set_kbd_hint('')
    assert screen.handle_key("#") is True
    assert "open a source first" in screen._kbd_hint.text()


def test_a_stored_value_that_is_not_a_number_counts_from_zero_and_shows_no_count(
        qtbot, source):
    screen = _open(qtbot, source)
    screen._set_focus_slot(0)
    path = screen._page_paths[0][0]
    screen._page_paths[0] = (path, 'abc')
    assert screen._count_text(0) == ""
    assert screen.handle_key("+") is True
    assert screen._current_value(0) == 1

    screen._page_paths[1] = (screen._page_paths[1][0], 'xyz')
    screen._set_focus_slot(1)
    screen._ask_count = lambda current: 7
    assert screen.handle_key("#") is True
    assert screen._current_value(1) == 7
    screen._worker.stop(wait=True)


def test_a_tile_with_a_count_paints_its_tag_and_a_repeated_count_changes_nothing(
        qtbot, source):
    screen = _open(qtbot, source)
    thumb = screen._thumbs[0]
    assert thumb.set_count_text("12") is True
    assert thumb.set_count_text("12") is False
    assert thumb.count_text() == "12"
    thumb.resize(60, 60)
    picture = thumb.grab()
    assert not picture.isNull()
    assert thumb.set_count_text("") is True
    assert thumb.count_text() == ""
    assert isinstance(QPixmap(picture), QPixmap)
    screen._worker.stop(wait=True)


def test_a_remembered_choice_keeps_only_the_names_still_on_offer():
    assert annotate_mod._known(None, ['a', 'b']) is None
    assert annotate_mod._known(['a', 'z'], ['a', 'b']) == ['a']
    assert annotate_mod._known(['z'], ['a', 'b']) is None


def test_a_ring_is_drawn_dashed_for_a_suggestion_and_skipped_when_the_tile_is_too_small():
    from PySide6.QtGui import QImage, QPainter

    image = QImage(40, 40, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        annotate_mod._Thumbnail._stroke(painter, '#ff0000', 4, 2.0, 40, 40, dashed=True)
        annotate_mod._Thumbnail._stroke(painter, '#00ff00', 4, 30.0, 40, 40)
    finally:
        painter.end()
    drawn = {image.pixelColor(x, y).name() for x in range(40) for y in range(40)
             if image.pixelColor(x, y).alpha()}
    assert any(name.startswith('#ff') for name in drawn)
    assert not any(name == '#00ff00' for name in drawn)
