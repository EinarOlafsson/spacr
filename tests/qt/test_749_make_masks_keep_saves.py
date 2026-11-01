"""Keep is Make Masks' one commit: it saves the mask and moves on.

The maintainer's 2026-11-01 decision: ANNOTATION's Keep and NAVIGATION's
separate Save mask button left it unclear whether Keep saved. Keep now
writes the mask, records the verdict and moves to the next image; the
Saved mark beside it says whether the mask on screen is the one on disk;
Ctrl+S still saves and stays; Skip and Discard are unchanged.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QPushButton

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen


def _settled(screen):
    return (not screen._loading and screen._reader_worker is None
            and screen._load_worker is None)


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt import prefs

    settings_path = tmp_path / "editor.ini"
    monkeypatch.setattr(
        prefs, "_s", lambda: QSettings(str(settings_path), QSettings.IniFormat))
    folder = tmp_path / "fields"
    folder.mkdir()
    for index in range(3):
        imageio.imwrite(folder / f"field_{index}.tif",
                        np.full((64, 64), index + 1, dtype=np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: _settled(screen))
    assert not screen._btn_save_on_navigation.isChecked()
    yield screen, folder
    screen._magnifier.close()
    screen.close_folded()


def _saved(folder, index):
    return engine.mask_save_path(str(folder), f"field_{index}.tif")


def _edit(screen, label=7):
    screen._canvas.mask[12:20, 12:20] = label
    screen._on_stroke_finished()


def _saved_shown(screen):
    return not screen._saved_state.isHidden()


def test_there_is_no_separate_save_mask_button(opened):
    screen, _folder = opened
    texts = [b.text().replace("&", "") for b in screen.findChildren(QPushButton)]
    assert "Save mask" not in texts
    assert not hasattr(screen, "_btn_save")
    nav = [b.text() for b in screen._navigation_group.findChildren(QPushButton)
           if b.parentWidget() is screen._navigation_group]
    assert nav == ["Previous", "Next"]
    assert screen._act_save_mask in screen._btn_keep.split_menu().actions()
    assert screen._saved_state.parentWidget() is screen._annotation_group
    assert screen._saved_state.text() == "Saved ✓"


def test_keep_writes_the_mask_and_moves_to_the_next_image(opened, qtbot):
    screen, folder = opened
    _edit(screen)
    assert not os.path.exists(_saved(folder, 0))

    screen._btn_keep.click()
    qtbot.waitUntil(lambda: _settled(screen))

    assert screen._current_index == 1
    assert np.all(imageio.imread(_saved(folder, 0))[12:20, 12:20] == 1)
    assert engine.curation_verdict(
        str(folder), str(folder / "field_0.tif")) is True
    assert not _saved_shown(screen)


def test_keep_on_an_unedited_field_writes_its_mask(opened, qtbot):
    screen, folder = opened
    screen._btn_keep.click()
    qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 1
    assert os.path.isfile(_saved(folder, 0))


def test_saved_mark_follows_keep_edits_and_navigation(opened, qtbot):
    screen, folder = opened
    assert not _saved_shown(screen)
    _edit(screen)
    screen._btn_keep.click()
    qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 1 and not _saved_shown(screen)

    screen._btn_prev.click()
    qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 0
    assert _saved_shown(screen)

    _edit(screen, label=9)
    assert not _saved_shown(screen)
    screen._act_undo.trigger()
    assert _saved_shown(screen)


def test_keep_on_the_last_field_stays_and_shows_saved(opened, qtbot):
    screen, folder = opened
    for _ in range(2):
        screen._btn_next.click()
        qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 2
    _edit(screen)
    screen._btn_keep.click()
    assert screen._current_index == 2
    assert os.path.isfile(_saved(folder, 2))
    assert _saved_shown(screen)


def test_ctrl_s_saves_and_stays_and_shows_saved(opened):
    screen, folder = opened
    _edit(screen)
    want = QKeySequence("Ctrl+S")
    found = [sc for sc in screen.findChildren(QShortcut) if sc.key() == want]
    assert len(found) == 1
    found[0].activated.emit()
    assert screen._current_index == 0
    assert np.all(imageio.imread(_saved(folder, 0))[12:20, 12:20] == 1)
    assert engine.curation_verdict(str(folder), str(folder / "field_0.tif")) is None
    assert _saved_shown(screen)
    _edit(screen, label=9)
    assert not _saved_shown(screen)
    screen._act_save_mask.trigger()
    assert _saved_shown(screen)


def test_shortcut_list_names_no_missing_button(opened):
    from spacr.qt.screens.make_masks import SHORTCUT_HINTS

    hints = dict(SHORTCUT_HINTS)
    assert hints["Ctrl+S"] == "Save the mask"
    assert not any("Save mask" in text for text in hints.values())


def test_a_failed_keep_save_shows_no_saved_mark(opened, monkeypatch):
    screen, folder = opened
    _edit(screen)

    def fail(*_args, **_kwargs):
        raise OSError("read-only mask destination")

    monkeypatch.setattr(engine, "save_mask", fail)
    monkeypatch.setattr(screen, "_warn", lambda *args: None)
    screen._btn_keep.click()
    assert screen._current_index == 0
    assert not _saved_shown(screen)
    assert engine.curation_verdict(str(folder), str(folder / "field_0.tif")) is None


def test_discard_still_writes_no_mask_and_moves_on(opened, qtbot):
    screen, folder = opened
    _edit(screen)
    screen._btn_discard.click()
    qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 1
    assert not os.path.exists(_saved(folder, 0))
    assert engine.curation_verdict(
        str(folder), str(folder / "field_0.tif")) is False


def test_skip_still_writes_no_mask_and_moves_on(opened, qtbot):
    screen, folder = opened
    screen._queue = SimpleNamespace(folder=str(folder))
    screen._sync_button_states()
    assert screen._btn_skip.isEnabled()
    _edit(screen)
    screen._btn_skip.click()
    qtbot.waitUntil(lambda: _settled(screen))
    assert screen._current_index == 1
    assert not os.path.exists(_saved(folder, 0))
