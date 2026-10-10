"""Optional mask saving follows the field the curator actually leaves."""
from __future__ import annotations

import os
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QSettings

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MODE_BOX, MakeMasksScreen


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt import prefs

    settings_path = tmp_path / "editor.ini"
    monkeypatch.setattr(
        prefs, "_s", lambda: QSettings(str(settings_path), QSettings.IniFormat))
    folder = tmp_path / "fields"
    folder.mkdir()
    for index in range(2):
        image = np.full((64, 64), index + 1, dtype=np.uint16)
        imageio.imwrite(folder / f"field_{index}.tif", image)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    yield screen, folder, settings_path
    screen._magnifier.close()
    screen.close_folded()


def _edit_mask(screen, label=7):
    screen._canvas.mask[12:20, 12:20] = label


def _saved_mask(folder, index):
    return engine.mask_save_path(str(folder), f"field_{index}.tif")


def test_toggle_is_right_of_next_and_default_off_without_changing_manual_navigation(
        opened):
    screen, folder, _settings_path = opened
    toggle = screen._btn_save_on_navigation
    row = screen._tool_row_layout
    assert toggle.objectName() == "MakeMasksSaveOnNavigate"
    assert row.indexOf(toggle) == row.indexOf(screen._btn_next) + 1
    assert not toggle.isChecked()
    _edit_mask(screen)

    screen._btn_next.click()

    assert screen._current_index == 1
    assert not os.path.exists(_saved_mask(folder, 0))


def test_enabled_next_saves_only_an_edited_mask(opened):
    screen, folder, _settings_path = opened
    screen._btn_save_on_navigation.click()

    screen._btn_next.click()
    assert screen._current_index == 1
    assert not os.path.exists(_saved_mask(folder, 0))

    _edit_mask(screen)
    screen._btn_prev.click()

    assert screen._current_index == 0
    assert np.all(imageio.imread(_saved_mask(folder, 1))[12:20, 12:20] == 1)


@pytest.mark.parametrize("keep", [True, False], ids=["keep", "discard"])
def test_curating_saves_the_edited_mask_before_its_verdict(
        opened, monkeypatch, keep):
    screen, folder, _settings_path = opened
    screen._btn_save_on_navigation.click()
    _edit_mask(screen)
    record = engine.record_curation
    saved_before_verdict = []

    def record_after_save(*args, **kwargs):
        saved_before_verdict.append(os.path.isfile(_saved_mask(folder, 0)))
        return record(*args, **kwargs)

    monkeypatch.setattr(engine, "record_curation", record_after_save)

    button = screen._btn_keep if keep else screen._btn_discard
    button.click()

    assert screen._current_index == 1
    assert saved_before_verdict == [True]
    assert np.all(imageio.imread(_saved_mask(folder, 0))[12:20, 12:20] == 1)
    assert engine.curation_verdict(
        str(folder), str(folder / "field_0.tif")) is keep


@pytest.mark.parametrize("action", ["next", "previous", "keep", "discard"])
def test_failed_mask_save_keeps_the_field_and_verdict_unchanged(
        opened, monkeypatch, action):
    screen, folder, _settings_path = opened
    if action == "previous":
        screen._on_next()
    index = screen._current_index
    image = screen._canvas.image.copy()
    _edit_mask(screen)
    mask = screen._canvas.mask.copy()
    screen._btn_save_on_navigation.click()
    recorded = []
    warnings = []
    def fail(*_args, **_kwargs):
        raise OSError("read-only mask destination")

    monkeypatch.setattr(engine, "save_mask", fail)
    monkeypatch.setattr(engine, "record_curation",
                        lambda *_a, **_kw: recorded.append(True))
    monkeypatch.setattr(screen, "_warn",
                        lambda *args: warnings.append(args))

    if action == "next":
        screen._btn_next.click()
    elif action == "previous":
        screen._btn_prev.click()
    else:
        (screen._btn_keep if action == "keep" else screen._btn_discard).click()

    assert screen._current_index == index
    assert np.array_equal(screen._canvas.image, image)
    assert np.array_equal(screen._canvas.mask, mask)
    assert not os.path.exists(_saved_mask(folder, index))
    assert recorded == []
    assert warnings and "read-only mask destination" in str(warnings[-1])


def test_box_tool_still_saves_unsaved_mask_pixels_before_next(opened):
    screen, folder, _settings_path = opened
    _edit_mask(screen)
    screen._set_mode(MODE_BOX)
    screen._btn_save_on_navigation.click()

    screen._btn_next.click()

    assert screen._current_index == 1
    assert np.all(imageio.imread(_saved_mask(folder, 0))[12:20, 12:20] == 1)


def test_manual_save_makes_later_navigation_clean(opened, monkeypatch):
    screen, folder, _settings_path = opened
    _edit_mask(screen)
    assert screen._on_save() == _saved_mask(folder, 0)
    screen._btn_save_on_navigation.click()

    def fail(*_args, **_kw):
        raise AssertionError("already saved mask was written twice")

    monkeypatch.setattr(engine, "save_mask", fail)
    screen._btn_next.click()

    assert screen._current_index == 1


def test_skip_keeps_its_no_mask_contract_with_toggle_on(opened):
    screen, folder, _settings_path = opened
    screen._queue = SimpleNamespace(folder=str(folder))
    screen._btn_save_on_navigation.click()
    _edit_mask(screen)

    screen._on_skip()

    assert screen._current_index == 1
    assert not os.path.exists(_saved_mask(folder, 0))


def test_toggle_persists_for_the_next_editor(opened, qtbot):
    screen, _folder, settings_path = opened
    screen._btn_save_on_navigation.click()
    assert QSettings(str(settings_path), QSettings.IniFormat).value(
        "make_masks/save_on_navigation", type=bool) is True

    next_editor = MakeMasksScreen()
    qtbot.addWidget(next_editor)
    assert next_editor._btn_save_on_navigation.isChecked()
    next_editor._btn_save_on_navigation.click()
    assert QSettings(str(settings_path), QSettings.IniFormat).value(
        "make_masks/save_on_navigation", type=bool) is False
