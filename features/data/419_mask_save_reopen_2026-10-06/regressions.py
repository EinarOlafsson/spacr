"""Real Draw/Brush edits survive Save, Next and a fresh screen."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import Qt

from spacr.qt import mask_engine as engine
from spacr.qt.screens.make_masks import MakeMasksScreen, MODE_DRAW, MODE_BRUSH


@pytest.mark.parametrize("extension", [".tif", ".tiff", ".png"])
@pytest.mark.parametrize("mode", [MODE_DRAW, MODE_BRUSH])
def test_saved_mouse_edits_replace_legacy_mask_on_reopen(
        qtbot, qt_theme_applied, tmp_path, extension, mode):
    folder = tmp_path / "images"
    folder.mkdir()
    (folder / "masks").mkdir()
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    for name in ("field_00", "field_01"):
        imageio.imwrite(folder / (name + extension), image)
        imageio.imwrite(folder / "masks" / (name + extension), np.zeros((64, 64), np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(folder))
    screen._canvas.resize(600, 400)
    screen._canvas.show()
    screen._set_mode(mode)
    points = [(15, 15), (32, 15), (32, 32), (15, 32), (15, 15)]
    mapped = [screen._canvas._image_to_canvas(x + .25, y + .25) for x, y in points]
    qtbot.mousePress(screen._canvas, Qt.LeftButton, pos=mapped[0])
    for point in mapped[1:]:
        qtbot.mouseMove(screen._canvas, pos=point)
    qtbot.mouseRelease(screen._canvas, Qt.LeftButton, pos=mapped[-1])
    assert screen._canvas.mask.any(), "the physical gesture must create labels"
    screen._btn_save.click()
    saved_path = Path(engine.mask_save_path(str(folder), "field_00" + extension))
    saved = imageio.imread(saved_path)
    assert saved.any()
    screen._btn_next.click()
    assert screen._current_index == 1
    screen.close()
    reopened = MakeMasksScreen()
    qtbot.addWidget(reopened)
    assert reopened._open_folder(str(folder))
    np.testing.assert_array_equal(reopened._canvas.mask, saved)
    np.testing.assert_array_equal(imageio.imread(folder / ("field_00" + extension)), image)


def test_clear_masks_is_immediately_left_of_discard_and_confirmed(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    imageio.imwrite(tmp_path / "field.tif", np.ones((32, 32), np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert screen._open_folder(str(tmp_path))
    layout = screen._btn_discard.parentWidget().layout()
    assert layout.indexOf(screen._btn_clear) >= 0
    assert layout.indexOf(screen._btn_clear) + 1 == layout.indexOf(screen._btn_discard)
    screen._canvas.mask[3:8, 3:8] = 3
    screen._canvas.mask[15:20, 15:20] = 9
    original = screen._canvas.mask.copy()
    screen._history.push(original)
    prompts = []

    def confirm(title, message):
        prompts.append((title, message))
        return False

    monkeypatch.setattr(screen, "_confirm", confirm)
    screen._btn_clear.click()
    assert len(prompts) == 1 and "2" in prompts[0][1]
    np.testing.assert_array_equal(screen._canvas.mask, original)
    monkeypatch.setattr(screen, "_confirm", lambda *_a: True)
    screen._btn_clear.click()
    assert not screen._canvas.mask.any()
    screen._on_undo()
    np.testing.assert_array_equal(screen._canvas.mask, original)
