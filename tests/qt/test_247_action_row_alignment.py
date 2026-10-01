"""Make Masks save/navigation remain aligned with editor actions."""
import numpy as np
import pytest
import tifffile


@pytest.mark.parametrize("width", [1280, 1600, 1920])
def test_save_navigation_share_action_row(qtbot, qt_theme_applied, tmp_path, width):
    from spacr.qt.screens.make_masks import MakeMasksScreen
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((48, 48), np.uint16))
    tifffile.imwrite(tmp_path / "b.tif", np.ones((48, 48), np.uint16))
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(tmp_path))
    screen.resize(width, 900)
    screen.show()
    qtbot.waitExposed(screen)
    buttons = (screen._btn_save, screen._btn_prev, screen._btn_next)
    center = screen._btn_undo.mapTo(screen, screen._btn_undo.rect().center()).y()
    for button in buttons:
        assert screen._tool_row_layout.indexOf(button) >= 0
        screen._tool_scroll.ensureWidgetVisible(button)
        qtbot.wait(20)
        assert abs(button.mapTo(screen, button.rect().center()).y() - center) <= 1
        assert button.visibleRegion().boundingRect() == button.rect()
    screen._canvas.mask[1:3, 1:3] = 37
    screen._btn_save.click()
    saved = tifffile.imread(tmp_path / "masks/a.tif")
    assert np.max(saved) == 1  # Existing saver numbers new objects from one.
    assert np.count_nonzero(saved) == 4
    screen._btn_next.click()
    assert screen._current_index == 1
    screen._btn_prev.click()
    assert screen._current_index == 0
