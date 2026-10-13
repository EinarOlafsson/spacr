"""The Hits control must expose a usable results panel before any run exists."""
from PySide6.QtCore import Qt
import pytest

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens import regression


@pytest.mark.parametrize("folded", [False, True])
def test_hits_button_reveals_the_initially_hidden_results(qtbot, qt_theme_applied, folded):
    screen = AppScreen(app_key="regression")
    qtbot.addWidget(screen)
    screen.resize(1500, 1000)
    screen.show()
    strip = regression.install_folds(screen)
    assert strip is not None
    assert screen._figures_card.isHidden()
    if folded:
        screen._figures_card.show()
        screen._figures_card.folder.set_shut(True, by_user=False)

    qtbot.mouseClick(strip.button_for("hit_list"), Qt.LeftButton)
    panel = regression.results_panel(screen)
    assert panel is not None
    assert panel.tabs.currentWidget() is panel.hits
    qtbot.waitUntil(panel.hits.isVisible, timeout=3000)
    assert screen._figures_card.isVisible()
    assert not screen._figures_card.folder.shut
    assert not screen._worker_thread_is_running()
