"""The first Make Masks fold makes its page strip in a turn of its own.

Making the strip moves the screen's whole body onto it, and Qt restyles
every widget of a body it moves. That and the folded page joining the strip
were one freeze, which kept the Cellpose fold over the capture gate. The
strip is now made first and the window breathes before the page joins it;
what ends up on screen -- the strip, its pages, their sheets and palettes --
is the same as before.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication

from spacr.qt import screens as screens_package
from spacr.qt.screens import map_barcodes
from spacr.qt.screens.make_masks import FoldedModulePanel, MakeMasksScreen

pytestmark = pytest.mark.qt


def _close(made):
    try:
        made._magnifier.close()
        made.close_folded()
        made.close()
        made.deleteLater()
    except RuntimeError:
        pass
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


@pytest.fixture
def screens(qtbot, qt_theme_applied):
    made = []

    def build():
        screen = MakeMasksScreen()
        qtbot.addWidget(screen)
        screen.resize(1400, 900)
        screen.show()
        QApplication.processEvents()
        made.append(screen)
        return screen

    yield build
    for screen in made:
        _close(screen)


def _what_shows(screen, panel):
    pages = screen._fold_pages
    roles = (QPalette.Window, QPalette.WindowText, QPalette.Base,
             QPalette.Button)
    return {
        "count": pages.count(),
        "panel_index": pages.indexOf(panel),
        "tabs": [pages.tabText(i) for i in range(pages.count())],
        "current": pages.currentWidget() is panel,
        "screen_sheet": screen.styleSheet(),
        "pages_sheet": pages.styleSheet(),
        "panel_sheet": panel.styleSheet(),
        "palette": [pages.palette().color(role).name() for role in roles]
        + [panel.palette().color(role).name() for role in roles],
    }


def test_the_strip_is_made_a_breath_before_the_page_joins_it(
        screens, monkeypatch):
    screen = screens()
    seen = []
    def note(_worked, _breathed):
        pages = getattr(screen, "_fold_pages", None)
        seen.append(None if pages is None else pages.count())

    monkeypatch.setattr(screens_package, "_note_the_breath", note)
    assert getattr(screen, "_fold_pages", None) is None

    panel = screen.open_folded("model_compare")

    assert 1 in seen, f"no breath between the strip and its page: {seen}"
    assert screen._fold_pages.count() == 2
    assert screen._fold_pages.currentWidget() is panel


def test_the_strip_and_its_pages_look_as_they_did(screens):
    fresh = screens()
    panel = fresh.open_folded("model_compare")
    QApplication.processEvents()
    now = _what_shows(fresh, panel)

    eager = screens()
    screen = eager.folded_screen("model_compare")
    old_panel = FoldedModulePanel("model_compare", screen,
                                  panel.windowTitle(),
                                  actions=eager._fold_actions("model_compare"))
    eager._fold_dialogs["model_compare"] = old_panel
    eager.seed_folded("model_compare")
    assert map_barcodes.show_as_page(old_panel, eager,
                                     panel.windowTitle()) is old_panel
    old_panel.show()
    QApplication.processEvents()
    before = _what_shows(eager, old_panel)

    assert now == before
