"""External Masks opens as a full page on Import, with room for its console.

Lessons 31 (External Masks) and 79 (module inputs and outputs) open External
Masks from Import's fold strip and read the console after a six-figure run.
The console viewport stayed at 155 px where the readable tour needs 240 px,
because the runtime column only got about 40 % of the window. Two causes,
both measured here in a 2560 x 1440 main window:

* Import scrolls on a cramped display (item 707), which moves its layout
  into a ``CrampedScreenScroll``. The fold strip looked for the page body in
  the screen's outer layout, found only the scroll, and opened the module as
  a 1180 x 760 window instead of a page.
* Import's stretched child was the mapping table alone, so even as a page
  the folded module sat between Import's input rows and its Import button.

Import's whole form is now one stretched page, and the page body is found
through the cramped scroll, so the folded module gets the window below
Import's masthead.
"""
from __future__ import annotations

import matplotlib
import pytest

pytest.importorskip("PySide6")

matplotlib.use("Agg")

import matplotlib.pyplot as plt                                 # noqa: E402
from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtTest import QTest                                # noqa: E402
from PySide6.QtWidgets import QApplication, QTabWidget          # noqa: E402

WINDOW = (2560, 1440)
CONSOLE_VIEWPORT_MIN = 240
FIGURES = 6


def _pump(n: int = 40) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture(params=[1.0, 1.5], ids=["font100", "font150"])
def import_window(request, qtbot, qt_theme_applied):
    """The main window at the lesson size, on Import, at a font scale."""
    from spacr.qt import app as gui
    from spacr.qt import preferences

    saved = preferences.get_font_scale()
    preferences.set_font_scale(request.param)
    preferences.apply_preferences_to_app(QApplication.instance())
    window = gui.MainWindow()
    qtbot.addWidget(window)
    window.resize(*WINDOW)
    window.show()
    _pump()
    window._on_nav_selected("foreign")
    qtbot.waitUntil(lambda: window._screens.get("foreign") is not None,
                    timeout=60000)
    _pump(80)
    try:
        yield window
    finally:
        window.close()
        preferences.set_font_scale(saved)
        preferences.apply_preferences_to_app(QApplication.instance())


def _open_external_masks(qtbot, window):
    from spacr.qt.screens.app_screen import AppScreen

    host = window._screens["foreign"]
    host._fold_strip.button_for("external_masks").click()
    qtbot.waitUntil(
        lambda: any(c.app_key == "external_masks" and c.isVisible()
                    for c in window.findChildren(AppScreen)),
        timeout=60000)
    _pump(80)
    screen = next(c for c in window.findChildren(AppScreen)
                  if c.app_key == "external_masks" and c.isVisible())
    return host, screen


def _read_the_console(screen) -> int:
    """The lesson's steps: fold System and Figures, open the console, give
    it the room."""
    usage = screen._usage_card
    if not usage.folder.shut:
        QTest.mouseClick(usage.title_label, Qt.LeftButton)
        _pump()
    if screen._console_folder.shut:
        QTest.mouseClick(screen._console_header, Qt.LeftButton)
        _pump()
    split = screen._runtime_splitter
    sizes = list(split.sizes())
    figures = split.indexOf(screen._figures_card)
    console = split.indexOf(screen._console_wrap)
    other = sum(size for slot, size in enumerate(sizes)
                if slot not in (figures, console))
    room = max(sum(sizes), split.height()) - other
    sizes[figures] = screen._figures_card.title_label.sizeHint().height()
    sizes[console] = room
    split.setSizes(sizes)
    _pump()
    screen._console.set_split_sizes(1400, 80)
    _pump()
    return screen._console._scroll.viewport().height()


def test_external_masks_opens_as_a_page_on_import(qtbot, import_window):
    """A page on Import's strip, not a separate 1180 x 760 window."""
    host, screen = _open_external_masks(qtbot, import_window)

    assert not screen.isWindow()
    pages = host._fold_pages
    assert isinstance(pages, QTabWidget)
    assert pages.indexOf(screen) > 0
    assert pages.currentWidget() is screen


def test_the_folded_page_gets_the_window_below_imports_masthead(
        qtbot, import_window):
    """Not the slot the mapping table had between Import's rows."""
    host, screen = _open_external_masks(qtbot, import_window)

    assert screen.height() >= 0.8 * host.height(), (
        screen.height(), host.height())
    assert screen._runtime_splitter.height() >= 0.7 * host.height()


def test_the_console_viewport_has_room_after_a_six_figure_run(
        qtbot, import_window):
    """Lessons 31 and 79 read the console after six figures arrived."""
    _host, screen = _open_external_masks(qtbot, import_window)
    for index in range(FIGURES):
        figure, axes = plt.subplots()
        axes.plot([0, 1], [index, index + 1])
        screen._on_figure_ready(figure)
    _pump(100)
    assert screen._figure_queue.count() == FIGURES

    height = _read_the_console(screen)

    assert screen._console.isVisible()
    assert height >= CONSOLE_VIEWPORT_MIN, height


def test_the_console_viewport_has_room_when_the_page_opens(
        qtbot, import_window):
    """Before any run the console is already readable at the lesson size."""
    _host, screen = _open_external_masks(qtbot, import_window)

    assert _read_the_console(screen) >= CONSOLE_VIEWPORT_MIN
