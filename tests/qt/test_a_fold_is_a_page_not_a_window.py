"""A folded module arrives as a page on its host, not a window over it.

"the prefered way is to integrate the folded module into the new module
... some new module could take space above the console or become a tab.
anything to integrate the new module naturally ... if you cannot find any
other way, then do your new window idea."

So the shared half of a fold puts the module's own screen on a page beside
the host's own, and only falls back to a window when the host has no body
to make pages out of. What these tests protect:

* THE HOST IS UNCHANGED BEHIND ITS PAGE. Its body keeps the stretch it
  had, so a module with no fold open looks as it did with a tab strip
  across the top rather than squashed into a corner.
* THE HOST'S OWN PAGE CANNOT BE CLOSED. There is nothing behind it.
* CLOSING A FOLDED PAGE KEEPS THE MODULE. A window that was closed took
  its state with it; a page put away keeps what it had loaded, and the
  same object comes back.
* AND THE WINDOW IS STILL THERE for a host that cannot carry pages, or a
  fold would silently open nothing at all.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from spacr.qt.screens import map_barcodes
from spacr.qt.screens.app_screen import AppScreen


def _host(qtbot):
    """A settings-form host with its fold strip installed."""
    screen = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(screen)
    assert map_barcodes.install_folds(screen) is not None
    return screen


def test_the_hosts_body_becomes_its_first_page(qtbot, qt_theme_applied):
    """The strip replaces the body in place, and the body keeps its stretch."""
    screen = _host(qtbot)
    body = map_barcodes._page_body(screen)
    outer = screen.layout()
    index = outer.indexOf(body)
    stretch = outer.stretch(index)

    pages = map_barcodes.host_pages(screen)

    assert isinstance(pages, QTabWidget)
    assert pages.objectName() == map_barcodes.PAGES_NAME
    assert pages.widget(0) is body
    assert outer.indexOf(pages) == index
    assert outer.stretch(outer.indexOf(pages)) == stretch
    assert pages.tabText(0) == "Map Barcodes"


def test_the_strip_is_made_once(qtbot, qt_theme_applied):
    """A second fold joins the strip the first one made."""
    screen = _host(qtbot)

    first = map_barcodes.host_pages(screen)

    assert map_barcodes.host_pages(screen) is first


def test_a_later_fold_refreshes_the_existing_hosts_qss_scope(
        qtbot, qt_theme_applied):
    """A registrar imported by fold two cannot disappear behind fold one."""
    from spacr.qt.theme import (
        clear_widget_qss_overlays,
        register_widget_qss,
        unregister_widget_qss,
    )

    screen = _host(qtbot)
    pages = map_barcodes.host_pages(screen)
    live_sheet = qt_theme_applied.styleSheet()
    name = "_SecondFoldProbe"
    register_widget_qss(
        name,
        lambda palette, opacity: f"QFrame#{name} {{ color: #ff00ff; }}",
        replace=True,
    )
    folded = QWidget()
    qtbot.addWidget(folded)
    try:
        assert name not in screen.styleSheet()
        assert map_barcodes.show_as_page(folded, screen, "Second") is folded
        assert screen._fold_pages is pages
        assert f"registered widget QSS: {name}" in screen.styleSheet()
        assert qt_theme_applied.styleSheet() == live_sheet
    finally:
        unregister_widget_qss(name)
        clear_widget_qss_overlays(qt_theme_applied)


def test_a_screen_that_names_itself_gets_that_name_on_its_page(
        qtbot, qt_theme_applied):
    """A screen with no registry key says what to call its own page.

    Make Masks and Annotate build their own mastheads and carry no app
    key, so without this their page would be captioned with nothing.
    """
    screen = QWidget()
    qtbot.addWidget(screen)
    column = QVBoxLayout(screen)
    column.addWidget(QLabel("masthead"))
    column.addWidget(QWidget(), 1)
    screen._fold_page_title = "Hand Curation"

    pages = map_barcodes.host_pages(screen)

    assert pages.tabText(0) == "Hand Curation"


def test_the_hosts_own_page_survives_a_close_request(qtbot, qt_theme_applied):
    """Closing it would leave the module with nothing on screen."""
    screen = _host(qtbot)
    opener = screen._fold_openers[0]
    opener.open()
    pages = screen._fold_pages

    pages.tabCloseRequested.emit(0)

    assert pages.count() == 2
    assert pages.tabText(0) == "Map Barcodes"


def test_closing_a_folded_page_keeps_the_module_it_held(
        qtbot, qt_theme_applied):
    """Put away, not thrown away: the next press finds what was typed."""
    screen = _host(qtbot)
    opener = screen._fold_openers[0]
    folded = opener.open()
    folded._settings_model.set_value_for_key("count_data", "/tmp/reads.csv")
    pages = screen._fold_pages

    pages.tabCloseRequested.emit(pages.indexOf(folded))
    assert pages.count() == 1

    assert opener.open() is folded
    assert (folded._settings_model.collect()["count_data"]
            == ["/tmp/reads.csv"])
    assert pages.count() == 2


def test_a_host_with_no_body_falls_back_to_a_window(qtbot, qt_theme_applied):
    """The window is the last resort, and it is still there to fall back to."""
    bare = QWidget()
    qtbot.addWidget(bare)
    QVBoxLayout(bare).addWidget(QLabel("nothing stretches here"))

    assert map_barcodes._page_body(bare) is None
    assert map_barcodes.host_pages(bare) is None

    # Not registered with qtbot: it becomes a child window of `bare`
    # below, so `bare` is what closes it -- and a second close of a widget
    # Qt has already deleted is a RuntimeError at teardown.
    from spacr.qt.theme import (
        clear_widget_qss_overlays,
        register_widget_qss,
        unregister_widget_qss,
    )

    folded = AppScreen(app_key="barcode_qc")
    opener = map_barcodes.FoldOpener(
        bare, "barcode_qc", lambda _window: folded)
    live_sheet = qt_theme_applied.styleSheet()
    name = "_FoldWindowProbe"
    register_widget_qss(
        name,
        lambda palette, opacity: f"QFrame#{name} {{ color: #ff00ff; }}",
        replace=True,
    )
    try:
        shown = opener.open()

        assert shown is folded
        assert shown.isWindow()
        assert shown.windowTitle() == "Barcode QC"
        assert f"registered widget QSS: {name}" in shown.styleSheet()
        assert qt_theme_applied.styleSheet() == live_sheet
    finally:
        unregister_widget_qss(name)
        clear_widget_qss_overlays(qt_theme_applied)


def test_a_screen_with_no_layout_carries_no_pages(qtbot):
    """Asking a widget that was never laid out costs nothing."""
    assert map_barcodes._page_body(QWidget()) is None
    assert map_barcodes.host_pages(QWidget()) is None


def test_the_page_strip_is_styled_by_the_time_it_exists(
        qtbot, qt_theme_applied):
    """An unstyled tab strip is a black rectangle, not a slightly-off one.

    A fold is opened long after the application stylesheet was composed.
    Its rule is installed on the host screen before the tab strip is made,
    without rebuilding the application sheet around every cached screen.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt.theme import (stylesheet, unregister_widget_qss,
                                widget_qss_names)

    app = QApplication.instance()
    unregister_widget_qss(map_barcodes.PAGES_NAME)
    # The sheet as it was before this module registered anything: not
    # empty, which `ensure_widget_qss_applied` reads as "nothing has
    # styled the application", but without the block under test.
    app.setStyleSheet(stylesheet())
    live_sheet = app.styleSheet()
    assert f"QTabWidget#{map_barcodes.PAGES_NAME}" not in app.styleSheet()
    screen = _host(qtbot)

    try:
        map_barcodes.host_pages(screen)

        assert map_barcodes.PAGES_NAME in widget_qss_names()
        assert app.styleSheet() == live_sheet
        assert f"QTabWidget#{map_barcodes.PAGES_NAME}" in screen.styleSheet()
    finally:
        # The themed application is a session fixture; leaving it holding
        # this test's sheet would follow every test after it.
        app.setStyleSheet(stylesheet())
