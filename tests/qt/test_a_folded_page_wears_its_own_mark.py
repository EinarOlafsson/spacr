"""A folded module's page carries the icon its tile used to.

A module that folds into a host gives up its tile, and the icon is the
thing a user already recognises it by. A page carrying only a title asks
them to re-learn a name for something they could have identified at a
glance -- and the icon costs nothing, because the same registry art the
fold button draws is still shipped.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QTabWidget, QWidget       # noqa: E402


@pytest.fixture()
def host(qtbot):
    """A host screen carrying the page area folds open into."""
    from spacr.qt.screens.map_barcodes import host_pages

    made = QWidget()
    qtbot.addWidget(made)
    from PySide6.QtWidgets import QVBoxLayout

    QVBoxLayout(made)
    pages = QTabWidget(made)
    made.layout().addWidget(pages)
    made._fold_pages = pages
    assert host_pages(made) is pages
    return made


def _page(qtbot, key):
    page = QWidget()
    qtbot.addWidget(page)
    page.app_key = key
    return page


def test_the_page_wears_the_modules_own_mark(host, qtbot):
    from spacr.qt.screens.map_barcodes import show_as_page

    page = _page(qtbot, "agreement")
    show_as_page(page, host, "Annotator Agreement")

    pages = host._fold_pages
    index = pages.indexOf(page)
    assert index >= 0
    assert not pages.tabIcon(index).isNull(), (
        "the page carries a title and no mark")


def test_the_mark_is_the_one_the_registry_ships(host, qtbot):
    """Not a second picture chosen here, which would drift."""
    from spacr.qt import iconset
    from spacr.qt.screens.map_barcodes import show_as_page

    page = _page(qtbot, "agreement")
    show_as_page(page, host, "Annotator Agreement")
    index = host._fold_pages.indexOf(page)

    drawn = host._fold_pages.tabIcon(index).pixmap(32, 32).toImage()
    shipped = iconset.app_icon("agreement").pixmap(32, 32).toImage()
    assert drawn == shipped


def test_the_title_is_still_there(host, qtbot):
    """The mark is added to the caption, not swapped for it."""
    from spacr.qt.screens.map_barcodes import show_as_page

    page = _page(qtbot, "agreement")
    show_as_page(page, host, "Annotator Agreement")
    index = host._fold_pages.indexOf(page)

    assert host._fold_pages.tabText(index) == "Annotator Agreement"


def test_a_page_with_no_key_still_opens(host, qtbot):
    """A screen that never declared one is a page without a mark, not an
    exception -- the mark is decoration and the page is the point."""
    from spacr.qt.screens.map_barcodes import show_as_page

    page = QWidget()
    qtbot.addWidget(page)

    assert show_as_page(page, host, "Something") is page
    assert host._fold_pages.indexOf(page) >= 0


def test_a_key_the_registry_never_heard_of_still_opens(host, qtbot):
    from spacr.qt.screens.map_barcodes import show_as_page

    page = _page(qtbot, "not_a_module_at_all")

    assert show_as_page(page, host, "Mystery") is page
    assert host._fold_pages.indexOf(page) >= 0


def test_reopening_does_not_add_a_second_page(host, qtbot):
    from spacr.qt.screens.map_barcodes import show_as_page

    page = _page(qtbot, "agreement")
    show_as_page(page, host, "Annotator Agreement")
    show_as_page(page, host, "Annotator Agreement")

    assert host._fold_pages.count() == 1
    assert not host._fold_pages.tabIcon(0).isNull()
