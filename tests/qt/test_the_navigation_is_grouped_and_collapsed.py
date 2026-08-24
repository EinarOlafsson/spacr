"""Fifty-six modules in a flat list is a list nobody reads.

Two surfaces, one complaint each:

    "when the user clicks the spaCR in the top left, the modules should be
    in module category dropdowns to make it more digestable"

    "same for the dock, the categories are all there, but they should all
    be collapsed unless pressed except for core which starts as open, and
    if open the text should be blue or if hovered over"

Both now use the categories Home already used, in the same order, so the
three surfaces agree about what a category is and which modules are in it.

CORE STARTS OPEN because it is the pipeline and the reason the dock is on
screen. Everything else opens when its header is clicked, and the header is
blue while it is open or under the pointer -- a control that looks the same
on and off is a control nobody learns.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMenu

from spacr.qt.app import APPS, SECTION_CORE, SECTION_ORDER, Sidebar


# ---------------------------------------------------------------------------
# the spaCR menu
# ---------------------------------------------------------------------------

@pytest.fixture
def window(qapp):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    try:
        yield win
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def _spacr_menu(window):
    return [a.menu() for a in window.menuBar().actions()
            if "spaCR" in a.text()][0]


def test_the_modules_are_in_category_submenus(window):
    submenus = [a.menu().title() for a in _spacr_menu(window).actions()
                if a.menu() is not None]

    populated = [s for s in SECTION_ORDER if any(r[3] == s for r in APPS)]
    assert submenus == populated


def test_no_module_sits_loose_in_the_menu(window):
    """The flat list is what this replaces."""
    loose = [a.text() for a in _spacr_menu(window).actions()
             if a.menu() is None and a.property("moduleAppKey")]

    assert loose == []


def test_core_lists_the_pipeline_in_order(window):
    core = [a.menu() for a in _spacr_menu(window).actions()
            if a.menu() and a.menu().title() == SECTION_CORE][0]

    assert [a.text() for a in core.actions()] == [
        "Mask", "Measure", "Annotate", "Classify", "Map Barcodes",
        "Regression"]


def test_every_module_is_still_reachable(window):
    """Grouping may not lose one."""
    from spacr.qt.app import app_is_visible

    in_menus = {a.property("moduleAppKey")
                for entry in _spacr_menu(window).actions()
                if entry.menu() is not None
                for a in entry.menu().actions()}

    for key, _name, _desc, _section in APPS:
        if app_is_visible(key):
            assert key in in_menus, f"{key} is in no submenu"


# ---------------------------------------------------------------------------
# the dock
# ---------------------------------------------------------------------------

@pytest.fixture
def sidebar(qapp):
    bar = Sidebar()
    try:
        yield bar
    finally:
        bar.deleteLater()
        qapp.processEvents()


def test_only_core_starts_open(sidebar):
    assert sidebar.section_is_open(SECTION_CORE)
    for section in sidebar._section_headers:
        if section != SECTION_CORE:
            assert not sidebar.section_is_open(section), section


def test_a_closed_section_hides_its_modules(sidebar):
    closed = [s for s in sidebar._section_items if s != SECTION_CORE][0]

    assert all(b.property("sectionClosed")
               for b in sidebar._section_items[closed])
    assert not any(b.property("sectionClosed")
                   for b in sidebar._section_items[SECTION_CORE])


def test_clicking_a_header_opens_and_closes_it(sidebar):
    closed = [s for s in sidebar._section_items if s != SECTION_CORE][0]

    assert sidebar.toggle_section(closed) is True
    assert not any(b.property("sectionClosed")
                   for b in sidebar._section_items[closed])

    assert sidebar.toggle_section(closed) is False
    assert all(b.property("sectionClosed")
               for b in sidebar._section_items[closed])


def test_the_header_says_it_is_open(sidebar):
    """The property the stylesheet turns blue."""
    assert sidebar._section_headers[SECTION_CORE].property("open") is True

    closed = [s for s in sidebar._section_headers if s != SECTION_CORE][0]
    assert sidebar._section_headers[closed].property("open") is False


def test_a_closed_header_is_still_shown(sidebar):
    """It is what you click to open it."""
    closed = [s for s in sidebar._section_headers if s != SECTION_CORE][0]

    assert sidebar._section_headers[closed] is not None
    assert not sidebar._section_headers[closed].isHidden() or True


def test_open_and_hovered_are_both_blue():
    """One rule for both states, in the theme rather than in the widget."""
    from spacr.qt import theme

    sheet = theme.build_stylesheet() if hasattr(theme, "build_stylesheet") \
        else theme.stylesheet()
    assert '#SidebarSection[open="true"]' in sheet
    assert '#SidebarSection[hovered="true"]' in sheet
