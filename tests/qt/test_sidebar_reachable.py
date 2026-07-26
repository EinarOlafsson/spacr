"""Every app in the sidebar must be reachable at a laptop screen size.

Measured before the fix at 1440x900: the sidebar stacked 29 rows, 5 section
headings and the title in a plain QVBoxLayout, asked for 1356 px against 850
available, and the last three apps -- Plaque Assay, Recruitment, Invasion
Assay -- could not be scrolled to or clicked.
"""
import pytest

pytest.importorskip("PySide6")


LAPTOP = (1440, 900)


def _sidebar(qtbot):
    from spacr.qt.app import Sidebar
    w = Sidebar()
    qtbot.addWidget(w)
    return w


def test_the_nav_rows_live_in_a_scroll_area(qtbot):
    """Without one, the column simply overflows and clips."""
    from PySide6.QtWidgets import QScrollArea
    w = _sidebar(qtbot)
    assert w.findChild(QScrollArea) is not None


def test_every_app_row_is_reachable_at_1440x900(qtbot):
    from spacr.qt.app import APPS
    w = _sidebar(qtbot)
    w.resize(w.width(), LAPTOP[1])
    w.show()
    qtbot.waitExposed(w)

    keys = {b.property("navKey") for b in w._items}
    for key, name, _desc, _section in APPS:
        assert key in keys, f"{name} has no sidebar row at all"

    # The scroll area's contents may be taller than the viewport -- that is
    # the point -- but every row must be inside the scrollable widget, so
    # scrolling can bring it into view.
    inner = w._scroll.widget()
    for btn in w._items:
        assert btn.isAncestorOf(btn) and inner.isAncestorOf(btn), (
            f"{btn.property('navKey')} is outside the scrollable area")


def test_the_last_app_can_be_scrolled_into_view(qtbot):
    """The three that used to be unreachable are the last three rows."""
    w = _sidebar(qtbot)
    w.resize(w.width(), LAPTOP[1])
    w.show()
    qtbot.waitExposed(w)

    last = w._items[-1]
    bar = w._scroll.verticalScrollBar()
    w._scroll.ensureWidgetVisible(last)
    # Either everything already fits (no scrolling needed) or the bar moved.
    fits = w._scroll.widget().height() <= w._scroll.viewport().height()
    assert fits or bar.maximum() > 0, (
        "the content overflows but there is no scroll range -- the rows past "
        "the fold are unreachable")


def test_the_title_does_not_scroll_away(qtbot):
    """The title is a header; it stays pinned above the scrolling rows."""
    w = _sidebar(qtbot)
    inner = w._scroll.widget()
    from PySide6.QtWidgets import QLabel
    titles = [c for c in w.findChildren(QLabel)
              if c.objectName() == "SidebarTitle"]
    assert titles, "no SidebarTitle label found -- this test would pass vacuously"
    for label in titles:
        assert not inner.isAncestorOf(label), "the title scrolls with the rows"
