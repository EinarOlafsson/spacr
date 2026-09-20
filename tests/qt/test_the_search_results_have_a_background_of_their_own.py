"""The search results sit on a surface, not on whatever is behind them.

Reported by the maintainer on 2026-09-20: "the dropdowns from the search need
to have a background, now they overlap with the log and its text and the
entries are difficult to read."

The cause was not a wrong colour, it was NO RULE. ``_build`` sets
``autoFillBackground`` and ``WA_StyledBackground`` on the results frame, and
both of those paint nothing at all unless a stylesheet selector matches the
widget -- and no selector in ``theme.stylesheet`` named ``HelpSearchResults``,
``HelpSearchResultList`` or ``HelpSearchNote``. The frame drew nothing, so the
console underneath it drew through, and the result rows read as words mixed
into the log.

These tests read PIXELS off the rendered window rather than asserting that a
string appears in the stylesheet, because a selector that is present and does
not match is exactly the failure being fixed. The console is put directly
behind the popup first, so a transparent frame fails rather than passes on a
window that happens to be empty there.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor


def _window_pixel(window, point):
    """The colour the WINDOW actually renders at ``point``."""
    image = window.grab().toImage()
    return QColor(image.pixel(int(point.x()), int(point.y())))


@pytest.fixture
def window(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    return win


@pytest.fixture
def field_with_results(window, qtbot):
    """The real field, showing real rows, with its popup up."""
    from spacr.qt.help_index import HelpEntry
    from spacr.qt.help_search import field_of

    found = field_of(window)
    assert found is not None, "no search field was installed beside Help"
    found.set_index([
        HelpEntry(kind="module", title=f"Measure {n}", subtitle="a module",
                  payload={"app": "measure"})
        for n in range(8)
    ])
    found.type_and_search("Measure")
    qtbot.waitUntil(found.popup().isVisible, timeout=2000)
    return found


def test_the_results_frame_paints_the_themes_surface_and_not_nothing(
        window, field_with_results):
    """A pixel inside the popup is the theme's surface, opaque."""
    from spacr.qt.preferences import resolve_effective_theme
    from spacr.qt.theme import palette_for

    popup = field_with_results.popup()
    here = popup.mapTo(window, popup.rect().center())
    painted = _window_pixel(window, here)

    expected = QColor(palette_for(resolve_effective_theme())["surface_hi"])
    assert painted.alpha() == 255
    for got, want, band in ((painted.red(), expected.red(), "red"),
                            (painted.green(), expected.green(), "green"),
                            (painted.blue(), expected.blue(), "blue")):
        assert abs(got - want) <= 2, (band, painted.name(), expected.name())


def test_what_is_behind_the_popup_does_not_show_through_it(
        window, field_with_results, qtbot):
    """The same pixel reads differently with the popup up and down.

    The point of the report: the log was visible THROUGH the results. If the
    frame paints, hiding it has to change what that pixel shows.
    """
    popup = field_with_results.popup()
    here = popup.mapTo(window, popup.rect().center())
    with_popup = _window_pixel(window, here)

    field_with_results.hide_popup()
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)
    without_popup = _window_pixel(window, here)

    assert with_popup.name() != without_popup.name(), (
        "the popup renders the same pixels as the window behind it, which is "
        "what 'you can read the log through the results' looks like")
