"""The spaCR mark and wordmark sit on a panel, not on a hole in the page.

The masthead's labels were transparent and the frame behind them was
nothing at all, so what showed through was the blanket window fill: a
black band across the top of the Home screen, drawn behind the logo and
squared off against a page whose every other surface is a rounded
translucent panel.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget                    # noqa: E402


@pytest.fixture()
def home(qtbot):
    from spacr.qt.app import APPS, _icon_for_app
    from spacr.qt.widgets.home import HomePage

    page = HomePage(list(APPS), _icon_for_app)
    qtbot.addWidget(page)
    page.resize(1100, 760)
    page.show()
    return page


@pytest.fixture()
def hero(home):
    found = home.findChild(QWidget, "Hero")
    assert found is not None, "the masthead has no name to find it by"
    return found


def test_it_carries_a_surface_of_its_own(hero):
    assert "background-color" in hero.styleSheet()


def test_the_surface_is_translucent(hero):
    """Opaque would still be a box; it has to sit ON the page."""
    rule = hero.styleSheet()
    assert "rgba(" in rule, f"the masthead is opaque: {rule}"


def test_the_corners_are_rounded(hero):
    from spacr.qt.theme import RADIUS

    assert f"border-radius: {RADIUS['lg']}px" in hero.styleSheet()


def test_it_is_the_same_surface_the_rest_of_the_page_uses(hero):
    """Matched by construction, not by a second colour written out here.

    A masthead with its own hardcoded panel colour drifts the day the
    palette moves or the pane-opacity preference changes.
    """
    from spacr.qt.preferences import resolve_effective_theme
    from spacr.qt.theme import pane_surface

    expected = pane_surface("surface_alt", resolve_effective_theme())
    assert expected in hero.styleSheet()


def test_it_follows_the_theme_rather_than_resolving_its_own(hero):
    """It read a near-white panel onto a dark page by asking twice."""
    from spacr.qt.preferences import resolve_effective_theme
    from spacr.qt.theme import palette_for

    surface = palette_for(resolve_effective_theme())["surface_alt"]
    red, green, blue = (int(surface[i:i + 2], 16) for i in (1, 3, 5))
    assert f"rgba({red}, {green}, {blue}" in hero.styleSheet()


def test_the_type_is_not_flush_against_the_rounded_corner(hero):
    """A panel has an edge now, and type against it reads as clipped."""
    margins = hero.layout().contentsMargins()
    assert margins.left() > 0 and margins.top() > 0
    assert margins.right() > 0 and margins.bottom() > 0
