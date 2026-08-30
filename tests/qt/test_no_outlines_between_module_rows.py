"""The Home grid has spacing, not decorative outlines.

Instruction 311 distinguishes the edge of an interactive state from the rim
of a transparent container.  These checks therefore read the pixels the user
sees: every resting module edge blends into its own surface, the four-pixel
row gap remains intact, focus paints the accent, and the selected-tab edge
moves with the selection.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from spacr.qt import preferences, theme
from spacr.qt.theme import SPACING
from spacr.qt.widgets.home import (
    AppTile,
    HomePage,
    RecentRunsPanel,
    SystemPanel,
    TotalsPanel,
)

PALETTES = ("dark", "light", "space", "cell", "glass")
_APPS = [(f"app_{index}", f"Module {index}", "Description", "Core")
         for index in range(8)]
_LONELY_CORNER = QPoint(4000, 2400)


def _rendered_home(qtbot, monkeypatch, palette: str) -> HomePage:
    """A quiet, real Home page with enough launchers to make three rows."""
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: palette)
    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: False)
    monkeypatch.setattr(preferences, "get_font_scale", lambda: 1.0)
    monkeypatch.setattr(preferences, "effective_pane_alpha", lambda: 0.6)
    monkeypatch.setattr(TotalsPanel, "read", lambda _self: {})
    monkeypatch.setattr(RecentRunsPanel, "read", lambda _self: [])
    monkeypatch.setattr(SystemPanel, "gpu_util", staticmethod(lambda: "n/a"))
    monkeypatch.setattr(SystemPanel, "gpu_vram", staticmethod(lambda: "n/a"))
    monkeypatch.setattr(SystemPanel, "disk_used", staticmethod(lambda: "n/a"))

    page = HomePage(_APPS, lambda _key: None)
    qtbot.addWidget(page)
    page.setStyleSheet(theme.stylesheet(
        palette, surface_opacity=0.6, load_widget_registrars=False))
    page.resize(1000, 760)
    # Keep the process-global synthetic pointer away: hover legitimately
    # paints a maturity border and is not the resting state under test.
    page.move(_LONELY_CORNER)
    page.show()
    qtbot.waitExposed(page)
    QApplication.processEvents()
    return page


def _visible_home_tiles(page: HomePage) -> list[AppTile]:
    return [tile for tile in page._tabs.widget(0).findChildren(AppTile)
            if tile.isVisible()]


def _vertical_pair(page: HomePage):
    """Nearest two tiles in the first column, in page coordinates."""
    placed = [(tile.mapTo(page, QPoint(0, 0)), tile)
              for tile in _visible_home_tiles(page)]
    placed.sort(key=lambda item: (item[0].y(), item[0].x()))
    for upper_origin, upper in placed:
        below = [(origin, tile) for origin, tile in placed
                 if origin.x() == upper_origin.x()
                 and origin.y() > upper_origin.y()]
        if below:
            lower_origin, lower = min(below, key=lambda item: item[0].y())
            return upper_origin, upper, lower_origin, lower
    raise AssertionError("Home did not lay out two module rows")


def _row_colour(image: QImage, y: int, left: int, right: int) -> QColor:
    """One colour for a text-free horizontal slice through a tile."""
    colours = {image.pixelColor(x, y).rgba()
               for x in range(left, right + 1)}
    assert len(colours) == 1, f"pixel row {y} is not uniform: {colours}"
    return QColor.fromRgba(next(iter(colours)))


def _channel_jump(one: QColor, two: QColor) -> int:
    return max(abs(one.red() - two.red()),
               abs(one.green() - two.green()),
               abs(one.blue() - two.blue()),
               abs(one.alpha() - two.alpha()))


@pytest.mark.parametrize("palette", PALETTES)
def test_home_row_boundaries_have_no_one_pixel_outline(
        qtbot, monkeypatch, palette):
    """Read a real two-row boundary in every flat and glass paint path."""
    page = _rendered_home(qtbot, monkeypatch, palette)
    upper_at, upper, lower_at, lower = _vertical_pair(page)

    upper_bottom = upper_at.y() + upper.height() - 1
    lower_top = lower_at.y()
    gap = lower_top - upper_bottom - 1
    assert gap == SPACING["xs"] == 4
    assert page._grids[0][1].verticalSpacing() == SPACING["xs"]

    image = page.grab().toImage()
    left = max(upper_at.x(), lower_at.x()) + 20
    right = min(upper_at.x() + upper.width(),
                lower_at.x() + lower.width()) - 21

    # At the centre of a rounded tile, a decorative border is a one-pixel
    # colour jump. Glass keeps its intentional vertical material gradient,
    # so compare each edge with the immediately adjacent interior row rather
    # than demanding the two different ends of that gradient be identical.
    upper_edge = _row_colour(image, upper_bottom, left, right)
    upper_inside = _row_colour(image, upper_bottom - 1, left, right)
    lower_edge = _row_colour(image, lower_top, left, right)
    lower_inside = _row_colour(image, lower_top + 1, left, right)
    assert _channel_jump(upper_edge, upper_inside) <= 4
    assert _channel_jump(lower_edge, lower_inside) <= 4

    gap_colours = [_row_colour(image, y, left, right).rgba()
                   for y in range(upper_bottom + 1, lower_top)]
    assert len(set(gap_colours)) == 1, (
        f"{palette}: a hairline is still painted in the module-row gap")


@pytest.mark.parametrize("palette", PALETTES)
def test_keyboard_focus_still_draws_a_meaningful_tile_ring(
        qtbot, monkeypatch, palette):
    page = _rendered_home(qtbot, monkeypatch, palette)
    tile = _visible_home_tiles(page)[0]
    page._tabs.tabBar().setFocus(Qt.OtherFocusReason)
    QApplication.processEvents()
    x = tile.width() // 2
    resting = tile.grab().toImage().pixelColor(x, 0)

    tile.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    assert tile.hasFocus()
    focused = tile.grab().toImage().pixelColor(x, 0)
    assert focused == QColor(theme.palette_for(palette)["accent"])
    assert focused != resting


@pytest.mark.parametrize("palette", PALETTES)
def test_the_transparent_home_pane_has_no_decorative_rim(
        qtbot, monkeypatch, palette):
    page = _rendered_home(qtbot, monkeypatch, palette)
    tabs = page._tabs
    image = tabs.grab().toImage()
    y = tabs.tabBar().height() + 50

    # Grabbed by itself, the transparent pane is alpha zero. A frame rule
    # makes exactly its first column opaque (or translucent on Glass).
    assert image.pixelColor(0, y) == image.pixelColor(1, y)
    assert image.pixelColor(0, y).alpha() == 0


@pytest.mark.parametrize("palette", PALETTES)
def test_the_selected_tab_indicator_moves_with_the_selected_tab(
        qtbot, monkeypatch, palette):
    page = _rendered_home(qtbot, monkeypatch, palette)
    bar = page._tabs.tabBar()

    def top_edge_colours():
        image = bar.grab().toImage()
        return [image.pixelColor(bar.tabRect(index).center().x(),
                                 bar.tabRect(index).top())
                for index in range(2)]

    home_selected = top_edge_colours()
    page._tabs.setCurrentIndex(1)
    QApplication.processEvents()
    category_selected = top_edge_colours()

    assert home_selected[0] != home_selected[1]
    assert home_selected == list(reversed(category_selected))
