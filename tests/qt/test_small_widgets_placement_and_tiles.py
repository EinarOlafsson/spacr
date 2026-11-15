"""Where the small chrome widgets land, and what a tile paints on itself.

The DNA-rain button and its settings popover, the hover tooltip and the
activity spinner all place themselves next to a widget they do not own; that
widget may have no layout, the wrong kind of layout, no screen under it, or a
C++ half that is already gone.  Each case states the exact position or the
exact refusal.  Tiles are checked by their initials and caption, and the
crop-tile chrome by its cover arithmetic and by the pixels it paints.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case builds its own widgets.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QPoint, QRect, QRectF, QSettings, QSize, Qt  # noqa: E402
from PySide6.QtGui import (QColor, QGuiApplication, QIcon, QImage,  # noqa: E402
                           QPainter, QPixmap)
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel,  # noqa: E402
                               QPushButton, QVBoxLayout, QWidget)

from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.widgets import hover_tooltip as ht  # noqa: E402
from spacr.qt.widgets import tile_chrome as tc  # noqa: E402
from spacr.qt.widgets.activity_spinner import (  # noqa: E402
    ActivitySpinner, attach_activity_spinner)
from spacr.qt.widgets.dna_rain import DnaRainSettingsBar, _place_beside  # noqa: E402
from spacr.qt.widgets.dna_rain_settings import (  # noqa: E402
    ANCHOR_GAP, DnaRainSettingsPopover)
from spacr.qt.widgets.hover_tooltip import HoverTooltip  # noqa: E402
from spacr.qt.widgets.tile import Tile  # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------
# dna_rain._place_beside
# --------------------------------------------------------------------------

def test_the_dna_button_is_placed_before_its_anchor_or_refused(qapp, subtests):
    """_place_beside inserts before the anchor, appends in a grid, refuses without a slot."""
    with subtests.test(msg="no_anchor"):
        assert _place_beside(QPushButton("DNA"), None) is False

    with subtests.test(msg="anchor_whose_parent_lays_nothing_out"):
        parent = QWidget()
        anchor = QPushButton("AI", parent)
        assert _place_beside(QPushButton("DNA"), anchor) is False

    with subtests.test(msg="anchor_not_in_its_parents_layout"):
        parent = QWidget()
        QVBoxLayout(parent)
        anchor = QPushButton("AI", parent)
        assert _place_beside(QPushButton("DNA"), anchor) is False

    with subtests.test(msg="grid_appends_the_button"):
        parent = QWidget()
        grid = QGridLayout(parent)
        anchor = QPushButton("AI")
        grid.addWidget(anchor, 0, 0)
        button = QPushButton("DNA")
        assert _place_beside(button, anchor) is True
        assert (grid.indexOf(anchor), grid.indexOf(button)) == (0, 1)

    with subtests.test(msg="box_row_puts_the_button_before_the_anchor"):
        parent = QWidget()
        row = QVBoxLayout(parent)
        anchor = QPushButton("AI")
        row.addWidget(anchor)
        button = QPushButton("DNA")
        assert _place_beside(button, anchor) is True
        assert (row.indexOf(button), row.indexOf(anchor)) == (0, 1)


# --------------------------------------------------------------------------
# dna_rain_settings.DnaRainSettingsPopover._position_near
# --------------------------------------------------------------------------

def _expected_popover_pos(popover, top_left, bottom_left, centre_x, screen):
    x = centre_x - popover.width() // 2
    y = top_left.y() - popover.height() - ANCHOR_GAP
    if screen is not None:
        area = screen.availableGeometry()
        x = min(max(area.left(), x), area.right() - popover.width() + 1)
        if y < area.top():
            y = bottom_left.y() + ANCHOR_GAP
        y = min(max(area.top(), y), area.bottom() - popover.height() + 1)
    return QPoint(x, y)


def _anchor_points(anchor):
    top_left = anchor.mapToGlobal(anchor.rect().topLeft())
    bottom_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
    return top_left, bottom_left, top_left.x() + anchor.width() // 2


def test_the_rain_settings_popover_is_placed_against_its_anchor(qtbot, monkeypatch, subtests):
    """_position_near sits above a live anchor, clamps to the screen, survives a dead anchor or no screen."""
    def popover_for(vertical=False):
        popover = DnaRainSettingsPopover(DnaRainSettingsBar(vertical=vertical))
        qtbot.addWidget(popover)
        popover.adjustSize()
        return popover

    def anchor_at(x, y, w, h):
        anchor = QPushButton("settings")
        qtbot.addWidget(anchor)
        anchor.setFixedSize(w, h)
        anchor.move(x, y)
        return anchor

    screen = QGuiApplication.primaryScreen()
    for case_id, geometry in [("live_anchor_at_origin", (0, 0, 80, 24)),
                              ("live_anchor_far_down", (400, 500, 80, 24)),
                              ("live_anchor_beyond_the_right_edge", (5000, 300, 60, 24))]:
        with subtests.test(msg=case_id):
            popover = popover_for(vertical=True)
            anchor = anchor_at(*geometry)
            expected = _expected_popover_pos(popover, *_anchor_points(anchor), screen)
            popover._position_near(anchor)
            assert popover.pos() == expected
            area = screen.availableGeometry()
            assert area.left() <= popover.x() <= area.right()

    with subtests.test(msg="near_and_far_anchors_give_different_places"):
        popover = popover_for()
        popover._position_near(anchor_at(0, 0, 80, 24))
        first = popover.pos()
        popover._position_near(anchor_at(400, 500, 80, 24))
        assert popover.pos() != first

    with subtests.test(msg="destroyed_anchor_is_an_anchor_at_the_origin"):
        popover = popover_for()
        dead = QPushButton("settings")
        shiboken6.delete(dead)
        popover._position_near(dead)
        assert popover.pos() == _expected_popover_pos(
            popover, QPoint(0, 0), QPoint(0, 0), 0, screen)

    with subtests.test(msg="no_screen_at_all_places_it_unclamped"):
        popover = popover_for(vertical=True)
        anchor = QWidget()
        qtbot.addWidget(anchor)
        anchor.resize(60, 24)
        with monkeypatch.context() as mp:
            mp.setattr(QGuiApplication, "screenAt", staticmethod(lambda point: None))
            mp.setattr(QGuiApplication, "primaryScreen", staticmethod(lambda: None))
            top_left = anchor.mapToGlobal(anchor.rect().topLeft())
            popover._position_near(anchor)
        assert popover.x() == top_left.x() + anchor.width() // 2 - popover.width() // 2
        assert popover.y() == top_left.y() - popover.height() - ANCHOR_GAP


# --------------------------------------------------------------------------
# hover_tooltip.HoverTooltip placement and anchor claiming
# --------------------------------------------------------------------------

@pytest.fixture
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in os.path.normpath(store.fileName())
    return store


def test_the_hover_tooltip_docks_under_its_anchor_and_claims_it(
        _isolated_qsettings, qtbot, monkeypatch, subtests):
    """_position_under clamps or flips above; _claim_anchor hides the native tooltip only for a live anchor."""
    def fresh():
        popup = HoverTooltip()
        qtbot.addWidget(popup)
        return popup

    geometry = QGuiApplication.primaryScreen().availableGeometry()

    with subtests.test(msg="no_anchor_docks_under_the_origin"):
        tooltip = fresh()
        tooltip._position_under(None)
        assert tooltip.pos() == QPoint(
            min(max(geometry.left(), 0), geometry.right() - tooltip.width()), 4)

    with subtests.test(msg="no_screen_moves_to_the_anchor_point"):
        tooltip = fresh()
        tooltip.move(37, 41)

        class _NoScreens:
            @staticmethod
            def screenAt(_point):
                return None

            @staticmethod
            def primaryScreen():
                return None

        with monkeypatch.context() as mp:
            mp.setattr(ht, "QGuiApplication", _NoScreens)
            tooltip._position_under(None)
        assert tooltip.pos() == QPoint(0, 0)

    with subtests.test(msg="no_room_below_flips_above_the_anchor"):
        tooltip = fresh()

        class _DiesAfterTheFirstAsk:
            def __init__(self):
                self.asks = 0

            def rect(self):
                return QRect(0, 0, 40, 20)

            def mapToGlobal(self, _point):
                self.asks += 1
                if self.asks > 1:
                    raise RuntimeError("Internal C++ object already deleted")
                return QPoint(geometry.left(), geometry.bottom())

        anchor = _DiesAfterTheFirstAsk()
        tooltip.resize(200, 120)
        tooltip._position_under(anchor)
        assert anchor.asks == 2
        assert tooltip.pos() == QPoint(geometry.left(), geometry.bottom() - 120 - 4)

    from spacr.qt.tooltip_policy import OPT_OUT_PROPERTY

    def live():
        label = QLabel("Cell diameter")
        qtbot.addWidget(label)
        return label

    def dead():
        label = live()
        shiboken6.delete(label)
        return label

    for case_id, make_anchor, expected_hides in [
            ("live_anchor_hides_the_native_tooltip", live, [True]),
            ("no_anchor_takes_no_duty", lambda: None, []),
            ("destroyed_anchor_takes_no_duty", dead, [])]:
        with subtests.test(msg=case_id):
            tooltip = fresh()
            hidden = []

            class _ToolTip:
                @staticmethod
                def hideText():
                    hidden.append(True)

            anchor = make_anchor()
            with monkeypatch.context() as mp:
                mp.setattr(ht, "QToolTip", _ToolTip)
                tooltip._claim_anchor(anchor)
            assert hidden == expected_hides
            if expected_hides:
                assert anchor.property(OPT_OUT_PROPERTY) is True


# --------------------------------------------------------------------------
# activity_spinner.attach_activity_spinner
# --------------------------------------------------------------------------

def test_the_activity_spinner_attaches_once_beside_the_clear_button(qtbot, subtests):
    """attach_activity_spinner reuses a live spinner, replaces a dead one, refuses rows it cannot insert into."""
    def row_with_clear_button():
        host = QWidget()
        qtbot.addWidget(host)
        layout = QHBoxLayout(host)
        button = QPushButton("Clear console", host)
        host._btn_clear = button
        layout.addWidget(button)
        return host, layout, button

    with subtests.test(msg="attaching_twice_returns_the_same_spinner"):
        host, layout, button = row_with_clear_button()
        first = attach_activity_spinner(host)
        assert isinstance(first, ActivitySpinner)
        assert attach_activity_spinner(host) is first
        assert len(host.findChildren(ActivitySpinner)) == 1

    with subtests.test(msg="a_descendant_finds_its_ancestors_spinner"):
        host, _layout, _button = row_with_clear_button()
        spinner = attach_activity_spinner(host)
        child = QWidget()
        qtbot.addWidget(child)
        child.setParent(host)
        del host._btn_clear
        assert attach_activity_spinner(child) is spinner

    with subtests.test(msg="a_spinner_whose_c_half_is_gone_is_replaced"):
        host, _layout, _button = row_with_clear_button()
        stale = attach_activity_spinner(host)
        shiboken6.delete(stale)
        fresh = attach_activity_spinner(host)
        assert isinstance(fresh, ActivitySpinner)
        assert fresh is not stale

    with subtests.test(msg="a_button_in_no_layout_gets_none"):
        host = QWidget()
        qtbot.addWidget(host)
        row = QWidget(host)
        host._btn_clear = QPushButton("Clear console", row)
        assert attach_activity_spinner(host) is None

    with subtests.test(msg="a_grid_row_gets_none_and_no_stray_widget"):
        host = QWidget()
        qtbot.addWidget(host)
        row = QWidget(host)
        grid = QGridLayout(row)
        button = QPushButton("Clear console", row)
        grid.addWidget(button, 0, 0)
        host._btn_clear = button
        assert attach_activity_spinner(host) is None
        assert row.findChildren(ActivitySpinner) == []


# --------------------------------------------------------------------------
# tile.Tile
# --------------------------------------------------------------------------

@contextmanager
def _at_design_scale():
    """Measure designed geometry without inheriting or leaking saved zoom."""
    settings = prefs._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    try:
        prefs.set_font_scale(1.0)
        prefs.set_gui_scale(1.0)
        yield
    finally:
        for key in keys:
            if key in previous:
                settings.setValue(key, previous[key])
            else:
                settings.remove(key)


def _icon(colour=Qt.red, side=32):
    pixmap = QPixmap(side, side)
    pixmap.fill(colour)
    return QIcon(pixmap)


def test_a_tile_shows_its_icon_or_the_initials_of_its_name(qtbot, subtests):
    """Tile shows the icon with no text, else one initial per word (first characters if no words)."""
    with _at_design_scale():
        with subtests.test(msg="icon_tile_shows_the_icon_and_no_initials"):
            tile = Tile("Annotator Agreement", icon=_icon(), icon_size=48, tile_size=110)
            qtbot.addWidget(tile)
            assert tile._button.icon().isNull() is False
            assert tile._button.text() == ""
            assert tile._button.iconSize() == QSize(48, 48)
            assert tile.text == "Annotator Agreement"

        for name, initials in [("Annotator Agreement", "AA"), ("Curate", "C"), ("3d", "3")]:
            with subtests.test(msg=f"initials_of_{name}"):
                tile = Tile(name)
                qtbot.addWidget(tile)
                assert tile._button.icon().isNull() is True
                assert tile._button.text() == initials
                assert tile._caption.text() == name
                assert tile._button.toolTip() == name
                assert tile.text == name


# --------------------------------------------------------------------------
# tile_chrome.cover_rect and paint_tile
# --------------------------------------------------------------------------

def _solid(colour, w=40, h=40):
    pixmap = QPixmap(w, h)
    pixmap.fill(QColor(colour))
    return pixmap


def test_cover_rect_scales_a_crop_to_fill_its_box(qapp, subtests):
    """cover_rect scales to cover the box, centred, and leaves the box alone for an empty crop."""
    cases = [
        ("wide_crop_overflows_the_sides", (200, 100), (0, 0, 100, 100), (-50, 0, 200, 100)),
        ("tall_crop_overflows_top_and_bottom", (50, 200), (0, 0, 100, 100), (0, -150, 100, 400)),
        ("matching_crop_is_unchanged", (40, 40), (4, 4, 40, 40), (4, 4, 40, 40)),
        ("null_pixmap_leaves_the_box", None, (0, 0, 20, 20), (0, 0, 20, 20)),
    ]
    for case_id, size, box, expected in cases:
        with subtests.test(msg=case_id):
            pixmap = QPixmap() if size is None else _solid("red", *size)
            box_rect = QRectF(*box)
            out = tc.cover_rect(pixmap, box_rect)
            assert (out.x(), out.y(), out.width(), out.height()) == pytest.approx(expected)
            assert out.center() == box_rect.center()


def test_paint_tile_clips_the_crop_and_draws_its_rings(qapp, subtests):
    """paint_tile clips the crop to a rounded corner, skips empty or too-small crops, rings only as asked."""
    def paint(**kwargs):
        image = QImage(60, 60, QImage.Format_ARGB32)
        image.fill(QColor("white"))
        painter = QPainter(image)
        try:
            tc.paint_tile(painter, **kwargs)
        finally:
            painter.end()
        return image

    white = QColor("white")

    with subtests.test(msg="crop_is_clipped_to_a_rounded_corner"):
        image = paint(w=60, h=60, pixmap=_solid("#0000ff", 60, 60), border_colour="")
        middle = image.pixelColor(30, 30)
        assert middle.blue() > 200 and middle.red() < 60
        assert image.pixelColor(tc.TILE_INSET, tc.TILE_INSET) == white

    with subtests.test(msg="empty_current_slot_keeps_its_rings"):
        image = paint(w=60, h=60, pixmap=None, border_colour="#00ff00",
                      ring_colour="#ffffff", current=True)
        assert image.pixelColor(30, 30) == white
        state = image.pixelColor(tc.HOVER_RING_WIDTH + 1, 30)
        assert state.green() > 150 and state.red() < 120

    with subtests.test(msg="null_pixmap_is_an_empty_slot"):
        image = paint(w=60, h=60, pixmap=QPixmap(), border_colour="#00ff00")
        assert image.pixelColor(30, 30) == white

    with subtests.test(msg="tile_too_small_for_its_chrome_draws_no_crop"):
        image = paint(w=6, h=6, pixmap=_solid("#0000ff"), border_colour="#00ff00")
        assert image.pixelColor(3, 3) == white

    with subtests.test(msg="current_ring_sits_outside_the_state_ring"):
        image = paint(w=60, h=60, pixmap=_solid("#000000", 60, 60),
                      border_colour="#00ff00", ring_colour="#ff0000", current=True)
        outer = image.pixelColor(1, 30)
        inner = image.pixelColor(tc.HOVER_RING_WIDTH + 1, 30)
        assert outer.red() > 150 and outer.green() < 120
        assert inner.green() > 150 and inner.red() < 120

    with subtests.test(msg="not_current_gets_no_outer_ring"):
        image = paint(w=60, h=60, pixmap=_solid("#000000", 60, 60),
                      border_colour="#00ff00", ring_colour="#ff0000", current=False)
        assert image.pixelColor(1, 30) == white
