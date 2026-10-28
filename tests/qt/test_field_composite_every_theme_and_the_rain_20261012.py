"""The composited field frame equals the classic paint in every theme.

Three faults found after the card, chrome and switch compositing of
2026-10-12, both fixed here:

- Glass: a card body is a vertical gradient. Qt samples a gradient from its
  device position, and the composite was painted with the field at the
  image's origin, while the window paints the field 33 px down: about
  10,000 pixels on Mask differed by one level. The composite is now painted
  at the field's place in its window (``_CardPlan.origin``), and a field
  that moves in its window falls back to the classic paint until it is
  replanned.
- Map Barcodes: the DNA rain covers the field and paints its whole area,
  yet the cards above it were composited into the field under the rain, so
  every card background, field chrome and switch track vanished a few
  ticks after each settle (and came back while the field was hidden and
  shown again). A widget that paints its whole area now hides the field
  from the plan, and anything painted classically keeps what overlaps it
  after it classic. The rain also claims ``WA_OpaquePaintEvent`` again
  after the stylesheet's polish, as the field does.
- Spaceout: field chrome and switches are painted from a palette the
  spaceout drift re-hues as the clock runs, so a composited picture of them
  went stale (216,509 pixels on Mask under Fungal growth). Under the
  spaceout dressing only card backgrounds are composited.

Checked on small scenes, on the real window in every theme the Theme
control offers (Home and Mask at pane opacity 0, 50 and 100 %), and on Map
Barcodes under the rain. A theme whose field has no shading thread (the
starfield of Cirrus) never composites and keeps the classic paint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QRect, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QFrame, QWidget  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.theme import glass_material  # noqa: E402
from spacr.qt.widgets import ambient  # noqa: E402
from spacr.qt.widgets.dna_rain import DnaRainWidget  # noqa: E402
from spacr.qt.widgets.toggle import Toggle  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _cards_composited(monkeypatch):
    """Card compositing on: these tests are about it.

    Off by default since 2026-10-12 (``ambient._composite_cards_on``): on
    the workstation's GNOME capture host (2560x1440, real display) it cost
    8-10 painted fps, Home 48-50 against 57-58 and Mask 41-43 against
    51-52. The machinery stays, behind ``SPACR_COMPOSITE_CARDS=1``.
    """
    monkeypatch.setenv("SPACR_COMPOSITE_CARDS", "1")

THEMES = (
    "dark", "light", "glass", "high_contrast", "system",
    "cell:microtubules", "cell:filopodia",
    "lantern", "halcyon", "solstice", "undertow", "meridian", "cirrus",
    "nocturne", "aphelion", "pulsar", "vesper",
    "data_art_impulse_lens", "data_art_genetic_advection",
    "data_art_point_atlas",
)

SHEET = """
QWidget {{ background: transparent; }}
QFrame#Card {{ background: {glass}; border: 1px solid rgba(255, 255, 255, 70);
              border-radius: 14px; }}
QFrame#Hover {{ background: rgba(1, 2, 3, 100); }}
QFrame#Hover:hover {{ background: rgba(200, 0, 0, 100); }}
QFrame#Bar {{ background: #101820; }}
""".format(glass=glass_material("#2a3340", 0.55))


def _card(parent, rect, name="Card"):
    frame = QFrame(parent)
    frame.setObjectName(name)
    frame.setGeometry(rect)
    return frame


def _grab(widget):
    return widget.grab().toImage().convertToFormat(QImage.Format_RGB32)


@pytest.fixture
def lowered(qtbot):
    """A field 33 px down its window, as under the main window's menu bar,
    with a glass card over it."""
    window = QWidget()
    window.setStyleSheet(SHEET)
    window.resize(480, 400)
    bar = _card(window, QRect(0, 0, 480, 33), "Bar")
    bar.setAutoFillBackground(True)
    shell = QWidget(window)
    shell.setGeometry(0, 33, 480, 367)
    host = QWidget(shell)
    host.setProperty("spacrIndependentBackdrop", True)
    host.setGeometry(0, 0, 480, 367)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    backdrop.lower()
    card = _card(host, QRect(17, 21, 300, 233))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    yield window, shell, backdrop, card
    backdrop._card_tints.invalidate()


def _settle(qtbot, backdrop):
    tints = backdrop._card_tints

    def ready():
        tints.tick(backdrop._producer_box[0])
        return tints.active

    qtbot.waitUntil(ready, timeout=10000)
    return tints


def test_a_gradient_card_composites_exactly_where_the_field_sits(lowered,
                                                                 qtbot):
    window, shell, backdrop, card = lowered
    tints = _settle(qtbot, backdrop)
    assert tints._cards == [card]
    assert tints.plan.origin == QPoint(0, 33)
    assert tints.plan.base.size() == window.size()
    backdrop._producer_box[0].stop()
    composited = _grab(window)
    assert backdrop._last_composed[0] is tints.plan
    tints.invalidate()
    assert composited == _grab(window)


def test_a_field_moved_in_its_window_falls_back_until_replanned(lowered,
                                                                qtbot):
    window, shell, backdrop, card = lowered
    tints = _settle(qtbot, backdrop)
    plan = tints.plan
    backdrop._producer_box[0].stop()
    _grab(window)
    assert backdrop._last_composed[0] is plan
    shell.move(0, 40)
    moved = _grab(window)
    assert tints.active and backdrop._last_composed[0] is plan
    tints.enabled = False
    tints.invalidate()
    assert moved == _grab(window)
    tints.enabled = True
    backdrop.stop()
    backdrop.start()
    tints = _settle(qtbot, backdrop)
    assert tints.plan.origin == QPoint(0, 40)
    tints.plan = tints.plan._replace(origin=QPoint(0, 7))
    tints.tick(backdrop._producer_box[0])
    assert tints.plan is None and not tints.active
    tints = _settle(qtbot, backdrop)
    assert tints.plan.origin == QPoint(0, 40)


@pytest.fixture
def rainy(qtbot):
    """A field under a DNA rain, one card painted before the rain and one
    after it."""
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.setStyleSheet(SHEET)
    host.resize(460, 340)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    under = _card(host, QRect(20, 20, 200, 150))
    rain = DnaRainWidget(host, seed=3)
    rain.setGeometry(0, 0, 460, 250)
    over = _card(host, QRect(30, 180, 300, 140))
    clear = _card(host, QRect(340, 260, 110, 75))
    rain.lower()
    under.lower()
    backdrop.lower()
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    rain.stop()
    yield host, backdrop, rain, under, over, clear
    backdrop._card_tints.invalidate()


def test_cards_over_the_rain_keep_their_own_backgrounds(rainy, qtbot):
    host, backdrop, rain, under, over, clear = rainy
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)
    tints = _settle(qtbot, backdrop)
    assert tints._cards == [under, clear]
    assert not over.testAttribute(Qt.WA_NoSystemBackground)
    assert rain in tints._watched
    backdrop._producer_box[0].stop()
    rain.stop()
    composited = _grab(host)
    tints.enabled = False
    tints.invalidate()
    classic = _grab(host)
    assert composited == classic
    backdrop.hide()
    backdrop.show()
    rain.stop()
    under_the_rain = rain.geometry()
    assert (_grab(host).copy(under_the_rain)
            == classic.copy(under_the_rain))


def test_the_rain_counts_as_covering_even_when_qt_thinks_it_translucent(
        rainy, qtbot):
    host, backdrop, rain, under, over, clear = rainy
    rain.setAttribute(Qt.WA_OpaquePaintEvent, False)
    assert ambient._CardTints._paints_over(rain)
    tints = _settle(qtbot, backdrop)
    assert over not in tints._cards and under in tints._cards


def test_showing_a_covering_widget_falls_back(rainy, qtbot):
    host, backdrop, rain, under, over, clear = rainy
    tints = _settle(qtbot, backdrop)
    rain.hide()
    assert not tints.active
    tints = _settle(qtbot, backdrop)
    assert over in tints._cards
    other = DnaRainWidget(host, seed=4)
    other.setGeometry(0, 0, 50, 50)
    other.show()
    assert not tints.active


def test_what_overlaps_a_classic_background_stays_classic(qtbot):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.setStyleSheet(SHEET)
    host.resize(460, 340)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    backdrop.lower()
    hover = _card(host, QRect(10, 10, 200, 150), "Hover")
    covered = _card(host, QRect(100, 60, 200, 150))
    inside = _card(covered, QRect(10, 10, 120, 100))
    switch = Toggle("", host)
    switch.setGeometry(190, 140, 40, 22)
    free = _card(host, QRect(320, 200, 120, 120))
    shown = Toggle("", host)
    shown.setGeometry(330, 20, 40, 22)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    try:
        tints = _settle(qtbot, backdrop)
        assert tints._cards == [free]
        assert switch not in tints._parts and shown in tints._parts
        assert hover in tints._watched and inside in tints._watched
        assert not covered.testAttribute(Qt.WA_NoSystemBackground)
        backdrop._producer_box[0].stop()
        composited = _grab(host)
        tints.invalidate()
        assert composited == _grab(host)
    finally:
        backdrop._card_tints.invalidate()


def test_a_widget_that_says_it_covers_its_area_paints_over(qtbot):
    covering = QWidget()
    qtbot.addWidget(covering)
    covering.setAutoFillBackground(False)
    covering.setAttribute(Qt.WA_OpaquePaintEvent, False)
    covering._spacr_covers_its_area = True
    assert ambient._CardTints._paints_over(covering)
    covering._spacr_covers_its_area = False
    assert not ambient._CardTints._paints_over(covering)


def _translucent_pixmap():
    image = QImage(64, 48, QImage.Format_ARGB32)
    image.fill(QColor(30, 60, 90, 255))
    image.setPixelColor(3, 5, QColor(30, 60, 90, 10))
    return QPixmap.fromImage(image, Qt.NoOpaqueDetection)


def test_the_rain_claims_opaque_painting_after_polish(qtbot):
    host = QWidget()
    host.setStyleSheet("QWidget { background: transparent; }")
    host.resize(200, 150)
    rain = DnaRainWidget(host, seed=1)
    rain.follow_parent()
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.setAttribute(Qt.WA_OpaquePaintEvent, False)
    host.setStyleSheet("QWidget { background: transparent; color: red; }")
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.setAttribute(Qt.WA_OpaquePaintEvent, False)
    rain.setEnabled(False)
    assert not rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.hide()
    rain.show()
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.set_backdrop(_translucent_pixmap())
    assert not rain.testAttribute(Qt.WA_OpaquePaintEvent)
    host.setStyleSheet("QWidget { background: transparent; }")
    assert not rain.testAttribute(Qt.WA_OpaquePaintEvent)
    opaque = QImage(64, 48, QImage.Format_RGB32)
    opaque.fill(QColor(1, 2, 3))
    rain.set_backdrop(opaque)
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.set_backdrop(None)
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)


def test_the_theme_list_is_the_theme_control(qapp):
    assert tuple(token for _label, token
                 in preferences.theme_choices()) == THEMES


@pytest.fixture
def themed(qapp, tmp_path, monkeypatch):
    """A private preference store; the application's look put back after."""
    from spacr.qt.widgets.field_fade import (invalidate_field_fade,
                                             uninstall_field_fade)

    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()

    def apply(theme, opacity):
        preferences.set_theme_choice(theme)
        preferences.set_pane_opacity(opacity)
        preferences.apply_preferences_to_app(qapp)

    yield apply
    uninstall_field_fade(qapp)
    invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _window(qtbot):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1600, 1000)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_card_tints"):
        pytest.skip("the window backdrop is not the field in this setup")
    return window, backdrop


def _show(window, key):
    window._on_nav_selected(key)
    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()


def _composited(backdrop, qtbot):
    """Wait, with the field ticking, until the cards stay composited."""
    tints = backdrop._card_tints
    for _attempt in range(60):
        qtbot.waitUntil(lambda: tints.active, timeout=20000)
        qtbot.wait(300)
        if tints.active:
            return tints
    raise AssertionError("the cards never stayed composited")


@pytest.mark.parametrize("theme", THEMES)
def test_home_and_mask_composite_identically_in_every_theme(themed, qtbot,
                                                             theme):
    themed(theme, 0.0)
    window, backdrop = _window(qtbot)
    for key in ("__home__", "mask"):
        _show(window, key)
        for opacity in (0.0, 0.5, 1.0):
            themed(theme, opacity)
            if not (isinstance(backdrop.engine, ambient._BufferedEngine)
                    or backdrop._stars_on_graphics()):
                qtbot.wait(500)
                assert not backdrop._card_tints.active, (theme, key)
                continue
            tints = _composited(backdrop, qtbot)
            assert tints.plan.origin == backdrop.mapTo(window, QPoint(0, 0))
            backdrop._timer.stop()
            backdrop._producer_box[0].stop()
            composited = _grab(window)
            assert backdrop._last_composed[0] is tints.plan
            tints.invalidate()
            classic = _grab(window)
            backdrop.stop()
            backdrop.start()
            assert composited == classic, (theme, key, opacity)


@pytest.mark.parametrize("theme", ["dark", "glass"])
def test_map_barcodes_keeps_its_cards_over_the_rain(themed, qtbot, theme):
    themed(theme, 0.5)
    window, backdrop = _window(qtbot)
    _show(window, "map_barcodes")
    rains = [w for w in window.findChildren(DnaRainWidget) if w.isVisible()]
    assert rains and all(r.testAttribute(Qt.WA_OpaquePaintEvent)
                         for r in rains)
    tints = backdrop._card_tints
    qtbot.waitUntil(lambda: tints.plan is not None, timeout=20000)
    qtbot.wait(500)
    covered = [w for w in window.findChildren(QFrame)
               if w.objectName() == "Card" and w.isVisible()]
    assert covered
    assert not any(w.testAttribute(Qt.WA_NoSystemBackground)
                   for w in covered)
    for rain in rains:
        rain.stop()
    backdrop._timer.stop()
    backdrop._producer_box[0].stop()
    backdrop.advance_frame(0)
    frame_time = backdrop.time()
    assert tints.active
    before = _grab(window)
    backdrop.set_animating(False)
    backdrop.hide()
    backdrop.show()
    for rain in rains:
        rain.stop()
    assert not backdrop.is_running()
    assert not backdrop.shading_thread_alive()
    assert backdrop.time() == frame_time
    assert _grab(window) == before
    backdrop.set_animating(True)
    backdrop.stop()
    backdrop.start()


def test_a_change_event_other_than_a_restyle_leaves_the_rain_alone(qtbot):
    rain = DnaRainWidget(None, seed=2)
    qtbot.addWidget(rain)
    rain.setAttribute(Qt.WA_OpaquePaintEvent, False)
    rain.changeEvent(QEvent(QEvent.EnabledChange))
    assert not rain.testAttribute(Qt.WA_OpaquePaintEvent)
    rain.changeEvent(QEvent(QEvent.StyleChange))
    assert rain.testAttribute(Qt.WA_OpaquePaintEvent)


def test_no_static_part_is_composited_under_the_spaceout_dressing(
        qtbot, monkeypatch):
    switch = Toggle("")
    qtbot.addWidget(switch)
    switch.resize(40, 22)
    assert ambient._static_part(switch) is not None
    monkeypatch.setattr(ambient, "spaceout_enabled", lambda: True)
    assert ambient._static_part(switch) is None


def test_mask_composites_identically_under_the_spaceout_dressing(themed,
                                                                 qtbot):
    from spacr.qt import theme as theme_module

    theme_module.enable_spaceout()
    try:
        themed("data_art_fungal_growth", 0.5)
        window, backdrop = _window(qtbot)
        _show(window, "mask")
        tints = _composited(backdrop, qtbot)
        assert tints._parts == []
        backdrop._timer.stop()
        backdrop._producer_box[0].stop()
        composited = _grab(window)
        assert backdrop._last_composed[0] is tints.plan
        tints.invalidate()
        assert composited == _grab(window)
        backdrop.stop()
    finally:
        theme_module.disable_spaceout()
