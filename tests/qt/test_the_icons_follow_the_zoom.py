"""433 -- hold Z, turn the wheel, and the icons go where the text goes.

THE DEFECT, IN ONE SENTENCE. An icon size is a widget PROPERTY, written
once when the widget is built, so a font scale that grew every caption left
every glyph beside those captions exactly where it was. Reported on
2026-09-05 against the gesture 378 built: "is better i think but the icons
seem like they dont track perfectly".

WHAT THESE TESTS MEASURE, AND WHERE. The gesture is driven the way a user
drives it -- a real Z key event, real wheel events, a real settle -- on the
four widget families that carry an icon through it: the dock rows, both home
tiles, and the folded-module strip. Nothing here asserts on a helper's
return value; every assertion is a ``QSize`` read back off a widget that the
gesture has been through.

THE ROUND TRIP IS THE ONE THAT MATTERS. Ten notches up and ten back down
must return every icon to the exact pixel it started on, which is only true
because each size is recomputed from a stored BASE rather than from the size
the widget is currently wearing -- twenty roundings of ``round(px * 1.05)``
do not cancel.

THE LIVE HALF IS MEASURED TOO, by what it does NOT do. 378's request carried
the condition "only if possible to do fast without lag", so the icon sweep
belongs to the settle. ``test_a_notch_leaves_every_icon_alone`` fails the day
somebody moves it into the per-notch pass.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QFontMetrics, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import (QApplication, QLineEdit, QVBoxLayout,
                               QWidget)

from spacr.qt import live_zoom
from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture(scope="module", autouse=True)
def _leave_the_application_as_we_found_it(qapp):
    """Repolish the shared QApplication once this file is done with it.

    One test lets the settle's real stylesheet rebuild run, and the settle
    hands back every font it borrowed -- so without this the process's
    long-lived widgets are left inheriting rather than styled for every
    later test that measures a pixel. One rebuild at the end costs a
    second; one per test would cost twenty.
    """
    yield
    from spacr.qt.theme import stylesheet
    qapp.setStyleSheet(stylesheet())
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write the developer's real font scale. The settle persists it."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write real preferences")
    prefs.set_font_scale(1.0)
    return store


@pytest.fixture
def settle_without_restyling(monkeypatch):
    """Let the settle do its icon half, and not its stylesheet half.

    ``apply_preferences_to_app`` restyles every live widget in the process,
    which for a session-scoped ``qapp`` means every later test measuring a
    pixel. The icon sweep is one step INSIDE it, so the stub calls that one
    step for real and skips the rest: the settle is still the thing that
    reaches the icons, by the route it really takes, and the shared
    application is left alone. ``test_the_rebuild_is_what_carries_the_icons``
    runs the unstubbed function to prove the two are wired together.
    """
    applied = []

    def _stub(app=None):
        app = app or QApplication.instance()
        applied.append(app)
        prefs._rescale_icon_sizes(app)

    monkeypatch.setattr(prefs, "apply_preferences_to_app", _stub)
    return applied


@pytest.fixture
def zoom(qt_theme_applied):
    """A filter that always ends its gesture, whatever the test did."""
    live = live_zoom.LiveZoomFilter()
    yield live
    live.settle()


@pytest.fixture
def icon_screen(qt_theme_applied):
    """One shown tree holding every widget family that carries an icon.

    Built together rather than one per test because the gesture is
    application-wide: a sweep that reached the dock and missed the tiles
    would pass four separate single-widget tests and fail the screen.
    """
    from spacr.qt import iconset
    from spacr.qt.widgets.dock import Dock
    from spacr.qt.widgets.fold_strip import FoldStrip
    from spacr.qt.widgets.tile import HTile, Tile

    mark = iconset.icon("settings")
    root = QWidget()
    layout = QVBoxLayout(root)
    dock = Dock(
        [("mask", "Make Masks", "segment", "Core"),
         ("measure", "Measure", "quantify", "Core")],
        icon_for=lambda _key: iconset.icon("settings"))
    htile = HTile("Make Masks", "segment an image", icon=mark, icon_size=52)
    tile = Tile("Make Masks", mark, icon_size=64, tile_size=120)
    strip = FoldStrip([("mask", lambda: None), ("measure", lambda: None)])
    for widget in (dock, htile, tile, strip):
        layout.addWidget(widget)
    root.show()
    QApplication.processEvents()
    yield root, dock, htile, tile, strip
    root.hide()
    root.deleteLater()
    QApplication.processEvents()


def _key(kind, key=Qt.Key_Z, modifiers=Qt.NoModifier, autorepeat=False):
    """One key event, shaped the way the window manager delivers it."""
    return QKeyEvent(kind, key, modifiers, "z", autorepeat)


def _wheel(notches: int = 1):
    """One wheel event carrying ``notches`` detents, Qt's own 120ths."""
    return QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
        QPoint(0, int(notches * 120)), Qt.NoButton, Qt.NoModifier,
        Qt.NoScrollPhase, False)


def _hold(zoom, widget):
    """Press Z the way the user does, and confirm the gesture armed."""
    assert zoom.eventFilter(widget, _key(QEvent.KeyPress)) is False, (
        "the gesture must never swallow the key itself -- Z is a letter")
    assert zoom._held


def _turn(zoom, widget, notches: int) -> None:
    """Turn the wheel one detent at a time, as a real wheel does.

    One event of twenty notches and twenty events of one are the same
    scale and NOT the same rounding history, and the difference is the
    whole subject of this file.
    """
    step = 1 if notches > 0 else -1
    for _ in range(abs(notches)):
        zoom.eventFilter(widget, _wheel(step))


def _icons(root) -> dict:
    """Every icon size under ``root``, keyed by the widget that wears it.

    Keyed on the widget itself rather than on a name: two dock rows are
    the same class at the same size, and a dict that collapsed them would
    pass while one of them was left behind.
    """
    sizes = {}
    for widget in root.findChildren(QWidget):
        try:
            size = widget.iconSize()
        except AttributeError:
            continue
        if size.width() > 0 and not widget.icon().isNull():
            sizes[widget] = (size.width(), size.height())
    return sizes


def test_the_screen_really_carries_icons(icon_screen):
    """The fixture is the measurement; an empty one would pass everything.

    Every assertion below is of the form "these sizes changed", which is
    vacuously true of no sizes at all. Six is the count this screen has:
    two dock rows, an HTile, a Tile's button and two fold buttons.
    """
    root = icon_screen[0]
    sizes = _icons(root)
    assert len(sizes) >= 6, (
        f"only {len(sizes)} icon-bearing widgets on the screen -- the rest "
        f"of this file would pass on an empty dictionary")


def test_a_settled_zoom_takes_the_icons_with_it(
        zoom, icon_screen, settle_without_restyling):
    """The defect, driven as the user drives it.

    Four notches is 20 %, which is past every rounding on every size here,
    so an icon that has not moved has not moved at all.
    """
    root = icon_screen[0]
    before = _icons(root)

    _hold(zoom, root)
    _turn(zoom, root, +4)
    zoom.settle()
    QApplication.processEvents()

    after = _icons(root)
    assert settle_without_restyling, "the settle never asked for a rebuild"
    for widget, was in before.items():
        assert after[widget][0] > was[0], (
            f"{widget.objectName() or type(widget).__name__} kept a "
            f"{was[0]} px icon while the text around it grew 20 %")


def test_a_notch_leaves_every_icon_alone(zoom, icon_screen):
    """The live half stays cheap: text moves per notch, icons do not.

    378's request carried the condition "only if possible to do fast
    without lag", and the answer it produced was "text live, spacing on
    release". The icons belong with the spacing. This fails the day the
    sweep is moved into the per-notch pass, whatever it costs there.
    """
    root, _dock, htile, _tile, _strip = icon_screen
    before = _icons(root)
    text_before = QFontMetrics(htile.font()).height()

    _hold(zoom, root)
    _turn(zoom, root, +4)
    QApplication.processEvents()

    assert QFontMetrics(htile.font()).height() > text_before, (
        "the live half did not grow the text, so this proves nothing "
        "about what it left alone")
    assert _icons(root) == before, (
        "an icon moved during the live half of the gesture")


def test_the_icons_come_back_to_the_pixel_they_started_on(
        zoom, icon_screen, settle_without_restyling):
    """Ten notches up, ten notches down, and nothing has drifted.

    THE PROOF THAT THE BASE IS REAL. Recomputing an icon from the size it
    is wearing compounds the rounding: twenty applications of
    ``round(px * 1.05)`` and its inverse do not return a 20 px icon to
    20 px. Every size here is derived from the number the widget was built
    with, so the scale alone decides the answer and the direction the
    wheel reached it from cannot matter.
    """
    root = icon_screen[0]
    before = _icons(root)

    _hold(zoom, root)
    _turn(zoom, root, +10)
    zoom.settle()
    QApplication.processEvents()
    peak = _icons(root)
    assert peak != before, "ten notches up moved nothing; nothing to undo"

    _hold(zoom, root)
    _turn(zoom, root, -10)
    zoom.settle()
    QApplication.processEvents()

    assert prefs.get_font_scale() == pytest.approx(1.0), (
        "the round trip did not end on the scale it started from")
    assert _icons(root) == before, (
        "an icon did not come back to the size it started at")


def test_the_way_a_scale_was_reached_cannot_change_an_icon(icon_screen):
    """One jump to 150 % and ten notches to 150 % agree to the pixel.

    The same question as the round trip, asked from the other side: a
    value that depends on its own history is the bug, and two histories
    ending at one scale is the cheapest way to see it.
    """
    root = icon_screen[0]
    app = QApplication.instance()

    prefs.set_font_scale(1.5)
    prefs._rescale_icon_sizes(app)
    in_one_step = _icons(root)

    prefs.set_font_scale(1.0)
    prefs._rescale_icon_sizes(app)
    scale = 1.0
    for _ in range(10):
        scale = round(scale + live_zoom.FONT_SCALE_STEP, 4)
        prefs.set_font_scale(scale)
        prefs._rescale_icon_sizes(app)

    assert _icons(root) == in_one_step, (
        "an icon size remembers how the scale was reached")


def test_a_widget_built_after_a_zoom_matches_the_ones_already_there(
        zoom, icon_screen, settle_without_restyling):
    """A screen opened at 150 % comes up at 150 %, not at 100 %.

    The sweep only reaches widgets that exist when it runs. Everything
    built later has to size itself correctly from the start, which it does
    by reading the scale at construction -- so this asserts on a widget
    the gesture has never seen, measured against one it has.
    """
    from spacr.qt import iconset
    from spacr.qt.widgets.fold_strip import FoldStrip
    from spacr.qt.widgets.tile import HTile, Tile

    root, _dock, htile, tile, strip = icon_screen

    _hold(zoom, root)
    _turn(zoom, root, +10)
    zoom.settle()
    QApplication.processEvents()

    mark = iconset.icon("settings")
    fresh_htile = HTile("Measure", "count them", icon=mark, icon_size=52)
    fresh_tile = Tile("Measure", mark, icon_size=64, tile_size=120)
    fresh_strip = FoldStrip([("mask", lambda: None)])
    try:
        assert fresh_htile.iconSize() == htile.iconSize()
        assert fresh_tile._button.iconSize() == tile._button.iconSize()
        assert (fresh_strip.buttons[0].iconSize()
                == strip.buttons[0].iconSize())
        assert fresh_strip.layout().spacing() == strip.layout().spacing()
    finally:
        for widget in (fresh_htile, fresh_tile, fresh_strip):
            widget.deleteLater()
        QApplication.processEvents()


def test_the_rebuild_is_what_carries_the_icons(icon_screen):
    """The sweep is inside ``apply_preferences_to_app``, not beside it.

    WHY IT LIVES THERE RATHER THAN IN ``settle``. That function is the one
    step both routes to a new scale already take -- the wheel's settle and
    the Preferences slider on its way out of the dialog -- so putting the
    sweep there fixes the same defect for the control as for the gesture,
    and there is no second place that can be forgotten.

    This is the one test in the file that lets the real rebuild run; the
    module fixture repolishes the shared application afterwards.
    """
    root = icon_screen[0]
    before = _icons(root)

    prefs.set_font_scale(1.6)
    prefs.apply_preferences_to_app(QApplication.instance())
    QApplication.processEvents()

    after = _icons(root)
    assert after != before, (
        "apply_preferences_to_app rebuilt the stylesheet and left every "
        "icon at the size it was")
    for widget, was in before.items():
        assert after[widget][0] > was[0]


def test_the_window_chrome_is_repainted_rather_than_stretched(qt_theme_applied):
    """The three marks in the corner of the window follow the scale too.

    THEY WERE NOT FOUND BY GREPPING FOR ``setIconSize``, because they never
    called it -- they took Qt's 16 px small-icon default, which does not
    move with the font scale, while painting their glyph into an 18 px box.
    They turned up in a census of what a real window draws an icon on, and
    that is the only reason this test exists.

    REPAINTED, NOT RESIZED. The glyph is a pixmap the button draws itself,
    so growing only the icon size hands Qt a small pixmap to stretch. The
    assertion is on the pixmap the icon can actually supply, not on
    ``iconSize``, which would pass on a blur.
    """
    from PySide6.QtWidgets import QWidget as _QWidget

    from spacr.qt.app import CHROME_HOVER, MainWindow, _ChromeButton

    parent = _QWidget()
    button = _ChromeButton(parent, MainWindow._close_icon,
                           CHROME_HOVER["CloseWindow"])
    try:
        before = button.icon().availableSizes()[0].width()
        assert button.iconSize().width() == before, (
            "the mark is already being drawn at a size it was not painted "
            "at, so this test cannot tell a repaint from a stretch")

        prefs.set_font_scale(2.0)
        prefs._rescale_icon_sizes(QApplication.instance())

        assert button.iconSize().width() > before
        assert button.icon().availableSizes()[0].width() \
            == button.iconSize().width(), (
                "the mark was resized without being repainted -- Qt is "
                "stretching the old pixmap")
    finally:
        prefs.set_font_scale(1.0)
        parent.deleteLater()
        QApplication.processEvents()


def test_the_chrome_keeps_its_hover_colour_across_a_zoom(qt_theme_applied):
    """A mark lit under the pointer is still lit after the wheel turns.

    The repaint has to be told which of the two paintings to make. A
    repaint that assumed "resting" would drop the hover colour of whatever
    the pointer happened to be over -- and the pointer is over something
    during a wheel gesture by definition.
    """
    from PySide6.QtWidgets import QWidget as _QWidget

    from spacr.qt.app import CHROME_HOVER, MainWindow, _ChromeButton

    asked = []

    def painter(size=18, colour=None):
        asked.append(colour)
        return MainWindow._close_icon(size=size, colour=colour)

    parent = _QWidget()
    button = _ChromeButton(parent, painter, CHROME_HOVER["CloseWindow"])
    try:
        button._show(True)
        assert asked[-1] == CHROME_HOVER["CloseWindow"]

        prefs.set_font_scale(1.5)
        prefs._rescale_icon_sizes(QApplication.instance())

        assert asked[-1] == CHROME_HOVER["CloseWindow"], (
            "the zoom repainted a lit mark as a resting one")
    finally:
        prefs.set_font_scale(1.0)
        parent.deleteLater()
        QApplication.processEvents()


def test_an_icon_base_survives_a_scale_it_cannot_round_cleanly(icon_screen):
    """10 % is the floor of the Zoom range and rounds a 20 px icon to 2.

    The floor is where a size derived by repeated multiplication collapses
    to nothing and never comes back. Nothing may reach 0 px -- Qt draws
    nothing at all there -- and everything must still return.
    """
    root = icon_screen[0]
    app = QApplication.instance()
    before = _icons(root)

    prefs.set_font_scale(prefs.FONT_SCALE_MIN)
    prefs._rescale_icon_sizes(app)
    floored = _icons(root)
    assert all(w > 0 and h > 0 for w, h in floored.values()), (
        "an icon collapsed to zero pixels at the bottom of the range")

    prefs.set_font_scale(1.0)
    prefs._rescale_icon_sizes(app)
    assert _icons(root) == before, (
        "an icon did not come back from the bottom of the range")


def test_nothing_spacr_draws_an_icon_on_is_left_behind(qtbot, qt_theme_applied):
    """A census of a real window, because grepping missed three buttons.

    THE METHOD THAT FOUND THE DEFECT, KEPT AS THE TEST. Searching for
    ``setIconSize`` finds the widgets that set one; it cannot find the ones
    that take Qt's 16 px small-icon default and never say so, and three of
    those are the minimise, full-screen and close marks in the corner of
    every window. Asking a built window which of its widgets is drawing an
    icon finds both kinds.

    THE EXEMPTIONS ARE QT'S OWN WIDGETS AND NOTHING ELSE. A menu bar that
    has run out of room grows an overflow button, and a line edit with a
    clear mark grows one too; both are created by Qt inside its own
    classes, both size themselves from the style, and neither is ours to
    register. Every other icon in the window has to carry a base or a hook,
    or the zoom leaves it behind -- which is this instruction's whole
    subject, so a new one arriving is a thing to decide about rather than
    to discover later in a screenshot.
    """
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    QApplication.processEvents()

    left_behind = []
    for widget in window.findChildren(QWidget):
        try:
            if widget.iconSize().width() <= 0 or widget.icon().isNull():
                continue
        except AttributeError:
            continue
        if widget.property(prefs._KEY_ICON_BASE_W) is not None:
            continue
        if callable(getattr(widget, "_apply_icon_scale", None)):
            continue
        if widget.objectName().startswith("qt_"):
            continue
        if isinstance(widget.parent(), QLineEdit):
            continue
        left_behind.append(
            f"{type(widget).__name__}#{widget.objectName()} at "
            f"{widget.iconSize().width()} px")

    assert not left_behind, (
        "these widgets draw an icon that the interface scale cannot "
        "reach:\n  " + "\n  ".join(sorted(left_behind)))
