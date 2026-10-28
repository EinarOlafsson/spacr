"""The splash's own defences, driven rather than assumed.

The loading screen is painted before the theme is guaranteed to have been
resolved, so every colour it asks for carries the literal it replaced and
every lookup is allowed to fail. The paths below are the failing ones: a
palette module that raises, a colour spelling Qt cannot parse, a logo that is
not on disk. None of them may reach the user, because there is no window yet
to report them in.

The painting is driven onto a real ``QPixmap`` and the pixels are read back,
so "the sentence lights up as the modules land" is measured rather than
claimed.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QWidget

from spacr.qt.widgets import loading_screen as ls

pytestmark = pytest.mark.qt


# -- the palette is allowed to be unreachable --------------------------------

def test_a_palette_that_raises_leaves_the_splash_with_its_literal(
        qapp, monkeypatch):
    """A theme lookup that blows up yields the fallback, not a traceback.

    This is the first thing spaCR paints. There is no window to show an
    error in yet, so the only survivable answer is the colour the literal
    named before the palette existed.
    """
    import spacr.qt.theme as theme

    def _explode(*args, **kwargs):
        raise RuntimeError("the theme has not been resolved yet")

    monkeypatch.setattr(theme, "palette_for", _explode)

    assert ls._role("splash_bg", "#003737") == "#003737"
    assert ls._role_color("splash_bg", "#003737").name() == "#003737"
    assert ls._role_brush("splash_track").name() == "#ffffff"


def test_a_palette_missing_the_role_falls_back_too(qapp, monkeypatch):
    """A palette that answers, but not about this role, is not an answer."""
    import spacr.qt.theme as theme

    monkeypatch.setattr(theme, "palette_for",
                        lambda *a, **k: {"splash_ink": "#112233"})

    assert ls._role("splash_bg", "#010203") == "#010203"
    assert ls._role("splash_ink", "#010203") == "#112233"


def test_an_empty_palette_entry_is_not_used(qapp, monkeypatch):
    """``""`` in the palette means "unset", and paints the literal."""
    import spacr.qt.theme as theme

    monkeypatch.setattr(theme, "palette_for", lambda *a, **k: {"splash_bg": ""})

    assert ls._role("splash_bg", "#0a0b0c") == "#0a0b0c"


# -- the colour spellings the palette may hand back --------------------------

def test_a_four_component_rgba_string_keeps_its_alpha():
    colour = ls._rgba("rgba(10, 20, 30, 40)", QColor("#000000"))

    assert (colour.red(), colour.green(), colour.blue()) == (10, 20, 30)
    assert colour.alpha() == 40


def test_a_three_component_rgba_string_is_opaque():
    """``rgba(r, g, b)`` names no alpha, so the colour is fully opaque."""
    colour = ls._rgba("rgba(10, 20, 30)", QColor("#000000"))

    assert (colour.red(), colour.green(), colour.blue()) == (10, 20, 30)
    assert colour.alpha() == 255


def test_an_rgba_string_with_the_wrong_number_of_parts_falls_back():
    colour = ls._rgba("rgba(10, 20)", QColor("#445566"))

    assert colour.name() == "#445566"


def test_an_rgba_string_whose_parts_are_not_numbers_falls_back():
    """A misspelt palette entry paints the fallback rather than raising."""
    colour = ls._rgba("rgba(ten, twenty, thirty, one)", QColor("#445566"))

    assert colour.name() == "#445566"


def test_a_float_rgba_string_is_truncated_to_channels():
    colour = ls._rgba("rgba(10.9, 20.2, 30.7, 255)", QColor("#000000"))

    assert (colour.red(), colour.green(), colour.blue()) == (10, 20, 30)


def test_a_colour_qt_cannot_parse_falls_back():
    colour = ls._rgba("chartreuse-ish", QColor("#778899"))

    assert colour.name() == "#778899"


def test_a_hash_colour_is_taken_as_written():
    assert ls._rgba("#123456", QColor("#000000")).name() == "#123456"


# -- the logo -----------------------------------------------------------------

def test_a_null_logo_pixmap_is_not_kept(qapp, monkeypatch):
    """A file that is not an image leaves the splash with no mark at all.

    ``QPixmap`` on a missing path returns a null pixmap rather than raising,
    so the null check is the branch that matters.
    """
    monkeypatch.setattr(ls, "LOGO_FILE", "no_such_logo_file.png")

    screen = ls.LoadingScreen(total=2)

    assert screen._logo is None


def test_the_packaged_logo_is_loaded(qapp):
    """The shipped mark really is readable from the resource folder."""
    screen = ls.LoadingScreen(total=2)

    assert screen._logo is not None
    assert not screen._logo.isNull()


# -- painting ------------------------------------------------------------------

def _painted(screen) -> QPixmap:
    """Render ``screen`` to a pixmap so its paint path can be inspected."""
    screen.resize(900, 400)
    return screen.grab()


def test_the_splash_paints_its_background_over_the_whole_window(qapp,
                                                               monkeypatch):
    """The cover is opaque: it exists to hide a half-built window."""
    import spacr.qt.theme as theme

    monkeypatch.setattr(
        theme, "palette_for",
        lambda *a, **k: {"splash_bg": "#102030", "splash_ink": "#ffffff",
                         "splash_ink_dim": "#404040",
                         "splash_track": "#202020",
                         "splash_fill": "#00ff00"})

    screen = ls.LoadingScreen(total=3)
    image = _painted(screen).toImage()

    assert image.pixelColor(2, 2).name() == "#102030"
    assert image.pixelColor(image.width() - 3, 2).name() == "#102030"


def test_the_progress_rule_grows_with_the_fraction(qapp, monkeypatch):
    """The hairline under the sentence is filled to the completed share.

    Counted as fill pixels on the rule's own row, which is the only way to
    tell a rule that moved from a rule that was redrawn identically.
    """
    import spacr.qt.theme as theme

    monkeypatch.setattr(
        theme, "palette_for",
        lambda *a, **k: {"splash_bg": "#000000", "splash_ink": "#ffffff",
                         "splash_ink_dim": "#303030",
                         "splash_track": "#202020",
                         "splash_fill": "#ff0000"})

    def _fill_pixels(screen):
        image = _painted(screen).toImage()
        return sum(image.pixelColor(x, y).name() == "#ff0000"
                   for y in range(image.height())
                   for x in range(image.width()))

    screen = ls.LoadingScreen(total=4)
    screen.advance(1)
    quarter = _fill_pixels(screen)
    screen.advance(4)
    whole = _fill_pixels(screen)

    assert quarter > 0
    assert whole > quarter


def test_a_splash_with_no_logo_still_paints_the_sentence(qapp, monkeypatch):
    """The mark is optional; the words are not."""
    import spacr.qt.theme as theme

    monkeypatch.setattr(
        theme, "palette_for",
        lambda *a, **k: {"splash_bg": "#000000", "splash_ink": "#ffffff",
                         "splash_ink_dim": "#000000",
                         "splash_track": "#000000",
                         "splash_fill": "#000000"})

    screen = ls.LoadingScreen(total=3)
    screen._logo = None
    screen.advance(3)
    image = _painted(screen).toImage()

    # Glyph edges are antialiased, and Qt's offscreen font backend is free
    # to render a thin face without any fully saturated centre pixel.  What
    # matters is that ink made it onto the otherwise all-black image.
    ink = sum(image.pixelColor(x, y).value() > 0
              for y in range(image.height())
              for x in range(image.width()))

    assert ink > 0


def test_the_lit_and_the_unlit_phases_are_inked_differently(qapp,
                                                            monkeypatch):
    """Half-done paints both colours; nothing-done paints only the dim one."""
    import spacr.qt.theme as theme

    monkeypatch.setattr(
        theme, "palette_for",
        lambda *a, **k: {"splash_bg": "#000000", "splash_ink": "#00ff00",
                         "splash_ink_dim": "#0000ff",
                         "splash_track": "#000000",
                         "splash_fill": "#000000"})

    def _has(screen, channel):
        # Keep the colour probe about the sentence.  The packaged mark has
        # its own antialiased pixels and is not part of this contract.
        screen._logo = None
        image = _painted(screen).toImage()
        for y in range(image.height()):
            for x in range(image.width()):
                colour = image.pixelColor(x, y)
                channels = (colour.red(), colour.green(), colour.blue())
                wanted = channels[channel]
                others = channels[:channel] + channels[channel + 1:]
                if wanted > max(others):
                    return True
        return False

    waiting = ls.LoadingScreen(total=3)
    assert waiting.lit_phases() == 0
    assert _has(waiting, 2)
    assert not _has(waiting, 1)

    half = ls.LoadingScreen(total=3)
    half.advance(2)
    assert half.lit_phases() == 2
    assert _has(half, 1)
    assert _has(half, 2)


def test_a_resize_repaints_the_cover(qapp, monkeypatch):
    """The splash follows the window it covers, and asks to be redrawn.

    The artwork is centred on the widget's own rectangle, so a cover that
    is resized without a repaint request keeps the logo and the phase dots
    where the old geometry put them until something else happens to
    invalidate it.
    """
    screen = ls.LoadingScreen(total=2)
    screen.resize(400, 200)
    screen.show()
    qapp.processEvents()
    repaints = []
    monkeypatch.setattr(screen, "update",
                        lambda *args: repaints.append(True))

    screen.resize(800, 600)
    qapp.processEvents()

    assert repaints == [True]
    assert screen.size().width() == 800
    monkeypatch.undo()
    image = screen.grab().toImage()
    assert image.width() == 800
    screen.hide()


def test_a_parented_splash_takes_the_window_it_covers(qapp):
    """Built with a parent, the cover starts the size of that parent."""
    host = QWidget()
    host.resize(640, 480)

    screen = ls.LoadingScreen(total=2, parent=host)

    assert screen.geometry() == host.rect()


# -- the sentence -------------------------------------------------------------

def test_the_strap_phrases_are_translated_one_at_a_time(qapp, monkeypatch):
    """The home screen reuses these words, so they go through the catalog."""
    import spacr.qt.i18n as i18n

    monkeypatch.setattr(i18n, "tr", lambda text: text.upper())

    assert ls.strap_phrases() == tuple(p.upper() for p in ls.STRAP_PHASES)


def test_the_whole_strap_line_joins_the_translated_phases(qapp, monkeypatch):
    import spacr.qt.i18n as i18n

    monkeypatch.setattr(i18n, "tr", lambda text: f"<{text}>")

    line = ls.strap_line()

    for phrase in ls.STRAP_PHASES:
        assert f"<{phrase}>" in line
    assert line.count("→") == len(ls.STRAP_PHASES) - 1


def test_the_deprecated_alias_still_names_the_splash_background():
    """An importer that spells it the old way gets the current colour."""
    assert ls.INSTALLER_GREEN == ls.SPLASH_BACKGROUND
    assert ls.splash_role is ls._role


@pytest.mark.parametrize('reported,expected', [
    (60, 60), (144, 144), (None, ls._DEFAULT_REFRESH_HZ),
    ('unreadable', ls._DEFAULT_REFRESH_HZ), (float('nan'), ls._DEFAULT_REFRESH_HZ),
    (0, ls._DEFAULT_REFRESH_HZ), (10000, ls._DEFAULT_REFRESH_HZ),
])
def test_screen_refresh_metadata_stays_within_supported_pacing_limits(reported, expected):
    """Bad screen API metadata cannot produce a nonfinite or zero timer budget."""
    from types import SimpleNamespace

    screen = None if reported is None else SimpleNamespace(refreshRate=lambda: reported)
    assert ls._screen_refresh_hz(screen) == expected


def test_display_update_dispatch_and_starvation_watchdog_retire_with_real_cover(qtbot, monkeypatch):
    """A display-capability shim exercises real window requests and owned timer cleanup."""
    from tests.qt.test_694_photomosaic_blow_away_startup_screen import _Clock, _played

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(420, 320)
    host.show()
    screen = ls.LoadingScreen(total=2, parent=host)
    clock = _Clock()
    pace = _Clock()
    screen._clock = clock
    screen.clock = clock
    screen._pace_clock = pace
    monkeypatch.setattr(ls, '_display_paced', lambda platform: True)
    assert screen.show_mosaic()
    _played(screen)
    handle = host.windowHandle()
    assert handle is not None and screen._paced_window is handle and screen._vsync
    requested, repainted = [], []
    original_request, original_update = handle.requestUpdate, screen.update

    def request():
        requested.append(True)
        original_request()

    def update(*args):
        repainted.append(True)
        original_update(*args)

    monkeypatch.setattr(handle, 'requestUpdate', request)
    monkeypatch.setattr(screen, 'update', update)
    screen._follow_the_display(handle)
    assert requested == [True]
    screen.grab()
    assert screen._presented[-1] == pace.now
    before_watchdog = len(repainted)
    pace.now += 2 * screen._frame_budget_ms() / 1000
    screen._tick()
    assert repainted[before_watchdog:] == [True]
    assert screen._tick_timer.isActive()
    screen.skip()
    assert screen._paced_window is None
    assert screen._tick_timer is None or not screen._tick_timer.isActive()
    screen._stop_pacing()


@pytest.mark.parametrize('operation', ['hold', 'release'])
def test_optional_import_priority_failure_does_not_break_the_cover(qtbot, monkeypatch, operation):
    """Unavailable background-import coordination cannot abort startup artwork."""
    from spacr.qt import gil_priority

    screen = ls.LoadingScreen(total=2)
    qtbot.addWidget(screen)

    def unavailable(*args):
        raise OSError('import coordination unavailable')

    target = 'hold_background_imports' if operation == 'hold' else 'release_background_imports'
    monkeypatch.setattr(gil_priority, target, unavailable)
    getattr(screen, '_hold_the_imports' if operation == 'hold' else '_release_the_imports')()
    assert screen.fraction() == 0
    screen.advance()
    assert screen.fraction() == .5


def test_removed_mosaic_resource_during_hold_cannot_prepare_tiles(qtbot, tmp_path):
    """A resource removed after its map loads leaves no stale prepared flight."""
    from tests.qt.test_694_photomosaic_blow_away_startup_screen import _Clock

    folder = tmp_path / 'mosaic'
    folder.mkdir()
    original = ls.MosaicMap.load()
    image = folder / 'image.jpg'
    image.write_bytes(open(original.image_path, 'rb').read())
    mosaic = ls.MosaicMap(image_path=str(image), grid=original.grid,
                          tile_px=original.tile_px, occupied=original.occupied)
    screen = ls.LoadingScreen(total=1)
    qtbot.addWidget(screen)
    clock = _Clock()
    screen._clock = clock
    screen.show_mosaic(mosaic)
    assert screen.blow_away(force=True)
    screen.grab()
    image.unlink()
    previous_key = screen._mosaic_key
    screen.resize(240, 180)
    assert screen._mosaic_layout()[3] != previous_key[0]
    screen._prepare()
    assert screen._prepared is None
    assert screen._mosaic is None
    screen.skip()


def test_retired_native_display_handle_can_be_disconnected_twice(qtbot, monkeypatch):
    """Window destruction between pacing setup and cleanup is an ordinary lifetime edge."""
    import shiboken6
    from types import SimpleNamespace
    from PySide6.QtGui import QWindow

    screen = ls.LoadingScreen(total=1)
    qtbot.addWidget(screen)
    handle = QWindow()
    owner = SimpleNamespace(windowHandle=lambda: handle)
    monkeypatch.setattr(screen, 'window', lambda: owner)
    monkeypatch.setattr(ls, '_display_paced', lambda platform: True)
    screen._start_pacing()
    assert screen._paced_window is handle
    shiboken6.delete(handle)
    screen._stop_pacing()
    assert screen._paced_window is None
    screen._stop_pacing()


def test_retired_window_lookup_uses_screen_fallback_without_a_handle(qtbot, monkeypatch):
    """A stale native owner cannot prevent a live cover from selecting a safe timer budget."""
    import shiboken6

    screen = ls.LoadingScreen(total=1)
    qtbot.addWidget(screen)
    old = QWidget()
    lookup = old.window
    shiboken6.delete(old)
    monkeypatch.setattr(screen, 'window', lookup)
    screen._start_pacing()
    assert screen._paced_window is None and not screen._vsync
    assert screen._refresh_hz > 0
    screen._schedule_tick()
    assert screen._tick_timer is None


@pytest.mark.parametrize('pixels', [18, 24])
def test_strap_arrow_keeps_a_pixel_aligned_visible_stem(qtbot, pixels):
    """Both even- and odd-pixel stems remain visible in actual offscreen paint."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QImage, QPainter

    screen = ls.LoadingScreen(total=1)
    qtbot.addWidget(screen)
    image = QImage(140, 60, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    font = QFont()
    font.setPixelSize(pixels)
    painter.setFont(font)
    try:
        screen._paint_strap_arrow(painter, 20, 40, 60, QColor('#ffffff'))
    finally:
        painter.end()
    ink = [(x, y) for y in range(image.height()) for x in range(image.width())
           if image.pixelColor(x, y).alpha() > 0]
    assert len(ink) > 20
    assert min(x for x, y in ink) >= 20 and max(x for x, y in ink) < 80


def test_late_display_paints_reduce_pacing_and_slow_costs_thin_real_tiles(qtbot, monkeypatch):
    """Controlled callback timestamps/cost inputs exercise adaptive dispatch and real tile paint."""
    from tests.qt.test_694_photomosaic_blow_away_startup_screen import _Clock, _played

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(420, 320)
    host.show()
    screen = ls.LoadingScreen(total=1, parent=host)
    clock, pace = _Clock(), _Clock()
    screen._clock = screen.clock = clock
    screen._pace_clock = pace
    monkeypatch.setattr(ls, '_display_paced', lambda platform: True)
    assert screen.show_mosaic()
    _played(screen)
    initial = screen._divisor
    for _ in range(ls._PACING_WINDOW + 2):
        pace.now += 4 * screen._frame_budget_ms() / 1000
        screen.grab()
    assert screen._divisor > initial
    requests = []
    original = screen._paced_window.requestUpdate

    def request():
        requests.append(True)
        original()

    monkeypatch.setattr(screen._paced_window, 'requestUpdate', request)
    screen._tick()
    assert requests == [True]
    screen._follow_the_display(screen._paced_window)
    assert requests == [True]
    for _ in range(8 * ls._SLOW_STREAK + 12):
        screen._adapt(100.0)
    assert screen._stride > 1
    clock.now += 1.0
    frame = screen.grab().toImage()
    assert not frame.isNull() and screen._drawn[-1] > 0
    assert screen._drawn[-1] <= len(screen._tiles) // screen._stride + 1
    screen.skip()


def test_timely_display_ticks_do_not_repaint_and_unrelated_windows_cannot_drive_the_cover(qtbot, monkeypatch):
    """A normal display cadence owns repaint requests; another window owns no cover callbacks."""
    from PySide6.QtGui import QWindow
    from tests.qt.test_694_photomosaic_blow_away_startup_screen import _Clock, _played

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(420, 320)
    host.show()
    screen = ls.LoadingScreen(total=1, parent=host)
    clock, pace = _Clock(), _Clock()
    screen._clock = screen.clock = clock
    screen._pace_clock = pace
    monkeypatch.setattr(ls, '_display_paced', lambda platform: True)
    screen.show_mosaic()
    _played(screen)
    screen.grab()
    assert screen._divisor == 1
    updates = []
    original = screen.update

    def update(*args):
        updates.append(args)
        original(*args)

    monkeypatch.setattr(screen, 'update', update)
    other = QWindow()
    try:
        screen._follow_the_display(other)
        screen._tick()
        assert updates == [] and screen._tick_timer.isActive()
    finally:
        other.destroy()
        screen.skip()


def test_unavailable_native_underlay_capture_restores_cover_visibility(qtbot, monkeypatch):
    """A null capture response leaves no stale backdrop and still restores the cover."""
    from PySide6.QtGui import QGuiApplication

    class UnavailableCapture(QWidget):
        def grab(self, *args):
            return QPixmap()

    parent = UnavailableCapture()
    qtbot.addWidget(parent)
    parent.resize(420, 320)
    parent.show()
    screen = ls.LoadingScreen(total=1, parent=parent)
    screen.show()
    monkeypatch.setattr(QGuiApplication, 'platformName', lambda: 'cocoa')
    screen._capture_underlay()
    assert screen._underlay is None
    assert screen.isVisible()
