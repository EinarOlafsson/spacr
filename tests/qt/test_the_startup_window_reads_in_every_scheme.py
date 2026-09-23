"""The startup window's words can be read in every scheme (item 415).

Reported on Windows, 2026-09-15: "spacr is in white mode on windows and the
text is dark gray in the startup spacr window when in dark mode, so it is not
legable. when the app opens everything is fine so it is just the startup
window."

Reproduced offscreen before the fix. The OS scheme had nothing to do with it;
the loading screen never read it. Two faults made the picture:

* The splash looked its colours up with ``palette_for()``, which takes a
  theme and defaults to DARK. Whatever theme spaCR was about to open in, the
  startup window was black. A light spaCR, including the default "Follow
  system" on a light Windows, got a black startup window. ``MainWindow`` then
  filled its first frame with that same black, under the light theme's
  near-black window text: 1.09:1.
* The unlit phases were solved to only 3:1. On black that is ``#6e6e6e``
  (4.12:1), and at the start of a load all three phases are unlit, so the
  whole sentence is dark gray on black.

Each case supplies one reported OS scheme and one stored spaCR theme. The
application palette is set independently because spaCR must not mistake its
own applied palette for the operating system's preference. It builds the real
``LoadingScreen``, renders it, and measures every string the paint path draws
against the background the paint path filled. Contrast is measured twice:
from the pen it was drawn with, and from the pixels that came out.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPainter, QPalette  # noqa: E402

pytestmark = pytest.mark.qt

#: WCAG 2.x AA for body text. The strap line is 15 px, which is body text.
MINIMUM_CONTRAST = 4.5

#: Stand-ins for what a light and a dark OS colour scheme hand a Qt
#: application, for every role a widget's own text could fall back to.
OS_PALETTES = {
    "light": {"Window": "#f3f3f3", "WindowText": "#000000",
              "Base": "#ffffff", "Text": "#000000",
              "Button": "#fdfdfd", "ButtonText": "#000000"},
    "dark": {"Window": "#202020", "WindowText": "#ffffff",
             "Base": "#2b2b2b", "Text": "#ffffff",
             "Button": "#2d2d2d", "ButtonText": "#ffffff"},
}

OS_SCHEMES = tuple(OS_PALETTES)

#: The stored spaCR theme. "system" is the default, and it is the report's
#: "white mode on windows" when the OS is light.
SPACR_THEMES = ("light", "dark", "system")

#: How far the load has got: nothing lit, one phase lit, all lit.
PROGRESS = (0, 2, 6)


def _effective(spacr_theme: str, os_scheme: str) -> str:
    """The theme the main window opens in for this pair."""
    return os_scheme if spacr_theme == "system" else spacr_theme


@pytest.fixture
def scheme(qapp, monkeypatch):
    """Put an OS scheme and a stored spaCR theme in place; restore both."""
    from spacr.qt import preferences, theme

    store = preferences._settings()
    real_config = os.path.join(os.path.expanduser("~"), ".config") + os.sep
    if os.path.abspath(store.fileName()).startswith(real_config):
        pytest.skip("the preference store is the real one; this test writes "
                    "the theme and runs only inside tests/qt/conftest.py's "
                    "sandbox")
    saved_palette = QPalette(qapp.palette())
    keys = (preferences._KEY_THEME, preferences._KEY_THEME_FOLLOW_SYSTEM_CHOSEN)
    saved = {key: (store.contains(key), store.value(key)) for key in keys}

    def put(os_scheme: str, spacr_theme: str, applied: bool) -> str:
        monkeypatch.setattr(theme, "system_colour_scheme",
                            lambda app=None: os_scheme)
        palette = QPalette(qapp.palette())
        for role, colour in OS_PALETTES[os_scheme].items():
            palette.setColor(getattr(QPalette.ColorRole, role), QColor(colour))
        qapp.setPalette(palette)
        preferences.set_theme(spacr_theme)
        if applied:
            from spacr.qt.theme import apply_qpalette

            apply_qpalette(qapp, theme=preferences.resolve_effective_theme())
        effective = _effective(spacr_theme, os_scheme)
        assert preferences.resolve_effective_theme() == effective
        return effective

    yield put

    qapp.setPalette(saved_palette)
    store = preferences._settings()
    for key, (present, value) in saved.items():
        if present:
            store.setValue(key, value)
        else:
            store.remove(key)
    store.sync()


def _render(screen, monkeypatch):
    """Paint ``screen``; return the image, the fills and the drawn strings."""
    from spacr.qt.widgets import loading_screen

    fills: list = []
    strings: list = []

    class Recording(QPainter):
        """The widget's own painter, noting what it fills and writes."""

        def fillRect(self, *args):
            fills.append(QColor(args[-1]).name())
            return super().fillRect(*args)

        def drawText(self, *args):
            x, y, text = args
            metrics = self.fontMetrics()
            strings.append((text, self.pen().color().name(), x, y,
                            metrics.horizontalAdvance(text),
                            metrics.ascent(), metrics.descent()))
            return super().drawText(*args)

    monkeypatch.setattr(loading_screen, "QPainter", Recording)
    image = screen.grab().toImage()
    return image, fills, strings


def _strongest_pixel(image, box, background: str) -> float:
    """The highest contrast any pixel inside ``box`` has on ``background``."""
    from spacr.qt.theme import _contrast

    ratio = image.devicePixelRatio()
    x0, y0, x1, y1 = (int(round(v * ratio)) for v in box)
    seen: dict = {}
    best = 1.0
    for y in range(max(0, y0), min(image.height(), y1 + 1)):
        for x in range(max(0, x0), min(image.width(), x1 + 1)):
            name = image.pixelColor(x, y).name()
            if name not in seen:
                seen[name] = _contrast(name, background)
            best = max(best, seen[name])
    return best


@pytest.mark.parametrize("applied", (True, False),
                         ids=("theme-applied", "theme-not-yet-applied"))
@pytest.mark.parametrize("spacr_theme", SPACR_THEMES)
@pytest.mark.parametrize("os_scheme", OS_SCHEMES)
def test_every_word_on_the_startup_window_reads_against_its_background(
        qapp, qtbot, monkeypatch, scheme, os_scheme, spacr_theme, applied):
    """Bright words on a dark theme, dark words on a light one, 4.5:1 or more.

    ``theme-applied`` is the launch order: ``launch`` applies the stored
    theme to the application before ``MainWindow`` builds the loading screen.
    ``theme-not-yet-applied`` builds it while the application still wears the
    OS palette. The words must not depend on which of the two came first.
    """
    from spacr.qt.theme import _contrast, _relative_luminance, palette_for
    from spacr.qt.widgets.loading_screen import LoadingScreen

    effective = scheme(os_scheme, spacr_theme, applied)
    window_bg = palette_for(effective)["bg"].lower()
    failures = []
    for done in PROGRESS:
        screen = LoadingScreen(total=PROGRESS[-1])
        qtbot.addWidget(screen)
        screen.resize(1200, 320)
        screen.advance(done)
        image, fills, strings = _render(screen, monkeypatch)

        assert fills, "the loading screen painted no background"
        background = fills[0].lower()
        assert image.pixelColor(2, 2).name().lower() == background
        assert background == window_bg, (
            f"OS {os_scheme}, spaCR {spacr_theme}: the startup window is "
            f"painted {background}, but the main window opens in the "
            f"{effective} theme on {window_bg}")
        assert strings, "the loading screen wrote nothing"

        for text, pen, x, y, advance, ascent, descent in strings:
            pen_ratio = _contrast(pen, background)
            pixel_ratio = _strongest_pixel(
                image, (x, y - ascent, x + advance, y + descent), background)
            if effective == "light":
                right_way_round = (_relative_luminance(pen)
                                   < _relative_luminance(background))
            else:
                right_way_round = (_relative_luminance(pen)
                                   > _relative_luminance(background))
            if (pen_ratio < MINIMUM_CONTRAST
                    or pixel_ratio < MINIMUM_CONTRAST
                    or not right_way_round):
                failures.append(
                    f"{done}/{PROGRESS[-1]} done, {text.strip() or '->'!r}: "
                    f"pen {pen} on {background} = {pen_ratio:.2f}:1, "
                    f"brightest pixel {pixel_ratio:.2f}:1"
                    + ("" if right_way_round else ", wrong way round"))

    assert not failures, (
        f"OS {os_scheme}, spaCR {spacr_theme} ({effective}): startup text "
        f"below {MINIMUM_CONTRAST}:1 --\n  " + "\n  ".join(failures))


def test_the_loading_screen_has_no_widget_of_its_own_to_inherit_a_colour(
        qapp, qtbot):
    """Every word is painted with an explicit pen, never a widget palette.

    A child label would take its text colour from the widget palette, which
    follows the OS scheme until spaCR's theme reaches it. If one is ever
    added, the test above must be taught to measure it too.
    """
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.loading_screen import LoadingScreen

    screen = LoadingScreen(total=3)
    qtbot.addWidget(screen)

    assert screen.findChildren(QWidget) == []


@pytest.mark.parametrize("spacr_theme", SPACR_THEMES)
@pytest.mark.parametrize("os_scheme", OS_SCHEMES)
def test_the_first_frame_behind_the_startup_window_carries_readable_text(
        qapp, scheme, os_scheme, spacr_theme):
    """``MainWindow`` fills its first frame with the splash background.

    It does that before its stylesheet reaches it, and its children inherit
    that Window colour together with the application's WindowText. Before
    415 that fill was the dark theme's black under every theme, so a light
    spaCR showed #0d0e10 text on #000000.
    """
    from spacr.qt.theme import _contrast, palette_for
    from spacr.qt.widgets.loading_screen import splash_role

    effective = scheme(os_scheme, spacr_theme, True)
    fill = splash_role("splash_bg", "#000000").lower()
    text = qapp.palette().color(QPalette.ColorRole.WindowText).name()

    assert fill == palette_for(effective)["bg"].lower(), (
        f"OS {os_scheme}, spaCR {spacr_theme}: the first frame is {fill}, "
        f"not the {effective} theme's window colour")
    assert _contrast(text, fill) >= MINIMUM_CONTRAST, (
        f"OS {os_scheme}, spaCR {spacr_theme}: window text {text} on the "
        f"first frame's {fill} is {_contrast(text, fill):.2f}:1")
