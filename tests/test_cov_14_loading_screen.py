"""The splash paints whatever the theme and the disk give it.

The loading screen is the FIRST thing drawn, before the theme is resolved and
before anything else could report a problem, so every input it reads has a
literal behind it: an unparseable colour, a missing logo file, a translation
catalog that has not been imported yet. A traceback from any of them replaces
the splash with a crash on a window the user has not seen yet.

Progress has the same shape. An unknown total is reported as zero progress
rather than dividing by it, and the last phase lights only at completion, so
the sentence never claims spaCR has finished loading while work is running.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_an_unparseable_ink_colour_falls_back_to_white(qapp, monkeypatch):
    """A palette entry Qt cannot parse still yields a paintable colour.

    An invalid QColor paints as black on the splash's black ground, which is
    a splash with no text on it.
    """
    import spacr.qt.theme as theme
    from spacr.qt.widgets import loading_screen

    monkeypatch.setattr(theme, "palette_for",
                        lambda *a, **k: {"splash_ink": "not-a-colour"})

    colour = loading_screen._ink(200)

    assert colour.isValid()
    assert (colour.red(), colour.green(), colour.blue()) == (255, 255, 255)
    assert colour.alpha() == 200


def test_a_readable_ink_colour_is_used(qapp, monkeypatch):
    """A parseable palette entry is what gets painted."""
    import spacr.qt.theme as theme
    from spacr.qt.widgets import loading_screen

    monkeypatch.setattr(theme, "palette_for",
                        lambda *a, **k: {"splash_ink": "#102030"})

    colour = loading_screen._ink(128)

    assert (colour.red(), colour.green(), colour.blue()) == (16, 32, 48)


def test_a_new_total_repaints_and_an_unchanged_one_does_not(qapp):
    """The denominator is only re-set when it actually changed."""
    from spacr.qt.widgets.loading_screen import LoadingScreen

    screen = LoadingScreen(total=0)
    screen.set_total(4)
    screen.advance()
    screen.advance()

    assert screen.fraction() == 0.5

    screen.set_total(4)

    assert screen.fraction() == 0.5


def test_a_negative_total_is_clamped_to_unknown(qapp):
    """A negative denominator is "unknown", not a negative fraction."""
    from spacr.qt.widgets.loading_screen import LoadingScreen

    screen = LoadingScreen(total=3)
    screen.set_total(-5)

    assert screen.fraction() == 0.0


def test_an_unknown_total_reports_no_progress(qapp):
    """Zero steps means zero progress rather than a division by zero."""
    from spacr.qt.widgets.loading_screen import LoadingScreen

    screen = LoadingScreen(total=0)
    screen.advance()

    assert screen.fraction() == 0.0
    assert screen.lit_phases() == 0


def test_the_last_phase_lights_only_at_completion(qapp):
    """All phases light at 1.0 and not before.

    A lit final phase while imports are still running tells the user spaCR is
    ready when it is not.
    """
    from spacr.qt.widgets.loading_screen import LoadingScreen, STRAP_PHASES

    screen = LoadingScreen(total=3)
    screen.advance(2)

    assert screen.lit_phases() < len(STRAP_PHASES)

    screen.advance(3)

    assert screen.fraction() == 1.0
    assert screen.lit_phases() == len(STRAP_PHASES)


def test_a_logo_that_cannot_be_read_does_not_stop_the_splash(qapp,
                                                             monkeypatch):
    """A failing logo load leaves the screen drawable with no mark."""
    from spacr.qt.widgets import loading_screen

    def _explode(*args, **kwargs):
        raise OSError("the resource folder is not readable")

    monkeypatch.setattr(loading_screen, "QPixmap", _explode)

    screen = loading_screen.LoadingScreen(total=2)

    assert screen._logo is None


def test_a_translation_catalog_that_fails_leaves_the_text_alone(qapp,
                                                               monkeypatch):
    """Untranslatable text is shown as written, not as an exception."""
    import spacr.qt.i18n as i18n
    from spacr.qt.widgets import loading_screen

    def _explode(text):
        raise RuntimeError("the catalog is not loaded")

    monkeypatch.setattr(i18n, "tr", _explode)

    assert loading_screen._translate("Loading spaCR") == "Loading spaCR"


def test_a_working_catalog_is_used(qapp, monkeypatch):
    """The same helper does go through the catalog when it works."""
    import spacr.qt.i18n as i18n
    from spacr.qt.widgets import loading_screen

    monkeypatch.setattr(i18n, "tr", lambda text: f"[{text}]")

    assert loading_screen._translate("Loading spaCR") == "[Loading spaCR]"
