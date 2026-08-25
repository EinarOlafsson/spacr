"""Theme lookups degrade to a legible default instead of raising in a paint.

Almost everything in :mod:`spacr.qt.theme` is called from a ``paintEvent`` or
from stylesheet generation, and the preference store it consults is imported
lazily because `preferences` imports this module back. An exception on that
path would be a traceback per repaint. Every branch driven here is the
fallback that keeps a widget drawn: dark theme, the designed scrim, an
unscaled font, and an un-restyled but still working application.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QAbstractScrollArea, QApplication, QWidget

from spacr.qt import preferences as prefs
from spacr.qt import theme


def _raise(*args, **kwargs):
    raise RuntimeError("the preference store is unavailable")


@pytest.fixture()
def no_preferences(monkeypatch):
    """Make every preference lookup this module does fail."""
    monkeypatch.setattr(prefs, "resolve_effective_theme", _raise)
    monkeypatch.setattr(prefs, "get_pane_opacity", _raise)
    monkeypatch.setattr(prefs, "get_font_scale", _raise)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_a_surface_colour_survives_an_unreadable_preference_store(
        no_preferences):
    """With no theme and no opacity readable, the dark surface is used.

    Dark is the safe default: the light palette's surfaces on a dark
    backdrop are the one combination that leaves text unreadable.
    """
    colour = theme.pane_surface("surface_alt")

    assert colour == theme.pane_surface("surface_alt", theme="dark",
                                        opacity=None)


def test_a_block_surface_survives_an_unreadable_preference_store(
        no_preferences):
    """Blocks take their opacity as an argument, so only the theme is looked up."""
    assert theme.block_surface("surface_alt", opacity=0.5) == \
        theme.block_surface("surface_alt", theme="dark", opacity=0.5)


def test_a_painted_panel_colour_survives_an_unreadable_preference_store(
        no_preferences):
    """The QColor accessor is what a custom-painted canvas fills with."""
    colour = theme.panel_qcolor("surface")

    assert colour == theme.panel_qcolor("surface", theme="dark", opacity=None)
    assert colour.isValid()


def test_painting_a_panel_survives_an_unreadable_preference_store(
        qtbot, no_preferences):
    """`paint_panel` is called inside paintEvent; it must not raise there."""
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(40, 30)
    image = QImage(40, 30, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        theme.paint_panel(painter, widget)
    finally:
        painter.end()

    assert image.pixelColor(20, 15).alpha() > 0


# ---------------------------------------------------------------------------
# Contrast search
# ---------------------------------------------------------------------------

def test_ink_that_can_never_reach_the_target_is_drawn_fully_opaque():
    """Ink the same colour as its background never reaches any contrast.

    Returning the top of the range keeps the text as legible as that pair
    allows, rather than leaving it at the dim floor for no benefit.
    """
    assert theme.splash_dim_alpha("#101010", "#101010") == 255


def test_ink_that_reads_easily_stays_dim():
    """The control: an unlit phase is meant to look unreached."""
    alpha = theme.splash_dim_alpha("#ffffff", "#000000")

    assert 110 <= alpha < 255


@pytest.mark.xfail(strict=True,
                   reason="theme._channels is defined twice; the second "
                          "definition shadows the first, so the malformed-"
                          "colour fallback that first one implements is "
                          "unreachable and the helper raises instead")
def test_a_malformed_colour_does_not_raise_out_of_a_composite():
    """Compositing is paint-time work, so a bad colour must degrade.

    The module's own fallback answers white for an unreadable colour, which
    keeps a splash phase visible; raising turns one bad palette entry into a
    traceback on every repaint.
    """
    assert theme._composite("not-a-colour", "#000000", 128) == "#7f7f7f"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def test_a_font_size_that_is_not_a_size_falls_back_to_the_body_size():
    """An unknown role must still produce a readable number of pixels."""
    assert theme.font_px("no-such-role", scale=1.0) == theme.FONT_SIZE["body"]
    assert theme.font_px(None, scale=1.0) == theme.FONT_SIZE["body"]


def test_a_font_scale_that_cannot_be_read_leaves_the_text_unscaled(
        no_preferences):
    """An unscaled label is cosmetic; an exception in a paint is not."""
    assert theme.font_px("body") == theme.FONT_SIZE["body"]


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

def test_marking_nothing_is_not_an_error():
    """Callers pass optional widgets straight through, so None is skipped."""
    theme.mark_surface(None)

    assert theme.is_surface(None) is False


def test_a_scroll_area_with_no_viewport_still_gets_marked(qtbot):
    """The area itself is tagged even when there is no viewport to tag.

    A scroll area normally paints through its viewport, so the viewport is
    the target; one without must not lose the tag altogether.
    """
    class _NoViewport(QAbstractScrollArea):
        def viewport(self):
            return None

    area = _NoViewport()
    qtbot.addWidget(area)

    theme.mark_surface(area)

    assert theme.is_surface(area) is True


# ---------------------------------------------------------------------------
# The widget QSS registry
# ---------------------------------------------------------------------------

def test_a_widget_qss_module_that_will_not_import_is_skipped(monkeypatch):
    """One broken module must not cost every other registered block.

    The loader runs once per process, and an exception in it would leave the
    whole registry half-filled with no way to refill it.
    """
    monkeypatch.setattr(theme, "_QSS_REGISTRARS_LOADED", False)
    monkeypatch.setattr(theme, "WIDGET_QSS_MODULES",
                        ("spacr.qt.widgets.there_is_no_such_module",))

    assert theme.load_widget_qss_registrars() == ()


def test_a_widget_qss_block_needs_a_name():
    """An unnamed block could never be found again, or checked for."""
    with pytest.raises(ValueError, match="needs a name"):
        theme.register_widget_qss("", lambda palette, opacity: "")


# ---------------------------------------------------------------------------
# Re-applying the stylesheet
# ---------------------------------------------------------------------------

@pytest.fixture()
def styled_app(qtbot, monkeypatch):
    """An application carrying a stylesheet with no registered-block markers."""
    app = QApplication.instance()
    before = app.styleSheet()
    app.setStyleSheet("QWidget { color: red; }")
    yield app
    app.setStyleSheet(before)


def test_with_no_application_there_is_nothing_to_restyle(monkeypatch):
    """A screen built before the application exists asks and is told no."""
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    assert theme.ensure_widget_qss_applied("SomeBlock") is False


def test_without_the_preference_module_the_sheet_is_left_alone(styled_app,
                                                               monkeypatch):
    """Regenerating the sheet needs the preferences; without them, no change."""
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if "preferences" in name or "apply_preferences_to_app" in (fromlist or ()):
            raise ImportError("preferences are unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)

    assert theme.ensure_widget_qss_applied("SomeBlock") is False


def test_a_restyle_that_fails_reports_failure_rather_than_raising(styled_app,
                                                                  monkeypatch):
    """The caller is a constructor; it must not die because styling did."""
    monkeypatch.setattr(prefs, "apply_preferences_to_app", _raise)

    assert theme.ensure_widget_qss_applied("SomeBlock") is False
    assert styled_app.styleSheet() == "QWidget { color: red; }"
