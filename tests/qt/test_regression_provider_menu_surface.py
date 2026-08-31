"""Provider chevrons beside the AI toggle have no black plate.

The control is a ``QToolButton``, not a ``QPushButton``.  Without a rule of
its own Qt's native dark-theme primitive fills most of its rectangle with the
palette's pure-black Window colour.  It is especially conspicuous between the
Regression inputs and Run controls when the surrounding page is translucent;
Annotate builds the same provider control on its bottom row. These tests render
both production controls: object names and stylesheet text alone cannot prove
that the native primitive stopped painting.
"""
from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QSettings
from PySide6.QtGui import QColor, QImage, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.theme import apply_qpalette, stylesheet

#: Nothing in the spaCR palette is magenta, so whatever survives this colour
#: in a direct widget render was painted by the provider control itself.
SENTINEL = QColor(255, 0, 255)


@pytest.fixture
def provider_surface_app(qt_theme_applied, monkeypatch, tmp_path):
    """A deterministic dark theme without reading real provider settings."""
    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr("spacr.qt.ai.configured_providers", lambda: [])
    preferences.set_theme("dark")
    preferences.set_ambient_enabled(False)
    preferences.set_pane_opacity(0.60)
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(
        stylesheet("dark", surface_opacity=0.60))
    return qt_theme_applied


def _assert_provider_chevron_has_no_plate(
        screen, qtbot, provider_surface_app, *, size):
    """Show one production screen and measure only its provider control."""
    qtbot.addWidget(screen)
    screen.resize(*size)
    screen.show()
    QApplication.processEvents()

    button = screen._ai_menu_btn
    assert button.width() > 0 and button.height() > 0

    image = QImage(button.size(), QImage.Format_ARGB32)
    image.fill(SENTINEL)
    button.render(
        image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    colours = Counter(
        image.pixelColor(x, y).name().lower()
        for y in range(button.height())
        for x in range(button.width())
    )
    total = image.width() * image.height()
    painted = total - colours[SENTINEL.name().lower()]

    # Before the dedicated tool-button rule the native primitive painted
    # essentially the whole rectangle. With the rule, only the arrow glyph
    # paints (about 9% here). The broad middle is deliberate portability
    # room, not enough room for any rectangular plate to return.
    assert painted / total < 0.50, (
        f"the provider chevron painted {painted / total:.1%} of its own "
        "rectangle, which is a plate rather than an arrow")

    assert button.objectName() == "AiProviderMenuButton"
    selector = "QToolButton#AiProviderMenuButton {"
    sheet = provider_surface_app.styleSheet()
    assert selector in sheet
    resting_rule = sheet.split(selector, 1)[1]
    resting_rule = resting_rule.split("}", 1)[0]
    assert "background: transparent" in resting_rule


def test_regression_provider_chevron_does_not_render_a_black_box(
        qtbot, provider_surface_app):
    """Regression transmits the action-row surface instead of native black."""
    _assert_provider_chevron_has_no_plate(
        AppScreen("regression"), qtbot, provider_surface_app,
        size=(1600, 1050))


def test_annotate_provider_chevron_does_not_render_a_black_box(
        qtbot, provider_surface_app):
    """Annotate's duplicated provider control follows the same theme rule."""
    from spacr.qt.screens.annotate import AnnotateScreen

    _assert_provider_chevron_has_no_plate(
        AnnotateScreen(), qtbot, provider_surface_app, size=(1400, 900))
