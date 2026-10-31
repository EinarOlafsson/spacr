"""Mask Generation's cards, composited into the field, look as they did.

The field's shading thread composites the translucent card backgrounds into
each frame and the cards paint transparent while that frame is on screen
(:class:`spacr.qt.widgets.ambient._CardTints`). On the real Mask screen, in
the dark and light themes and at every pane opacity, the window must be
pixel-identical to the classic paint, and a scroll or a resize must put the
cards back to painting themselves before the next paint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QScrollArea, QWidget  # noqa: E402

from spacr.qt import preferences  # noqa: E402

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


@pytest.fixture
def themed(qapp, tmp_path, monkeypatch):
    """A private preference store; the application's look put back after."""
    from spacr.qt.widgets.field_fade import (invalidate_field_fade,
                                             uninstall_field_fade)

    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()

    def apply(theme="dark", opacity=None):
        preferences.set_theme_choice(theme)
        if opacity is not None:
            preferences.set_pane_opacity(opacity)
        preferences.apply_preferences_to_app(qapp)

    yield apply
    uninstall_field_fade(qapp)
    invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _mask_window(qtbot):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1600, 1000)
    window.show()
    qtbot.waitExposed(window)
    from spacr.qt.widgets.loading_screen import LoadingScreen

    qtbot.waitUntil(lambda: not window.findChildren(LoadingScreen),
                    timeout=20000)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_card_tints"):
        pytest.skip("the window backdrop is not the field in this setup")
    window._on_nav_selected("mask")
    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()
    return window, backdrop


def _composited(backdrop, qtbot):
    """Wait, with the field ticking, until the cards stay composited."""
    tints = backdrop._card_tints
    for _attempt in range(60):
        qtbot.waitUntil(lambda: tints.active, timeout=20000)
        qtbot.wait(300)
        if tints.active:
            return tints
    raise AssertionError("the cards never stayed composited")


def _grab(window):
    return window.grab().toImage().convertToFormat(QImage.Format_RGB32)


def _same_as_classic(window, backdrop, qtbot):
    """Freeze one frame and grab it composited, then classic."""
    tints = _composited(backdrop, qtbot)
    assert tints._cards, "Mask has no composited cards"
    backdrop._timer.stop()
    backdrop._producer_box[0].stop()
    composited = _grab(window)
    assert backdrop._last_composed[0] is tints.plan
    tints.invalidate()
    classic = _grab(window)
    backdrop.stop()
    backdrop.start()
    return composited == classic


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_mask_composited_is_pixel_identical_to_classic(themed, qtbot, theme):
    themed(theme)
    window, backdrop = _mask_window(qtbot)
    assert _same_as_classic(window, backdrop, qtbot)


def test_every_pane_opacity_composites_identically(themed, qtbot):
    themed("dark", 0.0)
    window, backdrop = _mask_window(qtbot)
    for opacity in (0.0, 0.25, 0.5, 0.75, 1.0):
        themed("dark", opacity)
        assert _same_as_classic(window, backdrop, qtbot), opacity


def test_a_scroll_or_resize_paints_classic_until_it_settles(themed, qtbot):
    themed("dark")
    window, backdrop = _mask_window(qtbot)
    tints = _composited(backdrop, qtbot)
    cards = list(tints._cards)
    box = next(area for area in window.findChildren(QScrollArea)
               if area.objectName() == "SettingsBox")
    bar = box.verticalScrollBar()
    assert bar.maximum() > 0, "the settings box does not scroll"
    bar.setValue(bar.value() + 40)
    assert not tints.active and tints.plan is None
    assert not any(card.testAttribute(Qt.WA_NoSystemBackground)
                   for card in cards)
    assert backdrop._producer_box[0].cards is None
    backdrop.repaint()
    assert backdrop._last_composed is None or not tints.active
    assert _same_as_classic(window, backdrop, qtbot)
    tints = _composited(backdrop, qtbot)
    window.resize(1500, 950)
    assert not tints.active
    assert _same_as_classic(window, backdrop, qtbot)
