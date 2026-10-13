"""Mask Generation's field chrome and switches, composited, look as they did.

Since 2026-10-12 the field's shading thread composites, besides the card
backgrounds, the fade's chrome under every form field without the focus and
the track and knob of every settings switch
(:func:`spacr.qt.widgets.ambient._static_part`), and those widgets leave
them out while a composited frame is on screen. On the real Mask screen, in
the dark, light and high-contrast themes at pane opacity 0, 50 and 100 %,
the window must be pixel-identical to the classic paint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

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


def _mask_window(qtbot):
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


@pytest.mark.parametrize("theme", ["dark", "light", "high_contrast"])
def test_mask_fields_and_switches_composite_identically(themed, qtbot,
                                                        theme):
    from spacr.qt.widgets.field_fade import FIELD_TYPES, _LENT
    from spacr.qt.widgets.toggle import Toggle

    themed(theme, 0.0)
    window, backdrop = _mask_window(qtbot)
    for opacity in (0.0, 0.5, 1.0):
        themed(theme, opacity)
        tints = _composited(backdrop, qtbot)
        fields = [w for w in tints._parts if isinstance(w, FIELD_TYPES)]
        switches = [w for w in tints._parts if isinstance(w, Toggle)]
        assert fields and switches, (theme, opacity, tints._parts)
        assert all(field in _LENT for field in fields)
        assert all(switch._lent_to is not None for switch in switches)
        backdrop._timer.stop()
        backdrop._producer_box[0].stop()
        composited = _grab(window)
        assert backdrop._last_composed[0] is tints.plan
        tints.invalidate()
        classic = _grab(window)
        backdrop.stop()
        backdrop.start()
        assert composited == classic, (theme, opacity)
