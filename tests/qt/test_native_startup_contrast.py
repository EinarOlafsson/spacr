"""Local sampler checks; these do not claim native Windows acceptance."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtGui import QColor, QImage

from tools.validate_native_startup_contrast import (
    _contrast,
    _require_hosted_windows,
    _runner_preferences,
    _splash_measurements,
    _text_pixels,
)


@pytest.mark.parametrize("scheme", ["dark", "light"])
@pytest.mark.parametrize("progress", [0, 2, 6])
def test_sampler_finds_every_real_splash_phrase_and_arrow(qapp, scheme, progress):
    from spacr.qt import preferences, theme
    from spacr.qt.widgets.loading_screen import LoadingScreen

    old = preferences.get_theme()
    try:
        preferences.set_theme(scheme)
        screen = LoadingScreen(total=6)
        screen.resize(1200, 450)
        screen.advance(progress)
        evidence = _splash_measurements(screen, screen.grab().toImage(), theme.palette_for(scheme))
        assert len(evidence) == 5
        assert all(row["readable_pixels"] >= 8 for row in evidence)
        assert sum(row["role"] == "splash_ink" for row in evidence) == {0: 0, 2: 1, 6: 5}[progress]
        screen.close()
    finally:
        preferences.set_theme(old)


def test_readability_rejects_single_bright_pixel_blank_and_low_contrast():
    image = QImage(30, 20, QImage.Format.Format_RGB32)
    for colour in ("#000000", "#6e6e6e"):
        image.fill(QColor(colour))
        with pytest.raises(AssertionError, match="readable ink pixels"):
            _text_pixels(image, [0, 0, 30, 20], (0, 0, 0))
    image.fill(QColor("#000000"))
    image.setPixelColor(2, 2, QColor("white"))
    with pytest.raises(AssertionError, match="Only 1 readable"):
        _text_pixels(image, [0, 0, 30, 20], (0, 0, 0))


def test_sampler_rejects_a_clipped_capture():
    image = QImage(30, 20, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    with pytest.raises(AssertionError, match="outside capture"):
        _text_pixels(image, [0, 0, 31, 20], (0, 0, 0))


def test_contrast_uses_wcag_linear_luminance():
    assert _contrast((255, 255, 255), (0, 0, 0)) == 21
    assert _contrast((110, 110, 110), (0, 0, 0)) == pytest.approx(4.118529, abs=.00001)


@pytest.mark.skipif(sys.platform == "win32", reason="Linux/macOS refusal is the local safety check")
def test_local_desktop_cannot_run_registry_changing_acceptance():
    with pytest.raises(RuntimeError, match="disposable GitHub-hosted Windows"):
        _require_hosted_windows()
    with pytest.raises(RuntimeError, match="disposable GitHub-hosted Windows"):
        with _runner_preferences():
            pytest.fail("The preference guard must run before opening any store")
