"""The splash strap's arrows are stroked, so no font fallback can thin them.

The strap fonts rarely carry U+2192 in bold. On Linux Qt drew the arrow from
Noto Sans Math Regular, a hairline, and at the 90 % default scale
(7884c4a1f) the unlit arrow kept 4 legible pixels on the light splash:
test_sampler_finds_every_real_splash_phrase_and_arrow[0-light] and [2-light]
failed on nightly. The arrow is now drawn as strokes in its ink, inside the
advance the glyph used to take, with a whole-pixel stem on the pixel grid:
CI's DejaVu Sans at a smaller font scale left a 1.25 px antialiased stroke
under 16 legible pixels. Checked at the default scale and at the 50 %
floor a user can choose.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPainter  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("scale", [None, 0.5])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_no_arrow_glyph_is_drawn_and_each_arrow_reads(qapp, scheme, scale,
                                                      monkeypatch):
    from spacr.qt import preferences, theme
    from spacr.qt.widgets.loading_screen import LoadingScreen
    from tools.validate_native_startup_contrast import _splash_measurements

    texts = []
    real = QPainter.drawText

    def recording(self, *args):
        texts.append(args[-1] if args and isinstance(args[-1], str) else "")
        return real(self, *args)

    old = (preferences.get_theme(), preferences.get_gui_scale(),
           preferences.get_font_scale())
    try:
        preferences.set_theme(scheme)
        preferences.set_gui_scale(
            preferences.DEFAULT_GUI_SCALE if scale is None else scale)
        preferences.set_font_scale(
            preferences.DEFAULT_FONT_SCALE if scale is None else scale)
        screen = LoadingScreen(total=6)
        screen.resize(1200, 450)
        monkeypatch.setattr(QPainter, "drawText", recording)
        image = screen.grab().toImage()
        monkeypatch.undo()
        assert texts and not any("→" in text for text in texts)
        arrows = [row for row in _splash_measurements(
            screen, image, theme.palette_for(scheme)) if "→" in row["text"]]
        assert len(arrows) == 2
        assert all(row["readable_pixels"] >= 16 for row in arrows)
        screen.close()
    finally:
        preferences.set_theme(old[0])
        preferences.set_gui_scale(old[1])
        preferences.set_font_scale(old[2])
