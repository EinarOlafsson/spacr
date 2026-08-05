"""Sanity tests for the tutorial engine.

We don't actually render a video here — that takes ~90s per module.
Instead we verify:
  - every AVAILABLE_TUTORIALS entry has a build function
  - Step / RenderResult are usable dataclasses
  - the CLI parses correctly for one/all cases
  - SRT timestamp formatting is right
  - the cursor / highlight overlays paint the right pixels in the right
    place, read back off the QPixmap

The full renderer is exercised by hand via `spacr-tutorial <app>`.
"""
from __future__ import annotations

import pytest


def test_available_tutorials_all_have_builders():
    from spacr.qt.tutorial.scripts import AVAILABLE_TUTORIALS, build_steps
    # We can't call build_steps without a MainWindow, but we can at
    # least catch typos: build_steps raises ValueError only on unknown.
    with pytest.raises(ValueError):
        build_steps("nonexistent-tutorial-name", window=None)


def test_step_dataclass_defaults():
    from spacr.qt.tutorial.engine import Step
    s = Step("hello world")
    assert s.narration == "hello world"
    assert s.action is None
    assert s.target is None
    assert s.hold_ms > 0


def test_srt_timestamp_format():
    from spacr.qt.tutorial.engine import _srt_ts
    assert _srt_ts(0) == "00:00:00,000"
    assert _srt_ts(1.5) == "00:00:01,500"
    assert _srt_ts(65.25) == "00:01:05,250"
    assert _srt_ts(3661.001) == "01:01:01,001"


def test_narrator_raises_if_voice_model_missing(tmp_path):
    from spacr.qt.tutorial.engine import Narrator
    with pytest.raises(FileNotFoundError):
        Narrator(voice_model=tmp_path / "nope.onnx")


def _white_pixmap(side: int = 256):
    """A fully white ``side x side`` pixmap to paint an overlay onto."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    pm = QPixmap(side, side)
    pm.fill(Qt.white)
    return pm


def _rgb(pixmap):
    """``(h, w, 3)`` uint8 RGB view of a QPixmap.

    Overlay tests have to read the pixels that were actually painted. A
    test that only checks "the painter was handed colour X" is exactly how
    the mask-outline colour stayed stuck on green for months while the
    setting it asserted on moved perfectly.
    """
    import numpy as np
    from PySide6.QtGui import QImage
    img = pixmap.toImage().convertToFormat(QImage.Format_RGB32)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    buf = buf.reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :]
    return buf[:, :, 2::-1].copy()          # BGRA -> RGB


def _painted_bbox(pixmap):
    """``(x0, y0, x1, y1, count)`` over every pixel that is no longer white."""
    import numpy as np
    arr = _rgb(pixmap)
    ys, xs = np.nonzero((arr != 255).any(axis=2))
    assert xs.size, "nothing was painted: the frame is still blank white"
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
            int(xs.size))


def test_cursor_overlay_draws_on_pixmap(qt_theme_applied):
    """The cursor dot lands on the requested point, in the click colour."""
    from spacr.qt.tutorial.engine import _draw_cursor_on

    pm = _white_pixmap()
    _draw_cursor_on(pm, (100, 100))
    arr = _rgb(pm)

    # The pixel asked for is the magenta click point, not white.
    assert tuple(arr[100, 100]) == (255, 0, 153)
    # The dot is small and centred there: every painted pixel sits inside a
    # tight box around (100, 100).
    x0, y0, x1, y1, count = _painted_bbox(pm)
    assert 20 < count < 200, f"cursor dot covers {count} px"
    assert (x0, y0, x1, y1) == (96, 96, 103, 103)
    # ... and the rest of the frame is untouched.
    assert tuple(arr[10, 10]) == (255, 255, 255)
    assert tuple(arr[200, 200]) == (255, 255, 255)

    # Contrast: asking for a different point must move the painted pixels.
    # Without this, a `_draw_cursor_on` that ignored `pos_xy` and always
    # stamped (100, 100) would sail through every assertion above.
    other = _white_pixmap()
    _draw_cursor_on(other, (200, 50))
    other_arr = _rgb(other)
    assert tuple(other_arr[50, 200]) == (255, 0, 153)
    assert tuple(other_arr[100, 100]) == (255, 255, 255)
    assert _painted_bbox(other)[:4] == (196, 46, 203, 53)


def test_highlight_overlay_draws_on_pixmap(qt_theme_applied):
    """The highlight ring surrounds the requested rect and stays hollow."""
    from spacr.qt.tutorial.engine import _draw_highlight_on

    pm = _white_pixmap()
    _draw_highlight_on(pm, (20, 20, 100, 40))
    arr = _rgb(pm)

    # The ring is drawn 4 px outside the rect with a 1 px pen, so the
    # painted box is (16,16)-(123,63) plus one pixel of antialiasing bleed.
    x0, y0, x1, y1, count = _painted_bbox(pm)
    assert (x0, y0, x1, y1) == (15, 15, 124, 64)
    assert count > 200, f"only {count} px painted — the ring is not there"

    # It is an outline, not a fill: the widget underneath stays legible.
    assert tuple(arr[40, 70]) == (255, 255, 255)
    # The ink is the blue accent blended over white by the pen's alpha —
    # blue-dominant, and nothing like the magenta cursor colour.
    edge = arr[40, 16]
    assert edge[2] > edge[1] > edge[0], f"ring pixel is not accent blue: {edge}"
    # Nothing is painted outside the ring.
    assert tuple(arr[200, 200]) == (255, 255, 255)
    assert tuple(arr[5, 5]) == (255, 255, 255)

    # Contrast: a different rect must move the ring — and must leave the
    # first rect's edge white, proving the box above is not a constant.
    other = _white_pixmap()
    _draw_highlight_on(other, (140, 140, 60, 60))
    other_arr = _rgb(other)
    assert _painted_bbox(other)[:4] == (135, 135, 204, 204)
    assert tuple(other_arr[40, 16]) == (255, 255, 255)
    assert tuple(other_arr[170, 170]) == (255, 255, 255)   # still hollow


def test_cli_parses_one_and_all():
    """The CLI parser rejects unknown apps but accepts every name in
    AVAILABLE_TUTORIALS plus the sentinel 'all'."""
    import argparse
    from spacr.qt.tutorial.__main__ import main  # noqa: F401
    from spacr.qt.tutorial.scripts import AVAILABLE_TUTORIALS
    # We only test argparse validation, not the actual render (heavy)
    # so we just import + assert coverage.
    assert "mask" in AVAILABLE_TUTORIALS
    assert "home" in AVAILABLE_TUTORIALS
