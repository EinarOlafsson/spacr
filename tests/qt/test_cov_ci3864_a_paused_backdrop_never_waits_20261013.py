"""A paused backdrop's still frame never makes the GUI thread wait.

``test_a_paused_backdrop_does_not_reshade_20261013`` pins that a backdrop
paused for a run blits a kept frame instead of reshading. These pin the two
ways that blit declines, both of which must hand the paint back to the
ordinary path rather than block or draw nothing:

* the engine is being shaded on another thread (its lock is taken): the
  GUI thread does not queue behind it for a still;
* the engine has nothing to show yet (``shade`` answers ``None``): no
  empty still is kept, so the next paint tries again.

The backdrop is built but never shown, so no shading thread of its own
competes with the one this test starts.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter                          # noqa: E402

from spacr.qt.widgets import ambient                                # noqa: E402

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"


@pytest.fixture
def paused(qtbot, monkeypatch):
    """A buffered backdrop that holds its still, as a paused one on screen."""
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    widget = ambient.AmbientWidget(None, theme=THEME, fps=30)
    widget.resize(160, 100)
    qtbot.addWidget(widget)
    assert isinstance(widget._engine, ambient._BufferedEngine)
    widget._holds_still = True
    widget._still = None
    yield widget
    widget.stop()


def _paint_still(widget):
    canvas = QImage(160, 100, QImage.Format_RGB32)
    painter = QPainter(canvas)
    try:
        return widget._paint_still(painter, 160, 100)
    finally:
        painter.end()


def test_a_still_is_not_waited_for_while_another_thread_shades(
        paused, monkeypatch):
    shaded = []
    monkeypatch.setattr(paused._engine, "shade",
                        lambda *size: shaded.append(size))
    taken, release = threading.Event(), threading.Event()

    def shade_elsewhere():
        with paused._engine_lock:
            taken.set()
            release.wait(10)

    worker = threading.Thread(target=shade_elsewhere)
    worker.start()
    try:
        assert taken.wait(10)
        assert _paint_still(paused) is False
    finally:
        release.set()
        worker.join(10)
    assert shaded == [], "the GUI thread shaded while the engine was busy"
    assert paused._still is None


def test_an_engine_with_nothing_to_show_keeps_no_still(paused, monkeypatch):
    monkeypatch.setattr(paused._engine, "shade", lambda *size: None)
    assert _paint_still(paused) is False
    assert paused._still is None

    frame = QImage(160, 100, QImage.Format_RGB32)
    frame.fill(0)
    blits = []
    monkeypatch.setattr(paused._engine, "shade", lambda *size: frame)
    monkeypatch.setattr(paused._engine, "blit",
                        lambda painter, image, w, h: blits.append(image))
    assert _paint_still(paused) is True
    assert paused._still is not None and paused._still[1] is frame
    assert blits == [frame]
