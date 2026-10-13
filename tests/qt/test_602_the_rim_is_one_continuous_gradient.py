"""The lit rim round a glassed window is one continuous gradient.

Reported 2026-09-30: "the rim arround the window perifery looks like it is
made up of many small boxes. please make it a continuous gradient with no
boxes". The run was stroked as hundreds of short lines with round caps, and
every cap overlapped the next, so the brightness along the rim rose and fell
every few pixels. These tests render the card offscreen and walk the lit run
pixel by pixel: a smooth run brightens to one peak and fades once, so its
profile turns over a handful of times at most, where the beaded one turned
over dozens of times.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget


def _render(theme, width, height, mode, monkeypatch):
    """The card at ``width`` x ``height`` with its run on the top edge."""
    import spacr.qt.preferences as prefs
    from spacr.qt.theme import palette_for
    from spacr.qt.widgets.setup_card import SetupCard

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: theme)
    page = QColor(palette_for(theme)["bg"])

    class Host(QWidget):
        def paintEvent(self, event):                 # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), page)
            painter.end()

    host = Host()
    host.resize(width, height)
    card = SetupCard(host, mode=mode, arc=280, align="centre")
    card.setGeometry(0, 0, width, height)
    card._aim_at_the_cursor = lambda: False
    middle_of_top = card.perimeter_position(QPointF(width / 2.0, -50.0))
    card._at = card._towards = float(middle_of_top)
    image = host.grab().toImage()
    host.deleteLater()
    return image


def _profile(image):
    """Brightness of the rim along the top edge, one value per column."""
    width = image.width()
    centre = image.pixelColor(width // 2, image.height() // 2)
    out = []
    for x in range(30, width - 30):
        total = 0.0
        for y in range(0, 4):
            pixel = image.pixelColor(x, y)
            total += (abs(pixel.red() - centre.red())
                      + abs(pixel.green() - centre.green())
                      + abs(pixel.blue() - centre.blue()))
        out.append(total)
    return out


def _turns(values, floor):
    """How often the profile changes direction where it is lit."""
    lit = [v for v in values if v > floor]
    noise = max(lit, default=0.0) * 0.01
    steps = [b - a for a, b in zip(lit, lit[1:]) if abs(b - a) > noise]
    return sum(1 for a, b in zip(steps, steps[1:]) if (a > 0) != (b > 0))


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("size", [(980, 700), (600, 420), (420, 300)])
@pytest.mark.parametrize("mode", ["glow", "rainbow"])
def test_the_lit_run_has_no_seams(qapp, monkeypatch, theme, size, mode):
    """One rise and one fall: no box edges along the run."""
    profile = _profile(_render(theme, *size, mode, monkeypatch))
    peak = max(profile)
    assert peak > 120, "the run is not lit on the top edge at all"
    turns = _turns(profile, peak * 0.15)
    assert turns <= 4, (
        f"the rim turns over {turns} times along its run: it is drawn as "
        "boxes with seams rather than as one gradient")


def test_neighbouring_pixels_along_the_run_change_gently(qapp, monkeypatch):
    """No jump between two neighbouring columns of the lit run."""
    profile = _profile(_render("dark", 980, 700, "glow", monkeypatch))
    peak = max(profile)
    lit = [v for v in profile if v > peak * 0.15]
    jumps = [abs(b - a) for a, b in zip(lit, lit[1:])]
    assert lit and max(jumps) < peak * 0.12
