"""Is this X display usable for a pixel measurement, or is it lying to us?

RUN THIS BEFORE ANY on-screen MEASUREMENT, and instruction 381 is why.
That item can only be settled by `QScreen.grabWindow` under X --
`QWidget.grab()` cannot see the GL-backed `AmbientWidget` and renders a
WORKING backdrop as black, which already cost an hour and a wrong
conclusion once.

AND grabWindow HAS ITS OWN VERSION OF THE SAME TRAP. On a locked or
blanked session the X server hands back whatever is actually in front of
that rectangle. Measured on this box, 2026-09-11, with the maintainer's
session locked:

    grab 240x160, 0.0% of sampled pixels are the marker colour
      what came back instead: rgb~(16,16,16) 73%, rgb~(0,0,0) 8%

NEAR-BLACK. Which is EXACTLY 381's failure signature -- that file records
the regression as "2.5 % chromatic, 31.7 % pure black" against "29.6 %
chromatic, no pure black" for the good commit. A measurement taken on a
locked session would report the bug whether or not it was there, and would
go on reporting it after it was fixed.

So this paints a colour nothing on a desktop paints and checks that colour
comes back. Exit 0 means the display can be measured; exit 2 means anything
measured on it is about something else.

    DISPLAY=:0 QT_QPA_PLATFORM=xcb python tools/can_this_display_be_measured.py
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QWidget

#: How long to let the window manager actually map and composite the
#: probe before grabbing. 1.5 s is generous on purpose: this tool exists
#: to answer "can this display be measured at all", and a false
#: "unmeasurable" costs far more than a second.
SETTLE_MS = 1500

MARKER = QColor(7, 199, 123)      # nothing else on a desktop is this colour


def main():
    app = QApplication.instance() or QApplication([])
    w = QWidget()
    w.setWindowTitle("spacr display probe")
    w.setFixedSize(240, 160)
    w.setAutoFillBackground(True)
    palette = w.palette()
    palette.setColor(QPalette.Window, MARKER)
    w.setPalette(palette)
    w.move(40, 40)
    # RAISED AND KEPT ON TOP. `grabWindow` under X returns what is in FRONT of
    # the rectangle, not the widget's own painting, so an unraised probe
    # measures whatever happens to be stacked above it.
    w.setWindowFlags(w.windowFlags() | Qt.WindowStaysOnTopHint)
    w.show()
    w.raise_()
    w.activateWindow()
    # REAL ELAPSED TIME, NOT A SPIN. Forty processEvents() calls take
    # microseconds; a compositing WM needs MILLISECONDS to map, raise and
    # composite a new window. The spin was the whole bug: the grab happened
    # before the probe was on screen, came back as the dark UI underneath,
    # and near-black is the exact signature 381 is looking for -- so an
    # unmeasurable verdict was returned on a display that measures fine.
    settle = QElapsedTimer()
    settle.start()
    while settle.elapsed() < SETTLE_MS:
        app.processEvents()

    screen = w.screen() or QGuiApplication.primaryScreen()
    shot = screen.grabWindow(w.winId())
    image = shot.toImage()
    if image.isNull() or image.width() < 10:
        print("UNUSABLE: the grab came back empty")
        return 2
    hits = 0
    total = 0
    for y in range(0, image.height(), 8):
        for x in range(0, image.width(), 8):
            total += 1
            colour = image.pixelColor(x, y)
            if (abs(colour.red() - MARKER.red()) < 12
                    and abs(colour.green() - MARKER.green()) < 12
                    and abs(colour.blue() - MARKER.blue()) < 12):
                hits += 1
    share = 100.0 * hits / max(total, 1)
    seen = {}
    for y in range(0, image.height(), 8):
        for x in range(0, image.width(), 8):
            colour = image.pixelColor(x, y)
            key = (colour.red() // 16, colour.green() // 16,
                   colour.blue() // 16)
            seen[key] = seen.get(key, 0) + 1
    ranked = sorted(seen.items(), key=lambda row: -row[1])[:3]
    print(f"grab {image.width()}x{image.height()}, "
          f"{share:.1f}% of sampled pixels are the marker colour")
    print("  what came back instead: " + ", ".join(
        f"rgb~({r * 16},{g * 16},{b * 16}) {100.0 * n / max(total, 1):.0f}%"
        for (r, g, b), n in ranked))
    print("USABLE" if share > 80 else
          "UNUSABLE: the grab is not showing this window")
    w.close()
    return 0 if share > 80 else 2


sys.exit(main())
