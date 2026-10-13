"""How much of the home screen is actually coloured, read from the X server.

INSTRUCTION 381'S MEASURE, and the only honest one it has.
`QWidget.grab()` cannot see the GL-backed `AmbientWidget` at all and
renders a WORKING backdrop as black, so the picture has to come from the
compositor. That means `QScreen.grabWindow` on a window which has been
RAISED and given real time to map -- both of which the first version of
`can_this_display_be_measured.py` got wrong, and both of which produce a
convincing near-black that looks exactly like the bug.

    RUN `can_this_display_be_measured.py` FIRST. Exit 2 there means every
    number this prints is about something else.

Measured 2026-09-11 on the maintainer's display, home screen, 1600x1000,
with the one-backdrop dedup switched on and off:

    theme   dedup OFF        dedup ON, opaque      dedup ON, transparent
                             window block          window block
    dark    58.8% / 0.0%     3.0% / 20.4% black    60.1% / 0.0%
    cell    93.5% / 0.3%     93.5% / 0.2%          95.7% / 0.1%
    glass   61.7% / 0.0%     67.3% / 0.0%          72.8% / 0.0%

The middle column is 381's recorded failure signature reproduced on
demand, and it is DARK-THEME-ONLY: the image themes already took the
transparent block.

    DISPLAY=:0 QT_QPA_PLATFORM=xcb \
        python tools/measure_the_home_screen_is_not_black.py [theme]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

from PySide6.QtCore import QElapsedTimer, Qt                  # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

THEME = sys.argv[1] if len(sys.argv) > 1 else "cell"


def settle(app, ms):
    clock = QElapsedTimer()
    clock.start()
    while clock.elapsed() < ms:
        app.processEvents()


def main():
    app = QApplication.instance() or QApplication([])
    from spacr.qt import preferences, register_self_registering_modules
    from spacr.qt.app import MainWindow

    preferences.set_theme(THEME)
    preferences.set_ambient_enabled(True)
    preferences.apply_preferences_to_app(app)
    register_self_registering_modules()

    window = MainWindow()
    window.resize(1600, 1000)
    window.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    window.show()
    window.raise_()
    window.activateWindow()
    settle(app, 4000)

    screen = window.screen()
    image = screen.grabWindow(window.winId()).toImage()
    if image.isNull() or image.width() < 100:
        print("UNUSABLE: empty grab")
        return 2

    chromatic = black = total = 0
    for y in range(0, image.height(), 3):
        for x in range(0, image.width(), 3):
            colour = image.pixelColor(x, y)
            r, g, b = colour.red(), colour.green(), colour.blue()
            total += 1
            if r == 0 and g == 0 and b == 0:
                black += 1
            if max(r, g, b) - min(r, g, b) >= 12:
                chromatic += 1
    print(f"theme={THEME} grab={image.width()}x{image.height()} "
          f"chromatic={100.0 * chromatic / total:.1f}% "
          f"pure_black={100.0 * black / total:.1f}%")
    window.close()
    return 0


sys.exit(main())
