#!/usr/bin/env python
"""Photograph the dock from INSIDE a running spaCR, and say what is behind it.

WHY THIS EXISTS. The dock's "black box" was reported four times on
2026-09-03 and fixed wrongly three times, because every measurement
available to the session reported a clean dock while the maintainer's screen
showed a box. XWayland refuses screen captures to X clients and
``QCursor.setPos`` is a no-op under Wayland, so the session could neither see
the defect nor drive a hover on the machine that had it.

Run it and it writes two PNGs and a colour report:

    python tools/diagnose_dock.py

    dock_rest.png     the dock with nothing hovered
    dock_hover.png    the dock with the pointer on a row, if one is there

Both are drawn over MAGENTA, so anything the dock does not paint shows as
magenta and anything it does paint is obvious against it.

If the PNGs look CLEAN and the screen still shows a box, the box is being
painted by the compositor path rather than by the widget tree, and no widget
or stylesheet change will move it. If the PNGs show the box, it is in the
widget tree after all and the colour report says which widget paints it.
That distinction is the whole point; nothing else here matters.
"""
from __future__ import annotations

import sys
from pathlib import Path

OUT = Path.cwd()


def main() -> int:
    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtGui import QColor, QCursor, QPixmap
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    from spacr.qt.app import MainWindow
    from spacr.qt.theme import apply_qpalette, stylesheet

    apply_qpalette(app)
    app.setStyleSheet(stylesheet())
    win = MainWindow()
    win.resize(1440, 900)
    win.show()

    def shoot():
        dock = win._sidebar
        row = next((r for r in dock._items
                    if str(r.property("navKey")) == "mask"), None)
        print(f"platform            : {app.platformName()}")
        print(f"dock size           : {dock.width()}x{dock.height()}")

        for name, tag in (("dock_rest.png", "rest"),
                          ("dock_hover.png", "hover")):
            if tag == "hover" and row is not None:
                # Wayland refuses this; the report says so rather than
                # pretending the pointer moved.
                QCursor.setPos(row.mapToGlobal(
                    QPoint(row.width() // 2, row.height() // 2)))
                app.processEvents()
                dock.sync_hover()
            shot = QPixmap(dock.size())
            shot.fill(QColor("#ff00ff"))       # magenta shows any gap
            dock.render(shot)
            path = OUT / name
            shot.save(str(path))
            image = shot.toImage()
            y = 200 if row is None else row.mapTo(
                dock, QPoint(0, row.height() // 2)).y()
            runs = []
            for x in range(dock.width()):
                colour = image.pixelColor(x, y).name()
                if not runs or runs[-1][1] != colour:
                    runs.append((x, colour))
            print(f"{tag:20s}: {runs[:10]}")
            print(f"{'':20s}  wrote {path}")
        if row is not None:
            from PySide6.QtCore import Qt
            print(f"row hovered         : {row._hovered}")
            print("row NoSystemBackgrnd: "
                  f"{row.testAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)}")
        print()
        print("A run of one colour across the whole width is a clean dock.")
        print("A dark run between the slab's edges is the box.")
        app.quit()

    QTimer.singleShot(3000, shoot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
