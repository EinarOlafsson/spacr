"""Opening Make Masks shows no window but the main one.

On a real desktop a small window flashed for a moment when Make Masks was
clicked. Every ``Show`` of a top-level widget and every native window that
becomes visible while the screen is built and shown is counted here.
The flash was the Otsu method group: built without a parent, it was made
visible by the next group's build and so opened as a window of its own
until the detection card took it in.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _describe(obj):
    name = obj.objectName() if hasattr(obj, "objectName") else ""
    return f"{type(obj).__module__}.{type(obj).__name__}({name!r})"


@pytest.mark.timeout(180)
def test_opening_make_masks_shows_no_other_top_level_window(qapp):
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtGui import QGuiApplication, QWindow
    from PySide6.QtWidgets import QWidget

    from spacr.qt.app import MainWindow

    win = MainWindow(initial_app="__home__")
    win.resize(1400, 900)
    win.show()
    for _ in range(5):
        qapp.processEvents()

    shown = []

    class _Watch(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Show:
                if isinstance(obj, QWidget) and obj.isWindow() and obj is not win:
                    shown.append(_describe(obj))
                elif isinstance(obj, QWindow) and obj is not win.windowHandle():
                    shown.append(_describe(obj))
            return False

    before = {id(w) for w in QGuiApplication.allWindows()}
    seen_visible = []

    def _new_visible():
        for handle in QGuiApplication.allWindows():
            if id(handle) not in before and handle.isVisible() and handle is not win.windowHandle():
                seen_visible.append(_describe(handle))

    assert "make_masks" not in win._screens
    watch = _Watch()
    qapp.installEventFilter(watch)
    try:
        win._on_nav_selected("make_masks")
        _new_visible()
        for _ in range(20):
            qapp.processEvents()
            _new_visible()
    finally:
        qapp.removeEventFilter(watch)
    try:
        assert win._stack.currentWidget() is win._screens.get("make_masks")
        assert not shown, f"windows shown while Make Masks opened: {shown}"
        assert not seen_visible, f"native windows made visible: {sorted(set(seen_visible))}"
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()
