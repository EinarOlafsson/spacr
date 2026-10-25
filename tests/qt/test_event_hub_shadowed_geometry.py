"""Native geometry tracking survives a widget control hiding the method."""
from types import SimpleNamespace
import weakref

from PySide6.QtCore import QEvent, QObject, QRect
from PySide6.QtWidgets import QComboBox, QWidget

from spacr.qt import gil_priority as gil


def test_shadowed_geometry_keeps_real_move_resize_and_show_replay(qapp):
    widget = QWidget()
    widget.geometry = QComboBox(widget)
    events = []

    class Watcher(QObject):
        def eventFilter(self, watched, event):
            assert watched is widget
            events.append(event.type())
            return False

    watcher = Watcher()
    hub = SimpleNamespace(_objects={})
    kinds = {QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show}
    gil._hub_add_object(hub, widget, watcher, kinds)
    before = gil._hub_geometry_before(hub)
    assert len(before) == 1
    assert QRect(*before[0][2]) == QWidget.geometry(widget)
    original = QRect(*before[0][2])
    widget.setGeometry(QRect(original.x() + 11, original.y() + 13,
                             original.width() + 31, original.height() + 29))
    widget.show()
    gil._hub_geometry_after(hub, before)
    assert events == [QEvent.Type.Show, QEvent.Type.Move, QEvent.Type.Resize]
    events.clear()
    gil._hub_geometry_after(hub, gil._hub_geometry_before(hub))
    assert events == []
    widget.close()
