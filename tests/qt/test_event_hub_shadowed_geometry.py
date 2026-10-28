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


_QT_NAMES_THE_HUB_READS = ("geometry", "isHidden", "x", "y", "width", "height", "isWindow",
                           "isVisible", "testAttribute", "pos", "size", "isEnabled", "parent",
                           "isWidgetType", "installEventFilter", "removeEventFilter", "findChildren")


def _hiding_qt_methods(widget):
    for name in _QT_NAMES_THE_HUB_READS:
        setattr(widget, name, QComboBox())
    return widget


def test_attributes_hiding_qt_methods_never_reach_the_hub_169(qapp):
    """GitHub #169 (as #162): every Qt call the hub makes on a watched object
    goes through the Qt class, so an instance attribute holding a control
    under a Qt method's name is never called."""
    from PySide6.QtWidgets import QApplication

    root = QWidget()
    child = QWidget(root)
    _hiding_qt_methods(root)
    _hiding_qt_methods(child)
    events = []

    class Watcher(QObject):
        def eventFilter(self, watched, event):
            events.append(event.type())
            return False

    watcher = Watcher()
    hub = SimpleNamespace(_objects={})
    gil._hub_add_object(hub, child, watcher, {QEvent.Type.Move, QEvent.Type.Show})
    before = gil._hub_geometry_before(hub)
    assert len(before) == 1
    QWidget.move(child, 17, 19)
    gil._hub_geometry_after(hub, before)
    assert QEvent.Type.Move in events
    state = gil._widget_state(child)
    assert state[5] is root and state[2] == QWidget.pos(child)
    lone = Watcher()
    assert gil._watch_object_events(child, lone, {QEvent.Type.Paint})
    gil._stop_watching_object_events(child, lone)
    with gil._restyle_unheard(QApplication.instance(), root):
        QWidget.setStyleSheet(root, "QWidget { color: #abc; }")
    QWidget.close(root)


def test_deleted_wrapper_addresses_are_declined_without_reusing_a_live_identity(qapp, monkeypatch):
    from shiboken6 import delete
    import shiboken6

    dead = QWidget()
    delete(dead)
    assert gil._cpp_address(dead) is None
    live = QWidget()
    real_pointer = shiboken6.getCppPointer

    def unavailable(widget):
        if widget is live:
            raise RuntimeError('native pointer unavailable')
        return real_pointer(widget)

    monkeypatch.setattr(shiboken6, 'getCppPointer', unavailable)
    assert gil._cpp_address(live) is None
    assert live.isEnabled()
    live.close()


def test_unknown_wrapper_finalizer_cannot_drop_another_watch(qapp):
    widget, watcher = QWidget(), QObject()
    hub = SimpleNamespace(_objects={})
    gil._hub_add_object(hub, widget, watcher, {QEvent.Type.Move})
    original = hub._objects[id(widget)]
    gil._hub_forget_wrapper(hub, -1, -2)
    assert hub._objects[id(widget)] is original
    assert gil._hub_held_wrapper(hub, id(widget)) is widget
    gil._hub_forget_wrapper(hub, id(widget), id(widget))
    assert gil._hub_held_wrapper(hub, id(widget)) is None
    assert gil._hub_ask_object_watchers(hub, widget, QEvent(QEvent.Type.Move), QEvent.Type.Move) is False
    widget.close()
