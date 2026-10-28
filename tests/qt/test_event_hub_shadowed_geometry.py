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


def test_reused_watch_slot_retires_the_previous_wrapper_before_delivering(qapp):
    """An address slot left by an old wrapper cannot retain its observer."""
    old, current = QWidget(), QWidget()
    delivered = []

    class Watcher(QObject):
        def eventFilter(self, watched, event):
            delivered.append((self, watched, event.type()))
            return False

    previous, replacement = Watcher(), Watcher()
    key = id(current)
    entry = [weakref.ref(previous), frozenset({QEvent.Move}), True,
             weakref.ref(old), True]
    hub = SimpleNamespace(_objects={key: [entry]},
                          _wrappers={key: (weakref.ref(old), id(old))},
                          _wrapper_keys={id(old): key})
    gil._hub_add_object(hub, current, replacement, {QEvent.Move})
    assert entry[2] is False
    assert gil._hub_held_wrapper(hub, key) is current
    gil._hub_ask_object_watchers(hub, current, QEvent(QEvent.Move), QEvent.Move)
    assert delivered == [(replacement, current, QEvent.Move)]
    old.close()
    current.close()


def test_parent_metadata_failure_keeps_real_object_event_delivery(qapp, monkeypatch):
    parent, watched = QWidget(), QWidget()
    watched.setParent(parent)
    events = []

    class Watcher(QObject):
        def eventFilter(self, obj, event):
            events.append((obj, event.type()))
            return False

    observer = Watcher()
    original = QObject.parent

    def unavailable(obj):
        if obj is watched:
            raise RuntimeError('parent metadata retired during registration')
        return original(obj)

    monkeypatch.setattr(QObject, 'parent', unavailable)
    hub = SimpleNamespace(_objects={})
    gil._hub_add_object(hub, watched, observer, {QEvent.Move})
    gil._hub_ask_object_watchers(hub, watched, QEvent(QEvent.Move), QEvent.Move)
    assert events == [(watched, QEvent.Move)]
    assert gil._hub_discard_object(hub, watched, observer)
    parent.close()


def test_wrapper_finalization_after_hub_teardown_is_silent(qapp, monkeypatch):
    import gc
    import sys

    watched, observer = QObject(), QObject()
    hub = SimpleNamespace(_objects={})
    gil._hub_add_object(hub, watched, observer, {QEvent.Move})
    key = id(watched)
    reference = weakref.ref(watched)
    watched.destroyed.disconnect()
    del hub._objects
    failures = []
    monkeypatch.setattr(sys, 'unraisablehook', failures.append)
    del watched
    gc.collect()
    assert reference() is None
    assert key not in hub._wrappers and key not in hub._wrapper_keys
    assert failures == []


def test_an_event_during_watch_bucket_retirement_reaches_no_old_observer(qapp):
    events = []

    class Watcher(QObject):
        def eventFilter(self, obj, event):
            events.append(obj)
            return False

    watched, observer = QWidget(), Watcher()
    hub = SimpleNamespace(_objects={})
    gil._hub_add_object(hub, watched, observer, {QEvent.Move})
    bucket = hub._objects.pop(id(watched))
    assert not gil._hub_ask_object_watchers(hub, watched, QEvent(QEvent.Move), QEvent.Move)
    assert events == []
    hub._objects[id(watched)] = bucket
    assert gil._hub_discard_object(hub, watched, observer)
    watched.close()


def test_restyle_replay_uses_live_widgets_when_native_identity_metadata_expires(qapp, monkeypatch):
    import shiboken6
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QLabel

    root = QWidget()
    unavailable = QLabel('Pointer unavailable', root)
    replacement = QLabel('Replacement', root)
    unavailable.ensurePolished()
    replacement.ensurePolished()
    retired = QWidget()
    retired_state = gil._widget_state(retired)
    shiboken6.delete(retired)
    heard = []

    class Hub(QObject):
        _by_kind = {QEvent.StyleChange: ()}
        _object_kinds = ()

        def eventFilter(self, obj, event):
            if event.type() == QEvent.StyleChange and obj in (unavailable, replacement):
                assert shiboken6.isValid(obj)
                heard.append(obj)
            return False

    hub = Hub()
    monkeypatch.setattr(gil, '_application_event_hub', lambda app, create=False: hub)
    pointer = shiboken6.getCppPointer

    def read_pointer(widget):
        if widget is unavailable:
            raise RuntimeError('native identity unavailable')
        return pointer(widget)

    monkeypatch.setattr(shiboken6, 'getCppPointer', read_pointer)
    try:
        with gil._restyle_unheard(qapp, root):
            root.setStyleSheet('QLabel { color: #123456; }')
        assert unavailable in heard and replacement in heard
        assert unavailable.palette().color(QPalette.WindowText).name() == '#123456'
        heard.clear()
        stale_snapshot = {gil._cpp_address(replacement): (retired, retired_state)}
        gil._tell_the_watchers(hub, root, stale_snapshot)
        assert replacement in heard and retired not in heard
        assert replacement.text() == 'Replacement'
    finally:
        qapp.removeEventFilter(hub)
        root.close()
