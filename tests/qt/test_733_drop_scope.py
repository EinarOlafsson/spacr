"""Only the visible Navigator intercepts whole-window local file drops."""
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QWidget

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


class _NativeDrop(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.received = []

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        self.received.append([url.toLocalFile() for url in event.mimeData().urls()])
        event.acceptProposedAction()


def _drop(target, path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    enter = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, event)
    assert event.isAccepted()


def test_module_drop_reaches_native_widget_and_home_still_receives_global_drop(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.qt.app import MainWindow

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    path = tmp_path / 'measurements.csv'
    path.write_text('label,area\n1,12\n')
    original = path.read_bytes()
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1180, 820)
    window.show()
    routed = []
    dispatch = window._easy_mode_drop

    def route(paths):
        routed.append(paths)
        dispatch(paths)

    monkeypatch.setattr(window, '_easy_mode_drop', route)
    try:
        home = window._startup
        assert window._stack.currentWidget() is home
        _drop(window.centralWidget(), path)
        assert routed == [[str(path)]]
        qtbot.waitUntil(lambda: home._last_result is not None and not home._runner.is_busy(), timeout=5000)
        assert home.path_edit.text() == str(path)
        window._on_nav_selected('measure')
        module = window._screens['measure']
        assert window._stack.currentWidget() is module
        native = _NativeDrop(module)
        native.setGeometry(0, 0, 200, 100)
        native.show()
        _drop(native, path)
        assert native.received == [[str(path)]]
        assert routed == [[str(path)]]
        assert window._stack.currentWidget() is module
        replacement = tmp_path / 'other.csv'
        replacement.write_text('label,area\n2,14\n')
        dispatch([str(replacement)])
        assert window._stack.currentWidget() is module
        assert home.path_edit.text() == str(path)
        window._on_nav_selected('__home__')
        _drop(window.centralWidget(), path)
        assert routed == [[str(path)], [str(path)]]
        assert window._stack.currentWidget() is home
        assert path.read_bytes() == original
    finally:
        window.close()
