"""A translucent flag must be backed by a native alpha buffer (Preferences)."""
import ctypes
import sys

import pytest

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from spacr.qt.widgets import glass


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows layered native surface')
def test_native_windows_popup_keeps_layered_alpha_after_resize_and_reopen(qtbot, qapp):
    if qapp.platformName() != 'windows':
        pytest.skip('requires actual Windows QPA')
    dialog = QDialog()
    qtbot.addWidget(dialog)
    QVBoxLayout(dialog).addWidget(QLabel('A visible layered popup'))
    dialog.resize(420, 300)
    dialog.winId()
    assert glass.glass(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    identity = int(dialog.winId())
    for action in ('initial', 'resize', 'reopen'):
        if action == 'resize':
            dialog.resize(480, 360)
        elif action == 'reopen':
            dialog.hide()
            dialog.show()
        qtbot.wait(30)
        assert int(dialog.winId()) == identity
        assert user.GetWindowLongPtrW(identity, -20) & 0x80000, action
        assert dialog.grab().toImage().pixelColor(0, 0).alpha() == 0


def test_windows_alpha_recreates_widget_resources_once(qtbot, monkeypatch):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    child = QLabel('Preserved native child', dialog)
    dialog.setAttribute(Qt.WA_TranslucentBackground)
    dialog.setWindowFlag(Qt.FramelessWindowHint)
    dialog.winId()
    child.winId()
    child_geometry = child.geometry()
    destroyed = []
    original = dialog.destroy

    def destroy(*args):
        destroyed.append(args)
        original(*args)

    monkeypatch.setattr(glass.QApplication, 'platformName', lambda: 'windows')
    monkeypatch.setattr(dialog, 'destroy', destroy)
    assert glass._ensure_alpha_surface(dialog)
    assert destroyed == [(True, False)]
    assert child.text() == 'Preserved native child'
    assert child.parentWidget() is dialog
    assert child.geometry() == child_geometry
    assert int(child.winId()) != 0
    identity = int(dialog.winId())
    assert glass._ensure_alpha_surface(dialog)
    assert int(dialog.winId()) == identity
    assert destroyed == [(True, False)]


def test_glassing_an_already_created_opaque_surface_requests_real_alpha(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    QVBoxLayout(dialog).addWidget(QLabel('Preserved contents'))
    dialog.resize(420, 300)
    dialog.winId()  # Qt may do this before Polish on native X11.
    window = dialog.windowHandle()
    assert window is not None
    glass.glass(dialog)
    if glass.QApplication.platformName() == 'windows':
        assert dialog.windowHandle() is not window
        window = dialog.windowHandle()
    else:
        assert dialog.windowHandle() is window
    assert window.format().alphaBufferSize() >= 8
    dialog.show()
    qtbot.wait(20)
    assert dialog.mask().isEmpty()
    assert dialog.findChild(QLabel).text() == 'Preserved contents'
    identity = dialog.winId()
    dialog.resize(480, 360)
    dialog.hide()
    dialog.show()
    qtbot.wait(20)
    assert dialog.winId() == identity, 'a repaired surface must not recreate on every show'
    assert window.format().alphaBufferSize() >= 8
    assert dialog.mask().isEmpty()


def test_translucent_attribute_without_alpha_keeps_rounded_shape(qtbot, monkeypatch):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(420, 300)
    dialog.setAttribute(Qt.WA_TranslucentBackground)

    class OpaqueFormat:
        def alphaBufferSize(self):
            return -1

    class OpaqueWindow:
        def format(self):
            return OpaqueFormat()

    monkeypatch.setattr(dialog, 'windowHandle', lambda: OpaqueWindow())
    assert glass.round_the_corners(dialog)
    assert not dialog.mask().isEmpty()
    assert not dialog.mask().contains(QPoint(0, 0))
    assert dialog.mask().contains(QPoint(210, 150))
