"""A translucent flag must be backed by a native alpha buffer (Preferences)."""
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from spacr.qt.widgets import glass


def test_glassing_an_already_created_opaque_surface_requests_real_alpha(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    QVBoxLayout(dialog).addWidget(QLabel('Preserved contents'))
    dialog.resize(420, 300)
    dialog.winId()  # Qt may do this before Polish on native X11.
    window = dialog.windowHandle()
    assert window is not None
    glass.glass(dialog)
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
