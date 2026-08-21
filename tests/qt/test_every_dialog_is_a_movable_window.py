"""Every dialog is a window the user can drag where they like.

Asked 2026-08-21: "the settings for your data settings window should be
movable without moving the main window. this should be tru of all settings
windows or any popup window from spacr."

ONE PLACE, NOT ONE CALL SITE PER DIALOG. `detach_from_window_manager` already
existed and was called from six files while more than twenty others opened
dialogs without it. "All settings windows" is not a state reachable by adding
a twenty-first call -- a rule applied by hand holds until the next dialog is
written.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _window_type(widget):
    from PySide6.QtCore import Qt

    return widget.windowFlags() & Qt.WindowType.WindowType_Mask


class TestTheFilterDetaches:

    def test_a_dialog_becomes_a_window(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QDialog, QWidget

        from spacr.qt.dialogs import detach_all_dialogs

        detach_all_dialogs(QApplication.instance())
        parent = QWidget()
        qtbot.addWidget(parent)
        dialog = QDialog(parent)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)

        assert _window_type(dialog) == Qt.WindowType.Window

    def test_it_still_shows_and_keeps_its_parent(self, qtbot):
        """`setWindowFlags` on a VISIBLE widget destroys and recreates the
        native window, so Qt hides it. Detaching on `Show` makes the dialog
        flash or vanish; `Polish` is delivered before the window is mapped.
        """
        from PySide6.QtWidgets import QApplication, QDialog, QWidget

        from spacr.qt.dialogs import detach_all_dialogs

        detach_all_dialogs(QApplication.instance())
        parent = QWidget()
        qtbot.addWidget(parent)
        dialog = QDialog(parent)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)

        assert dialog.isVisible()
        assert dialog.parent() is parent

    def test_installing_twice_on_the_same_app_leaves_one_filter(self):
        from PySide6.QtWidgets import QApplication

        from spacr.qt import dialogs

        app = QApplication.instance()
        dialogs.detach_all_dialogs(app)
        assert dialogs.detach_all_dialogs(app) is False

    def test_a_new_application_gets_its_own_filter(self):
        """A filter belongs to one QApplication and dies with it. Tracking
        only the filter meant that after a teardown the module reported
        "already installed" while nothing was filtering -- which is why two
        of these tests passed alone and failed in a full run."""
        from spacr.qt import dialogs

        class _Pretend:
            def __init__(self):
                self.filters = []

            def installEventFilter(self, f):
                self.filters.append(f)

        first, second = _Pretend(), _Pretend()
        dialogs._DETACHER = None
        dialogs._DETACHED_APP = None
        try:
            assert dialogs.detach_all_dialogs(first) is True
            assert dialogs.detach_all_dialogs(first) is False
            assert dialogs.detach_all_dialogs(second) is True
            assert len(second.filters) == 1
        finally:
            dialogs._DETACHER = None
            dialogs._DETACHED_APP = None

    def test_the_filter_is_kept_alive(self):
        """An event filter that is garbage collected stops filtering,
        silently."""
        from PySide6.QtWidgets import QApplication

        from spacr.qt import dialogs

        dialogs.detach_all_dialogs(QApplication.instance())
        assert dialogs._DETACHER is not None


class TestTheFilterIsActuallyReached:
    """It installed, reported success and silently never fired once --
    `class F(QObject, Mixin)` puts QObject first in the MRO, so QObject's own
    `eventFilter` (which does nothing) won over the mixin's."""

    def test_the_filter_object_defines_its_own_event_filter(self):
        from PySide6.QtWidgets import QApplication

        from spacr.qt import dialogs

        dialogs.detach_all_dialogs(QApplication.instance())
        filter_object = dialogs._DETACHER
        # The method must come from spaCR's class, not from QObject.
        owner = type(filter_object).__mro__[0]
        assert "eventFilter" in owner.__dict__

    def test_a_dialog_shown_after_install_is_changed(self, qtbot):
        """The end-to-end statement of the same thing."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QDialog

        from spacr.qt.dialogs import detach_all_dialogs

        detach_all_dialogs(QApplication.instance())
        dialog = QDialog()
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)
        assert _window_type(dialog) != Qt.WindowType.Dialog


class TestTheHelperItself:

    def test_it_clears_the_dialog_bit_and_keeps_window(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog

        from spacr.qt.dialogs import detach_from_window_manager

        dialog = QDialog()
        qtbot.addWidget(dialog)
        detach_from_window_manager(dialog)

        assert _window_type(dialog) == Qt.WindowType.Window

    def test_it_is_safe_on_a_parentless_dialog(self, qtbot):
        from PySide6.QtWidgets import QDialog

        from spacr.qt.dialogs import detach_from_window_manager

        dialog = QDialog()
        qtbot.addWidget(dialog)
        assert detach_from_window_manager(dialog) is dialog

    def test_it_returns_the_object_so_it_can_be_used_inline(self, qtbot):
        from PySide6.QtWidgets import QDialog

        from spacr.qt.dialogs import detach_from_window_manager

        dialog = QDialog()
        qtbot.addWidget(dialog)
        assert detach_from_window_manager(dialog) is dialog
