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


@pytest.fixture
def dialog_parent(qtbot):
    """Keep the Qt owner alive until pytest-qt closes the registered widgets."""
    from PySide6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    yield parent


class TestTheFilterDetaches:

    @pytest.mark.parametrize("embedded", (True, False))
    def test_polish_preserves_embedded_pages_but_detaches_popups(self, qtbot, qapp, request, embedded):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtWidgets import QDialog, QLineEdit, QVBoxLayout, QWidget
        from spacr.qt import dialogs

        owner = QWidget()
        qtbot.addWidget(owner)
        outer = QVBoxLayout(owner)
        page = QDialog(owner)
        if embedded:
            page.setWindowFlags(Qt.Widget)
            outer.addWidget(page)
        content = QVBoxLayout(page)
        editor = QLineEdit("owned control", page)
        content.addWidget(editor)
        before_flags = page.windowFlags()
        assert page.parentWidget() is owner
        assert page.isWindow() is not embedded
        previous_filter, previous_app = dialogs._DETACHER, dialogs._DETACHED_APP
        if dialogs.detach_all_dialogs(qapp):
            installed = dialogs._DETACHER

            def restore_filter():
                from spacr.qt.gil_priority import _stop_watching_application_events

                _stop_watching_application_events(qapp, installed)
                dialogs._DETACHER, dialogs._DETACHED_APP = previous_filter, previous_app

            request.addfinalizer(restore_filter)
        qapp.sendEvent(page, QEvent(QEvent.Polish))
        owner.show()
        page.show()
        qtbot.waitUntil(lambda: page.isVisibleTo(owner))
        if embedded:
            assert not page.isWindow(), (before_flags, page.windowFlags())
            assert page.windowFlags() == before_flags
            assert page.window() is owner and page.parentWidget() is owner
            assert outer.indexOf(page) >= 0 and page.layout() is content
            assert not page.property(dialogs.FILTER_DETACHED)
            assert not page.property(dialogs.RESIZABLE)
        else:
            qtbot.addWidget(page)
            assert _window_type(page) == Qt.Window
            assert page.isWindow() and page.parentWidget() is owner
            assert page.property(dialogs.FILTER_DETACHED)
            assert page.property(dialogs.RESIZABLE)

    def test_a_dialog_becomes_a_window(self, qtbot, dialog_parent):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QDialog

        from spacr.qt.dialogs import detach_all_dialogs

        detach_all_dialogs(QApplication.instance())
        dialog = QDialog(dialog_parent)
        dialog.show()
        qtbot.waitExposed(dialog)

        assert _window_type(dialog) == Qt.WindowType.Window

    def test_it_still_shows_and_keeps_its_parent(self, qtbot, dialog_parent):
        """`setWindowFlags` on a VISIBLE widget destroys and recreates the
        native window, so Qt hides it. Detaching on `Show` makes the dialog
        flash or vanish; `Polish` is delivered before the window is mapped.
        """
        from PySide6.QtWidgets import QApplication, QDialog

        from spacr.qt.dialogs import detach_all_dialogs

        detach_all_dialogs(QApplication.instance())
        dialog = QDialog(dialog_parent)
        dialog.show()
        qtbot.waitExposed(dialog)

        assert dialog.isVisible()
        assert dialog.parent() is dialog_parent

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
