"""The dialog filter's edges: a dead window, a floorless form, a second show.

Everything here drives ``spacr.qt.dialogs`` through a path the ordinary
sweep over spaCR's real dialogs never reaches -- a dialog whose C++ side has
already gone, a form whose layout imposes no floor at all, an application
that will not take an event filter, and the second polish of a dialog that
has already been detached once.

THE ONCE-ONLY GUARD IS MEASURED ON THE WINDOW, not on a counter.
``setWindowFlags`` on a widget that is already visible destroys and recreates
its native window, and Qt hides the widget when it does -- so "detached a
second time" and "still on screen" are the same question asked twice, and the
control arm below shows the assertion has teeth.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _window_type(widget):
    from PySide6.QtCore import Qt

    return widget.windowFlags() & Qt.WindowType.WindowType_Mask


def _polish():
    from PySide6.QtCore import QEvent

    return QEvent(QEvent.Type.Polish)


def _dead_dialog():
    """A ``QDialog`` whose C++ object has been destroyed under its wrapper.

    What is left behind is what the application filter sees while a dialog
    is being torn down: every call through it raises ``RuntimeError``.
    """
    import shiboken6
    from PySide6.QtWidgets import QDialog

    dialog = QDialog()
    shiboken6.Shiboken.delete(dialog)
    return dialog


@pytest.fixture
def unfiltered_app(qtbot):
    """The application with spaCR's own dialog filters taken off.

    Glass rewrites a dialog's flags itself and the application-level
    detacher marks dialogs as it sees them; both would answer the questions
    below before the test could ask them.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt import dialogs
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    app = QApplication.instance()
    had_glass = uninstall_glass_everywhere()
    saved = (dialogs._DETACHER, dialogs._DETACHED_APP)
    if saved[0] is not None:
        app.removeEventFilter(saved[0])
    dialogs._DETACHER = dialogs._DETACHED_APP = None
    yield app
    dialogs._DETACHER, dialogs._DETACHED_APP = saved
    if saved[0] is not None and saved[1] is app:
        app.installEventFilter(saved[0])
    if had_glass:
        install_glass_everywhere()


# --------------------------------------------------------------------------
# detach_from_window_manager, where Qt is not there to be asked
# --------------------------------------------------------------------------

def test_a_dialog_is_handed_back_untouched_when_qt_cannot_be_imported(
        qtbot, monkeypatch):
    """Without ``QtCore`` there are no flags to rewrite, and no exception.

    This module is imported in places PySide6 is not, and the caller gets
    its dialog back either way.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    from spacr.qt.dialogs import detach_from_window_manager

    dialog = QDialog()
    qtbot.addWidget(dialog)
    before = dialog.windowFlags()

    with monkeypatch.context() as without_qtcore:
        without_qtcore.setitem(sys.modules, "PySide6.QtCore",
                               types.SimpleNamespace())
        assert detach_from_window_manager(dialog) is dialog

    assert dialog.windowFlags() == before
    assert _window_type(dialog) == Qt.WindowType.Dialog


def test_a_dialog_whose_window_is_already_gone_is_handed_back_not_raised():
    """A dialog being torn down must not take the caller down with it."""
    from spacr.qt.dialogs import detach_from_window_manager

    dead = _dead_dialog()

    assert detach_from_window_manager(dead) is dead


# --------------------------------------------------------------------------
# make_the_window_resizable, on a form that has no floor to drop
# --------------------------------------------------------------------------

def test_a_form_with_no_floor_of_its_own_is_given_no_opening_size(
        qtbot, unfiltered_app):
    """Nothing was lowered, so there is no size to put back on the next show.

    A dialog whose layout has no margins and holds a hand-positioned
    container imposes no minimum at all: there is no floor in its way, and
    storing one would make it open at a size it never had.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import (QDialog, QLineEdit, QSizeGrip, QVBoxLayout,
                                   QWidget)

    from spacr.qt.dialogs import (OPENS_AT, RESIZABLE, SCROLLS,
                                  make_the_window_resizable,
                                  open_at_its_natural_size)

    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    box = QWidget(dialog)          # laid out by hand: no layout of its own
    field = QLineEdit(box)
    field.setGeometry(4, 4, 120, 24)
    layout.addWidget(box)
    dialog.resize(400, 300)

    assert make_the_window_resizable(dialog) is True

    assert dialog.minimumSize() == QSize(0, 0)
    assert dialog.property(OPENS_AT) is None
    assert dialog.property(SCROLLS) is None
    assert dialog.property(RESIZABLE) is True
    assert dialog.findChildren(QSizeGrip)
    # And the show that follows leaves the size the user is looking at.
    assert open_at_its_natural_size(dialog) is False
    assert QWidget.size(dialog) == QSize(400, 300)


# --------------------------------------------------------------------------
# the application filter
# --------------------------------------------------------------------------

def test_a_dialog_already_detached_is_not_rebuilt_on_its_next_polish(
        qtbot, unfiltered_app):
    """A second detach would recreate the native window and hide it."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    from spacr.qt.dialogs import FILTER_DETACHED, _DetachEveryDialog

    watcher = _DetachEveryDialog()
    dialog = QDialog()
    qtbot.addWidget(dialog)

    assert watcher.eventFilter(dialog, _polish()) is False
    assert _window_type(dialog) == Qt.WindowType.Window
    assert dialog.property(FILTER_DETACHED) is True

    dialog.show()
    qtbot.waitExposed(dialog)
    watcher.eventFilter(dialog, _polish())

    assert dialog.isVisible()

    # The control: a watcher that has never seen this window DOES rewrite
    # its flags, and Qt hides it when it does. That is what the mark above
    # is protecting the user from.
    naive = QDialog()
    qtbot.addWidget(naive)
    naive.show()
    qtbot.waitExposed(naive)
    _DetachEveryDialog().eventFilter(naive, _polish())

    assert not naive.isVisible()


def test_a_dialog_at_a_departed_dialogs_address_is_still_detached(
        qtbot, unfiltered_app):
    """Identity is the dialog, not the address it happens to occupy.

    Every modal dialog in spaCR is a temporary, and CPython hands the
    address of a released one straight to the next: the loop below sees the
    same ``id`` come back on essentially every pass. A dialog identified by
    that number would be taken for one already dealt with and left attached
    to the main window.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    from spacr.qt.dialogs import _DetachEveryDialog

    watcher = _DetachEveryDialog()
    polish = _polish()
    reused = 0
    for _ in range(20):
        departed = QDialog()
        watcher.eventFilter(departed, polish)
        address = id(departed)
        departed.setParent(None)
        del departed

        fresh = QDialog()
        reused += id(fresh) == address
        watcher.eventFilter(fresh, polish)

        assert _window_type(fresh) == Qt.WindowType.Window
        fresh.setParent(None)
        del fresh

    assert reused, "the address was never reused; the test proved nothing"


def test_the_filter_passes_on_an_event_from_a_dialog_that_has_gone(
        unfiltered_app):
    """The filter sees every event in the application and may lose none.

    A dialog whose C++ object is destroyed answers every question with
    ``RuntimeError``; the event still has to reach whoever it was for.
    """
    from spacr.qt.dialogs import _DetachEveryDialog

    dead = _dead_dialog()

    assert _DetachEveryDialog().eventFilter(dead, _polish()) is False


# --------------------------------------------------------------------------
# installing the filter on an application that will not take it
# --------------------------------------------------------------------------

def test_an_application_that_refuses_the_filter_leaves_the_slot_empty(qapp):
    """A failed install must not look like a successful one.

    ``_DETACHER`` doubles as "already installed", so leaving a filter
    recorded against an application that never took it would make every
    later call report success while nothing filtered anything.
    """
    from spacr.qt import dialogs

    class _RefusingApplication:
        """An application torn down under the caller's feet."""

        def installEventFilter(self, _filter):
            raise RuntimeError("Internal C++ object already deleted.")

    saved = (dialogs._DETACHER, dialogs._DETACHED_APP)
    dialogs._DETACHER = dialogs._DETACHED_APP = None
    try:
        assert dialogs.detach_all_dialogs(_RefusingApplication()) is False
        assert dialogs._DETACHER is None
        assert dialogs._DETACHED_APP is None

        # And a real application afterwards still installs.
        assert dialogs.detach_all_dialogs(qapp) is True
        assert dialogs._DETACHED_APP is qapp
    finally:
        if dialogs._DETACHER is not None:
            qapp.removeEventFilter(dialogs._DETACHER)
        dialogs._DETACHER, dialogs._DETACHED_APP = saved
