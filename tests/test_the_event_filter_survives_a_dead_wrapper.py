"""The application-wide filter must not dereference a freed object.

Captured by the crash dump on 2026-08-19, milliseconds after a regression
closed [success] while that run's figure widgets were being torn down:

    Fatal Python error: Segmentation fault
    Current thread (most recent call first):
      feature_dictionary.py, line 776 in eventFilter
      app.py, line 3110 in launch

The frame EXISTS at the `def` line, so both wrappers had been built and the
crash is on the first bytecode -- `event.type()` on a QEvent whose C++ half
was already freed.

THE LIMIT IS PART OF THE CONTRACT: `isValid` reports a wrapper whose deletion
shiboken was told about, and turns that case from a segfault into a no-op. An
object freed without shiboken being told still dereferences.
"""
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QTableWidget

from spacr.qt.widgets.feature_dictionary import FeatureHelpFilter, _still_alive


def test_a_deleted_event_is_dropped_not_dereferenced(qapp):
    import shiboken6

    table = QTableWidget(2, 2)
    event = QEvent(QEvent.Type.ContextMenu)
    shiboken6.delete(event)

    assert FeatureHelpFilter().eventFilter(table, event) is False


def test_a_deleted_receiver_is_dropped(qapp):
    import shiboken6

    table = QTableWidget(2, 2)
    event = QEvent(QEvent.Type.ContextMenu)
    shiboken6.delete(table)

    assert FeatureHelpFilter().eventFilter(table, event) is False


def test_an_ordinary_event_is_still_ignored_cheaply(qapp):
    table = QTableWidget(2, 2)

    # Not a context menu: the common path, and it must stay a plain False.
    assert FeatureHelpFilter().eventFilter(
        table, QEvent(QEvent.Type.MouseMove)) is False


def test_the_liveness_question_is_answered_conservatively(qapp):
    import shiboken6

    widget = QTableWidget(1, 1)
    assert _still_alive(widget) is True
    shiboken6.delete(widget)
    assert _still_alive(widget) is False

    assert _still_alive(None) is False
    # A plain Python object cannot be asked, so it is treated as usable --
    # refusing an event that is fine would break the feature.
    assert _still_alive(object()) is True


def test_the_filter_is_still_application_wide(qapp):
    """Installing per-view was tried and reverted; this pins why.

    The feature answers a right-click on an arbitrary child widget INSIDE a
    cell by walking back to the table behind it, and a per-view install
    cannot see those events.
    """
    import inspect

    from spacr.qt.widgets import feature_dictionary as fd

    source = inspect.getsource(fd.install_context_menu_filter)
    assert "app.installEventFilter" in source
