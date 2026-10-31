"""``qtbot.waitExposed(widget)`` written as a statement after ``show()`` waits.

pytest-qt's ``waitExposed`` is a context manager that waits when its block
exits, so the bare call this suite uses everywhere waited for nothing; on an
X display the tests then painted or dragged a window that was not yet mapped.
``tests/conftest.py`` makes the bare call wait for an already-shown widget and
leaves the ``with`` form for a widget that is not shown yet as pytest-qt has it.
"""
import pytest

pytest.importorskip("PySide6")
pytestqt_qtbot = pytest.importorskip("pytestqt.qtbot")

from PySide6.QtCore import Qt                                  # noqa: E402
from PySide6.QtWidgets import QWidget                          # noqa: E402
from pytestqt.exceptions import TimeoutError as QtBotTimeout   # noqa: E402


class _Recording:
    """Stands in for ``QTest`` and records each exposure wait in order."""

    def __init__(self, real, events, answer=None):
        self._real = real
        self._events = events
        self._answer = answer

    def qWaitForWindowExposed(self, widget, timeout):          # noqa: N802
        self._events.append(("wait", widget.isVisible(), timeout))
        if self._answer is not None:
            return self._answer
        return self._real.qWaitForWindowExposed(widget, timeout)


@pytest.fixture
def waits(monkeypatch):
    """The exposure waits pytest-qt makes, in order."""
    events = []
    api = pytestqt_qtbot.qt_api.QtTest
    monkeypatch.setattr(api, "QTest", _Recording(api.QTest, events))
    return events


def test_the_wrapper_is_installed():
    assert getattr(pytestqt_qtbot.QtBot.waitExposed,
                   "_spacr_waits_when_shown", False)


def test_a_bare_call_after_show_waits_until_the_window_is_exposed(qtbot, waits):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(120, 80)
    widget.show()
    qtbot.waitExposed(widget, timeout=4000)
    assert waits == [("wait", True, 4000)]
    assert widget.windowHandle() is not None
    assert widget.windowHandle().isExposed()


def test_a_with_block_around_show_still_waits_when_it_ends(qtbot, waits):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.resize(120, 80)
    with qtbot.waitExposed(widget):
        assert waits == []
        widget.show()
    assert waits == [("wait", True, 5000)]
    assert widget.windowHandle().isExposed()


def test_a_with_block_around_an_already_shown_widget_does_not_wait_twice(qtbot, waits):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.show()
    with qtbot.waitExposed(widget):
        widget.show()
    assert waits == [("wait", True, 5000)]


def test_a_bare_call_before_show_is_pytest_qts_own_context_manager(qtbot, waits):
    widget = QWidget()
    qtbot.addWidget(widget)
    waiting = qtbot.waitExposed(widget)
    assert type(waiting) is pytestqt_qtbot._WaitWidgetContextManager
    assert waits == []


def test_a_window_that_is_never_exposed_fails_the_bare_call(qtbot, monkeypatch):
    events = []
    api = pytestqt_qtbot.qt_api.QtTest
    monkeypatch.setattr(api, "QTest", _Recording(api.QTest, events, answer=False))
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.show()
    with pytest.raises(QtBotTimeout, match="not exposed in 250 ms"):
        qtbot.waitExposed(widget, timeout=250)
    assert events == [("wait", True, 250)]


def test_a_real_window_kept_off_screen_times_out(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.show()
    assert widget.isVisible()
    with pytest.raises(QtBotTimeout, match="not exposed in 300 ms"):
        qtbot.waitExposed(widget, timeout=300)
