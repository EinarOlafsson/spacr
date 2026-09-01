"""Three guards that carried a bare pragma with no reason recorded.

Instruction 288. An unexplained ``# pragma: no cover`` is the kind most
worth checking: nothing says why anybody believed the line could not run,
so there is nothing to disagree with when it turns out it can.

* ``_RegressionBackendField.eventFilter`` reads ``combo.view()``, which
  raises when the combo's C++ half has gone -- an event filter outlives
  the widget it watches.
* ``_hover_popup_row`` reads ``event.position()``, which is Qt 6. The
  fallback is ``event.pos()`` for anything that only has the Qt 5
  spelling.
* ``load_the_example_screen`` falls back to ``apply_settings_dict`` when
  the input table cannot take paths per side.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# A combo whose C++ half has gone
# ---------------------------------------------------------------------------

def test_an_event_filter_survives_a_combo_that_is_already_gone(qtbot):
    """THE ARM. An event filter outlives the widget it watches, so this
    is reached during teardown rather than never.

    A REAL field, built the way the tests that already exist build one.
    `__new__` alone leaves no QObject behind `super().eventFilter`, which
    made the first draft of this test skip -- and a skip proves nothing.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens import settings_model as sm

    field = sm._RegressionBackendField(regression_type="ols")
    qtbot.addWidget(field)

    asked = []

    class _Gone:
        def view(self):
            asked.append(True)
            raise RuntimeError("Internal C++ object already deleted.")

    field.combo = _Gone()

    watched = QWidget()
    qtbot.addWidget(watched)

    result = field.eventFilter(watched, QEvent(QEvent.Type.Enter))

    assert asked == [True], "the combo's view was never asked for"
    assert result is False, (
        "the filter consumed an event it does not handle")


def test_a_live_combo_is_consulted_normally(qtbot):
    """So the arm above is about the failure, not a filter that always
    defers to its superclass."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens import settings_model as sm

    field = sm._RegressionBackendField(regression_type="ols")
    qtbot.addWidget(field)
    watched = QWidget()
    qtbot.addWidget(watched)

    assert field.combo.view() is not None, "the real combo has no view"
    assert field.eventFilter(watched, QEvent(QEvent.Type.Enter)) is False


# ---------------------------------------------------------------------------
# An event with only the Qt 5 spelling
# ---------------------------------------------------------------------------

def test_an_event_with_no_position_falls_back_to_pos(qtbot):
    """THE ARM. `position()` is Qt 6; `pos()` is what came before."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QListView

    from spacr.qt.screens.settings_model import _RegressionBackendField

    field = _RegressionBackendField.__new__(_RegressionBackendField)
    view = QListView()
    qtbot.addWidget(view)

    asked = []

    class _OldEvent:
        def pos(self):
            asked.append("pos")
            return QPoint(1, 1)

    field._hover_popup_row(view, _OldEvent())    # must not raise

    assert asked == ["pos"], "the Qt 5 fallback was never used"


def test_an_event_with_a_position_uses_it(qtbot):
    """So the fallback is a fallback, not the only path."""
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QListView

    from spacr.qt.screens.settings_model import _RegressionBackendField

    field = _RegressionBackendField.__new__(_RegressionBackendField)
    view = QListView()
    qtbot.addWidget(view)

    asked = []

    class _NewEvent:
        def position(self):
            asked.append("position")
            return QPointF(1.0, 1.0)

        def pos(self):
            asked.append("pos")
            raise AssertionError("pos() was preferred over position()")

    field._hover_popup_row(view, _NewEvent())

    assert asked == ["position"]
