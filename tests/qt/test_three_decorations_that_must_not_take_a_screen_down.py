"""Three guards where a failure must cost the feature, not the screen.

Instruction 288.

* ``FoldingSummaryView.copy_to_clipboard`` declines when there is no
  clipboard rather than raising into a button press.
* ``PivotBuilder.recompute`` reports a non-``PivotError`` fault as a
  message; ``PivotError`` is the expected refusal and has its own arm.
* ``loading_screen._role`` falls back to the literal colour it replaced,
  because the splash is the FIRST thing painted -- sometimes before the
  theme has been resolved, and always before anything else could report
  a problem.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# The clipboard
# ---------------------------------------------------------------------------

def test_copying_with_no_clipboard_declines_rather_than_raising(qtbot,
                                                                monkeypatch):
    """THE ARM. `QApplication.clipboard()` returns None on a platform or
    session with no clipboard at all."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.folding_summary import FoldingSummaryView

    view = FoldingSummaryView()
    qtbot.addWidget(view)
    view.setPlainText("something worth copying")

    monkeypatch.setattr(QApplication, "clipboard", staticmethod(lambda: None))

    assert view.copy_to_clipboard() is False


def test_copying_with_a_clipboard_actually_copies(qtbot):
    """So the decline above is about the clipboard being absent, not
    about the method never copying anything."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.folding_summary import FoldingSummaryView

    view = FoldingSummaryView()
    qtbot.addWidget(view)
    view.setPlainText("something worth copying")

    assert view.copy_to_clipboard() is True
    assert QApplication.clipboard().text() == "something worth copying"


def test_copying_nothing_is_declined_earlier(qtbot):
    """The neighbouring guard, so the two are not confused."""
    from spacr.qt.widgets.folding_summary import FoldingSummaryView

    view = FoldingSummaryView()
    qtbot.addWidget(view)
    view.setPlainText("   \n  ")

    assert view.copy_to_clipboard() is False


# ---------------------------------------------------------------------------
# The pivot
# ---------------------------------------------------------------------------

def _frame():
    return pd.DataFrame({
        "plate": ["p1", "p1", "p2", "p2"],
        "gene": ["a", "b", "a", "b"],
        "value": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def builder(qtbot):
    from spacr.qt.widgets.pivot_builder import PivotPanel

    widget = PivotPanel()
    qtbot.addWidget(widget)
    widget.set_frame(_frame())
    # A SPEC THAT REACHES pivot(). An empty spec is refused earlier with
    # its own message -- "drop a column onto Rows" -- so without this the
    # arms under test are never reached and both tests pass vacuously
    # against whatever that message happens to be.
    from spacr.qt.widgets.pivot_spec import PivotSpec

    widget.set_spec(PivotSpec(rows=("plate",), values=("value",)))
    assert not widget.spec().is_empty, "the spec is still empty"
    return widget


def test_an_unexpected_pivot_fault_becomes_a_message(builder, monkeypatch):
    """THE ARM: anything that is not a PivotError."""
    from spacr.qt.widgets import pivot_builder

    def _explode(*_args, **_kwargs):
        raise ZeroDivisionError("something went wrong inside pivot()")

    monkeypatch.setattr(pivot_builder, "pivot", _explode)

    assert builder.recompute() is None       # must not raise
    assert "could not build that table" in builder.notice.text()


def test_the_expected_refusal_reads_differently(builder, monkeypatch):
    """So the arm above is not catching the refusal path by accident.

    A PivotError carries its own explanation and must NOT be wrapped in
    "could not build that table" -- that wrapper is what marks a fault.
    """
    from spacr.qt.widgets import pivot_builder
    from spacr.qt.widgets.pivot_spec import PivotError

    def _refuse(*_args, **_kwargs):
        raise PivotError("pick a column to pivot on first")

    monkeypatch.setattr(pivot_builder, "pivot", _refuse)

    assert builder.recompute() is None
    assert builder.notice.text() == "pick a column to pivot on first"
    assert "could not build that table" not in builder.notice.text()


# ---------------------------------------------------------------------------
# The splash palette
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_be_read_gives_the_literal_back(monkeypatch):
    """THE ARM. The splash is painted before the theme is resolved, so a
    palette lookup that raises would replace it with a traceback."""
    import spacr.qt.theme as theme

    from spacr.qt.widgets.loading_screen import _role

    def _explode():
        raise RuntimeError("the theme is not resolved yet")

    monkeypatch.setattr(theme, "palette_for", _explode)

    assert _role("bg", "#101114") == "#101114"


def test_a_readable_theme_is_preferred_over_the_literal(monkeypatch):
    """So the fallback is a fallback and not the only path."""
    import spacr.qt.theme as theme

    from spacr.qt.widgets.loading_screen import _role

    monkeypatch.setattr(theme, "palette_for", lambda: {"bg": "#ABCDEF"})

    assert _role("bg", "#101114") == "#ABCDEF"


def test_a_role_the_theme_does_not_know_falls_back_too(monkeypatch):
    """An empty or missing value is not an answer, and must not be
    painted as one."""
    import spacr.qt.theme as theme

    from spacr.qt.widgets.loading_screen import _role

    monkeypatch.setattr(theme, "palette_for", lambda: {"bg": ""})

    assert _role("bg", "#101114") == "#101114"
    assert _role("no_such_role", "#101114") == "#101114"
