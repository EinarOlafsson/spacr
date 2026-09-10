"""Clearing the figures is destructive, so it reads as destructive.

Asked for on 2026-08-17: "sorry there is already a clear figures button, just
make it red like other "negative" butons".
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _label(qtbot):
    from spacr.qt.widgets.figure_queue import _ClearFiguresLabel

    widget = _ClearFiguresLabel()
    qtbot.addWidget(widget)
    return widget


def test_it_rests_red(qtbot):
    from spacr.qt.theme import active_palette

    widget = _label(qtbot)

    assert active_palette()["error"].lower() in widget.styleSheet().lower()


def test_the_red_is_the_theme_s_own(qtbot):
    """Not a hex typed into the widget. A literal here is a fourth opinion
    about what destructive looks like, and it stays dark on the light theme."""
    import inspect

    from spacr.qt.widgets.figure_queue import _ClearFiguresLabel

    source = inspect.getsource(_ClearFiguresLabel._restyle)
    assert 'palette["error"]' in source


def test_it_is_not_the_dim_resting_colour_any_more(qtbot):
    from spacr.qt.theme import active_palette

    widget = _label(qtbot)
    palette = active_palette()

    assert palette["fg_dim"].lower() not in widget.styleSheet().lower()


def test_the_click_still_flashes_the_accent(qtbot):
    """The flash is the app-wide "your click landed" mark, shared with the
    console's copy glyph. A control inventing its own would be inconsistent
    in the other direction."""
    from spacr.qt.theme import active_palette

    widget = _label(qtbot)
    widget.flash()

    assert active_palette()["accent"].lower() in widget.styleSheet().lower()


def test_it_still_emits_on_click(qtbot):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    widget = _label(qtbot)
    widget.resize(120, 24)
    fired = []
    widget.clicked.connect(lambda: fired.append(True))

    widget.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPoint(10, 10),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    # The colour change must not have cost the control its behaviour --
    # `qtbot.waitSignal` alone asserts nothing the hygiene guard can see, and
    # it is right to say so: a context manager that times out still leaves a
    # test that passed for the wrong reason if the body raised first.
    assert fired == [True]
    # A release OUTSIDE the label is a cancelled drag, not a click.
    widget.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.Type.MouseButtonRelease, QPoint(500, 500),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert fired == [True], "a release off the label counted as a click"
