"""A button explains itself at the foot of the window, not over itself.

The Home screen already worked this way: a module tile carries no
tooltip, and hovering it writes its description into a bar across the
bottom of the page. A popup would be a second copy of the same sentence,
drawn on top of the grid the reader is using to choose.

The same argument is stronger for a button. A tooltip appears over the
button -- exactly where the pointer already is, and where the user is
about to click -- so it hides the thing it describes, and it flickers in
and out as the pointer crosses a row of them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent                       # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel,    # noqa: E402
                               QPushButton, QWidget)

from spacr.qt.widgets.hint_bar import (DEFAULT_HINT,    # noqa: E402
                                       HintBar, explain_through_the_bar,
                                       hint_bar_of)


def test_the_sentence_moves_off_the_button(qtbot):
    bar = HintBar()
    qtbot.addWidget(bar)
    button = QPushButton("Clear RAM")
    qtbot.addWidget(button)
    button.setToolTip("Frees cached memory. Asks first.")

    assert bar.explain(button) == "Frees cached memory. Asks first."
    assert button.toolTip() == "", "the popup it replaces must be gone"
    assert bar.explains(button) == "Frees cached memory. Asks first."


def test_hovering_writes_it_and_leaving_puts_the_default_back(qtbot):
    bar = HintBar()
    qtbot.addWidget(bar)
    button = QPushButton("Go")
    qtbot.addWidget(button)
    bar.explain(button, "Starts the run and cannot be undone.")

    QApplication.sendEvent(button, QEvent(QEvent.Enter))
    assert bar.text() == "Starts the run and cannot be undone."
    QApplication.sendEvent(button, QEvent(QEvent.Leave))
    assert bar.text() == DEFAULT_HINT


def test_a_screen_reader_does_not_lose_the_sentence(qtbot):
    """Removing the tooltip must not cost the assistive text with it."""
    bar = HintBar()
    qtbot.addWidget(bar)
    button = QPushButton("Go")
    qtbot.addWidget(button)
    bar.explain(button, "Starts the run.")

    assert button.accessibleDescription() == "Starts the run."


def test_a_control_with_nothing_to_say_is_not_watched(qtbot):
    """Otherwise crossing it would blank the sentence beside it."""
    bar = HintBar()
    qtbot.addWidget(bar)
    silent = QPushButton("?")
    qtbot.addWidget(silent)

    assert bar.explain(silent) == ""
    assert bar.count() == 0


def test_crossing_between_two_buttons_does_not_flash_the_default(qtbot):
    """Leave and Enter arrive in that order often enough to matter."""
    bar = HintBar()
    qtbot.addWidget(bar)
    first, second = QPushButton("A"), QPushButton("B")
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    bar.explain(first, "Does the first thing.")
    bar.explain(second, "Does the second thing.")

    QApplication.sendEvent(first, QEvent(QEvent.Enter))
    QApplication.sendEvent(second, QEvent(QEvent.Enter))
    QApplication.sendEvent(first, QEvent(QEvent.Leave))
    assert bar.text() == "Does the second thing.", (
        "the bar showed the button the pointer had left, then blanked")


def test_a_helper_finds_the_bar_from_anywhere_in_the_window(qtbot):
    """So a form deep in a dialog need not be handed the bar."""
    window = QWidget()
    qtbot.addWidget(window)
    from PySide6.QtWidgets import QVBoxLayout
    column = QVBoxLayout(window)
    inner = QWidget()
    column.addWidget(inner)
    bar = HintBar(parent=window)
    column.addWidget(bar)
    button = QPushButton("Go", inner)
    button.setToolTip("Starts the run.")

    assert hint_bar_of(button) is bar
    assert explain_through_the_bar(button) is True
    assert button.toolTip() == ""


def test_a_window_with_no_bar_keeps_the_tooltip(qtbot):
    """Explaining itself awkwardly beats explaining itself nowhere."""
    orphan = QPushButton("Go")
    qtbot.addWidget(orphan)
    orphan.setToolTip("Starts the run.")

    assert explain_through_the_bar(orphan) is False
    assert orphan.toolTip() == "Starts the run."


def test_it_wears_the_same_name_the_home_screen_bar_wears(qtbot):
    """One thing wearing one style, not two that look alike today."""
    bar = HintBar()
    qtbot.addWidget(bar)
    assert bar.objectName() == "HintBar"
    assert isinstance(bar, QLabel)
