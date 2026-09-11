"""The guard instruction 380's remaining fix needs before it can be written.

380's last lever is to stop calling `setStyleSheet` on the QApplication and
set it per top-level window instead, marking hidden module screens dirty so
their `showEvent` applies it. 378 measured why: 107 ms for the visible
screen against 587 for the whole application.

THE REASON IT HAS NOT BEEN WRITTEN is named in 380 itself:
`app.setStyleSheet` covers every widget that exists AND every one created
later -- dialogs, popups, menus, a screen built after the change -- and
"the failure mode is a dialog opening in the previous theme, which is
exactly 'compromising functionality'".

AND THAT FAILURE MODE IS MEASURABLE HERE. 380 says it "needs a display to
verify and a session that can watch it happen", and that is true of whether
a theme LOOKS right. It is not true of this: a stylesheet colour resolves
into the widget's palette, and an offscreen widget reports it. So the
specific regression the fix risks can be caught by a test, and this file is
that test -- written BEFORE the change, which is what this item's own
method demands.

    WHAT THIS FILE DOES NOT DO is check that the theme is pretty, legible
    or correct. It checks ONE thing: that a widget created AFTER a theme
    change resolves to the SAME theme as one created before it.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import Qt                                # noqa: E402
from PySide6.QtGui import QPalette                           # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QLabel,  # noqa: E402
                               QMainWindow, QMenu, QWidget)

#: Two sheets that differ only in a colour nothing else uses, so a widget
#: reading the wrong one is unambiguous rather than a near-miss.
FIRST = "QLabel { color: rgb(11, 22, 33); }"
SECOND = "QLabel { color: rgb(200, 100, 50); }"
FIRST_HEX = "#0b1621"
SECOND_HEX = "#c86432"


def _resolved(widget) -> str:
    """The colour this label actually paints with, after a polish."""
    app = QApplication.instance()
    widget.ensurePolished()
    app.processEvents()
    return widget.palette().color(QPalette.WindowText).name()


@pytest.fixture
def app(qtbot):
    existing = QApplication.instance()
    yield existing
    existing.setStyleSheet("")


def test_the_harness_can_see_a_theme_at_all(app, qtbot):
    """The control. A test that cannot see the FIRST theme cannot see a
    regression into it either, and would pass for ever."""
    app.setStyleSheet(FIRST)
    label = QLabel()
    qtbot.addWidget(label)
    assert _resolved(label) == FIRST_HEX


def test_a_widget_made_after_the_change_reads_the_new_theme(app, qtbot):
    """THE PROPERTY THE PER-WINDOW FIX MUST NOT BREAK.

    Under `app.setStyleSheet` this is free -- the sheet covers widgets that
    do not exist yet. Under a per-window sheet it is the thing that has to
    be arranged, and this is where it would fail.
    """
    app.setStyleSheet(FIRST)
    before = QLabel()
    qtbot.addWidget(before)
    assert _resolved(before) == FIRST_HEX

    app.setStyleSheet(SECOND)
    after = QLabel()
    qtbot.addWidget(after)

    assert _resolved(after) == SECOND_HEX, (
        "a widget created after the theme change is painting the previous "
        "theme; a per-window stylesheet has not reached new widgets")
    assert _resolved(before) == SECOND_HEX, (
        "a widget that existed across the change kept the old theme")


def test_a_dialog_opened_after_the_change_reads_the_new_theme(app, qtbot):
    """The exact sentence 380 uses for the risk: 'a dialog opening in the
    previous theme'. A dialog is its own top-level window, which is
    precisely what a per-window sheet would have to remember."""
    app.setStyleSheet(FIRST)
    app.setStyleSheet(SECOND)

    dialog = QDialog()
    qtbot.addWidget(dialog)
    inner = QLabel(dialog)

    assert _resolved(inner) == SECOND_HEX, (
        "a dialog opened after the theme change is in the previous theme")


def test_a_window_built_after_the_change_reads_the_new_theme(app, qtbot):
    """And a whole module screen, which 380's fix would mark dirty and
    re-sheet on `showEvent` rather than paint immediately."""
    app.setStyleSheet(FIRST)
    app.setStyleSheet(SECOND)

    window = QMainWindow()
    qtbot.addWidget(window)
    body = QWidget()
    window.setCentralWidget(body)
    inner = QLabel(body)

    assert _resolved(inner) == SECOND_HEX


def test_it_catches_a_sheet_that_reached_only_one_window(app, qtbot):
    """PROOF THE GUARD WORKS, by doing the wrong thing on purpose.

    This is what a per-window implementation looks like when it forgets a
    window: the sheet is set on ONE widget's tree and a second top-level
    never hears about it. Without this case the four tests above would
    pass under `app.setStyleSheet` for ever and prove nothing about the
    change they exist to protect.
    """
    app.setStyleSheet(FIRST)
    first_window = QWidget()
    qtbot.addWidget(first_window)
    first_label = QLabel(first_window)

    # The "fix" applied per-window, to this window only.
    app.setStyleSheet("")
    first_window.setStyleSheet(SECOND)

    second_window = QWidget()
    qtbot.addWidget(second_window)
    second_label = QLabel(second_window)

    assert _resolved(first_label) == SECOND_HEX
    assert _resolved(second_label) != SECOND_HEX, (
        "the second window somehow got the sheet; this negative case is "
        "what makes the four tests above meaningful")


# ---------------------------------------------------------------------------
# The change itself, 2026-09-11. Everything above holds the PROPERTY; what
# follows exercises the implementation that has to keep it.
# ---------------------------------------------------------------------------


@pytest.fixture
def per_window(qtbot):
    """A clean application with the per-window sheet installed and removed."""
    from spacr.qt import theme

    app = QApplication.instance()
    app.setStyleSheet("")
    yield theme.apply_stylesheet_per_window
    theme.apply_stylesheet_per_window(app, "")
    app.setStyleSheet("")


def test_the_sheet_reaches_a_window_that_already_exists(per_window, qtbot):
    """The control for this half: without it the rest prove nothing."""
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    label = QLabel(window)
    window.show()

    assert per_window(app, FIRST) >= 1
    assert _resolved(label) == FIRST_HEX


def test_a_theme_change_reaches_a_window_that_lived_across_it(per_window,
                                                              qtbot):
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    label = QLabel(window)
    window.show()

    per_window(app, FIRST)
    assert _resolved(label) == FIRST_HEX
    per_window(app, SECOND)
    assert _resolved(label) == SECOND_HEX


@pytest.mark.parametrize("make", [
    pytest.param(lambda: QDialog(), id="a dialog"),
    pytest.param(lambda: QMainWindow(), id="a whole window"),
    pytest.param(lambda: QMenu(), id="a menu"),
    pytest.param(lambda: QWidget(None, Qt.ToolTip), id="a tooltip"),
])
def test_a_window_born_after_the_change_is_not_in_the_previous_theme(
        per_window, qtbot, make):
    """380's NAMED RISK, and the two it recorded as needing a display.

    "`app.setStyleSheet` covers every widget that exists AND every one
    created later -- dialogs, popups, menus, a screen built after the
    change. Per-window application has to reproduce that, and the failure
    mode is a dialog opening in the previous theme."

    A MENU AND A TOOLTIP ARE ORDINARY TOP-LEVEL WIDGETS, which is why they
    are here rather than in the "needs a display" note they were left in.
    Qt creates them itself and a test cannot open a NATIVE menu, but the
    property at stake is not how the menu looks -- it is whether the
    widget Qt creates gets the sheet, and a `QMenu` constructed directly
    answers that with the same Polish event the real one gets.
    """
    app = QApplication.instance()
    per_window(app, FIRST)
    per_window(app, SECOND)

    window = make()
    qtbot.addWidget(window)
    label = QLabel(window)
    window.show()
    app.processEvents()

    assert _resolved(label) == SECOND_HEX, (
        "a window created after the theme change opened in the previous "
        "theme, which is exactly what 380 says must not happen")


def test_the_application_sheet_is_left_empty(per_window, qtbot):
    """Where the saving comes from, stated as a property.

    `QApplication.setStyleSheet` repolishes every widget the process owns,
    including the thousands on module screens nobody is looking at. If this
    starts failing, the sheet is being set globally again and the cost is
    back whether or not anything looks different.
    """
    from spacr.qt.theme import window_stylesheet

    app = QApplication.instance()
    per_window(app, SECOND)
    assert app.styleSheet() == ""
    assert window_stylesheet(app) == SECOND


def test_applying_the_same_sheet_twice_does_not_re_sheet_a_window(
        per_window, qtbot):
    """The serial is what makes Polish and Show cost one application, not two.

    Without it every `Show` event on every top-level would set the sheet
    again -- a full repolish of that window's tree, which is the cost this
    change exists to avoid, paid on every popup.
    """
    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    window.show()

    assert per_window(app, FIRST) >= 1
    calls = []
    real = QWidget.setStyleSheet

    def counting(self, sheet):
        calls.append(self)
        return real(self, sheet)

    QWidget.setStyleSheet = counting
    try:
        for _ in range(5):
            window.show()
            app.processEvents()
    finally:
        QWidget.setStyleSheet = real
    assert window not in calls
