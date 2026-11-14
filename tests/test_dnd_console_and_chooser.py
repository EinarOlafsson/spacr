"""Drag-and-drop: which console hears a rejected drop, what it says, and the chooser.

``_find_console`` decides where a rejected drop is reported, walking the
screen, its window, the window's screens in visit order and finally any
console panel; ``_report_drop_problem`` builds the report and delivers it
(console, AI flow, a log pane, the status bar); ``choose_one_dialog`` asks
which of several valid candidates was meant. Each test walks a case table.
The modal chooser is driven through its real buttons by a timer that waits
for the modal to be up (the file lives outside tests/qt, whose conftest refuses
every QDialog.exec).
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (  # noqa: E402
    QEvent, QModelIndex, QPointF, Qt, QTimer,
)
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialogButtonBox, QLabel, QListWidget, QPlainTextEdit,
    QWidget,
)

from spacr.qt import dnd  # noqa: E402

pytestmark = pytest.mark.qt


class _Console:
    """The console surface the drop report reaches for."""

    def __init__(self, *, errors=True, provider="claude", ai_active=True):
        self.errors = []
        self.stdout = []
        self.ai = []
        self._ai_active = ai_active
        self._provider = provider
        if not errors:
            self.append_error = None

    def append_error(self, text):
        self.errors.append(text)

    def append_stdout(self, text):
        self.stdout.append(text)

    def _current_provider(self):
        return self._provider

    def open_error_flow(self, text, **kwargs):
        self.ai.append((text, kwargs))


# ---------------------------------------------------------------------------
# _find_console
# ---------------------------------------------------------------------------

def test_the_drop_report_goes_to_the_nearest_console_by_visit_order(
        qtbot, monkeypatch, subtests):
    """Own, window, visited screens newest first, unvisited newest first, then a panel."""
    from spacr.qt.widgets.console_panel import ConsolePanel

    keep = []  # qtbot holds widgets weakly; the windows must outlive arrange()

    def window():
        top = QWidget()
        qtbot.addWidget(top)
        keep.append(top)
        return top

    def screens(top, *names, consoles=()):
        made = {}
        for name in names:
            made[name] = QWidget(top)
            if name in consoles:
                made[name]._console = _Console()
        top._screens = made
        return made

    def own_console():
        top = window()
        screen = QWidget(top)
        screen._console = _Console()
        top._console = _Console()
        return screen, screen._console

    def window_console():
        top = window()
        top._console = _Console()
        return QWidget(top), top._console

    def borrowed():
        top = window()
        made = screens(top, "plain", "regression", consoles=("regression",))
        top._visit_order = ["plain", "regression"]
        return QWidget(top), made["regression"]._console

    def latest_visit(order, winner):
        def arrange():
            top = window()
            made = screens(top, "first", "second",
                           consoles=("first", "second"))
            top._visit_order = order
            return QWidget(top), made[winner]._console
        return arrange

    def visited_beats_unvisited(order, winner):
        def arrange():
            top = window()
            made = screens(top, "visited", "unvisited", "tool",
                           consoles=("visited", "unvisited"))
            top._visit_order = order
            return made["tool"], made[winner]._console
        return arrange

    def newest_unvisited():
        top = window()
        made = screens(top, "older", "newer", consoles=("older", "newer"))
        top._visit_order = None
        return QWidget(top), made["newer"]._console

    def panel():
        top = window()
        ConsolePanel(parent=top)
        last = ConsolePanel(parent=top)
        top._screens = {"tool": QWidget(top)}
        return QWidget(top), last

    def nothing():
        return QWidget(window()), None

    def not_a_widget():
        class Bare:
            pass
        return Bare(), None

    cases = [
        ("the screen's own console", own_console),
        ("the hosting window's console", window_console),
        ("a registered screen's console is borrowed", borrowed),
        ("the most recent visit wins", latest_visit(["first", "second"],
                                                    "second")),
        ("the order follows the visits", latest_visit(["second", "first"],
                                                      "first")),
        ("a visited screen beats an unvisited one",
         visited_beats_unvisited(["visited", "tool"], "visited")),
        ("with no visits the newest registration wins",
         visited_beats_unvisited([], "unvisited")),
        ("a missing visit order is no visits", newest_unvisited),
        ("the last console panel in the window", panel),
        ("a window with nowhere to say it", nothing),
        ("something that is not a widget", not_a_widget),
    ]
    for case, arrange in cases:
        with subtests.test(msg=case):
            screen, expected = arrange()
            assert dnd._find_console(screen) is expected

    with subtests.test(msg="a console panel that will not import"), \
            monkeypatch.context() as patch:
        top = window()
        ConsolePanel(parent=top)
        patch.setitem(sys.modules, "spacr.qt.widgets.console_panel", None)
        assert dnd._find_console(QWidget(top)) is None


# ---------------------------------------------------------------------------
# _report_drop_problem
# ---------------------------------------------------------------------------

def test_a_rejected_drop_report_reaches_the_console_ai_pane_and_status_bar(
        qtbot, monkeypatch, subtests, tmp_path):
    """The exact report text, and every surface it is delivered to or kept from."""
    routing = {"on": True}
    monkeypatch.setattr("spacr.qt.ai.settings.get_route_errors_through_ai",
                        lambda: routing["on"])
    path = tmp_path / "wrong.txt"
    base = (f"[drop rejected] {path}\nReason: not a table\n"
            "Suggestion: drop a CSV\n")
    with_alternatives = (
        f"[drop rejected] {path}\nReason: not a table\nSuggestion: drop a "
        f"CSV\nCompatible nearby paths: {tmp_path / 'a'}, {tmp_path / 'b'}\n")
    flow = {"active_app": "regression", "show_raw": False}

    cases = [
        # case, console kwargs (None = no console), routing on, pane attr,
        # alternatives, expected message, console errors?, ai routed?,
        # pane text
        ("a callable AI flag is asked", dict(ai_active=lambda: True), True,
         "_summary", (), base, "errors", True, ""),
        ("a plain True AI flag routes too", dict(ai_active=True), True, None,
         (), base, "errors", True, None),
        ("an AI flag that says no", dict(ai_active=lambda: False), True,
         None, (), base, "errors", False, None),
        ("routing switched off", dict(), False, None, (), base, "errors",
         False, None),
        ("no AI provider", dict(provider=None), True, None, (), base,
         "errors", False, None),
        ("a console with only stdout", dict(errors=False), True, None, (),
         base, "stdout", True, None),
        ("no console, a summary pane", None, True, "_summary",
         (tmp_path / "a", tmp_path / "b"), with_alternatives, None, False,
         with_alternatives.rstrip()),
        ("no console, a log pane", None, True, "_log", (), base, None, False,
         base.rstrip()),
        ("no console, a console-text pane", None, True, "_console_text", (),
         base, None, False, base.rstrip()),
    ]
    for (case, console_kwargs, routed, pane_attr, alternatives, message,
         delivered, ai, pane_text) in cases:
        with subtests.test(msg=case):
            routing["on"] = routed
            said = []

            class Screen(QWidget):
                def _set_status(self, text):
                    said.append(text)

            screen = Screen()
            qtbot.addWidget(screen)
            screen.app_key = "regression"
            console = None
            if console_kwargs is not None:
                console = _Console(**console_kwargs)
                screen._console = console
            pane = None
            if pane_attr is not None:
                pane = QPlainTextEdit(screen)
                setattr(screen, pane_attr, pane)

            got = dnd._report_drop_problem(screen, path, "not a table",
                                           "drop a CSV",
                                           alternatives=alternatives)
            assert got == message
            assert said == ["Drop rejected: not a table Suggestion: drop a "
                            "CSV"]
            if console is not None:
                assert getattr(console, delivered) == [message]
                assert console.ai == ([(message, flow)] if ai else [])
            if pane is not None:
                assert pane.toPlainText() == pane_text

    with subtests.test(msg="a status bar that throws loses only itself"):
        attempts = []

        class Throwing(QWidget):
            def _set_status(self, text):
                attempts.append(text)
                raise RuntimeError("status bar is gone")

        screen = Throwing()
        qtbot.addWidget(screen)
        screen._log = QPlainTextEdit(screen)
        message = dnd._report_drop_problem(screen, tmp_path / "x.txt", "no",
                                           "try again")
        assert attempts == ["Drop rejected: no Suggestion: try again"]
        assert screen._log.toPlainText() == message.rstrip()


# ---------------------------------------------------------------------------
# choose_one_dialog
# ---------------------------------------------------------------------------

def _answer_modal(act, tries=200):
    """Run ``act(dialog)`` once a modal window is up; never leave it open."""
    state = {"tries": 0}

    def poll():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            state["tries"] += 1
            if state["tries"] < tries:
                QTimer.singleShot(5, poll)
            return
        try:
            act(dialog)
        finally:
            if dialog.isVisible():
                dialog.close()

    QTimer.singleShot(0, poll)


def _click(dialog, standard):
    dialog.findChild(QDialogButtonBox).button(standard).click()


def _double_click_row(listing, row):
    """Press/release/double-click/release: what the offscreen view needs."""
    rect = listing.visualItemRect(listing.item(row))
    where = QPointF(rect.center())
    globally = listing.viewport().mapToGlobal(where)
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                 QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
        QApplication.sendEvent(listing.viewport(), QMouseEvent(
            kind, where, globally, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))


def test_the_which_one_chooser_returns_the_picked_option_or_none(
        qtbot, subtests):
    """OK returns the current row, a double click accepts, Cancel answers ``None``."""
    def pick(row):
        def act(dialog):
            dialog.findChild(QListWidget).setCurrentRow(row)
            _click(dialog, QDialogButtonBox.Ok)
        return act

    def double_click(row):
        return lambda dialog: _double_click_row(
            dialog.findChild(QListWidget), row)

    def no_selection(dialog):
        dialog.findChild(QListWidget).setCurrentIndex(QModelIndex())
        _click(dialog, QDialogButtonBox.Ok)

    options = ["cell", "nucleus", "pathogen"]
    cases = [
        ("OK on the first row by default",
         lambda dialog: _click(dialog, QDialogButtonBox.Ok), "cell"),
        ("OK on a chosen row", pick(1), "nucleus"),
        ("a double click accepts its row", double_click(2), "pathogen"),
        ("Cancel answers nothing",
         lambda dialog: _click(dialog, QDialogButtonBox.Cancel), None),
        ("OK with nothing selected answers nothing", no_selection, None),
    ]
    for case, act, expected in cases:
        with subtests.test(msg=case):
            parent = QWidget()
            qtbot.addWidget(parent)
            seen = {}

            def inspect(dialog, act=act):
                listing = dialog.findChild(QListWidget)
                seen["title"] = dialog.windowTitle()
                seen["rows"] = [listing.item(i).text()
                                for i in range(listing.count())]
                seen["current"] = listing.currentRow()
                seen["header"] = dialog.findChild(QLabel).text()
                act(dialog)

            _answer_modal(inspect)
            chosen = dnd.choose_one_dialog(
                parent, "plate1.db holds 3 tables.",
                "Which one should be loaded?", options)
            assert chosen == expected
            assert seen == {
                "title": "Which one?", "rows": options, "current": 0,
                "header": "<b>plate1.db holds 3 tables.</b><br>Which one "
                          "should be loaded?"}
