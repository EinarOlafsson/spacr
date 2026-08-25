"""Drag-and-drop plumbing: where a rejected drop is reported, and the choosers.

The two modal dialogs are driven rather than replaced: a timer waits for the
modal to be up and then clicks its real buttons, so the assertions are about
what the dialog returns to its caller.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QTimer, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialogButtonBox, QListWidget, QMessageBox, QPlainTextEdit,
    QWidget,
)

from spacr.qt import dnd  # noqa: E402

pytestmark = pytest.mark.qt


def _answer_modal(qtbot, act, tries=200):
    """Run ``act(dialog)`` once a modal window is actually up.

    ``QDialog.exec`` and ``QMessageBox.warning`` both spin their own event
    loop, and neither can be monkeypatched -- they are Shiboken methods, and
    assigning over them leaves the real modal running and wedges the run. A
    timer that fires inside that loop is how a test reaches the buttons.
    """
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
            # Never leave a modal up: an exception inside a timer slot is
            # printed and swallowed by Qt, and the run would then hang in
            # exec() with no failure to report.
            if dialog.isVisible():
                dialog.close()

    QTimer.singleShot(0, poll)


def _click(dialog, standard):
    dialog.findChild(QDialogButtonBox).button(standard).click()


# ---------------------------------------------------------------------------
# install_for
# ---------------------------------------------------------------------------

def test_a_handler_that_cannot_be_built_leaves_the_screen_usable(
        qtbot, monkeypatch, caplog):
    """A missing dropzone is a missing convenience, not a broken screen."""
    import spacr.qt.dnd_handlers as handlers

    def explode(_key):
        raise RuntimeError("no handler registry today")

    monkeypatch.setattr(handlers, "get_handler", explode)
    target = QWidget()
    qtbot.addWidget(target)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.dnd"):
        assert dnd.install_for(target, "graph_builder") is False
    assert target.acceptDrops() is False
    assert not hasattr(target, "_dnd_handler")


def test_a_working_handler_is_installed_on_the_widget(qtbot):
    target = QWidget()
    qtbot.addWidget(target)
    assert dnd.install_for(target, "graph_builder") is True
    assert target.acceptDrops() is True
    assert target._dnd_screen is target


# ---------------------------------------------------------------------------
# Finding somewhere to say it
# ---------------------------------------------------------------------------

class _Console:
    """The console surface `_report_drop_problem` actually reaches for."""

    def __init__(self):
        self.errors = []
        self.stdout = []
        self.ai = []
        self._ai_active = True

    def append_error(self, text):
        self.errors.append(text)

    def append_stdout(self, text):
        self.stdout.append(text)

    def _current_provider(self):
        return "claude"

    def open_error_flow(self, text, **kwargs):
        self.ai.append((text, kwargs))


def test_a_screen_that_is_not_a_widget_has_no_console(qtbot):
    class Bare:
        pass

    assert dnd._find_console(Bare()) is None


def test_the_hosting_window_lends_its_console(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    console = _Console()
    window._console = console
    screen = QWidget(window)
    assert dnd._find_console(screen) is console


def test_a_registered_screens_console_is_borrowed_by_a_tool(qtbot):
    """A standalone tool has no console of its own; a sibling screen's will do."""
    window = QWidget()
    qtbot.addWidget(window)
    plain, has_console = QWidget(window), QWidget(window)
    has_console._console = _Console()
    window._screens = {"plain": plain, "regression": has_console}
    window._visit_order = ["plain", "regression"]
    assert dnd._find_console(QWidget(window)) is has_console._console


@pytest.mark.xfail(strict=True, reason=(
    "_find_console searches reversed(visit_order + list(screens)), so the "
    "reversed _screens keys come first and the most recently visited screen "
    "is only consulted after every registered screen"))
def test_the_most_recently_visited_screen_lends_its_console(qtbot):
    """Whose console a rejected drop lands in must follow where the user was.

    Both screens carry one, so the only thing that can choose between them
    is ``_visit_order`` -- which is what the code says it is for.
    """
    window = QWidget()
    qtbot.addWidget(window)
    first, second = QWidget(window), QWidget(window)
    first._console, second._console = _Console(), _Console()
    window._screens = {"first": first, "second": second}
    tool = QWidget(window)

    window._visit_order = ["first", "second"]
    assert dnd._find_console(tool) is second._console
    window._visit_order = ["second", "first"]
    assert dnd._find_console(tool) is first._console


def test_a_console_panel_in_the_window_is_found_by_type(qtbot):
    from spacr.qt.widgets.console_panel import ConsolePanel

    window = QWidget()
    qtbot.addWidget(window)
    panel = ConsolePanel(parent=window)
    window._screens = {"tool": QWidget(window)}
    assert dnd._find_console(QWidget(window)) is panel


def test_a_window_with_nothing_to_say_it_in_reports_none(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    assert dnd._find_console(QWidget(window)) is None


def test_a_console_panel_that_will_not_import_is_not_a_crash(
        qtbot, monkeypatch):
    """The console lookup is a convenience; it must not break the report."""
    import sys

    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.console_panel", None)
    window = QWidget()
    qtbot.addWidget(window)
    assert dnd._find_console(QWidget(window)) is None


# ---------------------------------------------------------------------------
# Reporting a rejected drop
# ---------------------------------------------------------------------------

def test_a_callable_ai_flag_is_asked_rather_than_truth_tested(
        qtbot, monkeypatch, tmp_path):
    """``_ai_active`` is a method on the real console, not an attribute."""
    monkeypatch.setattr(
        "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: True)
    screen = QWidget()
    qtbot.addWidget(screen)
    console = _Console()
    console._ai_active = lambda: True
    screen._console = console
    screen.app_key = "regression"

    message = dnd._report_drop_problem(
        screen, tmp_path / "wrong.txt", "not a table", "drop a CSV")
    assert console.ai and console.ai[0][0] == message
    assert console.ai[0][1]["active_app"] == "regression"


def test_a_tool_with_only_a_log_pane_still_gets_the_report(qtbot, tmp_path):
    screen = QWidget()
    qtbot.addWidget(screen)
    screen._summary = QPlainTextEdit(screen)
    message = dnd._report_drop_problem(
        screen, tmp_path / "wrong.txt", "not a mask folder",
        "drop the masks/ folder", alternatives=[tmp_path / "masks"])
    assert "Compatible nearby paths" in message
    assert message.rstrip() in screen._summary.toPlainText()


def test_a_status_bar_that_throws_does_not_swallow_the_drop_report(
        qtbot, tmp_path):
    said = []

    class Screen(QWidget):
        def _set_status(self, text):
            said.append(text)
            raise RuntimeError("status bar is gone")

    screen = Screen()
    qtbot.addWidget(screen)
    screen._log = QPlainTextEdit(screen)
    message = dnd._report_drop_problem(
        screen, tmp_path / "x.txt", "no", "try again")
    assert said and said[0].startswith("Drop rejected: no")
    assert message.rstrip() in screen._log.toPlainText()


# ---------------------------------------------------------------------------
# CSV routing
# ---------------------------------------------------------------------------

def test_a_header_that_cannot_be_read_is_no_header_at_all(tmp_path):
    """A folder named like a CSV must not raise out of the drop path."""
    folder = tmp_path / "looks_like.csv"
    folder.mkdir()
    assert dnd._csv_header(folder) == []
    assert dnd._looks_like_settings_csv(folder) is False


def test_a_data_csv_with_no_file_input_to_take_it_routes_nowhere(
        qtbot, tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("alpha,beta\n1,2\n")

    class Model:
        pass

    model = Model()
    model._widgets = {"score_data": QPlainTextEdit(),
                      "metadata_files": QPlainTextEdit()}
    screen = QWidget()
    qtbot.addWidget(screen)
    screen._settings_model = model
    assert dnd._route_data_csv_to_inputs(data, screen) is None


def test_a_routed_data_csv_is_announced_in_the_console(qtbot, tmp_path):
    from spacr.qt.widgets.file_list import FilePathListWidget

    data = tmp_path / "plate1_dv.csv"
    data.write_text("path,pred,plate,row,col\na,0.5,p1,r1,c1\n")

    class Model:
        pass

    model = Model()
    widget = FilePathListWidget(kind="table")
    qtbot.addWidget(widget)
    model._widgets = {"score_data": widget}

    class Screen(QWidget):
        def apply_settings_dict(self, values):
            return len(values)

    screen = Screen()
    qtbot.addWidget(screen)
    screen._settings_model = model
    screen._console = _Console()
    dnd._apply_settings_csv(data, screen)
    assert widget.get_value() == [str(data)]
    assert screen._console.stdout == [
        f"[drop] added {data.name} to score_data\n"]


def test_a_screen_that_imports_no_settings_ignores_the_drop(tmp_path):
    class Screen:
        pass

    csv_path = tmp_path / "settings.csv"
    csv_path.write_text("Key,Value\nsrc,/tmp\n")
    assert dnd._apply_settings_csv(csv_path, Screen()) is None


def test_a_settings_csv_that_will_not_load_says_so_in_a_dialog(
        qtbot, monkeypatch, tmp_path):
    """The failure reaches both the console report and a real message box."""
    csv_path = tmp_path / "settings.csv"
    csv_path.write_text("Key,Value\nsrc,/tmp\n")

    import spacr.utils

    def refuse(*_args, **_kwargs):
        raise ValueError("column mismatch")

    monkeypatch.setattr(spacr.utils, "load_settings", refuse)

    class Screen(QWidget):
        def apply_settings_dict(self, values):
            return len(values)

    screen = Screen()
    qtbot.addWidget(screen)
    screen._summary = QPlainTextEdit(screen)

    seen = []

    def dismiss(dialog):
        seen.append(dialog.text() if isinstance(dialog, QMessageBox) else "")
        dialog.close()

    _answer_modal(qtbot, dismiss)
    dnd._apply_settings_csv(csv_path, screen)
    assert seen == ["column mismatch"]
    assert "Settings CSV import failed" in screen._summary.toPlainText()


def test_a_settings_csv_in_the_second_spelling_still_imports(
        qtbot, tmp_path):
    csv_path = tmp_path / "settings.csv"
    csv_path.write_text("setting_key,setting_value\nsrc,/tmp/data\n")

    class Screen(QWidget):
        def __init__(self):
            super().__init__()
            self.applied = None

        def apply_settings_dict(self, values):
            self.applied = values
            return len(values)

    screen = Screen()
    qtbot.addWidget(screen)
    screen._console = _Console()
    dnd._apply_settings_csv(csv_path, screen)
    assert screen.applied == {"src": "/tmp/data"}
    assert screen._console.stdout == [
        f"[drop] imported 1 settings from {csv_path.name}\n"]


# ---------------------------------------------------------------------------
# The choosers
# ---------------------------------------------------------------------------

def test_choosing_one_of_several_answers_returns_that_one(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)

    def pick_second(dialog):
        listing = dialog.findChild(QListWidget)
        listing.setCurrentRow(1)
        _click(dialog, QDialogButtonBox.Ok)

    _answer_modal(qtbot, pick_second)
    chosen = dnd.choose_one_dialog(
        parent, "plate1.db holds 4 tables.", "Which one should be loaded?",
        ["cell", "nucleus", "pathogen"])
    assert chosen == "nucleus"


def test_cancelling_the_which_one_question_answers_nothing(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    _answer_modal(qtbot, lambda dialog: _click(dialog,
                                               QDialogButtonBox.Cancel))
    assert dnd.choose_one_dialog(parent, "two tables", "which?",
                                 ["a", "b"]) is None


def _double_click_row(listing, row):
    """Send the press/release/double-click sequence a real view expects.

    ``QTest.mouseDClick`` produces no ``itemDoubleClicked`` on the offscreen
    platform; the explicit four-event sequence does, and it is the same path
    a window-system double click takes.
    """
    rect = listing.visualItemRect(listing.item(row))
    where = QPointF(rect.center())
    globally = listing.viewport().mapToGlobal(where)
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                 QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
        QApplication.sendEvent(listing.viewport(), QMouseEvent(
            kind, where, globally, Qt.LeftButton, Qt.LeftButton,
            Qt.NoModifier))


def test_double_clicking_an_answer_accepts_it(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)

    def double_click(dialog):
        listing = dialog.findChild(QListWidget)
        _double_click_row(listing, 2)

    _answer_modal(qtbot, double_click)
    assert dnd.choose_one_dialog(parent, "three masks", "which?",
                                 ["a", "b", "c"]) == "c"


def test_the_did_you_mean_chooser_returns_the_folder_that_would_work(qtbot,
                                                                    tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    alternatives = [tmp_path / "plate1", tmp_path / "plate2"]

    def pick_second(dialog):
        dialog.findChild(QListWidget).setCurrentRow(1)
        _click(dialog, QDialogButtonBox.Ok)

    _answer_modal(qtbot, pick_second)
    chosen = dnd.suggest_alternatives_dialog(
        parent, tmp_path / "plates", alternatives, why="no images here")
    assert chosen == alternatives[1]


def test_cancelling_did_you_mean_keeps_the_original_drop_rejected(qtbot,
                                                                  tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    _answer_modal(qtbot, lambda dialog: _click(dialog,
                                               QDialogButtonBox.Cancel))
    assert dnd.suggest_alternatives_dialog(
        parent, tmp_path / "plates", [tmp_path / "plate1"]) is None
