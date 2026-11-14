"""Drag-and-drop plumbing: where a rejected drop is reported, and the choosers.

The two modal dialogs are driven rather than replaced: a timer waits for the
modal to be up and then clicks its real buttons, so the assertions are about
what the dialog returns to its caller.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QTimer  # noqa: E402
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


def test_a_console_panel_in_the_window_is_found_by_type(qtbot):
    from spacr.qt.widgets.console_panel import ConsolePanel

    window = QWidget()
    qtbot.addWidget(window)
    panel = ConsolePanel(parent=window)
    window._screens = {"tool": QWidget(window)}
    assert dnd._find_console(QWidget(window)) is panel


# ---------------------------------------------------------------------------
# Reporting a rejected drop
# ---------------------------------------------------------------------------


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
    data.write_text("alpha,beta\n1,2\n", encoding="utf-8")

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
    data.write_text("path,pred,plate,row,col\na,0.5,p1,r1,c1\n", encoding="utf-8")

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
    csv_path.write_text("Key,Value\nsrc,/tmp\n", encoding="utf-8")
    assert dnd._apply_settings_csv(csv_path, Screen()) is None


def test_a_settings_csv_that_will_not_load_says_so_in_a_dialog(
        qtbot, monkeypatch, tmp_path):
    """The failure reaches both the console report and a real message box."""
    csv_path = tmp_path / "settings.csv"
    csv_path.write_text("Key,Value\nsrc,/tmp\n", encoding="utf-8")

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
    csv_path.write_text("setting_key,setting_value\nsrc,/tmp/data\n", encoding="utf-8")

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
