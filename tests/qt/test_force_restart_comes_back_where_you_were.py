"""Instruction 142 — Force restart, for when Stop does not stop.

    "add an extra button to the stop popup window called Force restart, if
     forse stop dosnt kill the running jobs then this should save the state
     (module and its settings) then quit spacr, restart space, go back to the
     modual that was open with the settings it had at closure. if other
     moduals are currently also running warn about these and tell the user
     that theire settings states will also be reloaded"

The property everything else rests on: THE SAVE IS VERIFIED BEFORE ANYTHING IS
KILLED. A restart that loses the settings is worse than a stuck run, because a
stuck run can at least be waited out — so a state that will not write cancels
the restart, and nothing is stopped.

Instruction 140 C has since put a cancellation checkpoint inside the mixed
fit, so most wedged fits now answer Stop. This is what is left for the ones
that do not.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr import restart_state                                  # noqa: E402
from spacr.qt.shutdown import (CANCEL, FORCE, GRACEFUL,          # noqa: E402
                               RESTART, restart_spacr)


@pytest.fixture(autouse=True)
def _own_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "spacr_home"))
    yield
    restart_state.discard()


# -- the state ---------------------------------------------------------------

def test_the_state_carries_the_module_and_its_settings():
    written = restart_state.save(module="regression",
                                 settings={"alpha": "auto", "level": "both"})
    assert written is not None

    state = restart_state.peek()
    assert state["module"] == "regression"
    assert state["settings"]["alpha"] == "auto"


def test_the_state_is_verified_by_reading_it_back_not_by_the_write_returning():
    """"the write returned" and "the file parses" are different claims."""
    written = restart_state.save(module="mask", settings={"src": "/data"})
    assert json.loads(written.read_text())["module"] == "mask"


def test_a_state_that_cannot_be_written_is_None_and_never_an_exception(
        monkeypatch, tmp_path):
    """A raise would arrive inside a dialog on the way out of a wedged app."""
    monkeypatch.setattr(restart_state, "state_path",
                        lambda: tmp_path / "no" / "such" / "dir" / "s.json")
    monkeypatch.setattr("pathlib.Path.mkdir",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    assert restart_state.save(module="regression", settings={}) is None


def test_settings_that_are_not_json_survive_the_trip(tmp_path):
    from pathlib import Path

    restart_state.save(module="measure",
                       settings={"src": Path("/data"), "channels": (0, 1, 2),
                                 "nested": {"a": {"b": Path("/x")}}})
    settings = restart_state.peek()["settings"]
    assert settings["src"] == "/data"
    assert settings["channels"] == [0, 1, 2]
    assert settings["nested"]["a"]["b"] == "/x"


def test_the_state_is_taken_not_read_so_it_cannot_reopen_forever():
    """A crash on the way back up must not wedge every future launch."""
    restart_state.save(module="regression", settings={})
    assert restart_state.take()["module"] == "regression"
    assert restart_state.peek() is None
    assert restart_state.take() is None


def test_a_stale_state_is_dropped_rather_than_reopened():
    restart_state.save(module="regression", settings={},
                       saved="2020-01-01T00:00:00+00:00")
    assert restart_state.take() is None
    assert restart_state.peek() is None      # and it is gone either way


def test_the_command_uses_the_running_interpreter_not_the_path():
    """Coming back as a different spaCR is worse than not coming back."""
    import sys

    assert restart_state.command()[0] == sys.executable
    # `spacr.qt`, NOT `spacr` (2026-08-20). `spacr/__main__.py` is the CLI
    # and its command DEFAULTS TO "gui", which dispatches to the legacy Tk
    # interface -- so `-m spacr` quit the Qt app and opened the old one.
    # Reported as "if i press stop and force restart i open the old tkinter
    # spacr", which is not a restart of anything.
    assert restart_state.command()[1:] == ["-m", "spacr.qt"]


# -- what the dialog says ----------------------------------------------------

def test_every_running_module_is_named_with_how_long_it_has_run():
    text = restart_state.describe_running([
        {"module": "Mask", "seconds": 840},
        {"module": "Measure", "seconds": 125},
    ])
    assert "Mask (running 14 min)" in text
    assert "Measure (running 2 min)" in text


def test_an_hours_long_run_reads_as_hours():
    assert "running 2 h 05 min" in restart_state.describe_running(
        [{"module": "Mask", "seconds": 7500}])


def test_the_warning_separates_losing_runs_from_losing_configuration():
    """They are different losses and a user weighs them separately."""
    text = restart_state.warning_text(
        [{"module": "Mask", "seconds": 840}],
        ["/data/results/ols_1"])

    assert "will NOT resume" in text and "Mask" in text
    assert "settings are saved and come back" in text
    assert "/data/results/ols_1" in text


def test_with_nothing_else_running_it_says_so_rather_than_going_quiet():
    assert "No other module is running" in restart_state.warning_text([])


# -- the button --------------------------------------------------------------

def _dialog_buttons(qtbot, **kwargs):
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt import shutdown

    seen = {}

    def capture(box):
        seen["labels"] = [b.text() for b in box.buttons()]
        seen["default"] = box.defaultButton().text()
        return QMessageBox.RejectRole

    original = QMessageBox.exec
    QMessageBox.exec = lambda self: capture(self)
    try:
        shutdown.ask_how_to_quit(None, what="Regression", verb="Stop", **kwargs)
    finally:
        QMessageBox.exec = original
    return seen


def test_force_restart_is_offered_and_is_the_last_button(qtbot):
    seen = _dialog_buttons(qtbot, offer_restart=True, restart_detail="x")
    labels = seen["labels"]
    assert "Force restart" in labels
    # LAST, because it is the most destructive thing on the dialog.
    assert labels.index("Force restart") > labels.index("Force stop")
    assert seen["default"] == "Cancel"


def test_a_plain_quit_dialog_does_not_grow_the_most_destructive_option(qtbot):
    assert "Force restart" not in _dialog_buttons(qtbot)["labels"]


# -- doing it ----------------------------------------------------------------

def test_nothing_is_killed_when_the_state_cannot_be_saved(monkeypatch):
    monkeypatch.setattr("spacr.restart_state.save",
                        lambda **kwargs: None)
    launched, exited = [], []
    assert restart_spacr("regression", {}, launcher=launched.append,
                         exiter=exited.append) is False
    assert launched == [] and exited == []


def test_nothing_is_killed_when_the_new_process_will_not_start():
    """The state is LEFT on disk: the user will start spaCR themselves."""
    def refuses(_command):
        raise OSError("no such file")

    exited = []
    assert restart_spacr("regression", {"a": 1}, launcher=refuses,
                         exiter=exited.append) is False
    assert exited == []
    assert restart_state.peek()["module"] == "regression"


def test_a_successful_restart_saves_then_launches_then_leaves():
    order = []
    assert restart_spacr("regression", {"alpha": "auto"},
                         launcher=lambda c: order.append("launch"),
                         exiter=lambda code: order.append(f"exit {code}")) is True
    assert order == ["launch", "exit 0"]
    assert restart_state.peek()["settings"]["alpha"] == "auto"


# -- and coming back ---------------------------------------------------------

def test_the_screen_saves_its_own_settings_and_says_nothing_stopped_on_failure(
        qtbot, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    said = []
    screen._say = said.append

    monkeypatch.setattr("spacr.restart_state.save", lambda **kwargs: None)
    assert screen.force_restart(launcher=lambda c: None,
                                exiter=lambda code: None) is False
    assert any("did NOT restart" in line for line in said)
    assert any("Nothing was stopped" in line for line in said)


def test_a_restart_from_the_screen_carries_the_settings_it_had(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    launched, exited = [], []

    assert screen.force_restart(launcher=launched.append,
                                exiter=exited.append) is True
    state = restart_state.peek()
    assert state["module"] == "regression"
    # The whole settings state, not a token subset.
    assert len(state["settings"]) > 20


def test_a_restarted_spacr_opens_the_module_with_the_settings_it_had(qtbot):
    """The whole point: same module, same settings, fresh process."""
    from spacr.qt.app import MainWindow

    restart_state.save(module="regression",
                       settings={"alpha": "auto", "fdr_alpha": 0.01})
    window = MainWindow()
    qtbot.addWidget(window)

    screen = window._screens.get("regression")
    assert screen is not None, "the saved module was not reopened"
    values = screen._settings_model.collect() or {}
    assert values["fdr_alpha"] == 0.01
    # `auto` is the value that found a SECOND settings writer: AppScreen has
    # its own `_apply_value`, and `float("auto")` raised into an `except` that
    # left the control at 1 -- the one value a penalised fit cannot use.
    assert values["alpha"] == "auto"
    assert restart_state.peek() is None


def test_a_module_named_on_the_command_line_wins_over_a_saved_state(qtbot):
    """spaCR ignoring what it was just told would be a worse surprise."""
    from spacr.qt.app import MainWindow

    restart_state.save(module="regression", settings={"fdr_alpha": 0.01})
    window = MainWindow(initial_app="mask")
    qtbot.addWidget(window)

    assert "mask" in window._screens
    # And the state is still there, so the next plain launch honours it.
    assert restart_state.peek()["module"] == "regression"


def test_the_run_is_not_restarted_only_the_configuration(qtbot):
    """142 C: the runs do not come back, and starting one unasked is worse."""
    from spacr.qt.app import MainWindow

    restart_state.save(module="regression", settings={"fdr_alpha": 0.01})
    window = MainWindow()
    qtbot.addWidget(window)

    screen = window._screens["regression"]
    assert getattr(screen, "_thread", None) is None
