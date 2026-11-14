"""Table-driven tests of the quit question and the force restart.

``ask_how_to_quit`` must report the button the user actually pressed (finish,
force, force restart or cancel) and word the question after the button that
opened it; ``restart_spacr`` must save first, launch a detached replacement
with the right options for the platform, keep the current process when the
launch fails, and only then exit.  Offscreen Qt, no real dialogs or processes.
"""
from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from spacr.qt import shutdown
from spacr.qt.shutdown import CANCEL, FORCE, GRACEFUL, RESTART

pytestmark = pytest.mark.qt

BASE_INFO = (
    "Finish current work stops when the running steps reach a point they can "
    "be interrupted safely. This can take minutes, and you will be asked "
    "again every five minutes.\n\nForce quit stops immediately. Anything "
    "being written right now is left half-written.")
RESTART_INFO = (
    "\n\nForce restart saves this module and its settings, closes spaCR, "
    "starts it again and reopens the module where you left it. Use it when "
    "Force stop does not stop.\n\n")


def test_ask_how_to_quit_reports_the_pressed_button(subtests, qapp, monkeypatch):
    """Every button maps to its answer; the dialog's words and buttons are exact."""
    cases = [
        # (id, press, kwargs, answer, labels, informative text)
        ("finishing current work is the graceful answer", "Finish current work",
         {"verb": "Stop"}, GRACEFUL,
         ["Finish current work", "Force stop", "Cancel"], BASE_INFO),
        ("force stop is the force answer", "Force stop", {"verb": "Stop"},
         FORCE, ["Finish current work", "Force stop", "Cancel"], BASE_INFO),
        ("cancel is the answer when nothing destructive was chosen", "Cancel",
         {"verb": "Stop"}, CANCEL,
         ["Finish current work", "Force stop", "Cancel"], BASE_INFO),
        ("force restart is its own answer and states its cost",
         "Force restart",
         {"verb": "Stop", "offer_restart": True,
          "restart_detail": "one run will be interrupted"}, RESTART,
         ["Finish current work", "Force stop", "Force restart", "Cancel"],
         BASE_INFO + RESTART_INFO + "one run will be interrupted"),
        ("the default verb is Quit and a detail heads the text", "Force quit",
         {"detail": "Still running:\n  x"}, FORCE,
         ["Finish current work", "Force quit", "Cancel"],
         "Still running:\n  x\n\n" + BASE_INFO),
        ("closing the box without a button is cancel", None,
         {"verb": "Stop", "offer_restart": True}, CANCEL,
         ["Finish current work", "Force stop", "Force restart", "Cancel"],
         BASE_INFO + RESTART_INFO),
    ]
    for case_id, press, kwargs, answer, labels, info in cases:
        with subtests.test(msg=case_id):
            seen = {}

            def fake_exec(box, press=press, seen=seen):
                buttons = {b.text(): b for b in box.buttons()}
                seen.update(
                    title=box.windowTitle(), text=box.text(),
                    info=box.informativeText(),
                    labels=sorted(buttons),
                    default=box.defaultButton().text(),
                    escape=box.escapeButton().text(),
                    danger=sorted(t for t, b in buttons.items()
                                  if b.objectName() == "DangerButton"))
                if press is not None:
                    buttons[press].click()
                return 0

            monkeypatch.setattr(QMessageBox, "exec", fake_exec, raising=False)
            got = shutdown.ask_how_to_quit(None, what="Regression", **kwargs)
            verb = kwargs.get("verb", "Quit")
            assert got == answer
            assert seen == {
                "title": f"{verb} Regression",
                "text": f"{verb} Regression?",
                "info": info,
                "labels": sorted(labels),
                "default": "Cancel",
                "escape": "Cancel",
                "danger": sorted(label for label in labels
                                 if label.startswith("Force")),
            }


def test_restart_spacr_saves_launches_detached_and_exits(subtests, monkeypatch,
                                                         caplog):
    """Exact launch options per platform; any failure keeps this process open."""
    cases = [
        # (id, os name, frozen, save result, launcher, popen raises,
        #  expected launches, returned, exits, discarded, log)
        ("posix default launcher starts a new session", "posix", False,
         "state.json", None, False,
         [(["python", "-m", "spacr.qt"],
           {"start_new_session": True, "close_fds": True})],
         True, [0], 0, "restarting spaCR: python -m spacr.qt"),
        ("a source install without a frozen flag keeps the environment",
         "posix", None, "state.json", None, False,
         [(["python", "-m", "spacr.qt"],
           {"start_new_session": True, "close_fds": True})],
         True, [0], 0, "restarting spaCR: python -m spacr.qt"),
        ("windows detaches without inheriting the old console", "nt", False,
         "state.json", None, False,
         [(["python", "-m", "spacr.qt"],
           {"start_new_session": True, "close_fds": True,
            "creationflags": 520})],
         True, [0], 0, "restarting spaCR: python -m spacr.qt"),
        ("a frozen windows launch resets the bundled environment", "nt", True,
         "state.json", None, False,
         [(["python", "-m", "spacr.qt"],
           {"start_new_session": True, "close_fds": True,
            "creationflags": 520,
            "env": {"KEEP_EXISTING": "yes",
                    "PYINSTALLER_RESET_ENVIRONMENT": "1"}})],
         True, [0], 0, "restarting spaCR: python -m spacr.qt"),
        ("a failed frozen windows launch keeps the process", "nt", True,
         "state.json", None, True,
         [(["python", "-m", "spacr.qt"],
           {"start_new_session": True, "close_fds": True,
            "creationflags": 520,
            "env": {"KEEP_EXISTING": "yes",
                    "PYINSTALLER_RESET_ENVIRONMENT": "1"}})],
         False, [], 1,
         "could not start spaCR again (replacement executable unavailable); "
         "NOT quitting"),
        ("a caller's launcher gets the command alone", "posix", False,
         "state.json", "custom", False,
         [(["python", "-m", "spacr.qt"], "custom")], True, [0], 0,
         "restarting spaCR: python -m spacr.qt"),
        ("an unwritten state never launches", "posix", False, None, None,
         False, [], False, [], 0,
         "the restart state could not be written; NOT restarting"),
    ]
    for (case_id, os_name, frozen, saved, launcher, popen_raises, launches,
         returned, exits, discarded, log) in cases:
        with subtests.test(msg=case_id):
            environment = {"KEEP_EXISTING": "yes"}
            launched, exited, discards, saves = [], [], [], []

            def popen(command, **options):
                launched.append((list(command), options))
                if popen_raises:
                    raise OSError("replacement executable unavailable")
                return object()

            def save(**kwargs):
                saves.append(kwargs)
                return saved

            custom = (lambda command: launched.append((list(command), "custom"))
                      ) if launcher else None
            caplog.clear()
            with monkeypatch.context() as patch:
                patch.setattr(shutdown, "os",
                              SimpleNamespace(name=os_name, environ=environment))
                if frozen is None:
                    patch.delattr(shutdown.sys, "frozen", raising=False)
                else:
                    patch.setattr(shutdown.sys, "frozen", frozen, raising=False)
                patch.setattr(subprocess, "DETACHED_PROCESS", 8, raising=False)
                patch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512,
                              raising=False)
                patch.setattr(subprocess, "Popen", popen)
                patch.setattr("spacr.restart_state.save", save)
                patch.setattr("spacr.restart_state.discard",
                              lambda: discards.append(True))
                patch.setattr("spacr.restart_state.command",
                              lambda: ["python", "-m", "spacr.qt"])
                with caplog.at_level(logging.WARNING, logger="spacr.qt.shutdown"):
                    got = shutdown.restart_spacr(
                        "regression", {"a": 1}, running=("r",),
                        run_folders=("f",), launcher=custom,
                        exiter=exited.append)
            assert got is returned
            assert saves == [{"module": "regression", "settings": {"a": 1},
                              "running": ("r",), "run_folders": ("f",)}]
            assert launched == launches
            assert exited == exits
            assert len(discards) == discarded
            assert environment == {"KEEP_EXISTING": "yes"}
            assert [r.getMessage() for r in caplog.records
                    if r.name == "spacr.qt.shutdown"] == [log]
