"""Verbose logging is on by default, so its switch has to work both ways.

Two defects stood between the default the maintainer asked for on
2026-08-28 ("verbose logging should be on by default") and a default that
does what the Preferences row says. Both were found in review on
2026-09-19, and both are tested here the way a user meets them.

1. THE FIRST LAUNCH UNDID THE PREFERENCE. ``launch()`` applied the
   preferences and THEN ran ``setup_logging``, which set the ``spacr`` logger
   to INFO, the master log's filter to INFO and above, and ``cellpose`` to
   WARNING. Measured through ``launch()`` in a fresh HOME: a ``spacr.io``
   DEBUG record reached no log file and cellpose could not report its model,
   until the user's first Preferences Save. The Logging tab's own switches
   were overridden the same way. The test runs the real ``launch()`` in a
   fresh process, because ``setup_logging`` runs once per process.

2. THE SWITCH COULD NOT BE TURNED OFF FROM ITS DEFAULT. The Logging tab was
   built from ``get_log_file_levels()``, which adds DEBUG while verbose is
   on, and Save wrote every switch back. So unticking verbose and pressing
   Save stored ``DEBUG,INFO,WARNING,ERROR,CRITICAL`` as the user's own
   choice, and DEBUG stayed on. Now the DEBUG file switch is held on (and
   disabled) while verbose is ticked, and Save stores the user's own choice.
   These tests drive the real dialog and press the real Save.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPointF, QSettings                     # noqa: E402
from PySide6.QtGui import QEnterEvent                             # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox      # noqa: E402

from spacr.qt import preferences as prefs                          # noqa: E402
from spacr.qt.widgets.hint_bar import HintBar                     # noqa: E402
from spacr.qt.widgets.toggle import Toggle                        # noqa: E402

ALL_LEVELS = "DEBUG,INFO,WARNING,ERROR,CRITICAL"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty preference store, and a record of what Save applied.

    ``apply_preferences_to_app`` and ``apply_level_policy`` are replaced so a
    Save here does not re-theme the application or re-gate the process's
    loggers for the tests that run after this one. What ``set_log_levels``
    hands the live handlers is recorded instead.
    """
    import spacr.logging_util as package_logging

    path = tmp_path / "qt.conf"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(prefs, "apply_preferences_to_app",
                        lambda *args, **kwargs: None)
    applied: list = []
    monkeypatch.setattr(
        package_logging, "apply_level_policy",
        lambda files, console=(): applied.append(
            (frozenset(files), frozenset(console))))
    return applied


def _open(qtbot):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    verbose = next(toggle for toggle in dialog.findChildren(Toggle)
                   if toggle.text() == "Enable verbose logging")
    debug_file = dialog.findChild(Toggle, "LogFileLevelDebug")
    debug_console = dialog.findChild(Toggle, "LogConsoleLevelDebug")
    assert debug_file is not None and debug_console is not None
    return dialog, verbose, debug_file, debug_console


def _save(dialog):
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()


def _stored_file_levels():
    return prefs._settings().value(prefs._KEY_LOG_FILE_LEVELS, None)


def test_unticking_verbose_from_its_default_and_saving_turns_debug_off(
        qtbot, store):
    """The reviewer's flow: fresh store, untick verbose, Save."""
    dialog, verbose, debug_file, _console = _open(qtbot)
    assert verbose.isChecked(), "verbose is not on by default"
    assert debug_file.isChecked() and not debug_file.isEnabled(), (
        "the DEBUG file switch is not held on while verbose is on")

    verbose.setChecked(False)
    assert debug_file.isEnabled()
    assert not debug_file.isChecked(), (
        "unticking verbose left DEBUG on although the user never chose it")

    _save(dialog)

    assert prefs.get_verbose_logging() is False
    assert "DEBUG" not in str(_stored_file_levels())
    assert logging.DEBUG not in prefs.get_log_file_levels()
    files, _console_levels = store[-1]
    assert logging.DEBUG not in files, (
        "Save handed DEBUG to the live handlers after verbose went off")


def test_a_save_with_verbose_on_does_not_store_verbose_debug(qtbot, store):
    """Pressing Save with nothing touched used to store DEBUG for good."""
    dialog, _verbose, _debug_file, _console = _open(qtbot)
    _save(dialog)

    assert prefs.get_verbose_logging() is True
    assert _stored_file_levels() == "INFO,WARNING,ERROR,CRITICAL"
    assert logging.DEBUG in prefs.get_log_file_levels()
    files, _console_levels = store[-1]
    assert logging.DEBUG in files, (
        "verbose is on and the live handlers were not given DEBUG")


def test_a_debug_the_user_chose_is_kept_through_verbose(qtbot, store):
    """A DEBUG the user switched on is theirs, verbose or not."""
    prefs._settings().setValue(prefs._KEY_LOG_FILE_LEVELS,
                               "DEBUG,WARNING,ERROR,CRITICAL")
    dialog, _verbose, _debug_file, _console = _open(qtbot)
    _save(dialog)
    assert _stored_file_levels() == "DEBUG,WARNING,ERROR,CRITICAL"

    dialog, verbose, debug_file, _console = _open(qtbot)
    verbose.setChecked(False)
    assert debug_file.isChecked() and debug_file.isEnabled()
    _save(dialog)
    assert prefs.get_verbose_logging() is False
    assert logging.DEBUG in prefs.get_log_file_levels()


def test_the_last_choice_comes_back_when_verbose_goes_off(qtbot, store):
    """Tick verbose, untick it: the DEBUG switch is as the user left it."""
    prefs.set_verbose_logging(False)
    prefs._settings().setValue(prefs._KEY_LOG_FILE_LEVELS, ALL_LEVELS)
    dialog, verbose, debug_file, _console = _open(qtbot)
    assert debug_file.isChecked() and debug_file.isEnabled()

    debug_file.setChecked(False)
    verbose.setChecked(True)
    assert debug_file.isChecked() and not debug_file.isEnabled()
    verbose.setChecked(False)
    assert not debug_file.isChecked() and debug_file.isEnabled()

    _save(dialog)
    assert _stored_file_levels() == "INFO,WARNING,ERROR,CRITICAL"


def test_a_console_debug_ticked_while_verbose_is_on_is_kept(qtbot, store):
    """The console may show what the files keep, and they keep DEBUG."""
    dialog, _verbose, _debug_file, debug_console = _open(qtbot)
    assert debug_console.isEnabled()
    debug_console.setChecked(True)
    _save(dialog)

    assert "DEBUG" in str(prefs._settings().value(
        prefs._KEY_LOG_CONSOLE_LEVELS, None))
    assert logging.DEBUG in prefs.get_log_console_levels()
    _files, console = store[-1]
    assert logging.DEBUG in console


def test_the_held_switch_says_why_in_the_strip(qtbot, store,
                                               qt_theme_applied):
    """Point at the disabled switch, as a user does, and read the strip."""
    dialog, _verbose, debug_file, _console = _open(qtbot)
    dialog.show()
    qtbot.waitExposed(dialog)
    bar = dialog.findChild(HintBar)
    assert bar is not None

    QApplication.sendEvent(
        debug_file, QEnterEvent(QPointF(4, 4), QPointF(4, 4), QPointF(4, 4)))

    assert "While verbose logging is on" in bar.text(), bar.text()
    assert "Turn verbose logging off to choose it yourself" in bar.text()


_LAUNCH_ONCE = r'''
"""Run the real launch() once and report what the log files received."""
import json
import logging
import os
import sys

repo_root, report_path = sys.argv[1:3]
sys.path.insert(0, repo_root)

import spacr
assert spacr.__file__.startswith(repo_root), spacr.__file__
from spacr.qt import app as app_mod

NAMES = ("spacr", "spacr.io", "spacr.qt", "cellpose")
report = {}


def check():
    from PySide6.QtWidgets import QApplication
    from spacr.logging_util import _FILE_FILTER
    report["levels"] = {name: logging.getLevelName(
        logging.getLogger(name).getEffectiveLevel()) for name in NAMES}
    report["master_filter"] = sorted(
        logging.getLevelName(level) for level in (_FILE_FILTER.levels
                                                  if _FILE_FILTER else ()))
    logging.getLogger("spacr.io").debug("PROBE-io-debug")
    logging.getLogger("spacr.io").error("PROBE-io-error")
    logging.getLogger("cellpose.models").info("PROBE-cellpose-info")
    for logger in (logging.getLogger(), logging.getLogger("spacr")):
        for handler in logger.handlers:
            try:
                handler.flush()
            except Exception:
                pass
    QApplication.instance().quit()


started = app_mod._timing.event_loop_started


def after_the_loop_starts(*args, **kwargs):
    from PySide6.QtCore import QTimer
    try:
        started(*args, **kwargs)
    finally:
        QTimer.singleShot(200, check)


app_mod._timing.event_loop_started = after_the_loop_starts
report["returncode"] = app_mod.launch(["--no-setup"])
logs = os.environ["SPACR_LOG_DIR"]
report["files"] = {}
for name in sorted(os.listdir(logs)):
    if name.endswith(".log"):
        with open(os.path.join(logs, name), encoding="utf-8",
                  errors="replace") as handle:
            text = handle.read()
        report["files"][name] = [tag for tag in (
            "PROBE-io-debug", "PROBE-io-error", "PROBE-cellpose-info")
            if tag in text]
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle)
'''

_ENVIRONMENT_THE_LAUNCH_MUST_NOT_INHERIT = (
    "SPACR_NO_GL", "SPACR_NO_BACKDROP", "SPACR_TIMING",
    "SPACR_WATCH_GUI_STALLS", "SPACR_BENCHMARK_JSON", "SPACR_LOG_LEVEL")


def _launch_once(tmp_path, stored: dict) -> dict:
    """Launch spaCR in a fresh process against ``stored`` preferences."""
    import spacr

    repo_root = str(Path(spacr.__file__).resolve().parents[1])
    config = tmp_path / "config"
    store = QSettings(str(config / "spacr" / "qt.conf"), QSettings.IniFormat)
    for key, value in stored.items():
        store.setValue(key, value)
    store.sync()
    del store

    launcher = tmp_path / "launch_once.py"
    launcher.write_text(_LAUNCH_ONCE, encoding="utf-8")
    report_path = tmp_path / "report.json"
    env = {name: value for name, value in os.environ.items()
           if name not in _ENVIRONMENT_THE_LAUNCH_MUST_NOT_INHERIT}
    env.update(QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path),
               XDG_CONFIG_HOME=str(config),
               SPACR_LOG_DIR=str(tmp_path / "logs"),
               SPACR_NO_SETUP="1", SPACR_LAPTOP_MODE="0")
    finished = subprocess.run(
        [sys.executable, str(launcher), repo_root, str(report_path)],
        env=env, capture_output=True, text=True, timeout=240)
    assert report_path.exists(), (
        f"the launch wrote no report:\n{finished.stderr[-4000:]}")
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.timeout(300)
def test_the_first_launch_keeps_the_verbose_default(tmp_path):
    """No Save needed: DEBUG and cellpose's INFO reach the log from launch."""
    report = _launch_once(tmp_path, {})

    assert report["returncode"] == 0
    assert report["levels"]["spacr"] == "DEBUG", report
    assert report["levels"]["spacr.io"] == "DEBUG", report
    assert report["levels"]["cellpose"] == "INFO", report
    assert "DEBUG" in report["master_filter"], report
    assert "PROBE-io-debug" in report["files"]["spacr.log"], report
    assert "PROBE-io-debug" in report["files"]["spacr-debug.log"], report
    assert "PROBE-cellpose-info" in report["files"]["spacr.log"], report


@pytest.mark.timeout(300)
def test_the_first_launch_keeps_the_users_own_file_levels(tmp_path):
    """The Logging tab's switches are what the log files keep from launch."""
    report = _launch_once(tmp_path, {
        prefs._KEY_VERBOSE_LOG: False,
        prefs._KEY_LOG_FILE_LEVELS: "INFO,WARNING",
    })

    assert report["returncode"] == 0
    assert report["master_filter"] == ["INFO", "WARNING"], report
    assert report["levels"]["cellpose"] == "WARNING", report
    assert "PROBE-io-debug" not in report["files"]["spacr.log"], report
    assert "PROBE-io-error" not in report["files"]["spacr.log"], (
        "ERROR reached the master log although the user switched it off")
