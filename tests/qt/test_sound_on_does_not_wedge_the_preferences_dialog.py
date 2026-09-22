"""Item 444: with sound on, Preferences opens and closes and nothing wedges.

Reported 2026-09-19 by the maintainer, against the sound engine landed the
same day: "for some reason if sound is on in spacr and i try to reopen
preferences, the application hangs and needs to be force quit. i see the of
the preferences window but it is not populated with preferences then i get
the wait force quit option and i have to force quit".

The cause was `QMediaDevices` and every `QSoundEffect` being constructed on
the audio thread. Qt Multimedia's device handling belongs to the thread that
owns the application's event loop -- with the FFmpeg backend it installs
socket notifiers -- so driving it from another thread is undefined
behaviour. Reproduced here offscreen against a warm sound cache it was not
a hang but a SIGSEGV, three runs out of three, with the audio thread inside
`_make_effect` and the GUI thread inside `PreferencesDialog._page`.

Every test here drives the REAL Qt Multimedia stack, because a stand-in
cannot install a socket notifier and cannot crash. They run in child
processes for the same reason: the failure they are written for takes the
interpreter with it, and a test that cannot survive its own subject tells
you nothing about the run it killed. Volume is zero and no event that plays
by itself is switched on, so nothing reaches the speakers; what is
exercised is the construction and loading that did the damage.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtMultimedia")

from spacr.qt.sound_synth import (
    BED,
    DEFAULT_THEME,
    FEEDBACK_EVENTS,
    SOUND_THEMES,
    ensure_rendered,
    sound_names,
)

ROOT = Path(__file__).resolve().parents[2]

#: Long enough for a child that is working, short enough that a child that
#: has wedged is reported as one rather than waited on.
LIMIT_S = 240


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Every sound of the default set, rendered once for this module.

    The children then start against a warm cache, which is also the state
    the crash needed: a cold one spends its first seconds synthesizing and
    may never reach the constructor that does the damage.
    """
    root = tmp_path_factory.mktemp("sounds")
    names = [name for event in FEEDBACK_EVENTS for name in sound_names(event)]
    names.extend(sound_names(BED))
    ensure_rendered(SOUND_THEMES[DEFAULT_THEME], names, root=root)
    return root


def run_child(body: str, tmp_path: Path, cache: Path):
    """Run ``body`` in a fresh interpreter with a private HOME and cache.

    :param body: the script, which must print ``DONE`` as its last line.
    :param tmp_path: the private HOME and config root.
    :param cache: a warm sound cache.
    :returns: the finished process.
    :raises AssertionError: when the child did not finish in
        :data:`LIMIT_S` seconds, which is what a wedge looks like here.
    """
    script = (
        "import faulthandler, sys\n"
        "faulthandler.enable()\n"
        f"faulthandler.dump_traceback_later({LIMIT_S - 30}, exit=True)\n"
        "import spacr\n"
        f"assert spacr.__file__.startswith({str(ROOT)!r}), spacr.__file__\n"
        "from PySide6.QtCore import QTimer, qInstallMessageHandler\n"
        "from PySide6.QtWidgets import QApplication\n"
        "SHOUTS = []\n"
        "def _listen(mode, context, message):\n"
        "    text = str(message)\n"
        "    for bad in ('SocketNotifier', 'another thread',\n"
        "                'Destroyed while', 'deleted directly'):\n"
        "        if bad in text:\n"
        "            SHOUTS.append(text)\n"
        "    sys.stderr.write('QT: ' + text + '\\n')\n"
        "qInstallMessageHandler(_listen)\n"
        "app = QApplication([])\n"
        "from spacr.qt.theme import enable_spaceout\n"
        "enable_spaceout()\n"
        "from spacr.qt import preferences as prefs\n"
        "prefs.set_sound_enabled(True)\n"
        "prefs.set_sound_volume(0.0)\n"
        "prefs.set_sound_event_enabled('bed', False)\n"
        "from spacr.qt.preferences import apply_preferences_to_app\n"
        "from spacr.qt.sound import sound_engine\n"
        ) + body + (
        "\nprint('SHOUTS', len(SHOUTS), SHOUTS)\n"
        "print('DONE')\n")
    env = dict(os.environ, HOME=str(tmp_path), QT_QPA_PLATFORM="offscreen",
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               XDG_DATA_HOME=str(tmp_path / "data"),
               SPACR_SOUND_CACHE=str(cache))
    try:
        return subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                              env=env, capture_output=True, text=True,
                              timeout=LIMIT_S)
    except subprocess.TimeoutExpired as expired:
        raise AssertionError(
            "the child never finished -- that is the wedge this file is "
            f"about:\n{expired.stdout}\n{expired.stderr}") from expired


def _wait_for_sounds() -> str:
    """Script that starts the engine and waits until its set is loaded."""
    return (
        "apply_preferences_to_app(app)\n"
        "engine = sound_engine()\n"
        "assert engine is not None, 'sound was on and no engine was built'\n"
        "import time\n"
        "until = time.monotonic() + 90\n"
        "while engine.available is None and time.monotonic() < until:\n"
        "    app.processEvents()\n"
        "    time.sleep(0.005)\n"
        "assert engine.available is True, engine.available\n"
        "effects = engine.player()._effects\n"
        "assert effects, 'no QSoundEffect was built, so nothing was tested'\n"
        "print('EFFECTS', len(effects))\n")


def _finished(done, *, expect="DONE") -> str:
    """Assert the child got to the end, and hand back its output."""
    assert done.returncode == 0, (
        f"the child died with {done.returncode} "
        f"(-11 is a segfault, -6 an abort):\n{done.stdout}\n{done.stderr}")
    assert expect in done.stdout, done.stdout + done.stderr
    assert "SHOUTS 0 []" in done.stdout, (
        "Qt complained about threads:\n" + done.stdout + done.stderr)
    return done.stdout


def test_preferences_opens_and_closes_ten_times_with_sound_on(tmp_path,
                                                              cache):
    """The maintainer's report, ten times over, with the real audio stack.

    Before the fix this child segfaulted on the FIRST dialog, with the
    audio thread in `_SoundPlayer`'s ancestor `_AudioWorker._make_effect`.
    """
    body = _wait_for_sounds() + (
        "from spacr.qt.preferences import PreferencesDialog\n"
        "for round_number in range(10):\n"
        "    dialog = PreferencesDialog(None)\n"
        "    QTimer.singleShot(60, dialog.reject)\n"
        "    dialog.exec()\n"
        "    dialog.deleteLater()\n"
        "    app.processEvents()\n"
        "    print('ROUND', round_number)\n")
    out = _finished(run_child(body, tmp_path, cache))
    assert "ROUND 9" in out, out


def test_the_preferences_dialog_still_fills_itself_in_with_sound_on(tmp_path,
                                                                    cache):
    """"i see the of the preferences window but it is not populated".

    An empty frame is the shape the maintainer saw, so the second opening
    is asked what it contains rather than only whether it returned.
    """
    body = _wait_for_sounds() + (
        "from spacr.qt.preferences import PreferencesDialog\n"
        "from PySide6.QtWidgets import QTabWidget\n"
        "counts = []\n"
        "for round_number in range(3):\n"
        "    dialog = PreferencesDialog(None)\n"
        "    QTimer.singleShot(60, dialog.reject)\n"
        "    dialog.exec()\n"
        "    tabs = dialog.findChildren(QTabWidget)\n"
        "    counts.append(max([t.count() for t in tabs] or [0]))\n"
        "    dialog.deleteLater()\n"
        "    app.processEvents()\n"
        "print('TABS', counts)\n")
    out = _finished(run_child(body, tmp_path, cache))
    row = [line for line in out.splitlines() if line.startswith("TABS ")][0]
    counts = ast.literal_eval(row[len("TABS "):])
    assert len(counts) == 3 and min(counts) > 1, row
    assert len(set(counts)) == 1, f"a later opening lost tabs: {row}"


def test_no_qt_multimedia_object_is_built_off_the_gui_thread(tmp_path, cache):
    """The pin, against the real classes rather than a stand-in.

    Every `QSoundEffect` is asked which thread it belongs to, and the
    device is asked the same question from where it was connected. A
    `QObject`'s thread affinity is the thread it was created on, so this
    fails if the construction moves back.
    """
    body = _wait_for_sounds() + (
        "engine = sound_engine()\n"
        "gui = app.thread()\n"
        "audio = engine.audio_thread()\n"
        "assert audio is not None and audio.objectName() == 'spacr-sound'\n"
        "homes = [e.thread() for e in engine.player()._effects.values()]\n"
        "assert homes, 'no effects to check'\n"
        "assert all(h is gui for h in homes), homes\n"
        "assert not any(h is audio for h in homes), homes\n"
        "from PySide6.QtMultimedia import QMediaDevices\n"
        "assert QMediaDevices().thread() is gui\n"
        "print('EFFECTS ON THE GUI THREAD', len(homes))\n")
    _finished(run_child(body, tmp_path, cache))


def test_quitting_with_sound_on_ends_the_audio_thread(tmp_path, cache):
    """No "Destroyed while thread 'spacr-sound' is still running".

    That warning and the core dump behind it were the second defect in
    item 444's run: a process that aborts at exit loses whatever the run
    journal had not flushed.
    """
    body = _wait_for_sounds() + (
        "QTimer.singleShot(50, app.quit)\n"
        "app.exec()\n"
        "engine = sound_engine()\n"
        "print('RUNNING', engine.audio_thread().isRunning())\n"
        "print('CLOSED', engine.closed)\n")
    out = _finished(run_child(body, tmp_path, cache))
    assert "RUNNING False" in out, out
    assert "CLOSED True" in out, out


def test_a_process_that_never_reaches_aboutToQuit_still_stops_the_thread(
        tmp_path, cache):
    """The exit path `aboutToQuit` does not cover.

    A script, a crash handler, a `sys.exit` from a menu action: the
    application's event loop is not always what ends the process, and a
    `QThread` whose last reference goes while it is running aborts. This
    child starts the engine and simply falls off the end of its script.
    """
    body = _wait_for_sounds() + "print('NO QUIT, JUST EXIT')\n"
    out = _finished(run_child(body, tmp_path, cache))
    assert "NO QUIT, JUST EXIT" in out, out
