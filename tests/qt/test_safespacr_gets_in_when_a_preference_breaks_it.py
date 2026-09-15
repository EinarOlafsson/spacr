"""Safe mode is a way IN when a stored preference is what breaks the start."""
from __future__ import annotations

import os

import pytest

from spacr.qt import preferences


@pytest.fixture
def a_clean_process(monkeypatch):
    """Safe mode is process-local; never let it leak into another test."""
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    monkeypatch.setattr(preferences, "_SAFE_OVERRIDES", {})
    yield
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)


def test_a_stored_value_that_breaks_the_start_is_not_read(
        a_clean_process, monkeypatch):
    """The whole point: the poisoned value is never consulted."""
    written = {}

    class _Poisoned:
        def value(self, key, default=None, type=None):
            raise AssertionError(
                f"safe mode read the stored value for {key!r}")

        def setValue(self, key, value):
            written[key] = value

        def remove(self, key):
            written.pop(key, None)

        def sync(self):
            written["synced"] = True

    monkeypatch.setattr(preferences, "QSettings",
                        lambda *a, **k: _Poisoned())
    preferences.enable_safe_mode()

    # An ordinary getter answers, without touching the store that raises.
    assert preferences.get_font_scale() == preferences.DEFAULT_FONT_SCALE


def test_saving_reaches_the_real_store(a_clean_process, monkeypatch):
    """Reads are shadowed; WRITES are not, or safe mode fixes nothing."""
    written = {}
    synced = []

    class _Real:
        def value(self, key, default=None, type=None):
            raise AssertionError("reads must not reach the real store")

        def setValue(self, key, value):
            written[key] = value

        def remove(self, key):
            written.pop(key, None)

        def sync(self):
            synced.append(True)

    monkeypatch.setattr(preferences, "QSettings", lambda *a, **k: _Real())
    preferences.enable_safe_mode()

    preferences.set_ambient_enabled(True)

    assert written, "the value the user re-saved never reached the store"
    assert synced, "a re-saved value was left unflushed"
    assert written[preferences._KEY_AMBIENT_ENABLED] is True


def test_the_backdrop_is_forced_off_not_merely_defaulted(
        a_clean_process, monkeypatch):
    """Defaults are not safe by themselves: ambient defaults to ON."""
    class _Empty:
        def value(self, key, default=None, type=None):
            return default

        def setValue(self, key, value):
            pass

        def remove(self, key):
            pass

        def sync(self):
            pass

    monkeypatch.setattr(preferences, "QSettings", lambda *a, **k: _Empty())

    # Without safe mode the default really is on -- so forcing it matters.
    assert preferences.get_ambient_enabled() is True

    preferences.enable_safe_mode()
    assert preferences.get_ambient_enabled() is False
    assert preferences.get_verbose_logging() is False
    assert preferences.get_preload_policy() == "on_demand"


def test_safe_mode_refuses_a_gl_canvas(monkeypatch):
    """The crash log points at the GL path, so safe mode must not build one."""
    from spacr.qt.widgets import fractal_travel

    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("DISPLAY", ":0")
    assert fractal_travel.platform_can_do_opengl() is True

    monkeypatch.setenv("SPACR_NO_GL", "1")
    assert fractal_travel.platform_can_do_opengl() is False


def test_the_launcher_disarms_gl_and_timing_before_qt(monkeypatch):
    """`safespacr` sets the environment before anything reads it."""
    from spacr.qt import safespacr

    monkeypatch.setenv("SPACR_TIMING", "1")
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    seen = {}

    def _fake_run(argv):
        seen["timing"] = os.environ.get("SPACR_TIMING")
        seen["no_gl"] = os.environ.get("SPACR_NO_GL")
        seen["no_backdrop"] = os.environ.get("SPACR_NO_BACKDROP")
        seen["safe"] = preferences.in_safe_mode()
        seen["argv"] = list(argv or [])
        return 0

    import spacr.qt as qt_pkg
    monkeypatch.setattr(qt_pkg, "run", _fake_run)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)

    assert safespacr.main([]) == 0
    assert seen["timing"] is None, "timing instrumentation survived safe mode"
    assert seen["no_gl"] == "1"
    # The process-local switch crash recovery already uses: every backdrop
    # install site reads it first, before it would import the backdrop module.
    assert seen["no_backdrop"] == "1"
    assert seen["safe"] is True
    # Reading preferences as defaults makes "has this profile been set up"
    # read as "no", so safe mode greeted a long-standing user with the
    # setup wizard in front of the settings they came to repair.
    assert "--no-setup" in seen["argv"]


#: The imports safe mode exists to avoid. Named one by one so a failure says
#: WHICH arrived rather than "something heavy got in".
#:
#: `torch` and `cuml` are the GPU probe; `cellpose` pulls torch; `cupy` is
#: CUDA directly. Instruction 296 lists "no GPU probe, no CUDA import" among
#: the things safe mode turns off, and 295 is why: the crash log points at the
#: GL/driver path, so a safe mode that imports a CUDA runtime to ask what the
#: card is has taken the risk it was built to avoid.
SAFE_MODE_MUST_NOT_IMPORT = ("torch", "cupy", "cuml", "cellpose",
                             "tensorflow")


@pytest.mark.slow
def test_safe_mode_opens_without_importing_a_gpu_runtime():
    """296's "no GPU probe, no CUDA import", asserted instead of intended.

    IN A FRESH EXEC, and that is the whole point. By the time this file
    runs, pytest has imported most of spaCR, so `sys.modules` in THIS
    process says nothing about what a launch costs -- the same reason
    `tests/test_perf_guard.py` measures in a subprocess and says so in its
    own docstring.

    Drives the real safe-mode preamble: `enable_safe_mode()`, the launcher's
    environment, the app registration pass and a MainWindow shown offscreen.
    That is everything `safespacr` does short of entering the event loop.

    Measured 2026-09-15: none of the five arrive. This test exists because
    that was TRUE AND UNGUARDED -- the guarantee is in 296's list, nothing
    held it, and the cheapest way for it to break is somebody adding a
    capability probe to the startup path for a good reason.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    import spacr

    repo_root = Path(spacr.__file__).resolve().parents[1]
    sentinel = "SAFE-MODE-JSON "
    code = f'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, {str(repo_root)!r})
from spacr.qt.preferences import enable_safe_mode
enable_safe_mode()
os.environ.pop("SPACR_TIMING", None)
os.environ["SPACR_NO_GL"] = "1"
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(["safespacr"])
import spacr.qt as q
q.register_self_registering_modules()
from spacr.qt.app import MainWindow
window = MainWindow()
window.resize(1200, 800)
window.show()
for _ in range(40):
    app.processEvents()
names = {SAFE_MODE_MUST_NOT_IMPORT!r}
print({sentinel!r} + json.dumps(
    [name for name in names if name in sys.modules]))
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, (
        f"the safe-mode subprocess failed:\n{out.stderr[-3000:]}")
    line = next((ln for ln in out.stdout.splitlines()
                 if ln.startswith(sentinel)), None)
    assert line is not None, (
        f"no result line; stdout tail:\n{out.stdout[-2000:]}")
    arrived = json.loads(line[len(sentinel):])

    assert arrived == [], (
        f"safe mode imported {arrived} — instruction 296 says it turns the "
        f"GPU probe and the CUDA import OFF, and 295 is why: the crash it "
        f"exists to escape is in that path. A safe mode that loads a GPU "
        f"runtime to ask what the card is has taken the risk it was built "
        f"to avoid.")


#: The animation whose backdrop the launcher below aborts on. Any of
#: `AMBIENT_THEMES` except the default would do; the default cannot, because
#: then a store with nothing written in it would die too, and 296 asks for a
#: start that a WRITTEN value breaks.
POISONED_ANIMATION = "cells"

#: What safe mode saves in its place: the default, a real backdrop, so the
#: repaired ordinary start is seen to build one rather than to skip it.
REPAIRED_ANIMATION = "blobs"

#: A preference the user set and never touches in safe mode.
UNTOUCHED_FONT_SCALE = 1.25

#: Environment a developer's shell might carry that would change what an
#: ordinary start does, cleared so the three launches see only the store.
_ENVIRONMENT_THE_LAUNCHES_MUST_NOT_INHERIT = (
    "SPACR_NO_GL", "SPACR_NO_BACKDROP", "SPACR_TIMING",
    "SPACR_WATCH_GUI_STALLS", "SPACR_BENCHMARK_JSON")

_LAUNCH_ONCE = r'''
"""One spaCR launch, offscreen, reporting what it built, read and started.

usage: launch_once.py <repo root> <normal|safe> <report.json>
"""
import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading

repo_root, phase, report_path = sys.argv[1:4]
POISONED, REPAIRED = sys.argv[4], sys.argv[5]
sys.path.insert(0, repo_root)
report = {"phase": phase, "threads": [], "startup_real_store_reads": []}


def write():
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=1)


_start = threading.Thread.start


def _recording_start(thread, *args, **kwargs):
    report["threads"].append(thread.name)
    return _start(thread, *args, **kwargs)


threading.Thread.start = _recording_start


class _AbortWhenThePoisonedBackdropIsBuilt(importlib.abc.MetaPathFinder):
    """Patch the backdrop module as it is imported, never before.

    Importing it here would put it in `sys.modules` for a safe mode that
    never asked for it, and the import list is one of the things measured.
    """

    def find_spec(self, name, path, target=None):
        if name != "spacr.qt.widgets.ambient":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            original_init = module.AmbientWidget.__init__

            def __init__(widget, *args, **kwargs):
                if kwargs.get("theme") == POISONED:
                    sys.stderr.write("INJECTED-BACKDROP-ABORT\n")
                    sys.stderr.flush()
                    os.abort()
                original_init(widget, *args, **kwargs)

            module.AmbientWidget.__init__ = __init__

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _AbortWhenThePoisonedBackdropIsBuilt())

import spacr  # noqa: E402

report["spacr"] = spacr.__file__
write()

from PySide6 import QtCore, QtWidgets  # noqa: E402

recording = {"on": phase == "safe"}
_value = QtCore.QSettings.value


def _recording_value(settings, key, *args, **kwargs):
    if recording["on"]:
        report["startup_real_store_reads"].append(str(key))
    return _value(settings, key, *args, **kwargs)


QtCore.QSettings.value = _recording_value

MODULES = re.compile(r"opengl|vispy|spacr\.qt\.widgets\.(ambient|fractal)|"
                     r"torch|cupy|cellpose|^spacr\.settings$|settings_model|"
                     r"^spacr\.qt\.imagery$", re.I)
WIDGETS = re.compile(r"Ambient|Fractal|OpenGL|GLWidget|Canvas|Tour|Consent")


def _snapshot(app):
    recording["on"] = False
    report["startup"] = {
        "modules": sorted(m for m in sys.modules if MODULES.search(m)),
        "widgets": sorted({type(w).__name__ for w in app.allWidgets()
                           if WIDGETS.search(type(w).__name__)}),
        "threads": list(report["threads"]),
        "environment": {name: os.environ.get(name) for name in
                        ("SPACR_NO_GL", "SPACR_NO_BACKDROP")},
    }
    write()


def _watchdog():
    report["exit"] = "watchdog"
    write()
    os._exit(3)


def _repair_through_preferences(app):
    _snapshot(app)
    window = next(w for w in app.topLevelWidgets()
                  if type(w).__name__ == "MainWindow")

    def in_the_dialog(tries=[0]):
        dialog = QtWidgets.QApplication.activeModalWidget()
        if dialog is None and tries[0] < 50:
            tries[0] += 1
            QtCore.QTimer.singleShot(200, in_the_dialog)
            return
        combo = dialog.findChild(QtWidgets.QComboBox, "AmbientTheme")
        report["dialog_showed"] = combo.currentData()
        combo.setCurrentIndex(combo.findData(REPAIRED))
        box = dialog.findChild(QtWidgets.QDialogButtonBox)
        box.button(QtWidgets.QDialogButtonBox.Save).click()
        report["saved"] = True

    QtCore.QTimer.singleShot(500, in_the_dialog)
    window._open_preferences()
    report["preferences_closed"] = True
    window.close()


def _look_and_leave(app):
    _snapshot(app)
    app.quit()


_exec = QtWidgets.QApplication.exec


def _driven_exec(app):
    QtCore.QTimer.singleShot(120000, _watchdog)
    drive = _repair_through_preferences if phase == "safe" else _look_and_leave
    QtCore.QTimer.singleShot(3000, lambda: drive(app))
    return _exec()


QtWidgets.QApplication.exec = _driven_exec

if phase == "safe":
    from spacr.qt.safespacr import main
    report["rc"] = main([])
else:
    from spacr.qt import run
    report["rc"] = run(["--no-setup"])
report["exit"] = "returned"
write()
'''


@pytest.fixture(scope="module")
def a_start_that_a_preference_broke(tmp_path_factory):
    """296's own verification, in three real launches against one store.

    Write a preference that makes the ordinary start die; launch `spacr`
    and watch it die; launch `safespacr`, change that preference in the
    Preferences dialog, press Save and close the window; launch `spacr`
    again and watch it start.

    WHY AN INJECTED ABORT, AND WHY THIS ONE. No stored value makes an
    ordinary start fail on the offscreen platform. Measured 2026-09-15:
    every key the preference modules define (115 of them) was written as a
    string, a list, a negative number, 1e308 and broken JSON in turn, and
    `spacr` started and exited 0 every time; every zero-argument getter
    returned rather than raised. The crash 296 was asked for is NATIVE --
    "Fatal Python error: Segmentation fault" and "Aborted" with no Python
    frame, on the drawing path (295; `spacr.qt.crash_recovery`) -- and it
    needs a GL driver on a real display, which no test run has.

    So the launcher reproduces that crash's shape and nothing more: when
    the backdrop for the STORED animation is built, the process aborts, with
    no exception for any `except` in spaCR to catch -- the property that
    made the real one unrecoverable from inside the application. Everything
    else is real: the preference is the real Animation key in a real store
    that the real launchers read, the repair goes through the real dialog
    and the real Save button, and the exit is the real window close. The
    abort lives in this test's launcher, not in spaCR, and fires for one
    animation only, so it is the written VALUE that kills the start and
    choosing another animation is what repairs it.

    IN FRESH PROCESSES, because safe mode is process-global and a native
    abort would take pytest down with it. The store and the crash markers
    are pointed into a temporary directory, the installer profile at a file
    that does not exist, and laptop mode off, since laptop mode drops the
    backdrop on a small machine (a CI runner) and the ordinary start would
    then survive for a reason that has nothing to do with the preference.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    from PySide6.QtCore import QSettings

    import spacr

    repo_root = Path(spacr.__file__).resolve().parents[1]
    sandbox = tmp_path_factory.mktemp("safespacr-296")
    store_path = sandbox / "config" / "spacr" / "qt.conf"
    launcher = sandbox / "launch_once.py"
    launcher.write_text(_LAUNCH_ONCE, encoding="utf-8")

    env = {name: value for name, value in os.environ.items()
           if name not in _ENVIRONMENT_THE_LAUNCHES_MUST_NOT_INHERIT}
    env.update(QT_QPA_PLATFORM="offscreen",
               XDG_CONFIG_HOME=str(sandbox / "config"),
               SPACR_LOG_DIR=str(sandbox / "logs"),
               SPACR_INSTALL_PROFILE=str(sandbox / "no-installer-profile"),
               SPACR_LAPTOP_MODE="0")

    store = QSettings(str(store_path), QSettings.IniFormat)
    store.setValue(preferences._KEY_AMBIENT_ENABLED, True)
    store.setValue(preferences._KEY_AMBIENT_THEME, POISONED_ANIMATION)
    store.setValue(preferences._KEY_FONT_SCALE, UNTOUCHED_FONT_SCALE)
    store.sync()
    del store

    def launch(phase, name):
        report_path = sandbox / f"{name}.json"
        finished = subprocess.run(
            [sys.executable, str(launcher), str(repo_root), phase,
             str(report_path), POISONED_ANIMATION, REPAIRED_ANIMATION],
            env=env, capture_output=True, text=True, timeout=300)
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.exists() else {})
        return {"returncode": finished.returncode,
                "stderr": finished.stderr[-4000:], "report": report}

    def stored(key):
        return QSettings(str(store_path), QSettings.IniFormat).value(key)

    broken = launch("normal", "broken")
    safe = launch("safe", "safe")
    after_safe = {"animation": stored(preferences._KEY_AMBIENT_THEME),
                  "font_scale": stored(preferences._KEY_FONT_SCALE)}
    repaired = launch("normal", "repaired")
    return {"repo_root": str(repo_root), "broken": broken, "safe": safe,
            "after_safe": after_safe, "repaired": repaired}


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_written_animation_kills_the_ordinary_start(
        a_start_that_a_preference_broke):
    """Step one: the store really does make `spacr` die."""
    runs = a_start_that_a_preference_broke
    broken = runs["broken"]

    assert broken["report"].get("spacr", "").startswith(runs["repo_root"])
    assert broken["returncode"] != 0 and (
        "INJECTED-BACKDROP-ABORT" in broken["stderr"]), (
        "the ordinary start survived the poisoned animation, so nothing "
        "below proves safe mode gets past one. Something other than the "
        f"store dropped the backdrop:\n{broken['stderr']}")
    assert broken["report"].get("exit") != "returned"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_safespacr_opens_anyway_and_saves_the_repair(
        a_start_that_a_preference_broke):
    """Step two: in, change it, Save, close -- and it is written."""
    runs = a_start_that_a_preference_broke
    safe = runs["safe"]
    report = safe["report"]

    assert report.get("spacr", "").startswith(runs["repo_root"])
    assert safe["returncode"] == 0, (
        f"safespacr did not get in:\n{safe['stderr']}")
    assert report.get("exit") == "returned" and report.get("rc") == 0
    assert report.get("dialog_showed") != POISONED_ANIMATION, (
        "the Preferences dialog showed the stored animation, so safe mode "
        "read the value it exists to escape")
    assert report.get("saved") and report.get("preferences_closed")
    assert runs["after_safe"]["animation"] == REPAIRED_ANIMATION, (
        "Save in safe mode did not reach the real store")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_ordinary_start_works_again_after_the_repair(
        a_start_that_a_preference_broke):
    """Step three: `spacr` starts, and it starts WITH a backdrop.

    The backdrop is checked for, not only the exit code. Two unclean exits
    in a row make `spacr.qt.crash_recovery` start without one, and a start
    that skipped the backdrop would pass here for the wrong reason.
    """
    runs = a_start_that_a_preference_broke
    repaired = runs["repaired"]
    startup = repaired["report"].get("startup", {})

    assert repaired["returncode"] == 0, repaired["stderr"]
    assert repaired["report"].get("exit") == "returned"
    assert "AmbientWidget" in startup.get("widgets", []), (
        "the repaired start built no backdrop, so it did not exercise the "
        "path that killed the broken one")
    assert startup.get("environment", {}).get("SPACR_NO_BACKDROP") is None


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_safespacr_builds_no_backdrop_loads_no_gl_and_starts_no_thread(
        a_start_that_a_preference_broke):
    """296's list, measured in the same safe launch that did the repair.

    No animated backdrop and no fractal widget built, and no fractal module
    imported; nothing of vispy, spaCR's OpenGL library, imported at all; and
    no background thread -- 296 names "no preloading, no background import
    thread", and the settings pre-warm is one. The backdrop MODULE has a
    test of its own below.
    """
    startup = a_start_that_a_preference_broke["safe"]["report"]["startup"]

    import re

    drawing = [name for name in startup["modules"] if re.search(
        r"vispy|opengl|spacr\.qt\.widgets\.fractal", name, re.I)]
    assert drawing == [], f"safe mode imported {drawing}"
    built = [name for name in startup["widgets"]
             if re.search(r"Ambient|Fractal|OpenGL|GLWidget|Canvas", name)]
    assert built == [], f"safe mode built {built}"
    assert startup["threads"] == [], (
        f"safe mode started background threads {startup['threads']}")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_safespacr_reads_nothing_from_the_real_store_while_it_starts(
        a_start_that_a_preference_broke):
    """"It must not load the same preferences path that a normal start does."

    Every read that reached a real `QSettings` before the window settled is
    recorded; the safe-mode shadow answers without reaching one, so the list
    is exactly the reads that escaped it. The first-run tour and the
    installer consent were those reads, and both put something in front of
    the settings the user came to repair.
    """
    report = a_start_that_a_preference_broke["safe"]["report"]

    assert report["startup_real_store_reads"] == [], (
        "safe mode read the real store at startup: "
        f"{report['startup_real_store_reads']}")
    in_front = [name for name in report["startup"]["widgets"]
                if "Tour" in name or "Consent" in name]
    assert in_front == [], f"safe mode opened {in_front} over the window"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_safespacr_does_not_import_the_backdrop_module(
        a_start_that_a_preference_broke):
    """Nothing is built from it, but 296 says no animated backdrop at all.

    The last importer was `apply_preferences_to_app` ->
    `apply_ambient_preferences`, which imported the module to look for
    widgets to hide before asking whether the backdrop was on. A widget of
    a class whose module was never imported cannot exist, so there is
    nothing to hide then.
    """
    startup = a_start_that_a_preference_broke["safe"]["report"]["startup"]

    assert "spacr.qt.widgets.ambient" not in startup["modules"]


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.xfail(strict=True, reason=(
    "296 gap, not fixed: the Preferences dialog shows safe mode's defaults "
    "and Save writes every control, so a safe-mode Save resets every "
    "preference the user did not touch. The fix belongs in "
    "spacr/qt/preferences.py (PreferencesDialog), outside this change."))
def test_a_safe_mode_save_keeps_the_preferences_it_did_not_change(
        a_start_that_a_preference_broke):
    """Re-saving one value must not quietly reset the rest to defaults."""
    stored = a_start_that_a_preference_broke["after_safe"]["font_scale"]

    assert float(stored) == UNTOUCHED_FONT_SCALE
