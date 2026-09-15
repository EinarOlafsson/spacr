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
    seen = {}

    def _fake_run(argv):
        seen["timing"] = os.environ.get("SPACR_TIMING")
        seen["no_gl"] = os.environ.get("SPACR_NO_GL")
        seen["safe"] = preferences.in_safe_mode()
        seen["argv"] = list(argv or [])
        return 0

    import spacr.qt as qt_pkg
    monkeypatch.setattr(qt_pkg, "run", _fake_run)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)

    assert safespacr.main([]) == 0
    assert seen["timing"] is None, "timing instrumentation survived safe mode"
    assert seen["no_gl"] == "1"
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
