"""The field's GPU renderer is released on its own thread before Qt goes away.

On the GNOME capture host lesson 08's Templates run ended with SIGSEGV after
a clean quit; crash.log showed only "<no Python frame>" (interpreter
finalization) and listed ``moderngl.mgl`` and ``glcontext.egl``: the field's
impulse-lens renderer, a moderngl context over EGL, had been chosen by
default. Nothing stopped the backdrop's shading thread at exit. It is a
daemon thread that owns that context, so the context was left to interpreter
finalization: the thread could still be inside the driver, or the context
was collected on another thread after the GL libraries started to unload.

Now ``aboutToQuit``, ``spacr.qt.app.launch`` once the event loop returns,
and an ``atexit`` hook registered after PySide6's all call
``_retire_shading_threads``: every shading thread is joined, and each one
releases its renderer on itself as it ends, while QApplication still exists.

These run against a fake renderer that records the thread it was created
and closed on and whether the application and the interpreter were still
whole. The real GPU path is not exercised here (this suite never uses a GPU).
Each case runs in its own interpreter, because a crash at exit would take
pytest with it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

import spacr                                                         # noqa: E402

REPO_ROOT = Path(spacr.__file__).resolve().parents[1]

FAKE = '''
import os, sys, threading
from PySide6.QtGui import QImage

LOG = os.environ["FAKE_GL_LOG"]


def note(line):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\\n")


class FakeGraphics:
    renderer = "fake renderer"

    def __init__(self):
        self._owner = threading.get_ident()
        note("create " + threading.current_thread().name)

    def _close(self):
        from PySide6.QtWidgets import QApplication
        note("close %s owner=%s app=%s finalizing=%s" % (
            threading.current_thread().name,
            threading.get_ident() == self._owner,
            QApplication.instance() is not None, sys.is_finalizing()))

    def __getattr__(self, name):
        if not name.startswith("_draw"):
            raise AttributeError(name)

        def draw(width, height, *args):
            note("draw")
            image = QImage(width, height, QImage.Format_RGB32)
            image.fill(0xff102030)
            return image
        return draw


def install():
    from spacr.qt.widgets import ambient
    ambient._FlowPointGraphics = FakeGraphics
    ambient._flow_graphics_preflight = lambda: True
    ambient._flow_compute_idle = lambda: True
    ambient._FLOW_GRAPHICS_PROBE = True
'''

PREAMBLE = f'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, {str(REPO_ROOT)!r})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_gl
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget
'''

FIELD = '''
app = QApplication.instance() or QApplication(["exit-test"])
fake_gl.install()
from spacr.qt.widgets import ambient
host = QWidget()
host.resize(320, 200)
field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=30)
field._engine._graphics_backend = "gpu"
field.setGeometry(host.rect())
host.show()


def drawn():
    with open(os.environ["FAKE_GL_LOG"], encoding="utf-8") as handle:
        return handle.read().count("draw")
'''


def _run(body: str, tmp_path: Path, timeout: int = 180):
    """Run ``body`` in a fresh interpreter; the result and the renderer log."""
    env = dict(os.environ)
    env.pop("SPACR_NO_GL", None)
    log = tmp_path / "gl.log"
    log.write_text("", encoding="utf-8")
    env.update(SPACR_HOME=str(tmp_path / "spacr-home"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               SPACR_LOG_DIR=str(tmp_path / "logs"),
               FAKE_GL_LOG=str(log),
               QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    (tmp_path / "fake_gl.py").write_text(FAKE, encoding="utf-8")
    script = tmp_path / "exit_case.py"
    script.write_text(PREAMBLE + textwrap.dedent(body), encoding="utf-8")
    out = subprocess.run([sys.executable, "-X", "faulthandler", str(script)],
                         capture_output=True, text=True, timeout=timeout,
                         env=env, cwd=str(tmp_path))
    return out, log.read_text(encoding="utf-8").splitlines()


def _assert_released_in_order(out, lines, before=None):
    assert out.returncode == 0, (
        f"rc {out.returncode}; stderr tail:\n{out.stderr[-3000:]}")
    assert "create spacr-ambient-shade" in lines, lines
    creates = [line for line in lines if line.startswith("create ")]
    closes = [line for line in lines if line.startswith("close ")]
    assert len(closes) == len(creates) >= 1, lines
    assert set(closes) == {
        "close spacr-ambient-shade owner=True app=True finalizing=False"}, lines
    if before is not None:
        assert before in lines, lines
        last = max(i for i, line in enumerate(lines) if line.startswith("close "))
        assert last < lines.index(before), lines


def test_quitting_the_event_loop_releases_the_renderer_first(tmp_path):
    out, lines = _run(FIELD + '''
deadline = time.monotonic() + 30
while drawn() < 3 and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.01)
QTimer.singleShot(0, app.quit)
app.exec()
fake_gl.note("event loop returned")
''', tmp_path)
    _assert_released_in_order(out, lines, before="event loop returned")


def test_a_script_that_ends_without_an_event_loop_releases_the_renderer(
        tmp_path):
    out, lines = _run(FIELD + '''
deadline = time.monotonic() + 30
while drawn() < 3 and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.01)
fake_gl.note("script ends")
''', tmp_path)
    _assert_released_in_order(out, lines)
    assert lines.index("script ends") < lines.index(
        "close spacr-ambient-shade owner=True app=True finalizing=False")


def test_the_real_launch_quits_with_rc_0_and_the_renderer_released(tmp_path):
    """``spacr.qt.app.launch``: the main window, a module, quit, exit."""
    out, lines = _run('''
fake_gl.install()
import spacr.qt.app as qt_app


def later():
    app = QApplication.instance()
    windows = [w for w in app.topLevelWidgets()
               if type(w).__name__ == "MainWindow"]
    fake_gl.note("windows %d" % len(windows))
    if windows:
        windows[0].open_module("measure")
    QTimer.singleShot(2500, app.quit)


original_exec = QApplication.exec


def exec_and_quit(self, *args):
    QTimer.singleShot(1500, later)
    return original_exec()


QApplication.exec = exec_and_quit
if __name__ == "__main__":
    code = qt_app.launch([])
    fake_gl.note("launch returned %s" % code)
    sys.exit(code)
''', tmp_path, timeout=300)
    assert "windows 1" in lines, (lines, out.stderr[-3000:])
    _assert_released_in_order(out, lines, before="launch returned 0")


def test_retiring_joins_every_shading_thread_and_stops_new_ones(
        qtbot, monkeypatch, tmp_path):
    """In process: the join, the release thread and nothing started after."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets import ambient

    released = []

    class Fake:
        renderer = "fake"

        def __init__(self):
            self._owner = threading.get_ident()

        def _close(self):
            released.append(threading.current_thread().name)

        def __getattr__(self, name):
            if not name.startswith("_draw"):
                raise AttributeError(name)
            from PySide6.QtGui import QImage

            def draw(width, height, *args):
                image = QImage(width, height, QImage.Format_RGB32)
                image.fill(0)
                return image
            return draw

    monkeypatch.setattr(ambient, "_FlowPointGraphics", Fake)
    monkeypatch.setattr(ambient, "_flow_graphics_preflight", lambda: True)
    monkeypatch.setattr(ambient, "_flow_compute_idle", lambda: True)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_SHADING_RETIRED", [False])
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    host = QWidget()
    host.resize(240, 160)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=30)
    field._engine._graphics_backend = "gpu"
    field.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitUntil(lambda: field._engine._point_graphics is not None,
                    timeout=10000)
    thread = field._producer_box[0]._thread
    assert ambient._retire_shading_threads(5.0) == 0
    assert not thread.is_alive()
    assert released == ["spacr-ambient-shade"]
    assert field._engine._point_graphics is None
    assert field._producer_box[0] is None
    field.stop()
    field._sync_run_state()
    assert field.shading_thread_alive()
    assert ambient._retire_shading_threads(5.0, final=True) == 0
    assert field._producer_box[0] is None
    field.stop()
    field._sync_run_state()
    assert field._producer_box[0] is None


def test_no_gl_keeps_the_field_off_the_graphics_driver(monkeypatch):
    from spacr.qt.widgets import ambient

    monkeypatch.setenv("SPACR_NO_GL", "1")
    monkeypatch.setattr(ambient, "_FLOW_GRAPHICS_PROBE", None)
    assert ambient._flow_graphics_possible() is False
    assert ambient._flow_graphics_preflight() is False
