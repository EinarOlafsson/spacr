"""If the process dies below Python, it must still say where.

Reported three times on 2026-08-19: a regression run closes [success] and the
process is gone milliseconds later. The log ends mid-session with no shutdown
lines, dmesg and coredumpctl have nothing, and Python prints nothing -- because
the process dies below Python, in Qt or a C extension.

Three hypotheses were tested and eliminated against real sessions (an
off-thread plt.show(), pyplot building Qt canvases on the worker, and
quitOnLastWindowClosed), each costing the maintainer a launch-and-reproduce
cycle. faulthandler costs nothing until a fatal signal arrives and then names
the frame.
"""
import os
import subprocess
import sys

import pytest


def test_it_installs_and_reports_where_it_writes(tmp_path, monkeypatch):
    import faulthandler

    from spacr.qt import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("spacr.logging_util.log_dir", lambda: str(tmp_path))

    path = app._install_crash_dump()

    assert path.endswith(app.CRASH_DUMP_NAME)
    assert os.path.exists(path)
    assert faulthandler.is_enabled()


def test_launch_installs_it_before_anything_can_crash():
    import inspect

    from spacr.qt import app

    source = inspect.getsource(app.launch)
    assert "_install_crash_dump()" in source
    assert source.index("_install_crash_dump()") < source.index("QApplication(")


def test_a_real_segfault_lands_in_the_file(tmp_path):
    """Not mocked: the whole value is that it works from a signal handler."""
    script = tmp_path / "die.py"
    script.write_text(
        "import spacr.logging_util as L\n"
        f"L.log_dir = lambda: {str(tmp_path)!r}\n"
        "from spacr.qt.app import _install_crash_dump\n"
        "_install_crash_dump()\n"
        "import ctypes\n"
        "ctypes.string_at(0)\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")

    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, env=env, timeout=300)

    assert result.returncode != 0
    dump = tmp_path / "spacr-crash.log"
    assert dump.exists()
    text = dump.read_text()
    assert "Fatal Python error" in text
    assert "ctypes" in text, "the stack must name the frame that died"
