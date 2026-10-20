"""Opening Regression after the start-up warm imports no numerical stack and
compiles nothing on the GUI thread.

Regression's first open pulled pandas, scipy, matplotlib and scikit-image
onto the GUI thread, about a second of frozen window. The start-up settings
warm now imports them on its worker thread, so opening the module afterwards
must add no module from those packages. Counted in a fresh process, since
the test process has long since imported them.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys
import textwrap

import pytest


_TESTED_ROOT = Path(__file__).resolve().parents[2]


def _tested_tree(env):
    """Make a fresh probe process import this tree's spacr, not an install.

    :param env: the probe's environment, changed in place.
    :returns: ``env``.
    """
    env["SPACR_TESTED_ROOT"] = str(_TESTED_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTED_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


GUARD = 'import os as _os, spacr as _spacr\nassert _os.path.realpath(_spacr.__file__).startswith(_os.path.realpath(_os.environ["SPACR_TESTED_ROOT"]) + _os.sep), ("probe imported spacr from " + _spacr.__file__)\n'

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

SCRIPT = textwrap.dedent("""
    import json, sys, time
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from spacr.qt import app as A
    from spacr.qt.first_run import mark_tour_seen
    mark_tour_seen()
    window = A.MainWindow()
    window.show()
    app.processEvents()
    import threading
    import numba
    import numba.core.dispatcher as dispatcher
    gui_compiles = []
    real_compile = dispatcher.Dispatcher.compile

    def counting(self, sig):
        if threading.current_thread() is threading.main_thread():
            gui_compiles.append(self.py_func.__name__)
        return real_compile(self, sig)

    dispatcher.Dispatcher.compile = counting
    A._start_settings_prewarm().join()
    before = set(sys.modules)
    window.open_module("__MODULE__")
    end = time.monotonic() + 3
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    tops = ("pandas", "scipy", "matplotlib", "skimage", "spacr.settings_advisor")
    new = [m for m in sys.modules if m not in before]
    out = {t: sorted(m for m in new if m == t or m.startswith(t + "."))
           for t in tops}
    out["gui_thread_compilations"] = gui_compiles
    sys.stdout.write("\\nRESULT " + json.dumps(out) + "\\n")
    sys.stdout.flush()
""")


@pytest.mark.parametrize("module", ["regression", "invasion"])
def test_opening_a_module_adds_no_heavy_module_after_the_warm(tmp_path, module):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               CUDA_VISIBLE_DEVICES="", HOME=str(tmp_path),
               XDG_CONFIG_HOME=str(tmp_path))
    done = subprocess.run([sys.executable, "-c", GUARD + SCRIPT.replace("__MODULE__", module)], env=_tested_tree(env),
                          capture_output=True, text=True, timeout=300)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    noise = "qt.text.font.db: OpenType support missing"
    errors = "\n".join(line for line in done.stderr.splitlines() if noise not in line)
    assert lines, f"probe exit {done.returncode}; stderr: {errors[-2000:]}; stdout: {done.stdout[-500:]}"
    added = json.loads(lines[-1][len("RESULT "):])
    assert added == {key: [] for key in added}
