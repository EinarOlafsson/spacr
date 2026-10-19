"""Opening Regression after the start-up warm imports no numerical stack.

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
import sys
import textwrap

import pytest

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
    A._start_settings_prewarm().join()
    before = set(sys.modules)
    window.open_module("regression")
    end = time.monotonic() + 3
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    tops = ("pandas", "scipy", "matplotlib", "skimage")
    new = [m for m in sys.modules if m not in before]
    print(json.dumps({t: sorted(m for m in new
                                if m == t or m.startswith(t + "."))
                      for t in tops}))
""")


def test_opening_regression_adds_no_heavy_module_after_the_warm(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               CUDA_VISIBLE_DEVICES="", HOME=str(tmp_path),
               XDG_CONFIG_HOME=str(tmp_path))
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    assert lines, done.stderr[-2000:]
    added = json.loads(lines[-1])
    assert added == {top: [] for top in added}
