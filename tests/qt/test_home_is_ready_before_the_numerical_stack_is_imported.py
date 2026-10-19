"""Home reports ready with no pandas, scipy, scikit-image or numba.

The installed-artifact smoke probe samples sys.modules at Home's readiness
record. The start-up warm imports those packages on a worker, so it holds
them back until Home has been ready for a moment. Run in fresh processes,
several times, because the old failure was a race.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent("""
    import json, sys
    from PySide6.QtWidgets import QApplication
    import spacr.qt as spacr_qt
    from spacr.qt import timing
    from spacr.qt.first_run import mark_tour_seen
    mark_tour_seen()
    HEAVY = ("pandas", "scipy", "skimage", "numba")
    seen = {}

    def observe(entry):
        if entry.get("detail") != "__home__" or seen:
            return
        seen["heavy"] = [m for m in HEAVY if m in sys.modules]
        seen["ready"] = True
        QApplication.instance().quit()

    timing.subscribe_readiness(observe)
    code = spacr_qt.run(["--no-setup"])
    sys.stdout.write("\\nRESULT " + json.dumps({"code": code, **seen}) + "\\n")
""")


def _once(tmp_path, index):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="", SPACR_TIMING="1", SPACR_TIMING_IMPORTS="0",
               HOME=str(tmp_path / str(index)),
               XDG_CONFIG_HOME=str(tmp_path / str(index)),
               SPACR_HOME=str(tmp_path / str(index) / "spacr"),
               PYTHONPATH=os.pathsep.join([str(ROOT)] + sys.path))
    (tmp_path / str(index)).mkdir()
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT ")]
    if not lines:
        return {"error": done.stderr[-800:]}
    return json.loads(lines[-1][7:])


def test_home_is_ready_before_the_warm_imports_the_numerical_stack(tmp_path):
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda i: _once(tmp_path, i), range(6)))
    assert results == [{"code": 0, "heavy": [], "ready": True}] * 6
