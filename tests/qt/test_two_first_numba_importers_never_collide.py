"""The warm thread and the kernel warmup thread can both import numba first.

Each of twenty fresh processes starts both at the same instant behind a
barrier. A half-initialised numba surfaces as an ImportError in either one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("numba")

SCRIPT = textwrap.dedent("""
    import threading
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    from spacr.qt.widgets import ambient
    barrier = threading.Barrier(2)
    errors = []

    def first():
        barrier.wait()
        try:
            ambient._warm_packed_scatter()
        except BaseException as exc:
            errors.append(repr(exc))

    def second():
        barrier.wait()
        try:
            ambient._import_numba()
            import numba
            numba.njit
        except BaseException as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    failed = ambient._PACKED_SCATTER_FAILED
    print("RESULT", errors, failed)
""")


def _once(tmp_path, index):
    env = dict(os.environ, SPACR_HOME=str(tmp_path / str(index)),
               QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(sys.path))
    done = subprocess.run([sys.executable, "-c", SCRIPT], env=env,
                          capture_output=True, text=True, timeout=300)
    lines = [x for x in done.stdout.splitlines() if x.startswith("RESULT")]
    return lines[-1] if lines else done.stderr[-1500:]


def test_twenty_fresh_processes_import_numba_from_two_threads_cleanly(tmp_path):
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: _once(tmp_path, i), range(20)))
    assert results == ["RESULT [] False"] * 20
