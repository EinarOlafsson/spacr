"""A finished job's Qt objects are freed by reference counting, not by gc.

Tutorial lessons 37, 42 and 14 segfaulted on the capture host with the GUI
thread inside the garbage collector. Offscreen on 2026-10-19 the first job a
process ran left its ``PipelineWorker``, its output drain and its run handle
alive after the screen had let go of them: importing ``matplotlib.pyplot``
inside :func:`spacr.qt.bridge.make_thread` raises and catches hundreds of
pyparsing exceptions whose traceback cycles hold, through ``f_back``, every
frame above the import, and so the job's locals. Those QObjects were then
destroyed from inside a later collection. The import now sweeps its own
young cycles before any worker exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_the_first_job_of_a_process_dies_by_reference_count(tmp_path):
    script = textwrap.dedent("""
        import gc, sys, time, weakref
        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv[:1])
        gc.disable()
        assert "matplotlib.pyplot" not in sys.modules
        from spacr.qt import bridge
        done = []
        thread, worker = bridge.make_thread(lambda settings: None, {},
                                            app_key="probe", journal=False)
        alive = weakref.ref(worker), weakref.ref(worker._out_drain)
        worker.finished.connect(lambda ok: done.append(ok))
        thread.start()
        end = time.monotonic() + 20
        while not done and time.monotonic() < end:
            app.processEvents()
        end = time.monotonic() + 1
        while time.monotonic() < end:
            app.processEvents()
        del thread, worker
        for _ in range(5):
            app.processEvents()
        print("FINISHED" if done else "NEVER FINISHED")
        print("ALIVE" if any(ref() is not None for ref in alive) else "FREED")
    """)
    probe = tmp_path / "probe.py"
    probe.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True,
        timeout=180, env={**os.environ,
                          "QT_QPA_PLATFORM": "offscreen"})
    assert "FINISHED" in result.stdout, result.stdout + result.stderr
    assert "FREED" in result.stdout, (
        "the first job's worker or output drain outlived its owners and "
        "would be destroyed inside the garbage collector\n"
        + result.stdout + result.stderr)

