"""Two ways the annotator took the whole process down, and their fixes.

Both faults are the same shape as the two this application has already been
through -- something outliving the thread or the loop that owns it -- and both
kill the interpreter rather than raising, so every test here runs the scenario
in a CHILD process and asserts on its exit status. An in-process test cannot
survive a SIGABRT to assert anything, and a hang has no exception to catch.

1. PICKING CELLPOSE.  A Cellpose outline is a model construction plus one
   native forward pass per channel, none of it interruptible.  The page worker
   asked ``isInterruptionRequested`` only BETWEEN crops, so a close that landed
   mid-crop waited out its whole 15 s budget, gave up, and parked a QThread
   that was still running.  Nothing then joined it, and interpreter shutdown
   destroyed the wrapper::

       QThread: Destroyed while thread 'annotate-page-3' is still running
       Fatal Python error: Aborted

   Captured that way on 2026-08-26 by choosing Cellpose in the annotator and
   closing the window while a page was still being outlined.

2. PRESSING COVERAGE.  The report was shown with ``QDialog.exec``, which runs
   a NESTED event loop.  ``QCoreApplication.quit`` -- what closing the last
   window does -- unwinds only the outermost loop, so the window vanished and
   the GUI thread stayed parked inside ``_on_coverage`` for ever.  The child
   process reproducing it had exactly one Python frame on its main thread::

       File ".../spacr/qt/screens/annotate.py", line 2752 in _on_coverage

   The same nested loop is also the window in which the report's own parent
   can be destroyed, which deletes the object whose ``exec`` is on the stack.

WHY THE CELLPOSE MODEL IS A STUB IN THE CHILD.  Everything the user touches is
real here: the Settings dialog, its ``Outline method`` combo, its OK button,
the page worker and the close.  What is replaced is the 1.2 GB ``cpsam``
checkpoint behind ``annotate_engine._get_cellpose_outline_model``, because a
test may not download it and CPU inference on it takes minutes.  The stub
stands in for the one property that causes the crash -- a call into native
code that runs longer than the close is willing to wait -- and the wiring that
reaches the real Cellpose call is asserted separately, against the real
function, in :func:`test_a_real_cellpose_outline_asks_before_every_channel`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


#: The child gets longer than `CLOSE_DRAIN_MS` (15 s) plus the stub's own work
#: so that a PASS is a real pass and not a race, while a hang or a crash still
#: lands well inside the outer subprocess timeout.
CHILD_BUDGET_S = 75
SUBPROCESS_TIMEOUT_S = 150


def _run_child(tmp_path: Path, body: str, name: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter and return the finished process.

    The child gets its own ``XDG_CONFIG_HOME`` so that driving real screens
    cannot write the developer's preferences, and no GPU: the display is
    driven by the same card.
    """
    script = tmp_path / name
    preamble = textwrap.dedent(
        """
        import faulthandler, os, sys, time
        faulthandler.enable()
        # A hang is as fatal as a crash and has no exception to catch, so the
        # child kills itself with a traceback rather than waiting to be
        # reaped: the test then sees a non-zero status and a stack.
        faulthandler.dump_traceback_later({budget}, exit=True)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        """
    ).format(budget=CHILD_BUDGET_S - 25)
    script.write_text(preamble + textwrap.dedent(body))

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    return subprocess.run(
        [sys.executable, str(script)],
        env=env, capture_output=True, text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


def _assert_exited_cleanly(proc: subprocess.CompletedProcess) -> None:
    """Fail with the child's own output, naming a signal when there was one."""
    if proc.returncode == 0:
        return
    how = (f"killed by signal {-proc.returncode}" if proc.returncode < 0
           else f"exit status {proc.returncode}")
    raise AssertionError(
        f"the child {how}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}")


# ---------------------------------------------------------------------------
# Shared child-side scaffolding: a synthetic experiment and a real screen
# ---------------------------------------------------------------------------

_MAKE_SOURCE = """
import sqlite3
import numpy as np
from PIL import Image

def make_source(root, n_crops):
    src = os.path.join(root, "expt")
    os.makedirs(os.path.join(src, "measurements"))
    os.makedirs(os.path.join(src, "data", "images"))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_crops):
        arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
        path = os.path.join(src, "data", "images",
                            "plate1_A0%d_f%d_cell.png" % (i % 3 + 1, i))
        Image.fromarray(arr).save(path)
        rows.append((path, i % 2, "plate1", "A0%d" % (i % 3 + 1), str(i)))
    con = sqlite3.connect(os.path.join(src, "measurements", "measurements.db"))
    con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY, '
                'annotate INTEGER, plateID TEXT, rowID TEXT, columnID TEXT)')
    con.executemany('INSERT INTO "png_list" VALUES (?,?,?,?,?)', rows)
    con.commit(); con.close()
    return src

def spin(app, ms):
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)
"""


# ---------------------------------------------------------------------------
# 1. Picking Cellpose
# ---------------------------------------------------------------------------

CELLPOSE_CHILD = _MAKE_SOURCE + """
import tempfile
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
src = make_source(tempfile.mkdtemp(), 3)

from spacr.qt import annotate_engine as engine
from spacr.qt.screens.annotate import AnnotateScreen

#: One channel's inference, long enough that a close landing inside it cannot
#: be waited out: `CLOSE_DRAIN_MS` is 15 s and three channels of this is 24 s.
EVAL_SECONDS = 8.0

class _SlowModel:
    "Stands in for cpsam: native work that does not come back for a while."
    def eval(self, image, **kwargs):
        time.sleep(EVAL_SECONDS)
        import numpy as np
        mask = np.zeros(image.shape[:2], dtype=np.int32)
        mask[4:12, 4:12] = 1
        return mask, None, None

# Seeded, not monkeypatched: `_get_cellpose_outline_model` returns the cached
# model when there is one, so this is the same door the real checkpoint comes
# through -- including the should_stop question asked before it.
engine._cellpose_outline_model = _SlowModel()

screen = AnnotateScreen()
screen._settings.image_size = (32, 32)
screen._settings.grid_rows, screen._settings.grid_cols = 1, 3
screen.resize(520, 380)
screen.show()
screen._open_source(src)
spin(app, 2500)
print("PAGE-PATHS", len(screen._page_paths), flush=True)

def pick_cellpose():
    dlg = app.activeModalWidget()
    dlg._outline.setText("r, g, b")
    dlg._outline_method.setCurrentText("cellpose")
    dlg.accept()

QTimer.singleShot(300, pick_cellpose)
screen._btn_settings.click()          # the user's route into the setting
print("OUTLINE-METHOD", screen._settings.outline_method, flush=True)
assert screen._settings.outline_method == "cellpose"

# Let the page worker get INSIDE a forward pass, then close underneath it.
spin(app, 2500)
print("WORKER-RUNNING", screen._page_worker is not None, flush=True)
assert screen._page_worker is not None, "no page worker to close underneath"

started = time.time()
screen.close()
print("CLOSED-IN", round(time.time() - started, 1), flush=True)
del screen
import gc; gc.collect()
spin(app, 1000)
from spacr.qt.bridge import parked_thread_count
print("PARKED", parked_thread_count(), flush=True)
print("CHILD-OK", flush=True)
"""


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_S + 30)
def test_picking_cellpose_and_closing_does_not_take_the_process_down(tmp_path):
    """The reported crash: choose Cellpose in the annotator.

    The child drives the Settings dialog's own combo and its OK button, waits
    until the page worker is inside a forward pass, and closes the screen
    there.  Before the fix this ended in ``QThread: Destroyed while thread
    'annotate-page-N' is still running`` and ``Fatal Python error: Aborted``,
    which is a signal, not a failed assertion.
    """
    proc = _run_child(tmp_path, CELLPOSE_CHILD, "child_cellpose.py")
    _assert_exited_cleanly(proc)
    assert "CHILD-OK" in proc.stdout, proc.stdout + proc.stderr
    assert "OUTLINE-METHOD cellpose" in proc.stdout
    assert "WORKER-RUNNING True" in proc.stdout
    # Nothing may be left parked: a parked thread is a QThread nobody can
    # destroy safely, and the point of the fix is that the close now succeeds.
    assert "PARKED 0" in proc.stdout, proc.stdout


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_S + 30)
def test_the_close_returns_instead_of_waiting_out_the_whole_page(tmp_path):
    """The close must cost one forward pass, not the rest of the page.

    Three crops of three channels at eight seconds each is 72 s of outlining.
    Asking to stop only between crops left up to 24 s of it to run, which is
    past the 15 s the close waits -- so the measurement is the fix: the close
    comes back inside one channel's work.
    """
    proc = _run_child(tmp_path, CELLPOSE_CHILD, "child_cellpose_timing.py")
    _assert_exited_cleanly(proc)
    line = [l for l in proc.stdout.splitlines() if l.startswith("CLOSED-IN")]
    assert line, proc.stdout
    closed_in = float(line[0].split()[1])
    assert closed_in < 15.0, (
        f"the close took {closed_in}s, so it waited out more than one "
        f"channel's inference:\n{proc.stdout}")


def test_a_real_cellpose_outline_asks_before_every_channel():
    """The wiring, against the real :func:`outline_image` and no stub at all.

    The child test above proves the crash is gone; this proves the question is
    asked where it has to be -- before the model is built and before each
    channel's forward pass -- without needing a checkpoint on disk.
    """
    import numpy as np
    from PIL import Image
    from spacr.qt import annotate_engine as engine

    asked = []
    img = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))

    def should_stop():
        asked.append(len(asked))
        return len(asked) >= 2      # let the first question through

    engine._cellpose_outline_model = None
    with pytest.raises(engine.OutlineCancelled):
        engine.outline_image(
            base_img=img, full_img=img, outline_channels=["r", "g", "b"],
            outline_method="cellpose", should_stop=should_stop)
    # Asked before the model was built, and the answer stopped the work
    # before cellpose was ever imported -- which is what makes an abandoned
    # page cost nothing rather than a 1.2 GB checkpoint read.
    assert len(asked) >= 2
    assert engine._cellpose_outline_model is None


def test_an_otsu_page_is_unaffected_by_the_stop_hook():
    """The cheap path must not have grown a way to fail.

    ``should_stop`` is only consulted around Cellpose calls, so an Otsu
    outline with a hook that always says stop still draws its outline.
    """
    import numpy as np
    from PIL import Image
    from spacr.qt import annotate_engine as engine

    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[6:18, 6:18, 1] = 200
    img = Image.fromarray(arr)
    out = engine.outline_image(
        base_img=img, full_img=img, outline_channels=["g"],
        outline_method="otsu", should_stop=lambda: True)
    assert np.asarray(out).shape == arr.shape
    assert not np.array_equal(np.asarray(out), arr), "no outline was drawn"


# ---------------------------------------------------------------------------
# 2. The Coverage button
# ---------------------------------------------------------------------------

COVERAGE_CHILD = _MAKE_SOURCE + """
import tempfile
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])
src = make_source(tempfile.mkdtemp(), 12)

from spacr.qt.screens.annotate import AnnotateScreen

screen = AnnotateScreen()
screen._settings.image_size = (32, 32)
screen.resize(640, 520)
screen.show()
screen._open_source(src)
spin(app, 2500)

def close_everything():
    # What the user does next: the application goes away while the report is
    # still up. `quit` unwinds the OUTERMOST loop only, so a report shown with
    # `exec` leaves the GUI thread inside `_on_coverage` with no window left.
    print("CLOSING-UNDER-THE-REPORT", flush=True)
    screen.close()
    app.quit()

QTimer.singleShot(600, close_everything)
print("PRESSING-COVERAGE", flush=True)
screen._btn_coverage.click()          # the button, not the handler
print("COVERAGE-RETURNED", flush=True)
spin(app, 800)
print("CHILD-OK", flush=True)
"""


@pytest.mark.timeout(SUBPROCESS_TIMEOUT_S + 30)
def test_pressing_coverage_and_closing_does_not_wedge_the_process(tmp_path):
    """The reported crash: press Coverage, then close.

    Before the fix the child never printed ``COVERAGE-RETURNED``: its main
    thread was inside the report's nested event loop with the window already
    gone, and it had to be killed.
    """
    proc = _run_child(tmp_path, COVERAGE_CHILD, "child_coverage.py")
    _assert_exited_cleanly(proc)
    assert "CLOSING-UNDER-THE-REPORT" in proc.stdout, proc.stdout
    assert "COVERAGE-RETURNED" in proc.stdout, (
        "the coverage press never returned:\n" + proc.stdout + proc.stderr)
    assert "CHILD-OK" in proc.stdout


def test_the_coverage_report_is_a_window_that_owns_its_own_end(qtbot,
                                                               tmp_path,
                                                               monkeypatch):
    """No nested loop, and no report outliving the screen it hangs off.

    ``WA_DeleteOnClose`` is what makes the report retire itself, and the Qt
    parent is what makes it go when the screen goes.  Both together are why
    the screen can be destroyed at any moment without deleting an object that
    is running an event loop.
    """
    from PySide6.QtCore import Qt
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)

    report = screen._show_report("Annotation coverage", "plate  n\np1     40")
    assert report.isVisible()
    assert report.testAttribute(Qt.WA_DeleteOnClose)
    assert report.parent() is screen
    assert "p1     40" in report._view.toPlainText()

    # A second press rewrites the open report rather than stacking another
    # window behind it.
    again = screen._show_report("Annotation coverage", "plate  n\np1     41")
    assert again is report
    assert "p1     41" in report._view.toPlainText()

    screen.close()
    qtbot.wait(50)
    assert not screen._reports


# ---------------------------------------------------------------------------
# The belt behind both: a parked thread must not meet interpreter shutdown
# ---------------------------------------------------------------------------

def test_a_parked_thread_is_joined_before_the_interpreter_tears_down(qtbot):
    """Parking keeps a running QThread referenced — until the globals go.

    ``drain_thread`` parks a thread it could not stop so that nothing destroys
    a running QThread. That reference lives in this module's globals, and
    interpreter shutdown clears them, so the wrapper was destroyed anyway:
    ``QThread: Destroyed while thread ... is still running``. The wait has to
    happen while the thread can still be joined.
    """
    from PySide6.QtCore import QThread
    from spacr.qt import bridge

    class _Stubborn(QThread):
        """Ignores ``quit()`` for a moment, the way native work does."""

        def run(self):
            import time as _t
            _t.sleep(1.2)

    thread = _Stubborn()
    thread.setObjectName("parked-under-test")
    thread.start()
    qtbot.waitUntil(thread.isRunning, timeout=2000)

    # Too short to succeed: this is the close giving up and parking it.
    assert bridge.drain_thread(thread, timeout_ms=50) is False
    assert bridge.parked_thread_count() >= 1
    assert bridge._PARKED_EXIT_HOOK_INSTALLED, (
        "parking without arming the exit drain is the abort that was fixed")

    # What the exit hook does, called directly so the assertion is on the
    # behaviour rather than on the interpreter shutting down.
    assert bridge.wait_for_parked_threads(timeout_ms=5000) == 0
    assert not thread.isRunning()
    assert bridge.parked_thread_count() == 0


def test_the_exit_drain_reports_what_it_could_not_join(qtbot):
    """A budget that runs out is counted, not silently ignored."""
    from PySide6.QtCore import QThread
    from spacr.qt import bridge

    class _VerySlow(QThread):
        def run(self):
            import time as _t
            _t.sleep(1.5)

    thread = _VerySlow()
    thread.start()
    qtbot.waitUntil(thread.isRunning, timeout=2000)
    assert bridge.drain_thread(thread, timeout_ms=10) is False

    # A budget shorter than the work leaves it counted as still running, so
    # the exit hook can say so instead of pretending everything was joined.
    assert bridge.wait_for_parked_threads(timeout_ms=100) == 1
    # And a real budget finishes the job, leaving nothing parked behind.
    assert bridge.wait_for_parked_threads(timeout_ms=5000) == 0
