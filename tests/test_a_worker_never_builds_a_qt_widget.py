"""A regression's QC figures must not build Qt widgets on the run's thread.

Traced 2026-08-19 from a crash dump, after the process segfaulted twice in
places that had nothing to do with the cause -- once inside an
application-wide event filter, once inside pandas' CSV parser. A guard on
`QObject.__init__` named the real one:

    WidgetGroup was CONSTRUCTED on 'Dummy-2'
      bridge.py run  ->  perform_regression
      ->  _run_guide_permutation_analysis  ->  write_diagnostic_suite
      ->  plot_inference_diagnostics  ->  write_figure  ->  render_figure
      ->  build_scene

`build_scene` makes a pg.GraphicsLayoutWidget. A QWidget built on a worker
LIVES on that worker; every later touch, including Qt destroying it, is
undefined. Qt reports the one case it can detect -- "QBasicTimer::start:
Timers cannot be started from another thread" -- and says nothing about the
rest, which is why the crash surfaced somewhere unrelated each time.

VERIFIED BY DISABLING THE GATE: with the thread check stubbed out, this file
does not fail -- it HANGS, and had to be killed. Building the widget on the
worker blocks rather than raising, which is the other half of what was
reported ("this hung my computer twice so i had to restart it"). A test that
can only be seen to work by hanging is worth saying so in.
"""
import threading

import pytest


def test_the_gate_is_open_on_the_gui_thread(qapp):
    from spacr.figures.scene import pyqtgraph_ready

    ok, why = pyqtgraph_ready()

    # Either it is ready, or it is refused for a reason that is NOT the
    # thread -- a machine without pyqtgraph is allowed.
    assert ok or "GUI thread" not in why


def test_the_gate_is_shut_on_a_worker(qapp):
    from spacr.figures.scene import pyqtgraph_ready

    answer = {}

    def ask():
        answer["r"] = pyqtgraph_ready()

    worker = threading.Thread(target=ask, name="a-run")
    worker.start()
    worker.join()

    ok, why = answer["r"]
    assert ok is False
    assert "GUI thread" in why
    assert "a-run" in why, "name the thread, so the log says which run it was"
    assert "matplotlib page is written instead" in why, (
        "say what happens instead; a refusal that loses the figure is worse")


def test_no_qt_widget_is_constructed_when_a_worker_renders(qapp, tmp_path):
    """The property that actually matters, exercised end to end."""
    pytest.importorskip("pyqtgraph")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PySide6.QtCore import QObject

    from spacr.figures.scene import render_figure

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])

    born = []
    real_init = QObject.__init__

    def counting_init(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            born.append(type(self).__name__)
        return real_init(self, *args, **kwargs)

    QObject.__init__ = counting_init
    try:
        out = {}

        def render():
            out["r"] = render_figure(figure, str(tmp_path / "x.png"),
                                     fmt="png")

        worker = threading.Thread(target=render, name="a-run")
        worker.start()
        worker.join()
    finally:
        QObject.__init__ = real_init
        plt.close(figure)

    written, report = out["r"]
    assert written is None, "the worker must not have written a Qt scene"
    assert born == [], f"Qt objects built on the worker: {sorted(set(born))}"


def test_a_headless_run_is_not_affected(qapp):
    """With no GUI the run IS the main thread, so the gate never fires."""
    from spacr.figures.scene import pyqtgraph_ready

    ok, why = pyqtgraph_ready()

    assert "GUI thread" not in why
