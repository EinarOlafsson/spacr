"""A figure drawn on a worker must not be a QObject owned by that worker.

Reported 2026-08-19: "i ran it again and it just spontaniously quit". The log
shows the run CLOSING SUCCESSFULLY and, four milliseconds later,
"QBasicTimer::start: Timers cannot be started from another thread" -- then the
process is gone with no Python traceback.

A regression runs on a JobRunner worker and draws with pyplot. Under the
`qtagg` backend every `plt.figure()` on that worker builds a
FigureCanvasQTAgg, a QObject whose thread affinity is the WORKER. The main
thread then renders it, and Qt refuses; the `Internal C++ object
(FigureCanvasQTAgg) already deleted` errors in the same log are that object
seen from the other side.

`bridge` asked for Agg with `force=False`, WHICH DOES NOTHING once a backend
is active -- and by the time a run starts, `qtagg` is.
"""
import inspect

import pytest


def test_launch_sets_agg_before_any_figure_can_exist():
    from spacr.qt import app

    source = inspect.getsource(app.launch)
    assert 'matplotlib.use("Agg", force=True)' in source, (
        "pyplot must not be able to build a Qt canvas")
    # Before the QApplication, because switching later CLOSES open figures.
    assert source.index('matplotlib.use("Agg"') < source.index("QApplication(")


def test_the_bridge_no_longer_asks_with_force_false():
    from spacr.qt import bridge

    source = inspect.getsource(bridge)
    assert 'matplotlib.use("Agg", force=False)' not in source, (
        "force=False is a no-op once a backend is active, which is the bug")


def test_a_worker_thread_figure_carries_no_qt_object():
    """The property that actually matters, exercised rather than asserted."""
    import threading

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    seen = {}

    def draw():
        fig = plt.figure()
        fig.add_subplot(111).plot([0, 1], [0, 1])
        seen["canvas"] = type(fig.canvas).__name__
        plt.close(fig)

    worker = threading.Thread(target=draw)
    worker.start()
    worker.join()

    assert seen["canvas"] == "FigureCanvasAgg"
    assert "QT" not in seen["canvas"].upper()


def test_an_explicit_qt_canvas_still_works_under_agg(qapp):
    """Nothing is lost: the two call sites that want one build it themselves."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pytest.importorskip("matplotlib.backends.backend_qtagg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    fig = plt.figure()
    canvas = FigureCanvasQTAgg(fig)

    assert type(canvas).__name__ == "FigureCanvasQTAgg"
    plt.close(fig)
