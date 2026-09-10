"""`plt.show()` on a worker thread must not enter a Qt event loop.

MEASURED 2026-08-19 on the maintainer's own four-plate screen. Their log:

    09:32:44,894  run closed [success] in 51.8s
    09:32:44,897  Qt warning: QBasicTimer::start: Timers cannot be started
                  from another thread

and the application gone immediately after, every run. Reproduced headlessly:
`ml.minimum_cell_simulation` calls `plt.show()`, and with the Qt backend the
stack ends in `matplotlib.backends.qt_compat._exec` -- blocked forever in a Qt
event loop.

`_matplotlib_show_router` keys captures by THREAD, so a show on a thread with
no capture fell through to the real matplotlib show. On a worker that is a GUI
event loop started off the GUI thread: illegal, warned about, and fatal.
"""

import threading

import spacr


import pytest

from spacr.qt import bridge


def test_a_capture_still_receives_the_call():
    seen = []
    bridge._register_matplotlib_show(_FakePlt(), seen.append)
    try:
        bridge._matplotlib_show_router("arg")
        assert seen == ["arg"]
    finally:
        bridge._unregister_matplotlib_show(seen.append)


class _FakePlt:
    show = staticmethod(lambda *a, **k: None)


def test_a_worker_with_no_capture_does_not_reach_the_real_show():
    """THE CRASH. The real show enters a Qt event loop."""
    called = []

    original = bridge._MPL_ORIGINAL_SHOW
    bridge._MPL_ORIGINAL_SHOW = lambda *a, **k: called.append(True)
    try:
        done = threading.Event()

        def on_worker():
            bridge._matplotlib_show_router()
            done.set()

        worker = threading.Thread(target=on_worker, name="pipeline-worker")
        worker.start()
        assert done.wait(10)
        worker.join(10)

        assert called == [], (
            "a worker thread reached the real plt.show, which enters a Qt "
            "event loop off the GUI thread and takes the process with it")
    finally:
        bridge._MPL_ORIGINAL_SHOW = original


def test_the_main_thread_fallback_still_works():
    """A notebook or a script calling plt.show() outside a run means it, and
    there is no worker to confuse."""
    called = []

    original = bridge._MPL_ORIGINAL_SHOW
    bridge._MPL_ORIGINAL_SHOW = lambda *a, **k: called.append(True)
    try:
        bridge._matplotlib_show_router()
        assert called == [True]
    finally:
        bridge._MPL_ORIGINAL_SHOW = original
