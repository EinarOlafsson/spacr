"""A backdrop nobody asked for must not cost more than the one they did.

`resolved_cpu_threads` deliberately takes about 78% of the machine -- 24 of
32 cores here -- and that is a reasonable bargain for someone who CHOSE the
CPU renderer. It is an unreasonable one for someone who asked for the GPU
and got this instead.

Measured at 800x600 before this guard: 4.0 s of wall clock burned 82 s of
CPU, about twenty cores, and the application creates one backdrop per
screen. That starves the GUI thread -- the window will not drag, modules
take minutes to open or never open, and the process has to be force-quit.

So an unasked-for CPU backdrop is drawn once and then stopped. A still
fractal is a fine backdrop; an unusable application is not.
"""

import pytest

from spacr.qt.widgets import fractal_travel


def _dispose(widget):
    """Stop the backdrop before dropping it.

    The CPU renderer owns a worker thread, and `deleteLater()` alone leaves
    it running -- Qt then aborts the interpreter with "QThread: Destroyed
    while thread is still running" once the test process exits.
    """
    shutdown = getattr(widget, "shutdown", None)
    if shutdown is not None:
        shutdown()
    widget.deleteLater()


def _refuse_the_gpu(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise fractal_travel.GpuBackendError("no usable GL context here")
    monkeypatch.setattr(fractal_travel, "_make_gpu_widget", refuse)
    monkeypatch.setattr(fractal_travel, "gpu_is_available", lambda: True)


@pytest.mark.parametrize("backend", ["auto", "gpu"])
def test_a_backdrop_the_user_did_not_ask_for_is_drawn_once_and_stopped(
        backend, monkeypatch):
    _refuse_the_gpu(monkeypatch)
    widget = fractal_travel.create_fractal_widget(
        fractal_travel.Settings(backend=backend))
    try:
        assert widget.backend_name == "cpu"
        assert widget.is_paused() is True, (
            "an unasked-for CPU backdrop must not animate")
    finally:
        _dispose(widget)


def test_a_cpu_backdrop_the_user_chose_still_animates():
    """The bargain is only unreasonable when it was not the user's bargain."""
    widget = fractal_travel.create_fractal_widget(
        fractal_travel.Settings(backend="cpu"))
    try:
        assert widget.backend_name == "cpu"
        assert widget.is_paused() is False
    finally:
        _dispose(widget)


def test_the_still_backdrop_can_still_be_resumed(monkeypatch):
    """Stopped is not broken: the usual controls still work on it."""
    _refuse_the_gpu(monkeypatch)
    widget = fractal_travel.create_fractal_widget(
        fractal_travel.Settings(backend="auto"))
    try:
        assert widget.is_paused() is True
        assert widget.resume() is True
        assert widget.is_paused() is False
    finally:
        _dispose(widget)
