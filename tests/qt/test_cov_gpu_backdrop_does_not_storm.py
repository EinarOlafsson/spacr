"""The GPU backdrop must fail during construction, not on every frame.

vispy links a shader program lazily, at the first draw, and that draw happens
inside a DrawEvent. vispy's dispatcher catches whatever a handler raises,
logs it, and RETRIES -- doubling a repeat counter each time -- so a shader
that cannot compile produced an endless

    ERROR: Invoking ... _Canvas.on_draw ... for DrawEvent
    ERROR: Invoking ... repeat 2, 4, 8, 16, 32 ...

over a window that never drew a frame, while `create_fractal_widget`'s
fallback to the CPU renderer sat unused -- it guards construction only, and
construction had already succeeded.

Wayland is the context that exposed it: Qt hands vispy a GLES context there,
vispy prepends `#version 120` (desktop GLSL), and the vertex shader dies with
"unsupported version 120". The same code under QT_QPA_PLATFORM=xcb gets a GLX
context and runs on the GPU, so this is a property of the platform plugin,
not of the machine -- and not something a test can rely on reproducing. Both
failure paths are therefore driven directly rather than by hoping the local
GL context refuses.
"""

import pytest

from spacr.qt.widgets import fractal_travel


class _RefusingProgram:
    """Stands in for a `gloo.Program` whose shaders will not link."""

    def __init__(self, *_args, **_kwargs):
        self.draws = 0

    def __setitem__(self, _name, _value):
        pass

    def draw(self, *_args, **_kwargs):
        self.draws += 1
        raise RuntimeError(
            "Shader compilation error in GL_VERTEX_SHADER:\n"
            "    on line 2: error C0201: unsupported version 120")


def _dispose(widget):
    """Shut the backdrop down properly: it owns a thread and a vispy timer."""
    shutdown = getattr(widget, "shutdown", None)
    if shutdown is not None:
        shutdown()
    widget.deleteLater()


def _gpu_or_skip():
    """Build the real GPU canvas, or skip where there is no usable context."""
    try:
        return fractal_travel._make_gpu_widget(
            fractal_travel.Settings(),
            fractal_travel.RuntimeControls(),
            fractal_travel.HardwareProfile(logical_cpus=4))
    except fractal_travel.GpuBackendError as error:
        # ONLY this one. `GpuBackendError` is precisely "this environment
        # cannot give us a GL context", which is the one condition worth
        # skipping for. Catching Exception here would turn a genuine break
        # in the factory into a skip -- which is the whole reason the
        # backdrop bug this file guards went unnoticed for so long.
        pytest.skip(f"no usable GL context here: {error}")


def test_a_program_that_will_not_link_is_refused_at_construction(monkeypatch):
    """The eager link must turn a bad shader into `GpuBackendError`.

    Without it the program links at the first DrawEvent instead, where the
    caller's fallback cannot see it.
    """
    # `gloo` is imported inside the factory, so the patch has to land on
    # vispy itself rather than on this module.
    import vispy.gloo
    monkeypatch.setattr(vispy.gloo, "Program", _RefusingProgram)
    with pytest.raises(fractal_travel.GpuBackendError) as caught:
        fractal_travel._make_gpu_widget(
            fractal_travel.Settings(),
            fractal_travel.RuntimeControls(),
            fractal_travel.HardwareProfile(logical_cpus=4))
    # The refusal has to carry the driver's words: the storm's real cost was
    # that "unsupported version 120" scrolled away under the repeats.
    assert "unsupported version 120" in str(caught.value)


def test_a_draw_that_fails_mid_run_stops_instead_of_storming():
    """`on_draw` must swallow the failure, mark itself dead and stop the timer.

    A context can also be lost after a successful link, and vispy's retry
    turns one such failure into thousands of log lines.
    """
    widget = _gpu_or_skip()
    canvas = widget._canvas
    try:
        canvas._program = _RefusingProgram()
        canvas.on_draw(None)                    # must not raise
        assert canvas._dead is True
        # Dead means dead: a second event must not even reach the program.
        before = canvas._program.draws
        canvas.on_draw(None)
        assert canvas._program.draws == before
    finally:
        _dispose(widget)


def test_the_public_factory_falls_back_instead_of_raising(monkeypatch):
    """A GPU that will not build must yield the CPU renderer, not an error."""
    def refuse(*_args, **_kwargs):
        raise fractal_travel.GpuBackendError("shaders will not compile")

    monkeypatch.setattr(fractal_travel, "_make_gpu_widget", refuse)
    monkeypatch.setattr(fractal_travel, "gpu_is_available", lambda: True)
    widget = fractal_travel.create_fractal_widget(
        fractal_travel.Settings(backend="gpu"))
    try:
        assert widget.backend_name == "cpu"
    finally:
        _dispose(widget)
