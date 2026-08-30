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


def test_an_es_context_is_refused_before_the_program_is_built(monkeypatch):
    """The refusal is a question asked of the context, not a trial draw.

    An earlier version forced the shaders to link by drawing once in
    `_Canvas.__init__`, so that a context which cannot run them failed
    where the caller's CPU fallback could catch it. That draws into a
    canvas Qt has not realized, and on an NVIDIA driver it takes the
    process down -- opening Mask or Measure died with "Segmentation fault
    (core dumped)", once per screen.

    Reading GL_SHADING_LANGUAGE_VERSION establishes the same fact and is
    safe. vispy prepends `#version 120`, desktop GLSL, to shaders carrying
    no version of their own, and an ES context rejects it.
    """
    import vispy.gloo

    real = vispy.gloo.gl.glGetParameter

    def say_es(name):
        if name == vispy.gloo.gl.GL_SHADING_LANGUAGE_VERSION:
            return "OpenGL ES GLSL ES 3.20"
        return real(name)

    monkeypatch.setattr(vispy.gloo.gl, "glGetParameter", say_es)
    with pytest.raises(fractal_travel.GpuBackendError) as caught:
        fractal_travel._make_gpu_widget(
            fractal_travel.Settings(),
            fractal_travel.RuntimeControls(),
            fractal_travel.HardwareProfile(logical_cpus=4))
    said = str(caught.value)
    # The refusal must name what it saw, so the reason is in the log rather
    # than only in someone's head.
    assert "OpenGL ES GLSL ES 3.20" in said
    assert "120" in said


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
