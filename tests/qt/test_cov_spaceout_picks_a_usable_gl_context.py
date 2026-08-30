"""spaceout must not land on a GL context whose shaders will not compile.

On a Wayland session Qt gives vispy a GLES context. vispy prepends
``#version 120`` -- desktop GLSL -- so the fractal vertex shader fails to
compile, the GPU backdrop is refused, and the CPU renderer takes over.

That is not a graceful degradation. Measured at 800x600, the CPU backdrop
burned 82 s of CPU in 4.0 s of wall clock -- about twenty cores -- which
starves the GUI thread: the window cannot be dragged, the application and
the preferences dialog are slow to open, and the Mandelbrot pattern dumps
core. The same session under ``xcb`` runs the same backdrop on the GPU for
1% of one core.

So the launcher asks for XWayland before Qt starts, and only when the
caller has expressed no preference of their own.
"""

import os

import pytest

from spacr.qt.spaceout import _prefer_a_context_the_shaders_compile_on as prefer


@pytest.fixture
def env(monkeypatch):
    """A clean slate: no platform, no session, no display."""
    for name in ("QT_QPA_PLATFORM", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE",
                 "DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    return os.environ


def test_a_wayland_session_with_xwayland_is_moved_to_xcb(env, monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    prefer()
    assert env["QT_QPA_PLATFORM"] == "xcb"


def test_the_session_type_alone_is_enough_to_recognise_wayland(env,
                                                               monkeypatch):
    """A session can be Wayland without WAYLAND_DISPLAY being exported."""
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":0")
    prefer()
    assert env["QT_QPA_PLATFORM"] == "xcb"


def test_a_choice_the_caller_made_is_never_overruled(env, monkeypatch):
    """Including a deliberate `wayland`, which is the whole point of asking."""
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    prefer()
    assert env["QT_QPA_PLATFORM"] == "wayland"


def test_wayland_with_no_xwayland_is_left_alone(env, monkeypatch):
    """Without DISPLAY there is nothing to fall back to; xcb would fail."""
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    prefer()
    assert "QT_QPA_PLATFORM" not in env


def test_an_x11_session_is_left_alone(env, monkeypatch):
    """It already gets a GLX context; there is nothing to avoid."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    prefer()
    assert "QT_QPA_PLATFORM" not in env
