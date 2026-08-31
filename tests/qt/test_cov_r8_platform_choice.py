"""Which Qt platform plugin spaCR asks for, and when it declines to.

WHY THIS IS LOAD-BEARING. On a native Wayland session Qt hands vispy an
OpenGL ES context; vispy compiles the fractal shaders as desktop GLSL 120
and ES answers "unsupported version 120", so the backdrop never draws a
frame. The same compositor rounds the opaque surface, which is where the
black corners outside the rounded edges came from. Under `xcb` both are
gone, confirmed by running all three entry points.

The failure mode these tests exist to catch is not "it forgets to set
xcb" -- that is loud. It is the opposite: OVERRULING a choice somebody
made deliberately. The test suite sets QT_QPA_PLATFORM=offscreen, and a
helper that stamped xcb over it would take the whole suite off a
headless machine.
"""
from __future__ import annotations

import pytest

from spacr.qt import _prefer_a_context_the_shaders_can_run_on as prefer


def _run_with(monkeypatch, env):
    """Run the helper against exactly ``env`` and report the platform."""
    import os

    for name in ("QT_QPA_PLATFORM", "WAYLAND_DISPLAY", "DISPLAY",
                 "XDG_SESSION_TYPE"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    prefer()
    return os.environ.get("QT_QPA_PLATFORM")


def test_a_wayland_session_with_an_x_server_asks_for_xcb(monkeypatch):
    """The case the whole thing exists for."""
    assert _run_with(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0",
                                   "DISPLAY": ":0"}) == "xcb"


def test_the_session_type_is_believed_even_without_wayland_display(
        monkeypatch):
    """XDG_SESSION_TYPE is the other way a session says it is Wayland."""
    assert _run_with(monkeypatch, {"XDG_SESSION_TYPE": "wayland",
                                   "DISPLAY": ":0"}) == "xcb"


def test_the_session_type_is_matched_without_regard_to_case(monkeypatch):
    """`Wayland` and `wayland` are the same session."""
    assert _run_with(monkeypatch, {"XDG_SESSION_TYPE": "Wayland",
                                   "DISPLAY": ":0"}) == "xcb"


def test_an_explicit_choice_is_never_overruled(monkeypatch):
    """THE ONE THAT PROTECTS THE TEST SUITE.

    `offscreen` is what conftest sets, and a helper that replaced it
    would take every Qt test off a headless machine. Anything the caller
    named is left exactly as they named it.
    """
    assert _run_with(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0",
                                   "DISPLAY": ":0",
                                   "QT_QPA_PLATFORM": "offscreen"}) == "offscreen"


def test_someone_who_chose_wayland_deliberately_keeps_it(monkeypatch):
    """Preferring no backdrop to XWayland is a legitimate choice."""
    assert _run_with(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0",
                                   "DISPLAY": ":0",
                                   "QT_QPA_PLATFORM": "wayland"}) == "wayland"


def test_wayland_with_no_x_server_is_left_alone(monkeypatch):
    """Asking for xcb with nothing to fall back to would not start at all.

    A session with no DISPLAY has no XWayland to ask for, so naming xcb
    would trade a missing backdrop for a missing application.
    """
    assert _run_with(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0"}) is None


def test_a_plain_x11_session_is_left_alone(monkeypatch):
    """Already a GLX context; there is nothing to prefer."""
    assert _run_with(monkeypatch, {"DISPLAY": ":0"}) is None


def test_a_session_that_is_neither_is_left_alone(monkeypatch):
    """No DISPLAY and no Wayland: a container, or a cron job."""
    assert _run_with(monkeypatch, {}) is None


def test_an_empty_platform_string_does_not_count_as_a_choice(monkeypatch):
    """`QT_QPA_PLATFORM=` is not somebody naming a plugin."""
    assert _run_with(monkeypatch, {"WAYLAND_DISPLAY": "wayland-0",
                                   "DISPLAY": ":0",
                                   "QT_QPA_PLATFORM": ""}) == "xcb"


class TestEveryEntryPointAsks:
    """A fix that reaches one command and not the others is not a fix.

    This machine has three separate spaCR installs, and an evening was
    lost to a platform preference that lived in `spaceout` while the user
    was running `spacr`. The helper is in `spacr.qt` for that reason, and
    these assert that each way in actually calls it.
    """

    def test_run_asks(self):
        import inspect

        from spacr.qt import run

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(run)

    def test_run_without_setup_asks(self):
        import inspect

        from spacr.qt import run_without_setup

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(run_without_setup)

    def test_spaceout_asks(self):
        import inspect

        from spacr.qt.spaceout import main

        assert "_prefer_a_context_the_shaders_can_run_on" in \
            inspect.getsource(main)

    def test_safespacr_inherits_it_by_going_through_run(self):
        """Safe mode has no platform logic of its own, and needs none."""
        import inspect

        from spacr.qt.safespacr import main

        assert "run(" in inspect.getsource(main)
