"""The fractal settings are fields, and a bad number is explained."""
from __future__ import annotations

import pytest

from spacr.qt import preferences as P


@pytest.fixture
def store(monkeypatch):
    values = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return values.get(key, default)

        def setValue(self, key, value):
            values[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return values


def test_a_large_scale_is_kept(store):
    """It was capped at 2.0, which turned a typed 40 into 2 silently."""
    P.set_fractal_settings(scale=40.0)
    assert P.get_fractal_settings()["scale"] == 40.0


def test_supersampling_is_a_setting(store):
    """Called out as "a super important setting"."""
    assert "supersampling" in P.get_fractal_settings()
    P.set_fractal_settings(supersampling=4)
    assert P.get_fractal_settings()["supersampling"] == 4


def test_the_mandelbrot_numbers_are_settings(store):
    values = P.get_fractal_settings()
    for name in ("seconds_per_decade", "base_iterations",
                 "iterations_per_decade", "max_iterations",
                 "precision_digits", "initial_scale", "zoom_rate",
                 "render_scale", "steering_strength",
                 "steering_interval_decades", "steering_duration",
                 "candidate_count"):
        assert name in values, name


def test_the_published_defaults_are_what_is_offered(store):
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    values = P.get_fractal_settings()
    for name in ("seconds_per_decade", "base_iterations", "max_iterations",
                 "precision_digits", "initial_scale", "candidate_count",
                 "steering_strength", "steering_duration"):
        assert values[name] == DEFAULTS[name], name


def test_a_number_that_cannot_work_is_explained_not_clamped_silently():
    """"the code will gracefully throw an error and tell the user"."""
    said = P.explain_a_fractal_number("supersampling", 0)
    assert "too small" in said
    assert "draws nothing" in said, said
    assert "supersampling" in said

    said = P.explain_a_fractal_number("max_iterations", 9999)
    assert "4096" in said, said

    # And a merely extravagant number is somebody's business but ours.
    assert P.explain_a_fractal_number("scale", 40.0) == ""
    assert P.explain_a_fractal_number("speed", 500000.0) == ""


def test_a_non_number_says_so():
    said = P.explain_a_fractal_number("scale", "wide")
    assert "not a number" in said


def test_mandelbrot_is_the_default_pattern():
    from spacr.qt.fractal_defaults import DEFAULT_PATTERN, FALLBACK_PATTERN

    assert DEFAULT_PATTERN == "mandelbrot"
    assert FALLBACK_PATTERN == "orbit"


def test_a_machine_without_a_gpu_gets_the_orbit_fold(monkeypatch):
    """Mandelbrot is GPU-only; the fallback has a CPU renderer."""
    from spacr.qt.widgets import fractal_travel as ft

    monkeypatch.setattr(ft, "platform_can_do_opengl", lambda: False)
    assert ft.pattern_for_this_machine("mandelbrot") == "orbit"
    assert ft.pattern_for_this_machine("mandelbrot", "cpu") == "orbit"
    # Every other pattern is left alone.
    assert ft.pattern_for_this_machine("space") == "space"
    assert ft.pattern_for_this_machine("cascade", "cpu") == "cascade"

    monkeypatch.setattr(ft, "platform_can_do_opengl", lambda: True)
    monkeypatch.setattr(ft, "gpu_is_available", lambda: True)
    assert ft.pattern_for_this_machine("mandelbrot") == "mandelbrot"


def test_up_and_down_change_the_zoom_rate():
    """As in the renderer this pattern came from."""
    from spacr.qt.widgets import fractal_travel as ft

    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        start = controls.zoom_rate
        faster = ft.nudge_zoom_rate(1)
        assert faster == pytest.approx(start * ft.ZOOM_STEP)
        slower = ft.nudge_zoom_rate(-1)
        assert slower == pytest.approx(start)
        # A key held down does not reach 10^12.
        for _ in range(200):
            ft.nudge_zoom_rate(1)
        assert controls.zoom_rate <= ft.MAX_ZOOM_RATE
        for _ in range(400):
            ft.nudge_zoom_rate(-1)
        assert controls.zoom_rate >= ft.MIN_ZOOM_RATE
    finally:
        ft._LIVE_CONTROLS.clear()


def test_no_backdrop_means_the_key_is_not_taken():
    """Or Up and Down would stop working everywhere else."""
    from spacr.qt.widgets import fractal_travel as ft

    ft._LIVE_CONTROLS.clear()
    assert ft.nudge_zoom_rate(1) == 0.0
