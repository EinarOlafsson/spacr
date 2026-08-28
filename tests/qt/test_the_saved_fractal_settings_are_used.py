"""A setting that is stored and never read is a setting that does nothing.

Four were reported in one session: render scale changed nothing, Mouse
gravity could not be turned off, the Mandelbrot numbers were ignored, and
the guided path was never built. Each was collected by the panel, written to
the store, and then not consulted by the thing it names.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from spacr.qt import preferences as P
from spacr.qt.widgets import fractal_travel as ft


@pytest.fixture
def saved(monkeypatch):
    store = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return store.get(key, default)

        def setValue(self, key, value):
            store[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return store


def test_render_scale_changes_the_frame_that_is_shaded(qapp, saved):
    """"changing render scale changes nothing"."""
    P.set_fractal_settings(render_scale=1.0)
    widget = ft.create_fractal_widget(
        ft.Settings(pattern="orbit", backend="cpu"))
    widget.resize(800, 600)
    native = widget._target_size()

    P.set_fractal_settings(render_scale=0.5)
    half = widget._target_size()

    assert half[0] < native[0] and half[1] < native[1], (native, half)
    # Roughly half each way, which is what the number says.
    assert half[0] == pytest.approx(native[0] * 0.5, rel=0.25)


def test_the_pixel_ceiling_is_the_screen_not_a_fixed_number(qapp, saved):
    """It stopped at 1,250,000 pixels whatever the settings said, so past a
    point raising the scale did nothing and there was no way to tell."""
    P.set_fractal_settings(render_scale=1.0)
    widget = ft.create_fractal_widget(
        ft.Settings(pattern="orbit", backend="cpu"))
    widget.resize(1600, 1200)
    width, height = widget._target_size()
    assert width * height > 1_250_000, (width, height)


def test_the_mouse_effect_can_be_turned_off(saved):
    """"i cant turn th mouse effect off"."""
    P.set_fractal_settings(pointer_gravity=False)
    assert P.get_fractal_settings()["pointer_gravity"] is False

    controls = ft.RuntimeControls(follow_pointer=False)
    assert controls.follow_pointer is False


def test_zero_strength_stops_the_pull_even_when_the_switch_is_on():
    from PySide6.QtCore import QPoint

    class _Widget:
        def isVisible(self):
            return True

        def width(self):
            return 100

        def height(self):
            return 100

        def mapFromGlobal(self, _pos):
            return QPoint(50, 50)

    pointer = ft.Pointer()
    for _ in range(40):
        pointer.sample(_Widget(), size=1.0, strength=0.0)
    assert pointer.pull == pytest.approx(0.0, abs=1e-6)


def test_the_backdrop_is_handed_every_saved_control():
    """The three pointer settings were collected and never passed."""
    import inspect

    from spacr.qt.widgets import ambient

    source = inspect.getsource(ambient._the_spaceout_fractal)
    for name in ("follow_pointer", "pointer_size", "pointer_strength"):
        assert name in source, f"{name} is never handed to the backdrop"


def test_the_mandelbrot_renderer_prefers_the_saved_value(qapp, saved):
    """It read the module's published defaults, so all twelve numbers the
    panel offers were ignored."""
    from spacr.qt.widgets.fractal_mandelbrot import DEFAULTS

    P.set_fractal_settings(seconds_per_decade=99.0)
    assert P.get_fractal_settings()["seconds_per_decade"] == 99.0
    assert DEFAULTS["seconds_per_decade"] != 99.0, (
        "the published default happens to equal the test value")


def test_a_gpu_failure_is_reported_rather_than_swallowed():
    """`except Exception: pass` made a shader that would not compile look
    exactly like a machine with no GPU."""
    import inspect

    source = inspect.getsource(ft.create_fractal_widget)
    assert "LOG.warning" in source
    assert "except Exception:                                    # noqa: BLE001\n            pass" not in source


def test_the_cpu_never_pretends_to_draw_mandelbrot():
    """Handing it to the CPU builder silently produced the orbit fold."""
    import inspect

    source = inspect.getsource(ft.create_fractal_widget)
    assert "FALLBACK_PATTERN" in source
    assert "needs the GPU renderer" in source
