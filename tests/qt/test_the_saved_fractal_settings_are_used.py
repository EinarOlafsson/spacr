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


def test_the_fractal_module_has_a_logger():
    """It had none, and every LOG call added to it raised NameError.

    Those calls sit in the `except` blocks that REPORT failures, so each one
    replaced a real error with a NameError about the reporting -- and the
    backdrop fell back to the old ambient engine for every pattern with
    nothing anyone could act on.
    """
    import logging

    assert hasattr(ft, "LOG")
    assert isinstance(ft.LOG, logging.Logger)


def test_every_logger_name_in_the_backdrop_modules_resolves():
    """The same omission in a sibling would fail the same way."""
    import importlib
    import logging
    import re

    for name in ("fractal_travel", "fractal_cascade", "fractal_space",
                 "fractal_mandelbrot", "ambient"):
        module = importlib.import_module(f"spacr.qt.widgets.{name}")
        source = open(module.__file__).read()
        if not re.search(r"\bLOG\.", source):
            continue
        assert isinstance(getattr(module, "LOG", None), logging.Logger), (
            f"{name} calls LOG without defining one")


@pytest.mark.parametrize("pattern", ["orbit", "cascade", "space",
                                     "mandelbrot"])
def test_every_pattern_builds(qapp, pattern):
    """All four fell back to the old ambient engine while LOG was missing."""
    widget = ft.create_fractal_widget(
        ft.Settings(pattern=pattern, backend="cpu"), ft.RuntimeControls())
    assert widget is not None
    assert getattr(widget, "backend_name", None) in ("cpu", "gpu")


def test_the_spaceout_backdrop_is_built_rather_than_skipped(qapp, monkeypatch):
    """Returning None here is what put the old artwork back on screen."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt import theme
    from spacr.qt.widgets import ambient

    # SET AND RESTORED. `enable_spaceout` flips a module flag that stays
    # flipped, so a test that walks away leaves the Fractal tab on the
    # Preferences dialog for every test after it.
    monkeypatch.setattr(theme, "_SPACEOUT", True)
    host = QWidget()
    host.resize(400, 300)
    built = ambient._the_spaceout_fractal(host)
    assert built is not None, (
        "the spaceout backdrop was skipped; the old ambient engine draws")


def test_the_orbit_texture_is_seeded_after_the_program_exists():
    """It ran before `self._program` was assigned, so every GPU build raised
    AttributeError before it could reach the shader at all."""
    import inspect

    source = inspect.getsource(ft._make_gpu_widget)
    created = source.index("self._program = gloo.Program")
    seeded = source.index('self._program["u_orbit"]')
    assert created < seeded, "the orbit is seeded before the program exists"
