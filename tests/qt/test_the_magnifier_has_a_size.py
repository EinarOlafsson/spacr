"""Item 605: a slider sets the size of the spaceout magnifying glass.

"for the other spaceout modes theis a giant magnigying glass where the
mouse is. there should be a slider deffining its size."

Pinned here: the setting is saved, read back and clamped; the Preferences
slider shows it, explains it and saves it; a running backdrop takes it
without a rebuild; every shader with the lens reads it, the CPU orbit fold
too; and the size scales the whole lens without changing its shape, with
the usual size exactly the lens it always was.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from spacr.qt import preferences as P
from spacr.qt.fractal_defaults import (DEFAULT_MAGNIFIER_SIZE,
                                       MAGNIFIER_SIZE_RANGE)
from spacr.qt.widgets import fractal_travel as F

ROOT = Path(__file__).resolve().parents[2]
WIDGETS = ROOT / "spacr" / "qt" / "widgets"
LENS_SHADERS = ("fractal_travel.py", "fractal_orbit_gpu.py",
                "fractal_cascade.py", "fractal_space.py")


@pytest.fixture
def store(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(P, "_SAFE_MODE", False, raising=False)
    monkeypatch.setattr(F, "_LIVE_CONTROLS", [])
    from spacr.qt import theme

    monkeypatch.setattr(theme, "_SPACEOUT", True)
    yield


def test_the_default_is_the_lens_it_always_was(store):
    assert DEFAULT_MAGNIFIER_SIZE == 1.0
    assert P.get_fractal_settings()["magnifier_size"] == 1.0
    assert F.RuntimeControls().magnifier_size == 1.0


def test_it_is_saved_read_back_and_held_in_range(store):
    P.set_fractal_settings(magnifier_size=2.25)
    assert P.get_fractal_settings()["magnifier_size"] == pytest.approx(2.25)
    low, high = MAGNIFIER_SIZE_RANGE
    P.set_fractal_settings(magnifier_size=40.0)
    assert P.get_fractal_settings()["magnifier_size"] == high
    P.set_fractal_settings(magnifier_size=0.0)
    assert P.get_fractal_settings()["magnifier_size"] == low


def test_a_running_backdrop_takes_it_without_a_rebuild(store):
    controls = F.RuntimeControls()
    F._LIVE_CONTROLS.append(controls)
    P.set_fractal_settings(magnifier_size=1.75)
    assert F.apply_saved_controls() == 1
    assert controls.magnifier_size == pytest.approx(1.75)


def test_the_preferences_slider_shows_explains_and_saves_it(store, qtbot):
    from PySide6.QtWidgets import QDialogButtonBox, QSlider

    P.set_fractal_settings(magnifier_size=1.5)
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    slider = dlg.findChild(QSlider, "FractalMagnifierSize")
    assert slider is not None, "no Magnifier size slider on the Fractal page"
    assert (slider.minimum(), slider.maximum()) == (25, 300)
    assert slider.value() == 150
    tip = slider.toolTip() or slider.accessibleDescription()
    assert tip.endswith("Default 100%.") and len(tip) <= 600
    assert "magnifying glass" in tip

    slider.setValue(220)
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert P.get_fractal_settings()["magnifier_size"] == pytest.approx(2.2)


@pytest.mark.parametrize("name", LENS_SHADERS)
def test_every_shader_with_the_lens_reads_its_size(name):
    source = (WIDGETS / name).read_text(encoding="utf-8")
    assert "uniform float u_lens;" in source
    body = source[source.index("vec2 toward_pointer"):]
    body = body[:body.index("}")]
    assert "u_lens" in body
    assert "(target - uv) / lens" in body
    assert "to_pointer * lens" in body


def test_the_gpu_canvas_hands_the_shader_the_live_size(gpu_backdrop):  # noqa: F811
    controls = F.RuntimeControls(magnifier_size=2.5)
    canvas = gpu_backdrop("orbit_gpu", controls=controls)._canvas
    canvas._update_uniforms(1.0)
    assert float(canvas._program["u_lens"]) == pytest.approx(2.5)
    controls.magnifier_size = 0.5
    canvas._update_uniforms(1.1)
    assert float(canvas._program["u_lens"]) == pytest.approx(0.5)


def _warp(x, y, pointer, pull, lens):
    """The shader's toward_pointer, in Python."""
    to_x = (pointer[0] - x) / lens
    to_y = (pointer[1] - y) / lens
    strength = 0.55 * pull / (to_x * to_x + to_y * to_y + 0.05)
    strength = min(0.9, max(-1.4, strength))
    return x + strength * to_x * lens, y + strength * to_y * lens


def test_the_size_scales_the_whole_lens_and_keeps_its_shape():
    pointer = (0.2, -0.1)
    for distance in (0.05, 0.3, 0.9):
        for lens in (0.5, 2.0, 3.0):
            x = pointer[0] + distance * lens
            moved = _warp(x, pointer[1], pointer, 1.0, lens)[0] - x
            usual = _warp(pointer[0] + distance, pointer[1], pointer, 1.0,
                          1.0)[0] - (pointer[0] + distance)
            assert moved == pytest.approx(lens * usual)


def test_the_cpu_orbit_fold_honours_it(monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "_fast_sin", F._fast_sin.py_func)
    monkeypatch.setattr(F, "_fast_cos", F._fast_cos.py_func)
    sample = F._orbit_sample.py_func
    monkeypatch.setattr(F, "_orbit_sample", sample)
    args = (60, 5, 64, 48, 2.0, 4.0, 1.5, 5, 0.1, 0.1, 1.0, 0.0)
    assert sample(*args) == sample(*args, 1.0)
    assert sample(*args, 2.5) != sample(*args, 1.0)

    render = F._render_into.py_func
    small = np.zeros((6, 8, 3), dtype=np.uint8)
    large = np.zeros((6, 8, 3), dtype=np.uint8)
    render(small, 1.5, 4.0, 1.5, 5, 0.25, 0.25, 0.2, 0.1, 1.0, 0.0, 0.5)
    render(large, 1.5, 4.0, 1.5, 5, 0.25, 0.25, 0.2, 0.1, 1.0, 0.0, 3.0)
    assert not np.array_equal(small, large)


def test_the_cpu_widget_sends_the_size_with_every_frame():
    source = (WIDGETS / "fractal_travel.py").read_text(encoding="utf-8")
    assert re.search(r'"lens": controls\.magnifier_size', source)
    assert "self.engine.lens = float(request.get(" in source


from tests.qt.test_cov_r5_fractal_travel import (  # noqa: E402,F401
    gpu_backdrop,
    stand_in_vispy,
)
