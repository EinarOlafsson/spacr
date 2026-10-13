"""Item 531: the Fractal tab's Supersampling row reaches every renderer.

Found by item 530: the row was saved but no renderer read it -- every shader
and every CPU kernel averaged a fixed 2x2 -- so the number did nothing, live
or after a restart. Now the shaders get an N x N grid compiled in, the CPU
kernels take N x N spatial samples (the orbit walks N x N phases over time),
and the value is a build setting, so saving it rebuilds the running backdrop
through 530's path.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import ambient as A
from spacr.qt.widgets import fractal_travel as F
from tests.qt.test_cov_r5_fractal_travel import (  # noqa: F401
    gpu_backdrop, stand_in_vispy)
from tests.qt.test_fractal_changes_apply_without_a_restart import (  # noqa: F401
    _save, _window_with_a_backdrop, spaceout)

pytestmark = pytest.mark.qt

SHADER_MODULES = {
    "orbit": "fractal_travel",
    "mandelbrot": "fractal_mandelbrot",
    "space": "fractal_space",
    "cascade": "fractal_cascade",
    "orbit_gpu": "fractal_orbit_gpu",
}


def _shader(pattern):
    module = importlib.import_module(
        f"spacr.qt.widgets.{SHADER_MODULES[pattern]}")
    return module.FRAGMENT_SHADER


def _needs_numba():
    pytest.importorskip("numba")


class TestTheSubPixelGrid:

    def test_one_a_side_is_the_pixel_centre(self):
        assert F._sub_pixel_offsets(1) == ((0.0, 0.0),)

    def test_two_a_side_is_the_published_grid(self):
        assert F._sub_pixel_offsets(2) == (
            (-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
        assert F._orbit_jitters(2) == F.JITTERS

    def test_three_a_side_is_nine_evenly_spaced_samples(self):
        offsets = F._sub_pixel_offsets(3)
        assert len(offsets) == 9
        xs = sorted({round(dx, 9) for dx, _dy in offsets})
        assert xs == [round(-1 / 3, 9), 0.0, round(1 / 3, 9)]

    @pytest.mark.parametrize("given, used", [
        (0, 1), (-3, 1), (1, 1), (3, 3), (7, 7), (2.6, 3), ("4", 4),
        ("nonsense", 2), (None, 2)])
    def test_the_setting_is_read_as_a_whole_number_of_at_least_one(
            self, given, used):
        assert F._samples_a_side(given) == used
        assert F.Settings(supersampling=given).validated().supersampling \
            == used

    def test_an_unfilled_settings_draws_the_published_2x2(self):
        assert F.Settings().validated().supersampling == 2


class TestTheShadersFollowTheSetting:

    @pytest.mark.parametrize("pattern", sorted(SHADER_MODULES))
    def test_two_leaves_the_published_shader_untouched(self, pattern):
        source = _shader(pattern)
        assert F._supersampled_shader(source, 2) is source

    @pytest.mark.parametrize("pattern", sorted(SHADER_MODULES))
    @pytest.mark.parametrize("samples", [1, 3, 4])
    def test_the_grid_is_n_by_n_and_nothing_else_changes(self, pattern,
                                                        samples):
        source = _shader(pattern)
        shader = F._supersampled_shader(source, samples)

        assert shader.count("void main") == 1
        head = source[:source.index("void main")]
        assert shader.startswith(head), "a function before main() changed"
        main = shader[shader.index("void main"):]
        assert f"sy < {samples};" in main and f"sx < {samples};" in main
        assert f"/ {float(samples * samples)!r}" in main
        assert "0.75" not in main and "0.25" not in main

    def test_a_shader_without_a_sample_grid_is_refused_loudly(self):
        with pytest.raises(ValueError):
            F._supersampled_shader("void main() { gl_FragColor = "
                                   "vec4(1.0); }", 3)
        with pytest.raises(ValueError):
            F._supersampled_shader("uniform float u_t;", 3)

    @pytest.mark.parametrize("pattern", ["orbit", "cascade", "space",
                                         "orbit_gpu"])
    @pytest.mark.parametrize("samples", [1, 3])
    def test_the_gpu_backdrop_compiles_the_grid_it_was_given(
            self, gpu_backdrop, pattern, samples):
        canvas = gpu_backdrop(pattern, supersampling=samples)._canvas
        assert canvas._program.fragment == F._supersampled_shader(
            _shader(pattern), samples)
        assert f"spatial {samples}x{samples}" in canvas.stats_text()

    def test_the_gpu_backdrop_keeps_the_published_shader_at_two(
            self, gpu_backdrop):
        canvas = gpu_backdrop("cascade", supersampling=2)._canvas
        assert canvas._program.fragment == _shader("cascade")
        assert "spatial 2x2" in canvas.stats_text()


class TestTheCpuKernelsFollowTheSetting:

    @pytest.mark.parametrize("samples, phases", [(1, 1), (2, 4), (3, 9)])
    def test_the_orbit_keeps_one_phase_per_sub_pixel_position(
            self, samples, phases):
        _needs_numba()
        engine = F.OrbitEngine(1)
        engine.samples = samples
        first = engine.render(8, 6, 0.0, 1.0, 0.5, 4)

        assert first.shape == (6, 8, 3)
        assert engine.ring.shape[0] == phases
        for _ in range(phases):
            engine.render(8, 6, 0.1, 1.0, 0.5, 4)
        assert engine.slot == (phases + 1) % phases

    def test_the_orbit_resizes_its_ring_when_the_setting_changes(self):
        _needs_numba()
        engine = F.OrbitEngine(1)
        engine.render(8, 6, 0.0, 1.0, 0.5, 4)
        assert engine.ring.shape[0] == 4

        engine.samples = 3
        engine.render(8, 6, 0.0, 1.0, 0.5, 4)
        assert engine.ring.shape[0] == 9
        assert engine.frames == 1, "a new grid must not blend the old ring"

    def test_the_orbit_blend_weights_favour_the_newest_frame(self):
        weights = F._orbit_blend_weights(9)
        assert weights.sum() == pytest.approx(1.0)
        assert list(weights) == sorted(weights, reverse=True)
        assert F._orbit_blend_weights(1).tolist() == [1.0]

    def test_the_cascade_takes_n_by_n_spatial_samples(self):
        _needs_numba()
        from spacr.qt.widgets.fractal_cascade import CascadeEngine

        frames = {}
        for samples in (None, 1, 2, 3):
            engine = CascadeEngine(1)
            if samples is not None:
                engine.samples = samples
            frames[samples] = engine.render(24, 18, 3.0, 4.0, 1.5, 4)

        assert np.array_equal(frames[None], frames[2]), \
            "an engine nobody configured must still draw the published 2x2"
        assert not np.array_equal(frames[1], frames[2])
        assert not np.array_equal(frames[3], frames[2])

    def test_the_space_flight_uses_the_setting_when_it_has_one(self):
        _needs_numba()
        from spacr.qt.widgets.fractal_space import SpaceEngine

        engine = SpaceEngine(1)
        assert engine._samples_for(100, 100) == 2
        assert engine._samples_for(2000, 1000) == 1
        engine.samples = 3
        assert engine._samples_for(2000, 1000) == 3
        engine.samples = 1
        assert engine._samples_for(100, 100) == 1

        frames = {}
        for samples in (1, 3):
            engine.samples = samples
            frames[samples] = engine.render(24, 18, 5.0, 4.0)
        assert frames[1].shape == frames[3].shape == (18, 24, 3)
        assert not np.array_equal(frames[1], frames[3])

    @pytest.mark.parametrize("pattern", ["orbit", "cascade", "space"])
    def test_the_cpu_backdrop_hands_its_engine_the_setting(self, qapp,
                                                          pattern):
        _needs_numba()
        widget = F._make_cpu_widget(
            F.Settings(pattern=pattern, backend="cpu", supersampling=3),
            F.RuntimeControls(), F.HardwareProfile(logical_cpus=2))
        try:
            assert widget._worker.engine.samples == 3
            assert "3x3" in widget.stats_text()
        finally:
            widget.shutdown()
            widget.deleteLater()


class TestSavingItAppliesWithoutARestart:

    def test_supersampling_is_a_build_setting(self):
        assert "supersampling" in A._BUILT_FROM_SETTINGS

    def test_a_saved_grid_rebuilds_the_running_backdrop(self, spaceout):
        _needs_numba()
        from spacr.qt import preferences as P

        P.set_fractal_settings(supersampling=1)
        window, host = _window_with_a_backdrop(spaceout)
        old = window._dock_backdrop
        assert old._worker.engine.samples == 1
        assert "temporal 1x1" in old.stats_text()

        assert _save(supersampling=3) == 1
        new = window._dock_backdrop
        assert new is not old
        assert new.parentWidget() is host
        assert new._worker.engine.samples == 3
        assert "temporal 3x3" in new.stats_text()

        assert _save(supersampling=3) == 0, "an unchanged grid rebuilt"
