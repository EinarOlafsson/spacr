"""A continuous deep zoom into one point on the Mandelbrot boundary."""
from __future__ import annotations

import re

import numpy as np
import pytest

mb = pytest.importorskip("spacr.qt.widgets.fractal_mandelbrot")
mp = pytest.importorskip("mpmath")


def test_it_is_offered_as_a_pattern():
    from spacr.qt.preferences import FRACTAL_PATTERNS
    from spacr.qt.widgets.fractal_travel import PATTERN_LABELS, PATTERNS

    assert "mandelbrot" in PATTERNS
    assert "mandelbrot" in FRACTAL_PATTERNS
    assert PATTERN_LABELS["mandelbrot"]


def test_the_centre_is_actually_on_the_boundary():
    """An interior point fades to flat colour as you descend into it and an
    exterior one escapes; either way the zoom stops finding anything."""
    centre = mb.exact_misiurewicz_center(120)
    mp.mp.dps = 120

    def iterate(c, n):
        z = mp.mpc(0)
        for _ in range(n):
            z = z * z + c
        return z

    # Misiurewicz, preperiod 4 period 1: f^5(0) == f^4(0).
    residual = abs(iterate(centre, 5) - iterate(centre, 4))
    assert residual < mp.mpf("1e-60"), mp.nstr(residual, 5)


def test_the_reference_orbit_stays_bounded():
    """A reference that escapes is not a reference: every pixel perturbs
    around it."""
    orbit = mb.ReferenceOrbit(max_iter=400, digits=120)
    assert orbit.is_bounded
    assert orbit.packed.shape == (1, 401, 4)
    assert orbit.packed.dtype == np.float32


def test_the_low_words_carry_the_precision_a_float_would_lose():
    """One float32 cannot hold Z at this depth; the shader adds the pair
    back, and a zero low word would mean nothing was kept."""
    orbit = mb.ReferenceOrbit(max_iter=200, digits=120)
    assert np.abs(orbit.packed[0, :, 1]).max() > 0.0
    assert np.abs(orbit.packed[0, :, 3]).max() > 0.0


def test_deeper_gets_more_iterations():
    """Near the boundary the escape time grows with magnification; a fixed
    budget draws the deep frames as solid interior."""
    shallow = mb.iteration_budget(0.0)
    deep = mb.iteration_budget(20.0)
    assert deep > shallow
    assert mb.iteration_budget(10_000.0) <= mb.DEFAULTS["max_iterations"]
    assert mb.iteration_budget(0.0) == mb.DEFAULTS["base_iterations"]


def test_the_scale_never_underflows_to_zero():
    """Past a float64's range every pixel would sample the same point."""
    assert mb.scale_at(0.0) == pytest.approx(mb.DEFAULTS["initial_scale"])
    assert mb.scale_at(10.0) < mb.scale_at(5.0)
    assert mb.scale_at(100_000.0) > 0.0


def test_the_zoom_is_continuous():
    depths = [mb.depth_decades(t) for t in (0, 12, 24, 48)]
    assert depths == sorted(depths)
    assert depths[0] == 0.0
    # One decade per `seconds_per_decade`, as the defaults name it.
    assert mb.depth_decades(mb.DEFAULTS["seconds_per_decade"]) == \
        pytest.approx(1.0)


def test_the_defaults_are_the_ones_asked_for():
    """Given on 2026-08-28, guided path."""
    d = mb.DEFAULTS
    assert d["supersampling"] == 2
    assert d["fps"] == 30
    assert d["seconds_per_decade"] == 24.0
    assert d["base_iterations"] == 300
    assert d["iterations_per_decade"] == 55.0
    assert d["max_iterations"] == 2200
    assert d["precision_digits"] == 320
    assert d["initial_scale"] == 1.25
    assert d["tile_rows"] == 32
    assert d["gpu_fp64"] is False
    assert d["path"] == "guided"
    assert d["steering_strength"] == 0.09
    assert d["steering_interval_decades"] == 0.40
    assert d["steering_duration"] == 3.8
    assert d["candidate_count"] == 24


def test_the_shader_perturbs_rather_than_iterating_c_directly():
    """Past ~15 decades a float has no bits left to tell pixels apart."""
    source = mb.FRAGMENT_SHADER
    assert "refz" in source, "no reference orbit is read"
    # dz[n+1] = 2*Z*dz + dz^2 + dc
    assert "2.0 * cmul(Z, dz) + cmul(dz, dz) + dc" in source


def test_the_pointer_steers_the_dive():
    source = mb.FRAGMENT_SHADER
    declared = set(re.findall(r"uniform\s+\w+\s+(u_\w+)\s*;", source))
    for name in ("u_pointer_x", "u_pointer_y", "u_pull", "u_push"):
        assert name in declared
    assert source.count("toward_pointer") >= 2
