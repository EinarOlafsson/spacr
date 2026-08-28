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


def test_the_dive_restarts_instead_of_going_black():
    """Reported 2026-08-28: the zoom reaches an end and the screen is black.

    Perturbation buys precision for the CENTRE, but the per-pixel offset is
    still a float32 in the shader, and that is what runs out.
    """
    limit = mb.MAX_USEFUL_DEPTH
    assert mb.depth_after_restart(0.0) == 0.0
    assert mb.depth_after_restart(limit * 0.5) == pytest.approx(limit * 0.5)
    # Past the end it is back near the surface, not stuck at the bottom.
    assert mb.depth_after_restart(limit) == pytest.approx(0.0)
    assert mb.depth_after_restart(limit + 3.0) == pytest.approx(3.0)
    # However far past: a frame that arrives after the machine slept lands
    # where it should rather than skipping a whole descent.
    assert 0.0 <= mb.depth_after_restart(limit * 7 + 1.5) < limit


def test_the_restart_happens_before_the_pixel_step_underflows():
    """A zero pixel step is one sample of one point, filling the screen."""
    scale = np.float32(mb.scale_at(mb.MAX_USEFUL_DEPTH,
                                   mb.DEFAULTS["initial_scale"]))
    # A pixel on a 1080-tall window.
    step = np.float32(scale / np.float32(540.0))
    assert step > 0.0, "the deepest frame already samples a single point"
    # And with room to spare: the scale is still a normal float32, not a
    # denormal, where precision degrades before it fails.
    assert scale > np.finfo(np.float32).tiny, (
        f"scale {scale:.3e} is denormal at the restart depth")


def test_the_restart_depth_is_a_setting_and_is_bounded(monkeypatch):
    """It is the one fractal number with a real ceiling: past it the
    picture cannot exist."""
    from spacr.qt import preferences as P

    assert "max_depth" in P.FRACTAL_LIMITS
    floor, ceiling, why = P.FRACTAL_LIMITS["max_depth"]
    assert ceiling is not None and ceiling <= 45
    assert "underflow" in why

    assert P.explain_a_fractal_number("max_depth", 100) != ""
    assert P.explain_a_fractal_number("max_depth", 20) == ""


def test_changing_the_settings_sends_the_dive_back_to_the_surface():
    """Reported 2026-08-28: it resumed at the depth it had reached.

    New numbers applied thirty decades down have nothing recognisable to
    act on, so the change looks as though it did nothing.
    """
    from spacr.qt.widgets import fractal_travel as ft

    controls = ft.RuntimeControls()
    ft._LIVE_CONTROLS.clear()
    ft._LIVE_CONTROLS.append(controls)
    try:
        first = controls.restart_token
        ft.restart_the_dive()
        assert controls.restart_token != first

        # A COUNTER, not a flag: two changes in quick succession are two
        # restarts, and nobody has to remember to clear it.
        second = controls.restart_token
        ft.restart_the_dive()
        assert controls.restart_token != second
    finally:
        ft._LIVE_CONTROLS.clear()


def test_restarting_with_no_backdrop_is_harmless():
    from spacr.qt.widgets import fractal_travel as ft

    ft._LIVE_CONTROLS.clear()
    assert ft.restart_the_dive() is None


def test_the_guided_path_actually_goes_somewhere_different():
    """Reported 2026-08-28: "it allways heads right to the center of 3
    lines... how can it allways end up in the same place if steering is
    implementad." It could not: only the fixed path was built.
    """
    orbit = mb.ReferenceOrbit(max_iter=900, digits=120)
    chosen = []
    for step in range(6):
        plan = mb.plan_guided_step(orbit, 1.25, 400, strength=0.09,
                                   candidates=24, step_index=step)
        assert plan is not None, f"step {step} found no boundary at all"
        chosen.append((round(plan[0], 4), round(plan[1], 4)))

    assert len(set(chosen)) >= 3, (
        f"six steps chose {len(set(chosen))} targets: {chosen}")


def test_a_step_heads_the_way_it_was_pointed():
    """The heading is a constraint, not a preference: scoring every point
    and penalising the distant ones let the best-scoring point in the frame
    win whatever direction the step was exploring."""
    orbit = mb.ReferenceOrbit(max_iter=900, digits=120)
    angles = []
    for step in range(8):
        plan = mb.plan_guided_step(orbit, 1.25, 400, strength=0.09,
                                   candidates=24, step_index=step)
        if plan is None:
            continue
        angles.append(np.arctan2(plan[1], plan[0]))
    assert len(angles) >= 4
    # They must not all point the same way.
    spread = np.ptp(np.unwrap(np.sort(np.asarray(angles))))
    assert spread > 0.4, f"every step headed the same way (spread {spread})"


def test_the_map_only_targets_the_boundary():
    """An interior point fades to flat colour and an exterior one escapes;
    only the edge keeps producing detail at every magnification."""
    orbit = mb.ReferenceOrbit(max_iter=900, digits=120)
    escaped, _iterations = mb.perturbation_escape_map(
        orbit, 96, 54, 1.25, 400)
    edge = mb.boundary_mask(escaped)
    assert edge.any()
    # Every boundary point is bounded and touches something that escapes.
    rows, cols = np.nonzero(edge)
    for row, col in list(zip(rows, cols))[:20]:
        assert not escaped[row, col]
        neighbours = escaped[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        assert neighbours.any()
    # And the frame's own edge is excluded: a point there may look like a
    # boundary only because the map stopped.
    assert not edge[0, :].any() and not edge[-1, :].any()
    assert not edge[:, 0].any() and not edge[:, -1].any()


def test_the_move_is_eased_not_a_jump():
    assert mb.eased(0.0) == 0.0
    assert mb.eased(1.0) == 1.0
    assert mb.eased(0.5) == pytest.approx(0.5)
    # Gentle at both ends, which is what makes it a steer.
    assert mb.eased(0.1) < 0.1
    assert mb.eased(0.9) > 0.9
    assert mb.eased(-5.0) == 0.0 and mb.eased(5.0) == 1.0
