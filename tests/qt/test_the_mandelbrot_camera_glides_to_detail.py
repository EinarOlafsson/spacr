"""Item 604: the spaceout Mandelbrot camera glides to complex regions.

"in spaceout mode the mandelbrot camera should smoothely center on regions
with high complexity, i.e. not single color. now the camera jumps which
looks wierd. please make the camera movement smooth."

Pinned here:

* the complexity score is zero on one colour and highest on fine mixtures;
* driven over a real reference orbit, frame by frame, the camera never
  moves the picture more than a sliver of the window, the zoom's velocity
  is continuous, and every target it heads for scores above the floor;
* at the end of a dive it glides back up rather than cutting to the top;
* a view with no structure in it is backed out of, not zoomed into;
* the canvas surveys on a worker thread and drops a survey made against a
  reference that has since moved;
* moving the reference moves the camera's offset the other way, so the
  picture does not jump.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from spacr.qt.widgets import fractal_mandelbrot as M

pytest.importorskip("mpmath")

FPS = 30
DT = 1.0 / FPS


def _flat(rows=27, columns=48, value=7):
    escaped = np.ones((rows, columns), dtype=bool)
    return escaped, np.full((rows, columns), value, dtype=np.int32)


def test_one_colour_scores_zero_and_fine_mixtures_score_high():
    escaped, iterations = _flat()
    flat = M._complexity_map(escaped, iterations)
    assert flat.max() == pytest.approx(0.0)

    rng = np.random.default_rng(3)
    busy = rng.integers(0, 2000, size=(27, 48)).astype(np.int32)
    tangled = M._complexity_map(escaped, busy)
    assert tangled.max() > 2.5 * M._COMPLEXITY_FLOOR / 1.5

    gradient = np.tile(np.arange(48, dtype=np.int32), (27, 1))
    smooth = M._complexity_map(escaped, gradient)
    assert smooth.max() < tangled.max()
    assert (smooth[smooth >= 0] < M._COMPLEXITY_FLOOR).all(), (
        "a smooth gradient through a few bands is not detail")

    interior = ~escaped
    assert M._complexity_map(interior, iterations).max() == pytest.approx(0.0)
    assert (flat[:4, :] == -1.0).all(), "the frame's edge is not scored"


@pytest.fixture(scope="module")
def orbit():
    return M.ReferenceOrbit(max_iter=500, digits=60)


def _fly(orbit, seconds, rate, max_depth=21.0, survey_rows=27):
    """Drive a glide camera the way the canvas does, synchronously."""
    glide = M._GlideCamera(max_depth=max_depth)
    trail = []
    targets = []
    for _ in range(int(seconds * FPS)):
        centre, depth = glide.advance(DT, rate)
        scale = M.scale_at(depth)
        budget = min(M.iteration_budget(depth), orbit.max_iter)
        if glide.wants_survey():
            glide.surveyed()
            before = glide.target
            glide.consider(M._survey(orbit, glide.centre, scale, budget,
                                     16.0 / 9.0, survey_rows))
            if glide.target is not None and glide.target != before:
                targets.append(glide.target_score)
        trail.append((centre, depth, scale, glide.zoom_velocity))
    return glide, trail, targets


def test_the_camera_never_jumps_and_only_heads_for_detail(orbit):
    glide, trail, targets = _fly(orbit, 40.0, rate=1.0 / 4.0)

    steps = [math.hypot(b[0][0] - a[0][0], b[0][1] - a[0][1]) / a[2]
             for a, b in zip(trail, trail[1:])]
    assert max(steps) <= glide.speed_limit * DT * 1.05, (
        f"a frame moved the picture {max(steps):.4f} of the half-height")

    depths = np.array([row[1] for row in trail])
    assert depths[-1] > 2.0, "the dive went nowhere"
    per_frame = np.abs(np.diff(depths))
    assert per_frame.max() <= 0.25 * DT * 1.05
    acceleration = np.abs(np.diff(np.diff(depths)))
    assert acceleration.max() < 2e-4, "the zoom's velocity jumped"

    velocities = np.array([[b[0][0] - a[0][0], b[0][1] - a[0][1]]
                           for a, b in zip(trail, trail[1:])])
    scales = np.array([row[2] for row in trail[1:]])
    change = np.hypot(*np.diff(velocities, axis=0).T) / scales[1:]
    assert change.max() < 0.25 * glide.speed_limit * DT, (
        "the centre's velocity jumped")

    assert targets, "no target was ever chosen"
    assert min(targets) >= glide.floor
    late = [M._complexity_map(*M.perturbation_escape_map(
        orbit, 48, 27, row[2], min(M.iteration_budget(row[1]), 500),
        row[0][0], row[0][1]))[13, 24] for row in trail[-90::30]]
    assert max(late) >= glide.floor, "the view is not centred on detail"


def test_the_end_of_a_dive_glides_back_up(orbit):
    glide, trail, _targets = _fly(orbit, 16.0, rate=1.0 / 2.0,
                                  max_depth=1.5)
    depths = np.array([row[1] for row in trail])
    peak = int(np.argmax(depths))
    assert depths[peak] >= 1.5 - 0.01
    assert depths[peak:].min() < 0.1, "it never came back to the surface"
    assert np.abs(np.diff(depths)).max() <= glide.ascent_rate * DT * 1.01
    assert np.abs(np.diff(np.diff(depths))).max() < 0.01, (
        "turning round was a jolt")


def test_a_view_with_no_structure_is_backed_out_of():
    glide = M._GlideCamera()
    glide.depth = 5.0
    scale = glide.scale()
    glide.consider({"scores": np.zeros((27, 48)), "centre": (0.0, 0.0),
                    "scale": scale, "aspect": 48 / 27})
    assert glide.flat and glide.target is None
    for _ in range(FPS * 4):
        glide.advance(DT, 1.0 / 24.0)
    assert glide.depth < 5.0
    assert glide.zoom_velocity < 0.0


def test_a_target_fading_toward_one_colour_is_replaced_early():
    glide = M._GlideCamera()
    scale = glide.scale()
    scores = np.zeros((27, 48))
    scores[13, 30] = 3.0
    glide.consider({"scores": scores, "centre": (0.0, 0.0), "scale": scale,
                    "aspect": 48 / 27})
    first = glide.target
    assert first is not None and glide.target_score == 3.0

    scores[13, 30] = 1.05 * glide.floor
    scores[13, 18] = 2.4
    glide.consider({"scores": scores, "centre": (0.0, 0.0), "scale": scale,
                    "aspect": 48 / 27})
    assert glide.target != first, (
        "it waited for the patch to go flat before turning")
    assert glide.target_score == pytest.approx(2.4)


def test_a_drag_takes_the_camera_and_a_restart_glides_back():
    glide = M._GlideCamera()
    glide.depth = 4.0
    glide.target = (1e-4, 0.0)
    glide.drag(0.5, -0.25, glide.scale())
    assert glide.taken and glide.target is None
    assert not glide.wants_survey()

    glide.restart()
    assert not glide.taken
    assert glide.ascending, "a restart cut to the surface"
    assert glide.depth == 4.0


# ---------------------------------------------------------------------------
# The canvas around it
# ---------------------------------------------------------------------------

from tests.qt.test_cov_r5_fractal_travel import (  # noqa: E402,F401
    _StandInOrbit,
    gpu_backdrop,
    mandel,
    stand_in_vispy,
)


def _until(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_canvas_surveys_off_the_frame_and_drops_a_stale_survey(
        mandel, monkeypatch):  # noqa: F811
    mandel.saved["path"] = "tour"
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)

    canvas._mandelbrot_uniforms(0.0)
    slot = canvas._glide_survey
    assert slot is not None
    assert _until(lambda: slot["done"])
    assert slot["survey"] is not None

    considered = []
    monkeypatch.setattr(canvas._glide, "consider",
                        lambda survey, _scale=1.25: considered.append(survey))
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    assert considered == [], "a survey of another reference was used"


def test_moving_the_reference_does_not_move_the_picture(mandel, monkeypatch):  # noqa: F811
    module = mandel.module
    mandel.saved["path"] = "tour"
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    glide = canvas._glide
    monkeypatch.setattr(glide, "consider", lambda *_a, **_k: None)
    glide.survey_every = 1e9
    glide.centre = (0.3, 0.4)
    glide.target = (0.3, 0.4)
    glide.velocity = (0.0, 0.0)

    fresh = _StandInOrbit(max_iter=8, digits=20)
    monkeypatch.setattr(module, "best_reference_in_view",
                        lambda *_a, **_k: (0.1, 0.2))
    monkeypatch.setattr(module, "rebased_orbit",
                        lambda *_a, **_k: ((0.1, 0.2), fresh))
    canvas._refine_due = 0.0
    canvas._mandelbrot_uniforms(0.0)
    assert _until(lambda: canvas._refined is not None)
    before = glide.centre
    canvas._mandelbrot_uniforms(0.0)

    assert canvas._orbit is fresh
    assert glide.centre[0] == pytest.approx(before[0] - 0.1, abs=1e-3)
    assert glide.centre[1] == pytest.approx(before[1] - 0.2, abs=1e-3)
    assert glide.target == (pytest.approx(0.2), pytest.approx(0.2))


def test_the_published_path_is_still_the_tour():
    assert M.DEFAULTS["path"] == "tour"
