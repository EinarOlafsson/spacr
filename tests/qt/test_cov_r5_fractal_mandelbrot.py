"""The corners of the Mandelbrot dive a real orbit cannot easily be driven to.

Everything here pins behaviour of :mod:`spacr.qt.widgets.fractal_mandelbrot`
that the pattern reaches only in particular views: a survey with nothing to
steer toward, structure that sits outside the ring the search looks in, an
anchor whose surface promise does not survive the descent, and a reference
picked from a view that holds no boundary at all.

WHY THE ORBITS ARE MADE RATHER THAN ITERATED. ``perturbation_escape_map``
reads ``Z[n]`` out of ``orbit.packed`` as three float32 words per component,
and that is the whole of the contract between it and
:class:`~spacr.qt.widgets.fractal_mandelbrot.ReferenceOrbit`. Building a real
orbit at 320 digits takes seconds and lands the escape geometry wherever the
Misiurewicz point puts it; handing the same function a chosen ``Z`` puts the
escape boundary exactly where a test needs it -- a wall down the middle of the
frame, a ring outside the search annulus, nothing at all -- in microseconds.
"""
from __future__ import annotations

import builtins
import math

import numpy as np
import pytest

mb = pytest.importorskip("spacr.qt.widgets.fractal_mandelbrot")

#: The survey size ``plan_guided_step`` and ``best_reference_in_view`` use.
SURVEY = (96, 54)


def _orbit(real, imag=None):
    """A reference orbit whose ``Z[n]`` is exactly ``real + i*imag``.

    Packed the way :class:`ReferenceOrbit` packs it: row 0 carries the high
    and middle words of each component and row 1 the low ones, and the map
    sums all three. Putting the whole value in the high word is legitimate --
    the remainders are then zero, which is what an exactly representable Z
    would give.
    """
    real = np.asarray(real, dtype=np.float64)
    imag = (np.zeros_like(real) if imag is None
            else np.asarray(imag, dtype=np.float64))

    class _Stub:
        pass

    orbit = _Stub()
    orbit.max_iter = len(real)
    orbit.packed = np.zeros((2, len(real), 4), dtype=np.float32)
    orbit.packed[0, :, 0] = real
    orbit.packed[0, :, 2] = imag
    return orbit


def _screen(width, height):
    """The map's pixel centres in screen units, as the module computes them."""
    xs = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0)
    ys = ((np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0)
    return xs, ys


#: ``Z[1] = 16`` puts the escape circle |Z + dc| = 16 through dc = 0, so the
#: half of the frame right of centre escapes on the second step and the half
#: left of it does not: a boundary straight down the middle of any view.
WALL = 16.0


def test_a_viewport_with_no_pixels_stops_before_the_first_step():
    """A zero-width survey answers with empty maps instead of iterating.

    The loop's opening ``live.any()`` guard is what makes that true: with no
    pixels there is nothing alive on the very first step, and every later
    step is already covered by the guard after the escape test.
    """
    orbit = _orbit([0.0, WALL])

    escaped, iterations = mb.perturbation_escape_map(orbit, 0, 5, 1.0, 2)
    assert escaped.shape == (5, 0) and iterations.shape == (5, 0)
    assert escaped.dtype == np.bool_ and iterations.dtype == np.int32
    assert escaped.size == 0

    # And the same orbit over a viewport that HAS pixels does iterate: the
    # emptiness above is the width, not an orbit that never escapes.
    escaped, iterations = mb.perturbation_escape_map(orbit, 4, 5, 1e4, 2)
    assert escaped.all()
    assert iterations.max() == 1


def test_transitions_are_counted_only_where_the_patch_has_neighbours():
    """A patch one pixel wide has no horizontal pair to compare, and one a
    pixel tall has no vertical pair. Counting either would be counting a
    difference that does not exist."""
    strip = np.array([[False, True, False, True, False, True, False]])
    times = np.zeros_like(strip, dtype=np.int32)

    # One row: every horizontal neighbour differs, so transitions is 1.0 --
    # the vertical term is skipped rather than scored as zero.
    across = mb.candidate_score(strip, times, 0, 3, 10)
    # One column, the same data transposed: now the vertical term is the one
    # that counts and the horizontal is skipped.
    down = mb.candidate_score(strip.T.copy(), times.T.copy(), 3, 0, 10)
    assert across == pytest.approx(down)
    # 2.4 * one transition, with no escape-time variation and no balance
    # (a perfectly alternating patch is 4/7 escaped, so balance is small).
    assert across == pytest.approx(2.4 + 0.8 * (1.0 - 2.0 * abs(4 / 7 - 0.5)))

    # Both directions together score higher than either alone, which is what
    # makes the skipped term a real omission rather than a zero.
    block = np.zeros((7, 7), dtype=bool)
    block[::2, :] = True
    block[:, ::2] ^= True
    assert mb.candidate_score(block, np.zeros((7, 7), np.int32), 3, 3, 10) \
        > across


def test_a_frame_with_nothing_to_steer_toward_plans_no_step():
    """A view where every pixel behaves identically has no filament in it,
    and the step is refused rather than aimed at noise."""
    flat = _orbit([0.0])                      # never escapes, nothing varies

    escaped, iterations = mb.perturbation_escape_map(
        flat, *SURVEY, 1.0, 1)
    assert not escaped.any()
    assert not mb.boundary_mask(escaped).any()
    assert not mb.structure_mask(escaped, iterations, 1).any()
    assert mb.plan_guided_step(flat, 1.0, 1, strength=0.09,
                               candidates=4) is None

    # A view that DOES hold a boundary is planned from the same call, so the
    # refusal above is the frame's emptiness and not a broken planner.
    ring = _orbit([0.0, 0.0])
    assert mb.plan_guided_step(ring, 20.0, 2, strength=0.09,
                               candidates=4) is not None


def test_a_target_outside_the_search_ring_is_still_a_target():
    """The ring the search prefers is a preference, not a requirement.

    Restricting candidates to 0.025..0.34 of the frame keeps a step from
    being a lurch, but a view whose only structure lies further out has to
    be steered to anyway -- otherwise the dive stops steering exactly where
    there is something to look at.
    """
    # Z[1] = 0 makes the escape test |dc| > 16, so at a half-height of 20 the
    # boundary is an ellipse well away from the centre of the frame.
    ring = _orbit([0.0, 0.0])
    width, height = SURVEY
    aspect = width / height
    escaped, _iterations = mb.perturbation_escape_map(
        ring, width, height, 20.0, 2)
    edge = mb.boundary_mask(escaped)
    assert edge.any(), "the ring orbit drew no boundary at all"

    xs, ys = _screen(width, height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = np.hypot(grid_x, grid_y)
    assert radius[edge].min() > 0.34, (
        "the structure was inside the search ring after all, so the "
        "fall-back this test is about was never needed")

    plan = mb.plan_guided_step(ring, 20.0, 2, strength=0.09,
                               candidates=8, step_index=1)
    assert plan is not None, "a frame full of boundary was left unsteered"
    chosen = math.hypot(plan[0] / aspect, plan[1])
    assert chosen > 0.34
    assert chosen == pytest.approx(radius[edge].min(), abs=0.2)


def test_an_unusable_reach_chooses_nothing_rather_than_nonsense():
    """Steering strength is a stored setting, and a settings file carrying
    Infinity puts every candidate an infinite distance away. The planner
    answers "nowhere" -- which asks again sooner -- instead of returning a
    target whose score is not a number."""
    ring = _orbit([0.0, 0.0])
    assert mb.plan_guided_step(ring, 20.0, 2, strength=float("inf"),
                               candidates=6) is None
    # The same view, the same call, a reach that means something: a target.
    assert mb.plan_guided_step(ring, 20.0, 2, strength=0.09,
                               candidates=6) is not None


def test_the_pattern_says_what_it_needs_when_mpmath_is_missing():
    """mpmath is what buys the reference orbit its 320 digits, and it is an
    optional dependency. Without it the failure has to name itself."""
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mpmath" or name.startswith("mpmath."):
            raise ModuleNotFoundError("blocked mpmath for test")
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        with pytest.raises(RuntimeError) as raised:
            mb.exact_misiurewicz_center(60)
    finally:
        builtins.__import__ = original

    assert "mpmath" in str(raised.value)
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)

    # With mpmath present the same call solves, so the message above is the
    # missing module and not a solve that was going to fail anyway.
    centre = mb.exact_misiurewicz_center(60)
    assert abs(complex(centre)
               - complex(float(mb.MISIUREWICZ_GUESS_REAL),
                         float(mb.MISIUREWICZ_GUESS_IMAG))) < 1e-6


def test_a_flat_survey_offers_no_anchor():
    """Nothing to look at means no opinion about where to start, rather than
    an arbitrary point in an empty frame."""
    flat = _orbit([0.0])
    escaped, iterations = mb.perturbation_escape_map(flat, 128, 72, 1.25, 1)
    assert not mb.structure_mask(escaped, iterations, 1).any()
    assert mb.a_more_interesting_anchor(flat, budget=1) is None

    # A frame with a wall through it does yield one.
    assert mb.a_more_interesting_anchor(_orbit([0.0, WALL]),
                                        budget=2) is not None


def test_an_anchor_is_never_taken_from_the_frames_own_edge():
    """A point at the rim is half outside the survey, so its neighbourhood
    would be scored on data the map does not have."""
    # Z[1] = 14.85i puts the escape circle through dc_im ~ 1.15, which at a
    # half-height of 1.25 is a sliver along the top of the frame and nothing
    # else.
    rim = _orbit([0.0, 0.0], [0.0, 14.85])
    escaped, iterations = mb.perturbation_escape_map(rim, 128, 72, 1.25, 2)
    interesting = mb.structure_mask(escaped, iterations, 2)
    assert interesting.any(), "the rim orbit drew no structure to reject"

    xs, ys = _screen(128, 72)
    grid_x, grid_y = np.meshgrid(xs, ys)
    radius = np.hypot(grid_x, grid_y)
    assert radius[interesting].min() > 0.85, (
        "some structure was inside the rim cut-off, so a None below would "
        "not be the cut-off doing it")

    assert mb.a_more_interesting_anchor(rim, budget=2) is None


def test_the_anchor_is_checked_at_depth_not_only_at_the_surface():
    """Surface structure does not predict what survives a descent, so every
    shortlisted point is surveyed again at a hundredth of the scale and the
    one that still varies there is chosen."""
    wall = _orbit([0.0, WALL])
    anchor = mb.a_more_interesting_anchor(wall, budget=2, candidates=200)
    assert anchor is not None
    dx, dy = anchor

    # It landed ON the wall: dc_re ~ 0 is where the boundary is, and the
    # anchor is given in screen units at the survey's half-height of 1.25.
    assert abs(dx * 1.25) < 0.05, dx

    # And the deep frame around it is genuinely two-coloured, which is the
    # test the chosen point had to pass.
    deep_escaped, _deep_iterations = mb.perturbation_escape_map(
        wall, 48, 27, 1.25 / 100.0, 2, dx * 1.25, dy * 1.25)
    share = float(deep_escaped.mean())
    assert 0.02 < share < 0.98, share


def test_when_nothing_survives_the_deeper_look_the_surface_best_is_kept():
    """A surface best is still a better guess than the middle of the frame."""
    # Everything escapes here -- the right half on the second step and the
    # left half on the third -- so the survey's structure is the escape-time
    # step between them, and a deeper look at any of it is one colour.
    banded = _orbit([0.0, WALL, 1000.0])
    escaped, iterations = mb.perturbation_escape_map(banded, 128, 72, 1.25, 3)
    assert escaped.all(), "something stayed bounded, so this is not the case"
    assert sorted(np.unique(iterations).tolist()) == [1, 2]
    assert mb.structure_mask(escaped, iterations, 3).any()

    anchor = mb.a_more_interesting_anchor(banded, budget=3)
    assert anchor is not None, "the surface best was thrown away"
    dx, dy = anchor
    assert (dx, dy) != (0.0, 0.0), "it fell back to the middle of the frame"

    # Every deep frame was one colour, which is why the surface best is what
    # came back.
    deep_escaped, _deep = mb.perturbation_escape_map(
        banded, 48, 27, 1.25 / 100.0, 3, dx * 1.25, dy * 1.25)
    assert deep_escaped.all()

    # And the point it kept is on the surface structure it was chosen for.
    xs, ys = _screen(128, 72)
    aspect = 128 / 72
    col = int(np.argmin(np.abs(xs * aspect - dx)))
    row = int(np.argmin(np.abs(ys - dy)))
    assert mb.structure_mask(escaped, iterations, 3)[row, col]


def test_a_view_with_no_edge_takes_a_reference_from_inside_the_set():
    """No boundary in view is not a reason to keep an escaping reference:
    somewhere inside at least stays valid until the camera moves again."""
    wall = _orbit([0.0, WALL])
    width, height = SURVEY
    aspect = width / height
    # A half-height of 0.01 with the wall placed just right of the first
    # column: the only bounded pixels are that column, and the frame's own
    # edge is excluded from the boundary, so the view holds no edge at all.
    scale, offset_re, offset_im = 0.01, 0.01740, 0.0

    escaped, _iterations = mb.perturbation_escape_map(
        wall, width, height, scale, 2, offset_re, offset_im)
    bounded = ~escaped
    assert bounded.any(), "nothing was bounded, so this is the other branch"
    assert set(np.nonzero(bounded)[1].tolist()) == {0}
    assert not mb.boundary_mask(escaped).any()

    chosen = mb.best_reference_in_view(wall, offset_re, offset_im, scale, 2)
    xs, ys = _screen(width, height)
    col = int(np.argmin(np.abs(xs * aspect * scale + offset_re - chosen[0])))
    row = int(np.argmin(np.abs(ys * scale + offset_im - chosen[1])))
    assert bounded[row, col], "the new reference is a point that escapes"


def test_a_view_with_nothing_bounded_takes_the_longest_lived_point():
    """The nearest thing to the set a view of pure exterior contains."""
    banded = _orbit([0.0, WALL, 1000.0])
    width, height = SURVEY
    aspect = width / height

    escaped, iterations = mb.perturbation_escape_map(
        banded, width, height, 1.25, 3)
    assert escaped.all(), "something was bounded, so this is another branch"
    assert not mb.boundary_mask(escaped).any()
    assert iterations.min() < iterations.max(), "nothing lived longer"

    chosen = mb.best_reference_in_view(banded, 0.0, 0.0, 1.25, 3)
    xs, ys = _screen(width, height)
    col = int(np.argmin(np.abs(xs * aspect * 1.25 - chosen[0])))
    row = int(np.argmin(np.abs(ys * 1.25 - chosen[1])))
    assert iterations[row, col] == iterations.max()
