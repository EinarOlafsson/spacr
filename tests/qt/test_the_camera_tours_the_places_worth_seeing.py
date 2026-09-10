"""Instruction 327 (3): twenty places, floated between smoothly.

Asked for: "map out which coordinates show interesting visuals like
spirals and texture and have say 20 regions on the image that the camera
will automatically smoothely float towards."

SMOOTHLY IS THE REQUIREMENT, so these test the shape of the motion, not
that a target exists. A linear blend is continuous in position and not in
velocity, and the eye sees that corner as a stop -- so the interpolation
eases in and out, and the tests assert the velocity goes to zero at each
end rather than merely that the path is unbroken.

And drift is OFF the moment the user takes the camera, because dragging
is a statement about where they want to be.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fractal_travel import RegionTour, default_region_tour

REGIONS = (
    ("a", 0.0, 0.0, 1e-5, 1.0),
    ("b", 1.0, 0.0, 1e-5, 1.0),
    ("c", 1.0, 1.0, 1e-5, 1.0),
)


def _tour(**kwargs):
    return RegionTour(REGIONS, dwell=10.0, travel=5.0, **kwargs)


# ---------------------------------------------------------------------------
# It moves, and it moves smoothly
# ---------------------------------------------------------------------------

def test_the_camera_sits_at_a_region_then_leaves_for_the_next():
    tour = _tour()

    assert tour.target_at(0.0) == (0.0, 0.0)
    assert tour.target_at(9.9) == (0.0, 0.0), "it left before its dwell was up"
    assert tour.target_at(15.0) == (1.0, 0.0), "it did not arrive"


def test_the_path_never_jumps():
    """THE REQUIREMENT. Sampled densely across two full circuits."""
    tour = _tour()
    step = 0.01
    previous = tour.target_at(0.0)
    worst = 0.0
    t = step
    while t < 2 * tour.period():
        here = tour.target_at(t)
        worst = max(worst, math.dist(here, previous))
        previous = here
        t += step

    # One region is 1.0 away from the next; a jump would be that size.
    assert worst < 0.01, f"the camera jumped {worst:.4f} in one 10ms step"


def test_it_arrives_and_leaves_with_no_velocity():
    """A linear blend is smooth in position and NOT in velocity. The
    corner at each end is what reads as a stop rather than a float."""
    tour = _tour()
    step = 0.001

    def speed(at):
        a = tour.target_at(at)
        b = tour.target_at(at + step)
        return math.dist(a, b) / step

    leaving = speed(10.0 + step)        # just after the dwell ends
    midway = speed(12.5)                # half way through the travel
    arriving = speed(15.0 - 2 * step)   # just before it arrives

    assert midway > 0.0
    assert leaving < midway / 5.0, (
        f"it sets off at {leaving:.4f} against {midway:.4f} midway")
    assert arriving < midway / 5.0, (
        f"it arrives at {arriving:.4f} against {midway:.4f} midway")


def test_every_region_is_visited():
    tour = _tour()
    seen = {tour.target_at(index * (tour.dwell + tour.travel))
            for index in range(len(REGIONS))}
    assert seen == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)}


def test_the_tour_loops_rather_than_running_off_the_end():
    tour = _tour()
    assert tour.target_at(0.0) == tour.target_at(tour.period())
    assert tour.target_at(3.0) == tour.target_at(tour.period() + 3.0)


def test_the_last_region_travels_back_to_the_first():
    """Otherwise the loop closes with a jump, which is the one thing
    this whole class exists to avoid."""
    tour = _tour()
    end = tour.period()
    assert math.dist(tour.target_at(end - 0.01),
                     tour.target_at(end + 0.01)) < 0.01


# ---------------------------------------------------------------------------
# The user always wins
# ---------------------------------------------------------------------------

def test_taking_the_camera_stops_the_tour_for_good():
    """Dragging is a statement about where they want to be, and a tour
    that resumes over it is the application arguing."""
    tour = _tour()
    assert tour.target_at(3.0) is not None

    tour.take_over()

    assert tour.active is False
    assert tour.target_at(3.0) is None, (
        "the tour handed back a coordinate after the user took over")
    assert tour.target_at(300.0) is None, "it resumed later on its own"


def test_ctrl_r_hands_the_camera_back():
    tour = _tour()
    tour.take_over()
    tour.restart()

    assert tour.active is True
    assert tour.target_at(3.0) == (0.0, 0.0)


def test_none_is_returned_rather_than_a_coordinate_to_ignore():
    """So a caller can leave the camera exactly where the user put it."""
    tour = RegionTour(())
    assert tour.active is False
    assert tour.target_at(0.0) is None


# ---------------------------------------------------------------------------
# The committed regions
# ---------------------------------------------------------------------------

def test_the_shipped_tour_has_regions_to_visit():
    tour = default_region_tour()
    assert tour.regions, "no regions were generated; run the finder script"
    assert tour.target_at(0.0) is not None


def test_every_shipped_region_is_a_usable_coordinate():
    from spacr.qt.widgets.fractal_regions import REGIONS as SHIPPED

    assert len(SHIPPED) >= 10, f"only {len(SHIPPED)} regions"
    for name, x, y, half_width, score in SHIPPED:
        assert name
        assert -2.5 < x < 1.0, f"{name} is outside the set's neighbourhood"
        assert -1.5 < y < 1.5, f"{name} is outside the set's neighbourhood"
        assert half_width > 0.0
        assert score > 0.0, f"{name} scored {score}, so it is not interesting"


def test_the_shipped_regions_are_spread_out():
    """Twenty coordinates inside one filament is one place, not twenty."""
    from spacr.qt.widgets.fractal_regions import REGIONS as SHIPPED

    for i, (name_a, ax, ay, _hw, _s) in enumerate(SHIPPED):
        for name_b, bx, by, _hw2, _s2 in SHIPPED[i + 1:]:
            assert math.hypot(ax - bx, ay - by) > 0.05, (
                f"{name_a} and {name_b} are the same place")


# ---------------------------------------------------------------------------
# AND SOMETHING CALLS IT. Everything above passed for days while nothing in
# `spacr/` imported RegionTour or REGIONS -- the twenty places were mapped,
# tested and never on screen. 327's own closing note said so: "the pieces
# are there; the wire is not." These test the wire.
# ---------------------------------------------------------------------------

def test_one_leg_of_the_tour_is_one_dive():
    """The move belongs at the surface, and this is what puts it there.

    A lateral step thirty decades down crosses a view narrower than the gap
    between two regions by thirty orders of magnitude: a white flash, which
    is the jump the request was about. Sizing the dwell as "the dive, less
    the move" leaves the camera holding one region for the whole descent
    and moving to the next as the dive restarts.
    """
    from spacr.qt.widgets.fractal_travel import tour_leg_seconds

    dwell, travel = tour_leg_seconds(24.0, 23.0, 9.0)
    assert travel == 9.0
    assert dwell + travel == pytest.approx(24.0 * 23.0), (
        "a leg no longer matches the dive it is supposed to sit inside")


def test_a_dive_shorter_than_the_move_still_produces_a_tour_that_moves():
    """A hand-edited settings file must not be able to stop the tour."""
    from spacr.qt.widgets.fractal_travel import tour_leg_seconds

    dwell, travel = tour_leg_seconds(0.1, 1.0, 60.0)
    assert dwell > 0.0 and travel > 0.0


def test_the_offset_is_measured_from_the_reference_orbit():
    """`u_center_offset` is relative to the orbit, not to the plane.

    The shader knows one point exactly -- the reference orbit's centre --
    and measures every pixel as a small offset from it. A region's absolute
    coordinate handed over unconverted would point the camera at whatever
    happens to sit that far from the reference instead.
    """
    from spacr.qt.widgets.fractal_travel import tour_offset

    tour = _tour()
    assert tour_offset(tour, 0.0, (0.0, 0.0)) == (0.0, 0.0)
    assert tour_offset(tour, 0.0, (0.25, -0.5)) == (-0.25, 0.5)


def test_the_offset_follows_the_reference_when_it_is_refined():
    """The reference MOVES, and the tour must not be dragged along with it.

    `_refine_the_reference` rebases the orbit onto the boundary as the dive
    descends and zeroes the camera's offset when it lands. A tour that
    integrated its own offset would be thrown off the region every time
    that happened; recomputing from the current centre is immune, and this
    is the assertion that says so: the same instant of the tour, two
    different references, one absolute destination.
    """
    from spacr.qt.widgets.fractal_travel import tour_offset

    tour = _tour()
    before = tour_offset(tour, 3.0, (0.1, 0.2))
    after = tour_offset(tour, 3.0, (0.4, -0.1))
    assert before is not None and after is not None
    assert (before[0] + 0.1, before[1] + 0.2) == pytest.approx(
        (after[0] + 0.4, after[1] - 0.1)), (
        "the same moment of the tour points at two different places "
        "depending on where the reference happens to be")


def test_a_taken_over_tour_offers_no_offset_at_all():
    from spacr.qt.widgets.fractal_travel import tour_offset

    tour = _tour()
    tour.take_over()
    assert tour_offset(tour, 0.0, (0.0, 0.0)) is None


def test_the_dive_can_be_told_to_tour():
    """"tour" is a value the settings accept and hand back.

    Persisted rather than asserted in the abstract: the path is read
    through `get_fractal_settings`, which validates against a fixed set,
    and a value missing from that set is silently replaced by the default
    -- so the renderer would never see it and nothing would fail.
    """
    from spacr.qt import preferences as P

    P.set_fractal_settings(path="tour")
    try:
        assert P.get_fractal_settings()["path"] == "tour"
    finally:
        P.set_fractal_settings(path="fixed")


def test_the_preferences_panel_offers_the_tour():
    """A mode nothing can select is a mode nobody has."""
    import inspect

    from spacr.qt import preferences as P

    source = inspect.getsource(P.PreferencesDialog._build_the_dialog)
    assert '("tour", "Tour the mapped regions")' in source, (
        "the Path dropdown no longer offers the tour")


def test_the_renderer_steers_by_the_tour():
    """The wire itself, asserted where it is made.

    The GPU canvas cannot be built without a GL context, so the branch is
    read out of the source rather than driven. That is weaker than running
    it and it is the assertion that would have failed for the days this
    feature existed and did nothing.
    """
    import inspect

    from spacr.qt.widgets import fractal_travel as F

    source = inspect.getsource(F._make_gpu_widget)
    assert 'if path == "tour":' in source
    assert "self._tour_centre(camera, orbit)" in source
    assert "tour_offset(" in source
    assert "tour.take_over()" in source, (
        "dragging no longer stops the tour, so the camera and the user "
        "fight over where to point")
    assert "tour.restart()" in source, "Ctrl+R no longer hands it back"
