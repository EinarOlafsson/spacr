"""Instruction 327 (4): the pointer warps locally and nothing snaps back.

Reported 2026-08-31: "the mouse gravity really only pushes the camera
abit and then if the mouse is to close to the sides of the screen the
camera snapps back. I was more looking for the visual feature itself
being centered on the mouse or for instance stars in space mode being
pulled towards the mouse... (like gravity)."

Every GPU shader translated the WHOLE PLANE -- ``uv - target * pull``
moves every pixel by the same amount, which is towing the viewport. Two
consequences, both reported: the shift grows with the pointer's distance
from centre, so near an edge the whole picture is dragged; and when the
pointer leaves, the pull decays to zero and the picture springs back.

The CPU orbit fold never had either problem, and the user says so: "the
orbit fold cpu effect is like a magnigying glass, which looks cool".
Its warp is the reference, and every shader is now a transliteration of
it.

The maths is modelled here in Python. A GLSL unit test would need a GL
context; what these assert are the PROPERTIES that make it read as
gravity rather than as a drag, plus that every shader carries the same
constants.
"""
from __future__ import annotations

import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHADERS = [
    ROOT / "spacr" / "qt" / "widgets" / "fractal_travel.py",
    ROOT / "spacr" / "qt" / "widgets" / "fractal_space.py",
    ROOT / "spacr" / "qt" / "widgets" / "fractal_cascade.py",
    ROOT / "spacr" / "qt" / "widgets" / "fractal_mandelbrot.py",
]


def warp(x, y, pointer_x, pointer_y, pull, push=0.0):
    """The shader's warp, in Python. Kept identical on purpose."""
    to_x = pointer_x - x
    to_y = pointer_y - y
    distance2 = to_x * to_x + to_y * to_y + 0.05
    strength = (0.55 * pull - 0.95 * push) / distance2
    strength = max(-1.4, min(0.9, strength))
    return x + strength * to_x, y + strength * to_y


def _moved(x, y, **kwargs):
    nx, ny = warp(x, y, **kwargs)
    return math.hypot(nx - x, ny - y)


def old_warp(x, y, pointer_x, pointer_y, pull, push=0.0):
    """What every shader did before: translate the WHOLE plane.

    Kept so the tests can compare against it rather than against a
    threshold picked by eye. Note it ignores x and y entirely -- that is
    the defect in one line.
    """
    shift = pull - 0.85 * push
    return x - pointer_x * shift, y - pointer_y * shift


def _old_moved(x, y, **kwargs):
    nx, ny = old_warp(x, y, **kwargs)
    return math.hypot(nx - x, ny - y)


# ---------------------------------------------------------------------------
# It is LOCAL: that is what stops the snap-back
# ---------------------------------------------------------------------------

def test_a_pixel_under_the_pointer_moves_and_a_far_one_barely_does():
    """THE WHOLE POINT. A uniform translation moves both equally."""
    near = _moved(0.5, 0.0, pointer_x=0.0, pointer_y=0.0, pull=1.0)
    far = _moved(5.0, 0.0, pointer_x=0.0, pointer_y=0.0, pull=1.0)

    assert near > 0.05, "the pointer does nothing under the cursor"
    assert far < near / 3.0, (
        f"a far pixel moved {far:.4f} against {near:.4f} near the cursor; "
        f"that is a drag, not a local warp")

    # AND THE OLD ONE MOVED THEM EQUALLY, which is the actual defect.
    assert _old_moved(0.5, 0.0, pointer_x=0.0, pointer_y=0.0,
                      pull=1.0) == _old_moved(5.0, 0.0, pointer_x=0.0,
                                              pointer_y=0.0, pull=1.0)


def test_the_falloff_is_inverse_square():
    """An inverse-square falloff is what makes it read as gravity rather
    than as a magnet -- the instruction says so in as many words."""
    at_one = _moved(1.0, 0.0, pointer_x=0.0, pointer_y=0.0, pull=1.0)
    at_two = _moved(2.0, 0.0, pointer_x=0.0, pointer_y=0.0, pull=1.0)

    # displacement = strength * r, and strength ~ 1/r^2, so it falls ~1/r
    assert at_two < at_one
    assert at_one / at_two == pytest.approx(2.0, rel=0.25), (
        f"falloff is {at_one / at_two:.2f}x over a doubling of distance")


def test_the_displacement_points_at_the_pointer():
    """Toward, not away -- the sign is the difference between a pull and
    a shove, and a click is what reverses it."""
    x, y = 1.0, 0.0
    nx, _ny = warp(x, y, pointer_x=0.0, pointer_y=0.0, pull=1.0)
    assert nx < x, "the pixel moved away from the pointer"


def test_a_click_pushes_instead_of_pulling():
    x, y = 1.0, 0.0
    nx, _ny = warp(x, y, pointer_x=0.0, pointer_y=0.0, pull=0.0, push=1.0)
    assert nx > x, "a push did not move the structure away"


# ---------------------------------------------------------------------------
# Nothing snaps back, because nothing was displaced globally
# ---------------------------------------------------------------------------

def test_a_pointer_at_the_edge_does_not_drag_the_whole_picture():
    """THE REPORTED DEFECT. Under the old translation the shift was
    proportional to the pointer's own offset, so an edge pointer moved
    everything a long way."""
    now = _moved(-1.0, 0.0, pointer_x=1.0, pointer_y=0.0, pull=1.0)
    before = _old_moved(-1.0, 0.0, pointer_x=1.0, pointer_y=0.0, pull=1.0)

    assert now < before / 3.0, (
        f"a pointer at the right edge still moves the left edge "
        f"{now:.3f}, against {before:.3f} before -- barely an improvement")
    assert before == pytest.approx(1.0), (
        "the old translation moved every pixel by the pointer's own "
        "offset, which is what dragged the picture")


def test_letting_go_moves_distant_pixels_almost_not_at_all():
    """The spring-back was the whole picture returning. With a local
    warp only the neighbourhood of the cursor was ever displaced."""
    snaps = []
    old_snaps = []
    for x in (2.0, 3.0, 5.0, 8.0):
        held = warp(x, 0.0, pointer_x=0.6, pointer_y=0.0, pull=1.0)
        released = warp(x, 0.0, pointer_x=0.6, pointer_y=0.0, pull=0.0)
        snaps.append(math.hypot(held[0] - released[0],
                                held[1] - released[1]))

        old_held = old_warp(x, 0.0, pointer_x=0.6, pointer_y=0.0, pull=1.0)
        old_released = old_warp(x, 0.0, pointer_x=0.6, pointer_y=0.0,
                                pull=0.0)
        old_snaps.append(math.hypot(old_held[0] - old_released[0],
                                    old_held[1] - old_released[1]))

    # SMALLER EVERYWHERE, and shrinking with distance. The absolute size
    # is not the claim -- the CPU orbit fold is the reference and the
    # user likes it as it is -- so what is asserted is the SHAPE: the
    # further from the pointer, the less there is to spring back from.
    assert all(new < old for new, old in zip(snaps, old_snaps))
    assert snaps == sorted(snaps, reverse=True), (
        f"the spring-back does not fall off with distance: {snaps}")

    # The old one could say neither: it was the same at every distance.
    assert len(set(round(s, 9) for s in old_snaps)) == 1, (
        "the old translation is no longer uniform, so this comparison "
        "needs rewriting")


def test_no_pull_and_no_push_leaves_every_pixel_exactly_alone():
    for x, y in ((0.0, 0.0), (1.0, -1.0), (4.0, 2.0)):
        assert warp(x, y, pointer_x=0.3, pointer_y=0.2,
                    pull=0.0, push=0.0) == (x, y)


def test_a_pixel_is_never_thrown_past_the_pointer():
    """`strength` is clamped at 0.9, so a pixel closes at most most of
    the way -- overshoot would read as the picture folding through
    itself."""
    for r in (0.01, 0.05, 0.2, 1.0):
        nx, _ny = warp(r, 0.0, pointer_x=0.0, pointer_y=0.0, pull=2.0)
        assert nx >= 0.0, f"a pixel at r={r} crossed the pointer to {nx}"


def test_the_pointer_itself_is_finite():
    """The 0.05 floor keeps the divide finite where to_pointer is zero."""
    assert warp(0.0, 0.0, pointer_x=0.0, pointer_y=0.0, pull=1.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Every renderer bends it the same way
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", SHADERS, ids=lambda p: p.name)
def test_no_shader_still_translates_the_whole_plane(path):
    """THE REGRESSION. The old form is one line and easy to reintroduce."""
    source = path.read_text(encoding="utf-8")
    for old in ("uv - target * (u_pull", "q - target * (u_pull"):
        assert old not in source, (
            f"{path.name} translates the whole plane again")


@pytest.mark.parametrize("path", SHADERS, ids=lambda p: p.name)
def test_every_shader_uses_the_same_constants(path):
    """A warp that differs per pattern is four behaviours, not one."""
    source = path.read_text(encoding="utf-8")
    assert "0.55 * u_pull - 0.95 * u_push" in source
    assert "clamp(strength, -1.4, 0.9)" in source
    assert "dot(to_pointer, to_pointer) + 0.05" in source


def test_the_python_model_matches_the_cpu_renderer():
    """The CPU orbit fold is the reference the shaders were copied from,
    so the model here must match ITS constants too."""
    source = (ROOT / "spacr" / "qt" / "widgets"
              / "fractal_travel.py").read_text(encoding="utf-8")
    assert "strength = (0.55 * pull - 0.95 * push) / distance2" in source
    assert "distance2 = to_x * to_x + to_y * to_y + 0.05" in source
