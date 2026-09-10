"""The fold-inversion cascade's CPU maths, run as plain Python.

Numba compiles ``_layer``, ``_sample`` and ``render_into`` to machine code,
and a compiled kernel executes no Python lines -- so the fold map, its three
orbit traps, the palette and the supersampler are invisible to a line tracer
even while the suite renders frames with them. Numba's own ``DISABLE_JIT``
switch turns the decorators into no-ops, so the SAME source lines run as
interpreted Python and can be measured one at a time.

Frames here are TINY, because interpreting a per-pixel kernel at backdrop
size would be a wall-clock cost for nothing.

No GPU is touched anywhere in this file: the GLSL backend is separate and
nothing here constructs a canvas.
"""
from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

pytest.importorskip("numba")

cascade = pytest.importorskip("spacr.qt.widgets.fractal_cascade")

SOURCE = cascade.__file__

WIDTH, HEIGHT = 16, 12

#: Sentinel for "numba was not in sys.modules to begin with".
_MISSING = object()

#: The twenty-two arguments ``_sample`` takes, at a fixed point of the flight.
SAMPLE_ARGS = dict(
    width=WIDTH, height=HEIGHT, t=5.0, dream=0.5, iterations=6,
    camera_cs=0.97, camera_sn=0.24, tx=0.05, ty=-0.03,
    shear_x=0.02, shear_y=-0.01, stretch_x=1.05, stretch_y=0.96,
    rotation_cs=0.92, rotation_sn=0.38, constant_x=0.715, constant_y=0.475,
    scale_a=1.0, scale_b=2.0, blend=0.5, palette_phase=0.2,
)


def _sample_at(module, px, py, **overrides):
    """``_sample`` at one sub-pixel, with the flight held still."""
    args = dict(SAMPLE_ARGS)
    args.update(overrides)
    return module._sample(
        px, py, args["width"], args["height"], args["t"], args["dream"],
        args["iterations"], args["camera_cs"], args["camera_sn"],
        args["tx"], args["ty"], args["shear_x"], args["shear_y"],
        args["stretch_x"], args["stretch_y"], args["rotation_cs"],
        args["rotation_sn"], args["constant_x"], args["constant_y"],
        args["scale_a"], args["scale_b"], args["blend"],
        args["palette_phase"])


def _execute_source(name: str, *, hide_numba: bool = False,
                    disable_jit: bool = False):
    """Run the shipped module source again under a private module name.

    Loading a second copy leaves ``spacr.qt.widgets.fractal_cascade`` in
    ``sys.modules`` exactly as it was, so nothing else in the session sees a
    module whose kernels are interpreted or whose numba is missing.
    """
    from numba import config as numba_config

    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    was_disabled = numba_config.DISABLE_JIT
    had_numba = sys.modules.get("numba", _MISSING)
    if disable_jit:
        numba_config.DISABLE_JIT = True
    if hide_numba:
        # ``None`` in sys.modules is the documented way to make an import of
        # that name fail, which is what a machine without numba looks like.
        sys.modules["numba"] = None
    try:
        spec.loader.exec_module(module)
    finally:
        numba_config.DISABLE_JIT = was_disabled
        if hide_numba:
            if had_numba is _MISSING:
                sys.modules.pop("numba", None)
            else:
                sys.modules["numba"] = had_numba
    return module


@pytest.fixture(scope="module")
def plain():
    """The module's own kernels, interpreted rather than compiled."""
    return _execute_source("_fractal_cascade_plain", disable_jit=True)


@pytest.fixture(scope="module")
def no_numba():
    """The module as it loads on a machine that has no numba at all."""
    return _execute_source("_fractal_cascade_no_numba", hide_numba=True)


# --------------------------------------------------------------------------
# The cheap trigonometry the kernel is built on


def test_the_fast_sine_tracks_the_real_one_within_a_thousandth(plain):
    """It replaces ``math.sin`` several times per sub-pixel, so it has to be
    close enough that the palette and the camera do not visibly wobble."""
    angles = np.linspace(-12.0, 12.0, 481)
    error = max(abs(plain._fast_sin(a) - math.sin(a)) for a in angles)
    assert error < 0.0012, f"the sine approximation is off by {error}"


def test_the_fast_cosine_is_the_sine_a_quarter_turn_on(plain):
    for angle in (-7.0, -1.25, 0.0, 0.4, 3.3, 9.9):
        assert plain._fast_cos(angle) == plain._fast_sin(
            angle + 0.5 * math.pi)
        assert abs(plain._fast_cos(angle) - math.cos(angle)) < 0.0012


def test_the_sine_folds_angles_far_outside_one_turn(plain):
    """The camera and palette phases grow without bound as ``t`` runs, so an
    approximation that only held near zero would drift into nonsense."""
    for angle in (200.0, -200.0, 1000.5):
        assert abs(plain._fast_sin(angle) - math.sin(angle)) < 0.0012


# --------------------------------------------------------------------------
# The fold map and its three orbit traps


def test_the_fold_is_blind_to_sign_and_to_which_axis_is_which(plain):
    """Each iteration takes ``abs`` of both coordinates and puts the larger
    first, so the map is symmetric about both axes and about the diagonal --
    which is what makes the pattern a kaleidoscopic fold at all."""
    reference = plain._layer(0.3, 0.9, 6, 0.92, 0.38, 0.715, 0.475)
    assert plain._layer(0.9, 0.3, 6, 0.92, 0.38, 0.715, 0.475) == reference
    assert plain._layer(-0.3, -0.9, 6, 0.92, 0.38, 0.715, 0.475) == reference
    assert plain._layer(-0.9, 0.3, 6, 0.92, 0.38, 0.715, 0.475) == reference


def test_no_iterations_leaves_every_trap_at_its_sentinel(plain):
    """The traps are running minima, so an unvisited orbit has to report the
    starting distance rather than zero -- a zero would read as a direct hit
    and paint the brightest thing on screen."""
    assert plain._layer(0.3, 0.9, 0, 0.92, 0.38, 0.715, 0.475) == (
        10.0, 10.0, 10.0, 0.0)


def test_more_iterations_can_only_tighten_the_traps(plain):
    """Each trap keeps the closest approach so far, so extending the orbit
    never moves one further away, and the energy sum only grows."""
    short = plain._layer(0.35, 0.8, 3, 0.92, 0.38, 0.715, 0.475)
    long = plain._layer(0.35, 0.8, 9, 0.92, 0.38, 0.715, 0.475)
    for near, far in zip(long[:3], short[:3]):
        assert near <= far
    assert long[3] > short[3]
    for name, trap in zip(("ring", "diagonal", "slant"), long[:3]):
        assert trap < 10.0, f"the {name} trap never left its sentinel"


# --------------------------------------------------------------------------
# One sub-pixel


def test_a_sample_is_three_bytes(plain):
    for px, py in ((0.25, 0.25), (7.75, 5.25), (WIDTH - 0.25, HEIGHT - 0.25)):
        pixel = _sample_at(plain, px, py)
        assert len(pixel) == 3
        assert all(isinstance(c, int) for c in pixel), pixel
        assert all(0 <= c <= 255 for c in pixel), (px, py, pixel)


def test_the_palette_phase_moves_the_colour_without_moving_the_structure(plain):
    """The phase enters the palette only, so the same fold comes back in a
    different hue rather than as a different shape."""
    here = _sample_at(plain, 7.75, 5.25, palette_phase=0.0)
    shifted = _sample_at(plain, 7.75, 5.25, palette_phase=1.4)
    assert here != shifted


def test_the_vignette_darkens_a_corner_more_than_the_centre(plain):
    """The brightness factor falls from 1.0 to 0.76 as the screen radius
    passes 0.60, which is what keeps the interface in front readable."""
    centre = sum(_sample_at(plain, WIDTH / 2, HEIGHT / 2, dream=0.0,
                            camera_cs=1.0, camera_sn=0.0, tx=0.0, ty=0.0,
                            shear_x=0.0, shear_y=0.0, stretch_x=1.0,
                            stretch_y=1.0))
    corner = sum(_sample_at(plain, 0.25, 0.25, dream=0.0, camera_cs=1.0,
                            camera_sn=0.0, tx=0.0, ty=0.0, shear_x=0.0,
                            shear_y=0.0, stretch_x=1.0, stretch_y=1.0))
    assert corner < centre


# --------------------------------------------------------------------------
# A whole frame


def test_every_pixel_of_the_buffer_is_written(plain):
    """The buffer is reused between frames, so a pixel the loop skipped would
    hold the previous frame for as long as the widget is that size."""
    first = np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)
    second = np.full((HEIGHT, WIDTH, 3), 231, dtype=np.uint8)
    plain.render_into(first, 5.0, 1.0, 0.5, 6)
    plain.render_into(second, 5.0, 1.0, 0.5, 6)
    assert np.array_equal(first, second)


def test_the_cascade_fills_the_frame_with_structure(plain):
    """Unlike the star field, this pattern is meant to be a full field: a
    frame that came out near-black would be a fold that never resolved."""
    output = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(output, 5.0, 1.0, 0.5, 6)
    assert output.mean() > 40
    assert output.std() > 5, "a flat frame is not a cascade"


def test_the_frame_is_a_pure_function_of_its_arguments(plain):
    once = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    twice = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(once, 5.0, 1.0, 0.5, 6)
    plain.render_into(twice, 5.0, 1.0, 0.5, 6)
    assert np.array_equal(once, twice)
    later = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(later, 9.0, 1.0, 0.5, 6)
    assert not np.array_equal(once, later)


def test_a_still_dream_leaves_the_frame_centred_on_the_fold(plain):
    """Every wander term is multiplied by ``dream``, so zero is the pattern
    with no drift at all rather than the pattern with a different drift.

    That is measurable rather than a matter of taste. The fold takes ``abs``
    of both coordinates, so the field is symmetric about the origin; the
    camera rotation, the shear and the stretch are all linear and keep that
    symmetry. Only the translation and the dream warp move the origin off the
    middle of the frame. At ``dream = 0`` both are exactly zero, so the frame
    has to come out identical under a half turn -- and any drift that leaked
    past the ``dream`` factor breaks it.
    """
    still = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(still, 5.0, 1.0, 0.0, 6)
    assert np.array_equal(still, still[::-1, ::-1, :]), (
        "the fold is off centre with no dream to move it")

    dreaming = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(dreaming, 5.0, 1.0, 1.0, 6)
    assert not np.array_equal(dreaming, dreaming[::-1, ::-1, :])
    assert not np.array_equal(dreaming, still)


def test_a_pull_and_a_push_that_cancel_leave_the_camera_where_it_was(plain):
    """The pointer contributes ``pull * 0.30 - push * 0.55`` per axis, so a
    click that exactly answers the pull is the frame with no pointer at all.

    That is the whole claim of moving the camera rather than warping the
    field: the two terms are one translation and they add.
    """
    neutral = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cancelled = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pulled = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(neutral, 5.0, 1.0, 0.5, 6)
    plain.render_into(cancelled, 5.0, 1.0, 0.5, 6, 0.8, -0.5, 0.55, 0.30)
    plain.render_into(pulled, 5.0, 1.0, 0.5, 6, 0.8, -0.5, 0.55, 0.0)
    assert np.array_equal(neutral, cancelled)
    assert not np.array_equal(neutral, pulled)


def test_a_deeper_orbit_draws_a_different_frame(plain):
    """``iterations`` is the depth the widget trades for frame rate, and it
    has to reach BOTH scale windows.

    The two windows cross-fade, so a depth that reached only one of them
    would still change the frame -- and would make the cascade shallower for
    half of every cycle without ever looking broken.
    """
    near_only = _sample_at(plain, 7.75, 5.25, blend=0.0, iterations=3)
    assert near_only != _sample_at(plain, 7.75, 5.25, blend=0.0,
                                   iterations=12)
    far_only = _sample_at(plain, 7.75, 5.25, blend=1.0, iterations=3)
    assert far_only != _sample_at(plain, 7.75, 5.25, blend=1.0, iterations=12)

    shallow = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    deep = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(shallow, 5.0, 1.0, 0.5, 3)
    plain.render_into(deep, 5.0, 1.0, 0.5, 12)
    assert not np.array_equal(shallow, deep)


# --------------------------------------------------------------------------
# The engine around the kernel, and the machine with no numba


def test_the_buffer_is_reused_until_the_widget_changes_size():
    """One allocation per resize rather than one per frame."""
    engine = cascade.CascadeEngine(2)
    engine.render(WIDTH, HEIGHT, 5.0, 1.0, 0.5, 4)
    first = engine.output
    engine.render(WIDTH, HEIGHT, 6.0, 1.0, 0.5, 4)
    assert engine.output is first
    engine.render(WIDTH + 2, HEIGHT, 6.0, 1.0, 0.5, 4)
    assert engine.output is not first
    assert engine.output.shape == (HEIGHT, WIDTH + 2, 3)


def test_the_engine_hands_back_a_copy_rather_than_its_own_buffer():
    """The frame travels to the GUI thread while the next one is drawn into
    the buffer, so handing out the buffer itself would tear."""
    engine = cascade.CascadeEngine(2)
    frame = engine.render(WIDTH, HEIGHT, 5.0, 1.0, 0.5, 4)
    assert frame is not engine.output
    assert np.array_equal(frame, engine.output)


def test_the_compiled_kernel_and_the_interpreted_one_draw_the_same_frame(plain):
    """What the tests above measure is what a user's machine renders."""
    interpreted = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    plain.render_into(interpreted, 5.0, 1.0, 0.5, 6, 0.0, 0.0, 0.0, 0.0)
    compiled = cascade.CascadeEngine(2).render(WIDTH, HEIGHT, 5.0, 1.0, 0.5, 6)
    difference = np.abs(compiled.astype(int) - interpreted.astype(int)).max()
    assert difference <= 1, f"the two backends disagree by {difference}/255"


def test_without_numba_the_module_still_imports(no_numba):
    """The GPU backend does not need numba, so a machine without it must
    still get a module with a shader in it rather than an ImportError."""
    assert no_numba.njit is None
    assert no_numba.prange is range
    assert "render_sample" in no_numba.FRAGMENT_SHADER
    assert no_numba.CascadeEngine is not None


def test_without_numba_the_cpu_backend_says_what_is_missing(no_numba):
    """A frame of flat fallback colour with no explanation is the failure
    this replaces."""
    output = np.empty((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError,
                       match="numba is required for the cascade CPU backend"):
        no_numba.render_into(output, 1.0, 1.0, 0.0, 4)
