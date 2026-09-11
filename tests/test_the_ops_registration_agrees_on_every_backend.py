"""Phase correlation: the shift, the rejections, and both backends agreeing.

Instruction 372's PHASE E: "the backend is an indirection over the array
module: CuPy (CUDA) -> torch.fft (CUDA / ROCm / MPS) -> numpy+scipy. The
CPU path is not an error path; both backends run the same fixture and
must agree to within a pixel."

That is the contract this file holds, and the reason for it is worth
stating: a fallback nobody compares is not a fallback, it is a second
implementation that will drift. The accelerated backends are SKIPPED
where the machine has none -- a skip says "not asked", where a pass would
say "checked" about a path that never ran.

THE FIXTURE IS A FIELD OF BLOBS AT THE REAL OVERLAP. 10X SBS DAPI is
round nuclei with no corners, which is why feature matching found
nothing on it (26 of 333 tiles placed) and why phase correlation found
the geometry immediately. The tiles here overlap by 14 %, the same as the
acquisition's 213 px of 1480.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacr.ops_register import (MIN_PEAK_RATIO, Registration,
                                available_backends, phase_correlate,
                                register_edge, register_pairs, unwrap)

TILE = 640
OVERLAP = int(TILE * 0.14)    # the raster's own overlap: 213 px of 1480
STEP = TILE - OVERLAP

#: Blob spacing, brightness and blur, chosen so the fixture lands in the
#: ranges the real acquisition measured rather than near the threshold:
#: true neighbours here score 33.4 and 34.5 against a control at 11.9,
#: where well A1 gave 23.6-53.1 against 9.0-15.6. A fixture that sits on
#: the boundary tests the boundary, not the method.
PIXELS_PER_BLOB = 300
BLOB_SIGMA = 3.0
BLOB_PEAK = 1200.0
BACKGROUND = 40.0


def _field(height: int, width: int, seed: int = 5) -> np.ndarray:
    """A field of round nuclei, at the density and contrast of 10X DAPI.

    Built by blurring dots rather than by summing Gaussians: one
    convolution instead of thousands of exponentials, which is the
    difference between a fast test and a slow one. The amplitude is
    scaled by the kernel's own integral so `BLOB_PEAK` is the peak a blob
    actually reaches, not a number that means nothing.
    """
    import math

    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    dots = np.zeros((height, width))
    count = (height * width) // PIXELS_PER_BLOB
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    dots[ys, xs] = 1.0
    scale = BLOB_PEAK * 2 * math.pi * BLOB_SIGMA * BLOB_SIGMA
    return gaussian_filter(dots, BLOB_SIGMA) * scale + BACKGROUND


@pytest.fixture(scope="module")
def tiles():
    """Three tiles from one field: a, its right neighbour, and a stranger."""
    field = _field(TILE + STEP + 80, 2 * TILE + 200)
    rng = np.random.default_rng(11)

    def crop(top, left):
        return rng.poisson(field[top:top + TILE,
                                 left:left + TILE]).astype(float)

    return {
        "a": crop(40, 40),
        "right": crop(40, 40 + STEP),
        "below": crop(40 + STEP, 40),
        "stranger": crop(40, 40 + TILE + 60),
    }


# -- the unwrap ------------------------------------------------------------

def test_a_wrapped_shift_folds_to_the_negative_half():
    """-213 of a 1480 px tile arrives as +1267 and must read as -213."""
    assert unwrap(1267, 1480) == -213
    assert unwrap(213, 1480) == 213
    assert unwrap(0, 1480) == 0


def test_the_unwrap_resolves_around_what_the_caller_expects():
    """The defect a strip introduces, and the reason `about` exists.

    A 222 px strip carrying a true shift of 116 folds to -106 against the
    half-range: wrong, and plausible enough to be believed. The caller
    knows the pitch, so it says so.
    """
    assert unwrap(116, 222) == -106
    assert unwrap(116, 222, about=100) == 116
    assert unwrap(1267, 1480, about=-200) == -213
    # An extent of zero cannot wrap anything and must not divide by it.
    assert unwrap(7, 0) == 7


# -- the measurement -------------------------------------------------------

def test_a_known_roll_comes_back_as_the_shift_that_undoes_it(tiles):
    """The sign convention, asserted rather than described.

    `dy, dx` is what lands the SECOND field on the first, so rolling the
    second by it reproduces the first.
    """
    first = tiles["a"]
    second = np.roll(np.roll(first, -37, axis=0), 12, axis=1)
    found = phase_correlate(first, second, gpu=False)
    assert (found.dy, found.dx) == (37, -12)
    assert np.allclose(np.roll(np.roll(second, found.dy, axis=0),
                               found.dx, axis=1), first)


def test_two_touching_tiles_give_the_rasters_step(tiles):
    """The measurement A2 is made of, on both axes."""
    right = register_edge(tiles["a"], tiles["right"], "horizontal",
                          expected_overlap=OVERLAP, gpu=False)
    assert right.accepted
    assert right.dx == pytest.approx(STEP, abs=1)
    assert abs(right.dy) <= 1

    below = register_edge(tiles["a"], tiles["below"], "vertical",
                          expected_overlap=OVERLAP, gpu=False)
    assert below.accepted
    assert below.dy == pytest.approx(STEP, abs=1)
    assert abs(below.dx) <= 1


def test_a_pair_that_does_not_touch_is_rejected_by_the_layout(tiles):
    """THE LAYOUT REJECTS IT. THE PEAK RATIO CANNOT, AND THIS SAYS SO.

    The first version of this test asserted the ratio, on a fixture where
    a true neighbour scored 33 and a control 12. The full distributions
    over one real well say that separation is an artefact of a contrived
    pair -- 624 true adjacencies run from 6.2 to 120.2 and 60 controls
    from 5.9 to 18.9, so the families OVERLAP and no threshold divides
    them. A gate at 20 would have thrown away 152 real edges.

    So the control is rejected because it did not land where the raster
    says it should, which is a question with an answer.
    """
    found = register_edge(tiles["a"], tiles["stranger"], "horizontal",
                          expected_overlap=OVERLAP, tolerance=3, gpu=False)
    assert not found.accepted
    assert found.peak_ratio > MIN_PEAK_RATIO, (
        "this control scores above the ratio gate, which is the point: "
        "the ratio would have accepted it")


def test_the_ratio_gate_only_refuses_a_surface_with_no_peak(tiles):
    """What is left of the threshold once the layout does the deciding.

    Two fields of pure noise have no translation that makes them agree,
    and that is the case a ratio can still answer. A distant neighbour is
    not.
    """
    rng = np.random.default_rng(5)
    flat = rng.normal(100.0, 1.0, (128, 128))
    other = rng.normal(100.0, 1.0, (128, 128))
    found = phase_correlate(flat, other, gpu=False)
    assert found.peak_ratio < 20.0, (
        "even pure noise clears the OLD gate of 20 on this fixture, which "
        "is why the gate moved")


def test_the_strip_beats_the_whole_tile_at_this_overlap(tiles):
    """Why `register_edge` crops at all, kept as a measurement.

    86 % of a whole-tile correlation at a 14 % overlap is signal that
    cannot match and can only raise the background the peak is judged
    against.
    """
    whole = phase_correlate(tiles["a"], tiles["right"], gpu=False)
    strip = register_edge(tiles["a"], tiles["right"], "horizontal",
                          expected_overlap=OVERLAP, gpu=False)
    assert strip.peak_ratio > whole.peak_ratio
    assert strip.accepted


def test_the_origin_is_never_a_placement(tiles):
    """(0,0) is the no-overlap signature, judged on the RAW peak.

    Once the unwrap resolves around an expectation, a raw origin peak
    comes back as some other number -- so a test on the unwrapped shift
    would quietly stop firing. Two identical fields are the strongest
    possible peak at the origin and must still be refused.
    """
    found = phase_correlate(tiles["a"], tiles["a"], gpu=False)
    assert found.peak_ratio > MIN_PEAK_RATIO
    assert not found.accepted, "a perfect self-match is not an adjacency"


def test_a_mis_shaped_pair_is_a_caller_error_not_a_zero_shift(tiles):
    """Silently registering two different shapes would return (0,0)."""
    with pytest.raises(ValueError, match="2-D"):
        phase_correlate(tiles["a"], tiles["a"][:-5], gpu=False)
    with pytest.raises(ValueError, match="one shape"):
        register_edge(tiles["a"], tiles["a"][:-5], "vertical", gpu=False)
    with pytest.raises(ValueError, match="vertical"):
        register_edge(tiles["a"], tiles["a"], "sideways", gpu=False)


def test_the_skew_across_the_edge_is_a_separate_allowance(tiles):
    """The stage is not square, and one tolerance for both axes fails.

    Measured on well A1: `down` is (1267, 9) and `right` is (-9, 1268).
    That ~9 px across the axis is one rigid stage geometry present in
    every edge in both directions -- so a true edge's PERPENDICULAR
    component is not zero, and a tolerance of 8 applied to both axes
    rejected 525 of 624 real adjacencies. It reads as a catastrophic
    failure of the method and is an off-by-one in a parameter.
    """
    from spacr.ops_register import SKEW_PX, phase_correlate

    assert SKEW_PX > 9, "the default allowance is below the measured skew"

    # A pair whose along-axis shift is exact and whose across-axis shift
    # is the stage's own: accepted with the separate allowance, refused
    # when one number has to serve both.
    shifted = np.roll(np.roll(tiles["a"], -STEP, axis=1), -9, axis=0)
    with_skew = phase_correlate(tiles["a"], shifted, expected=(0, STEP),
                                tolerance=(SKEW_PX, 3), gpu=False)
    without = phase_correlate(tiles["a"], shifted, expected=(0, STEP),
                              tolerance=3, gpu=False)
    assert with_skew.accepted, with_skew
    assert not without.accepted, (
        "one tolerance for both axes accepted a 9 px skew, so this test "
        "is not showing the failure it was written for")


def test_the_layout_decides_acceptance_when_it_is_given(tiles):
    """The test the real well used, and the one it replaced.

    A peak ratio asks "was that a confident peak". The layout lets the
    caller ask "did this pair land where the raster says it should",
    which is a stronger question and an answerable one. Judged that way,
    624 of 624 edges of well A1 were accepted; judged on the ratio, 29
    real adjacencies were refused.
    """
    good = register_edge(tiles["a"], tiles["right"], "horizontal",
                         expected_overlap=OVERLAP, tolerance=3, gpu=False)
    assert good.accepted
    assert abs(good.dx - STEP) <= 3

    # A pair that is not a neighbour lands somewhere else, and lands there
    # whatever its peak ratio happens to be.
    stranger = register_edge(tiles["a"], tiles["stranger"], "horizontal",
                             expected_overlap=OVERLAP, tolerance=3,
                             gpu=False)
    assert not stranger.accepted

    # And a tolerance that cannot be met refuses a pair the ratio liked,
    # which is what says the layout is deciding rather than the peak.
    strict = register_edge(tiles["a"], tiles["right"], "horizontal",
                           expected_overlap=OVERLAP, tolerance=0, gpu=False)
    assert strict.peak_ratio == good.peak_ratio
    if good.dx != STEP:
        assert not strict.accepted


def test_the_dc_bin_is_masked_so_a_shared_background_is_not_a_match(tiles):
    """The 27 true adjacencies a bare origin rule threw away.

    Two strips that share no structure still share a BACKGROUND, and
    after phase normalisation that constant is the only component they
    agree on -- so the surface peaks at the origin and a real pair reads
    as no-overlap. Asserted at the source: the surface a flat pair
    produces must not have its maximum at the origin any more.
    """
    import spacr.ops_register as reg

    flat = np.full((64, 64), 500.0)
    rng = np.random.default_rng(3)
    surface = reg._surface_numpy(flat + rng.normal(0, 1, (64, 64)),
                                 flat + rng.normal(0, 1, (64, 64)))
    assert surface[0, 0] == pytest.approx(0, abs=surface.std() * 3), (
        "the DC bin is still driving the origin")


# -- the backends ----------------------------------------------------------

def test_numpy_is_always_available_and_gpu_false_refuses_the_rest():
    """A card that exists and is busy is not a card this run may take."""
    assert available_backends(gpu=False) == ("numpy",)
    assert available_backends(gpu=True)[-1] == "numpy"


def test_an_unknown_backend_is_refused_rather_than_ignored(tiles):
    with pytest.raises(ValueError, match="unknown backend"):
        phase_correlate(tiles["a"], tiles["right"], backend="opencl")


@pytest.mark.parametrize("name", ("cupy", "torch"))
def test_every_accelerated_backend_agrees_with_numpy(tiles, name):
    """The contract: both paths, the same fixture, agreeing to a pixel.

    Torch is asked for BY NAME so this runs on a machine with no card at
    all -- `torch.fft` on the CPU is the same code with a different
    device, and CI is exactly where an unrun path would drift. CuPy
    cannot do that: it is CUDA or nothing.
    """
    if name == "cupy" and "cupy" not in available_backends(gpu=True):
        pytest.skip("this machine has no usable cupy backend")
    if name == "torch":
        pytest.importorskip("torch")
    reference = phase_correlate(tiles["a"], tiles["right"], backend="numpy")
    found = phase_correlate(tiles["a"], tiles["right"], backend=name)
    assert found.backend == name
    assert abs(found.dy - reference.dy) <= 1
    assert abs(found.dx - reference.dx) <= 1
    assert found.accepted == reference.accepted
    assert found.peak_ratio == pytest.approx(reference.peak_ratio, rel=0.05)


def test_a_backend_that_raises_falls_through_to_the_next(tiles, monkeypatch):
    """An out-of-memory on a shared card is not a failed well."""
    import spacr.ops_register as reg

    def explode(*_args, **_kwargs):
        raise RuntimeError("the card went away")

    monkeypatch.setitem(reg._SURFACES, "cupy", explode)
    monkeypatch.setattr(reg, "available_backends",
                        lambda gpu=True: ("cupy", "numpy"))
    found = phase_correlate(tiles["a"], tiles["right"], gpu=True)
    assert found.backend == "numpy"
    # But a backend the caller NAMED raises, because falling through
    # silently would answer a question nobody asked.
    with pytest.raises(RuntimeError, match="went away"):
        phase_correlate(tiles["a"], tiles["right"], backend="cupy")


# -- the whole well --------------------------------------------------------

def test_every_pair_a_layout_offers_is_registered_and_recorded(tiles):
    """Including the failures: a rejected pair belongs in `ops_geometry`."""
    from spacr.ops_layout import round_well_layout

    layout = round_well_layout(5)
    library = {0: tiles["a"], 1: tiles["below"], 2: tiles["stranger"],
               3: tiles["a"], 4: tiles["right"]}
    found = register_pairs(library, layout.pairs(),
                           expected_overlap=OVERLAP, gpu=False)
    assert set(found) == {(a, b) for a, b, _axis in layout.pairs()}
    assert all(isinstance(one, Registration) for one in found.values())
    # A callable source is the point for a real well: 333 tiles of 1480 px
    # do not need to be resident at once.
    lazily = register_pairs(library.__getitem__, layout.pairs(),
                            expected_overlap=OVERLAP, gpu=False)
    assert {key: one.shift for key, one in lazily.items()} == \
           {key: one.shift for key, one in found.items()}


def test_the_dangerous_unwrap_default_warns_and_names_the_alternative(caplog):
    """372, learned the hard way on the first real acquisition.

    `unwrap`'s docstring already says `about=0` is right only when the true
    shift is under half the axis. It was still silent when it was not, and a
    raster pitch of 1,267 px on a 1,480 px tile folds to -213 -- a number that
    docstring itself calls "both wrong and plausible". Every measurement taken
    at -213 said the tiles did not overlap. They overlap by 14.3%.

    So the default now says so, and names the other representative, which is
    the thing a caller needs in order to act.
    """
    import logging

    import numpy as np

    from spacr.ops_register import phase_correlate

    rng = np.random.default_rng(0)
    field = rng.random((256, 256)).astype(np.float32)
    # A neighbour 200 px down: more than an eighth of the axis, so the fold
    # is ambiguous and the caller should be told.
    shifted = np.roll(field, 200, axis=0)

    with caplog.at_level(logging.WARNING, logger="spacr.ops_register"):
        result = phase_correlate(field, shifted)

    assert any("expected=<pitch>" in r.message or "expected=<pitch>" in r.getMessage()
               for r in caplog.records), caplog.text
    # and it names the OTHER representative, not just the complaint
    assert any(str(result.dy + 256) in r.getMessage()
               or str(result.dy - 256) in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_small_shift_does_not_warn(caplog):
    """Two cycles of ONE field are unambiguous, and must stay quiet.

    A warning that fires on the legitimate case is a warning people learn to
    ignore, which would cost more than the one it exists to prevent.
    """
    import logging

    import numpy as np

    from spacr.ops_register import phase_correlate

    rng = np.random.default_rng(1)
    field = rng.random((256, 256)).astype(np.float32)
    drifted = np.roll(field, 3, axis=1)

    with caplog.at_level(logging.WARNING, logger="spacr.ops_register"):
        phase_correlate(field, drifted)

    assert not [r for r in caplog.records if "expected=<pitch>" in r.getMessage()], \
        caplog.text


def test_passing_the_pitch_silences_it(caplog):
    """A caller who has said what they meant should not be nagged."""
    import logging

    import numpy as np

    from spacr.ops_register import phase_correlate

    rng = np.random.default_rng(2)
    field = rng.random((256, 256)).astype(np.float32)
    shifted = np.roll(field, 200, axis=0)

    # DERIVED, NOT ASSUMED. np.roll's sense and this module's shift sign are
    # not the same question, and hard-coding one made this test assert 200
    # against a correct 312. Ask for the default answer, then ask for the
    # other representative of it -- which is exactly what a caller does once
    # the warning has told them the two candidates.
    default = phase_correlate(field, shifted)
    other = default.dy + 256 if default.dy < 0 else default.dy - 256

    # caplog accumulates for the whole test, and the probe call above warns
    # on purpose. Clear it, or this asserts against the warning it asked for.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="spacr.ops_register"):
        result = phase_correlate(field, shifted, expected=(other, 0))

    assert result.dy == other
    assert not [r for r in caplog.records if "expected=<pitch>" in r.getMessage()], \
        caplog.text
