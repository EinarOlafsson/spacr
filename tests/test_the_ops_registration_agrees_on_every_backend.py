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


def test_a_pair_that_does_not_touch_is_rejected(tiles):
    """Non-neighbours measured 9.0-15.6 against neighbours at 23.6-53.1."""
    found = register_edge(tiles["a"], tiles["stranger"], "horizontal",
                          expected_overlap=OVERLAP, gpu=False)
    assert not found.accepted
    assert found.peak_ratio < MIN_PEAK_RATIO


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
