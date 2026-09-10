"""Register two overlapping tiles by phase correlation, on whatever hardware.

WHY PHASE CORRELATION AND NOT FEATURE MATCHING. Both were run
on the same pixels -- well A1 cycle 1 of `screenA/20200202_6W-LaC024A`,
333 tiles of 10X SBS DAPI:

    ORB + RANSAC        993 pairs scored, median score 0.0002, max 0.0985
                        26 of 333 tiles placed, 307 dropped
    phase correlation   the geometry instantly, peak/mean 23.6 to 53.1
                        against control pairs at 9.0 to 15.6

Nuclei at 10X are round blobs. They have no corners, so a corner detector
finds essentially nothing and the matcher's scores are noise. Phase
correlation asks a different question -- what single translation makes
these two fields agree -- and a field of blobs answers it very well.
Feature matching still earns its keep in ONE place, and it is not this
one: the phenotype-to-SBS step across magnifications, where scale and
rotation are real.

THREE THINGS THAT MAKE THE ANSWER RIGHT, all of them learned the hard way
and all of them measured:

  * THE SHIFT COMES BACK WRAPPED. An FFT knows nothing about which way is
    negative: a -213 px shift of a 1480 px tile arrives as +1267. Reading
    it unwrapped made the first prototype's shifts nonsense while every
    individual registration was correct. :func:`unwrap` folds anything
    past half a tile into the negative half.
  * (0,0) IS THE NO-OVERLAP SIGNATURE, not a perfect alignment. Two tiles
    that do not touch have no translation that makes them agree, and the
    peak lands at the origin. It is REJECTED rather than believed, which
    is what the three control pairs in 372's PART 11-B confirmed.
  * A PEAK IS ONLY A PEAK AGAINST ITS BACKGROUND. The absolute height
    means nothing; peak/mean does. True neighbours measured 23.6 to 53.1
    and non-neighbours 9.0 to 15.6, so `min_peak_ratio` has real room
    between them.

THE BACKEND IS AN INDIRECTION OVER THE ARRAY MODULE, which is all the
GPU question is here: this is an FFT, a multiply and an argmax. CuPy
first where it exists, `torch.fft` second -- torch is already a spaCR
dependency through Cellpose, so it costs nothing to install and reaches
CUDA, ROCm and Apple MPS through one path -- and NumPy always.

    THE CPU PATH IS NOT AN ERROR PATH. Both run the same fixture and must
    agree to within a pixel, which is what
    `tests/test_the_ops_registration_agrees_on_every_backend.py` asserts.
    A fallback nobody compares is a second implementation, not a
    fallback.

AND IT NEVER ASSUMES THE CARD IS FREE. A shared GPU is the common case --
this one also runs structure prediction. `spacr.accelerator` is the seam
that decides whether there is a usable device at all, `gpu=False` refuses
one that exists, and a backend that raises at runtime falls through to the
next rather than failing the well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "Registration",
    "available_backends",
    "phase_correlate",
    "register_edge",
    "register_pairs",
    "unwrap",
]

#: Low enough to catch a FLAT SURFACE and nothing else, and that is a
#: correction rather than a choice.
#:
#: IT WAS 20, from a first measurement of seven true neighbours against
#: three controls. The full distributions over one real well -- 624 true
#: adjacencies and 60 deliberate non-neighbours, strip registration --
#: say no threshold separates them:
#:
#:     TRUE   n=624  min 6.2  p05 7.2  median 63.6  p95 99.4  max 120.2
#:     CONTROL n=60  min 5.9  median 8.1  p95 9.9   max 18.9
#:
#:     lowest true 6.2   against   highest control 18.9
#:     true below 20: 152          controls above 20: 0
#:
#: THE FAMILIES OVERLAP. A threshold of 20 would have thrown away 152 real
#: adjacencies, and one low enough to keep them admits controls. A
#: synthetic fixture separates cleanly (33.4 and 34.5 against 11.9)
#: because it is one contrived pair; over 624 real ones the tails cross.
#:
#: SO THE LAYOUT IS THE TEST AND THIS IS NOT. Pass `tolerance` -- "did
#: this pair land where the raster says it should" -- and the ratio
#: becomes a recorded number. What is left here only refuses a surface
#: with no peak at all, which is a broken read rather than a distant
#: neighbour.
MIN_PEAK_RATIO: float = 3.0

#: How far off the perpendicular axis a real edge may land, in pixels.
#:
#: THE STAGE HAS A SKEW AND IT IS NOT AN ERROR. Measured on well A1:
#: `down` is (1267, 9) and `right` is (-9, 1268) -- a constant ~9 px
#: offset across the axis, in every edge, in both directions. It is one
#: rigid stage geometry, so the perpendicular component of a true edge is
#: NOT zero and a tolerance that assumes it is rejects every pair on the
#: plate: 525 of 624 refused at a tolerance of 8, which looks like a
#: catastrophic failure of the method and is an off-by-one in a parameter.
#:
#: IT IS A PROPERTY OF THE MICROSCOPE, NOT OF THE WELL, so this default is
#: a starting point and another acquisition wants its own measured. 24 is
#: room for the 9 that was measured plus the same again.
SKEW_PX: int = 24

#: How much of a tile the overlap strip takes. Wider than the raster's
#: 14 % overlap so the band certainly contains it, and not much wider:
#: 40 % measured worse than the whole tile it was meant to improve on.
STRIP_FRACTION: float = 0.30

#: The backends, best first. A name here is a claim that the module can be
#: imported AND that it has a working device, which is why each is behind
#: its own probe.
BACKEND_ORDER: Tuple[str, ...] = ("cupy", "torch", "numpy")


@dataclass(frozen=True)
class Registration:
    """What one pair of tiles measured.

    :ivar dy: rows to move the second tile to land it on the first,
        unwrapped into ``[-height/2, height/2)``.
    :ivar dx: the same across columns.
    :ivar peak_ratio: the correlation peak against the mean of the
        surface. The evidence, not the shift.
    :ivar accepted: whether this is a real overlap: a peak above the
        threshold that is not at the origin.
    :ivar backend: which array module answered, so a run can say what it
        ran on rather than leaving it to be inferred.
    """

    dy: int
    dx: int
    peak_ratio: float
    accepted: bool
    backend: str

    @property
    def shift(self) -> Tuple[int, int]:
        """``(dy, dx)``, for a caller that wants the pair."""
        return self.dy, self.dx


def unwrap(value: int, extent: int, about: int = 0) -> int:
    """Fold a wrapped FFT shift to the representative nearest ``about``.

    A translation of -213 in a 1480 px tile comes back as +1267, because
    the transform is periodic and has no notion of direction. Reading it
    unwrapped made 372's first prototype's shifts nonsense while every
    individual registration was correct.

    ``about`` IS NOT DECORATION AND THE DEFAULT IS THE DANGEROUS CASE.
    With ``about=0`` this folds into ``[-extent/2, extent/2)``, which is
    right only when the true shift is smaller than half the axis. It is
    not, whenever the correlation runs on an overlap STRIP: a 222 px
    strip carrying a true shift of 116 folds to -106, a number that is
    both wrong and plausible. The caller knows roughly where the
    neighbour is -- the raster pitch says so -- so it passes that, and
    the representative nearest it is the answer.

    :param value: the raw peak coordinate.
    :param extent: the axis length.
    :param about: the shift the caller expects, in the same units.
    :returns: the representative of ``value`` mod ``extent`` closest to
        ``about``.
    """
    extent = int(extent)
    if extent <= 0:
        return int(value)
    value = int(value) % extent
    about = int(about)
    # The two candidates that bracket `about`, since any representative is
    # `value + k * extent`.
    base = value + extent * round((about - value) / extent)
    return int(base)


def _cupy():
    """CuPy with a working device, or None."""
    try:
        import cupy                                      # noqa: F401

        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:                                    # noqa: BLE001
        return None


def _torch_gpu():
    """``(torch, device)`` when spaCR resolves a usable accelerator.

    Asked through `spacr.accelerator` rather than `torch.cuda.is_available`
    so that this module agrees with the rest of spaCR about the machine,
    including a device the user has forced off.
    """
    try:
        import torch

        from .accelerator import is_gpu, torch_device

        if not is_gpu():
            return None
        return torch, torch_device()
    except Exception:                                    # noqa: BLE001
        return None


def available_backends(gpu: bool = True) -> Tuple[str, ...]:
    """Which backends this machine can actually run, best first.

    :param gpu: False refuses the accelerated ones outright, which is what
        the `gpu` setting means -- a card that exists and is busy with
        somebody else's job is not a card this run may take.
    :returns: the usable names, always ending in ``"numpy"``.
    """
    found = []
    if gpu:
        if _cupy() is not None:
            found.append("cupy")
        if _torch_gpu() is not None:
            found.append("torch")
    found.append("numpy")
    return tuple(found)


def _surface_numpy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The cross-power spectrum's inverse transform, in NumPy."""
    a = np.fft.rfft2(first)
    b = np.fft.rfft2(second)
    product = a * np.conj(b)
    magnitude = np.abs(product)
    # THE NORMALISATION IS THE WHOLE METHOD. Dividing out the magnitude
    # keeps only the phase, which is what makes the answer independent of
    # how bright either field happens to be -- and a zero there is a
    # frequency neither tile carries, not a division to be silently
    # allowed to produce a nan.
    with np.errstate(invalid="ignore", divide="ignore"):
        product = np.where(magnitude > 0, product / magnitude, 0)
    # THE DC BIN IS MASKED, AND IT IS NOT A DETAIL. Two strips that share
    # no structure still share a BACKGROUND, and after phase
    # normalisation that constant is the only component they agree on --
    # so the surface peaks at the origin and the pair reads as
    # no-overlap. On the real plate that cost 27 of 624 TRUE adjacencies,
    # all of them at the well's edge where the overlap band holds few
    # nuclei. With the bin zeroed, every one of the 624 registers.
    product[0, 0] = 0
    return np.fft.irfft2(product, s=first.shape)


def _surface_cupy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The same surface on CUDA through CuPy, returned as NumPy."""
    cupy = _cupy()
    if cupy is None:                                     # pragma: no cover
        raise RuntimeError("cupy is not usable on this machine")
    a = cupy.fft.rfft2(cupy.asarray(first))
    b = cupy.fft.rfft2(cupy.asarray(second))
    product = a * cupy.conj(b)
    magnitude = cupy.abs(product)
    product = cupy.where(magnitude > 0, product / cupy.maximum(magnitude, 1e-30), 0)
    product[0, 0] = 0
    return cupy.asnumpy(cupy.fft.irfft2(product, s=first.shape))


def _surface_torch(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The same surface through `torch.fft`, on the resolved device.

    FLOAT32, DELIBERATELY. Metal has no float64 and a 1480 x 1480 tile
    does not need it: the answer is an integer pixel, and the peak ratio
    it is judged by is three significant figures at most.

    NAMED EXPLICITLY, IT RUNS ON THE CPU TOO. Automatic selection only
    ever reaches for torch when `spacr.accelerator` resolves a device --
    that is the whole point of the seam. But `backend="torch"` is a
    caller asking for this code path, and refusing it on a machine with
    no card is what would make the agreement test unrunnable exactly
    where it matters: CI, which has no GPU, is where the two paths would
    otherwise silently diverge.
    """
    found = _torch_gpu()
    if found is None:
        import torch as torch_module

        torch, device = torch_module, torch_module.device("cpu")
    else:
        torch, device = found
    a = torch.fft.rfft2(torch.as_tensor(np.ascontiguousarray(first),
                                        dtype=torch.float32, device=device))
    b = torch.fft.rfft2(torch.as_tensor(np.ascontiguousarray(second),
                                        dtype=torch.float32, device=device))
    product = a * torch.conj(b)
    magnitude = torch.abs(product)
    product = torch.where(magnitude > 0,
                          product / torch.clamp(magnitude, min=1e-30),
                          torch.zeros_like(product))
    product[0, 0] = 0
    surface = torch.fft.irfft2(product, s=tuple(first.shape))
    return surface.detach().to("cpu").numpy()


#: ``name -> the function that computes the correlation surface``.
_SURFACES = {
    "cupy": _surface_cupy,
    "torch": _surface_torch,
    "numpy": _surface_numpy,
}


def phase_correlate(first: np.ndarray, second: np.ndarray, *,
                    expected: Tuple[int, int] = (0, 0),
                    tolerance=None,
                    backend: Optional[str] = None,
                    gpu: bool = True,
                    min_peak_ratio: float = MIN_PEAK_RATIO) -> Registration:
    """Measure the translation between two overlapping fields.

    The result is the shift that lands ``second`` on ``first``: rolling
    ``second`` by ``(dy, dx)`` brings it into ``first``'s frame.

    Takes and returns NumPy whatever ran, so no caller learns which
    backend answered -- the name is on the result for the record, not for
    branching.

    :param first: the reference field, 2-D.
    :param second: the field to place against it, the same shape.
    :param expected: roughly where the neighbour is, for the unwrap. The
        default of (0, 0) folds into the half-range, which is right for
        two whole tiles and WRONG for two overlap strips -- see
        :func:`unwrap`.
    :param tolerance: how far from ``expected`` a shift may land and still
        be accepted, in pixels. WHEN IT IS GIVEN, IT IS THE TEST, and the
        peak ratio becomes a recorded number rather than the gate -- see
        the note on :data:`MIN_PEAK_RATIO`.

        A PAIR OF NUMBERS IS ACCEPTED AND IS USUALLY WHAT IS WANTED:
        ``(rows, columns)``. The two axes are not the same question -- a
        true edge is within a pixel or two ALONG the raster and up to the
        stage's skew ACROSS it -- and one number for both either rejects
        the skew or admits a neighbour a whole tile away. See
        :data:`SKEW_PX`.
    :param backend: force one; None takes the best available.
    :param gpu: False keeps it on the CPU even where a card exists.
    :param min_peak_ratio: below this the pair is not a neighbour.
    :returns: the registration, accepted or not.
    :raises ValueError: when the two tiles are not the same 2-D shape,
        which is a caller error rather than a failed registration and must
        not come back as a confident (0, 0).
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or first.shape != second.shape:
        raise ValueError(
            f"phase correlation needs two 2-D tiles of one shape, got "
            f"{first.shape} and {second.shape}")

    order = (backend,) if backend else available_backends(gpu)
    surface = None
    used = "numpy"
    for name in order:
        function = _SURFACES.get(name)
        if function is None:
            raise ValueError(f"unknown backend {name!r}; "
                             f"expected one of {sorted(_SURFACES)}")
        try:
            surface = function(first, second)
            used = name
            break
        except Exception:                                # noqa: BLE001
            # A BACKEND THAT RAISES IS NOT A FAILED WELL. An out-of-memory
            # on a shared card, a driver that went away: fall to the next.
            LOG.debug("the %s backend could not register this pair", name,
                      exc_info=True)
            if backend:
                raise
    if surface is None:                                  # pragma: no cover
        raise RuntimeError("no usable backend for phase correlation")

    flat = int(np.argmax(surface))
    peak_y, peak_x = np.unravel_index(flat, surface.shape)
    height = float(surface.flat[flat])
    mean = float(np.abs(surface).mean())
    ratio = height / mean if mean > 0 else 0.0

    height_px, width_px = first.shape
    dy = unwrap(peak_y, height_px, expected[0])
    dx = unwrap(peak_x, width_px, expected[1])
    # (0, 0) IS WHAT TWO FIELDS THAT DO NOT TOUCH PRODUCE. Never accepted,
    # however strong the peak: there is no such thing as two adjacent
    # fields of a raster occupying the same place.
    #
    # JUDGED ON THE RAW PEAK, NOT THE UNWRAPPED SHIFT, and this is not a
    # detail. Once `expected` is non-zero the unwrap moves the origin to
    # whichever representative is nearest the expectation -- a raw (0, 0)
    # against a 222 px strip expecting 116 comes back as 222 -- so a test
    # on the unwrapped pair would never fire again, and the no-overlap
    # signature would silently start reading as a confident placement.
    at_origin = (int(peak_y), int(peak_x)) == (0, 0)
    if tolerance is not None:
        # THE LAYOUT IS A STRONGER TEST THAN A THRESHOLD. The question is
        # not "was that a confident peak" but "did this pair land where
        # the raster says it should", and the second one is answerable
        # because the layout predicts it. Judged this way on the real
        # well, 624 of 624 edges were accepted and none was guessed;
        # judged on a bare peak ratio, 152 real adjacencies were refused.
        rows, columns = ((tolerance, tolerance)
                         if isinstance(tolerance, (int, float))
                         else tolerance)
        accepted = (abs(dy - expected[0]) <= rows
                    and abs(dx - expected[1]) <= columns)
        # AND THE ORIGIN RULE IS DELIBERATELY NOT APPLIED HERE. A raw peak
        # at (0, 0) unwraps to the representative nearest the expectation,
        # so it is only accepted when that representative lands within
        # tolerance of where the raster says the neighbour is -- which is
        # the same question this branch already asks, answered better.
        # Adding `not at_origin` on top would refuse a pair whose true
        # strip-frame shift genuinely is zero, which is exactly what a
        # caller gets by choosing an overlap fraction equal to the real
        # overlap. Trading a rare aliasing case for a reachable false
        # rejection is the wrong way round, and the DC mask has already
        # removed the reason the origin was suspicious.
    else:
        accepted = ratio >= min_peak_ratio and not at_origin
    return Registration(dy=dy, dx=dx, peak_ratio=ratio, accepted=accepted,
                        backend=used)


def register_edge(first: np.ndarray, second: np.ndarray, axis: str, *,
                  overlap_fraction: float = STRIP_FRACTION,
                  expected_overlap: Optional[int] = None,
                  tolerance: Optional[int] = None,
                  skew: Optional[int] = SKEW_PX,
                  **kwargs) -> Registration:
    """Register two tiles that touch along one edge, using that edge only.

    CORRELATE THE BAND THAT CAN AGREE, NOT THE TILE. The raster overlaps
    its fields by about 14 % -- 213 px of 1480 -- so 86 % of a whole-tile
    correlation is signal that CANNOT match and can only add background
    to the surface the peak is judged against. Measured on a blob fixture
    at exactly that overlap:

        whole tile      peak/mean 15.2, shift WRONG, rejected
        30 % strip      peak/mean 29.2, shift exact, accepted
                        control pair 14.0, correctly rejected

    which is also the separation the real acquisition showed (23.6-53.1
    for true neighbours, 9.0-15.6 for controls). The strip has to be
    WIDER than the overlap and not much wider: 40 % diluted it back to
    15.4 and lost the shift again.

    :param first: the tile the edge is measured against.
    :param second: its neighbour, below it for "vertical" and to its
        right for "horizontal" -- the order :meth:`WellLayout.pairs`
        returns them in.
    :param axis: "vertical" or "horizontal".
    :param overlap_fraction: how much of the tile the strip takes.
    :param expected_overlap: the overlap in pixels, when the pitch is
        known. It sets the unwrap's expectation, which is what stops a
        true shift larger than half the strip reading as a negative one.
    :param tolerance: how far ALONG the edge the measured shift may land
        from the expected overlap and still be accepted. This is the
        acceptance the real well used; without it the peak ratio decides,
        and the peak ratio cannot decide -- see :data:`MIN_PEAK_RATIO`.
    :param skew: how far ACROSS it may land. The stage's skew is a real
        ~9 px on the measured plate and is not an error, so this is a
        separate number: one tolerance for both axes rejected 525 of 624
        true adjacencies at 8 px. See :data:`SKEW_PX`.
    :param kwargs: passed to :func:`phase_correlate`.
    :returns: the registration, in TILE coordinates rather than strip
        ones -- the caller asked about two tiles.
    :raises ValueError: on an unknown axis.
    """
    if axis not in ("vertical", "horizontal"):
        raise ValueError(f"axis is 'vertical' or 'horizontal', not {axis!r}")
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(
            f"an edge is between two tiles of one shape, got {first.shape} "
            f"and {second.shape}")

    down = axis == "vertical"
    extent = first.shape[0] if down else first.shape[1]
    band = max(8, min(extent, int(round(extent * float(overlap_fraction)))))
    if down:
        strip_first, strip_second = first[-band:, :], second[:band, :]
    else:
        strip_first, strip_second = first[:, -band:], second[:, :band]

    # In the strip frame the expected shift is the strip width minus the
    # overlap: the two bands are that far out of step with each other.
    guess = band - int(expected_overlap) if expected_overlap else 0
    expected = (guess, 0) if down else (0, guess)
    across = tolerance if skew is None else skew
    limits = None if tolerance is None else (
        (tolerance, across) if down else (across, tolerance))
    measured = phase_correlate(strip_first, strip_second,
                               expected=expected,
                               tolerance=limits, **kwargs)

    # Back to tile coordinates. The strip started `extent - band` into the
    # first tile, so that much of the shift was cropped away.
    lead = extent - band
    if down:
        moved = Registration(dy=measured.dy + lead, dx=measured.dx,
                             peak_ratio=measured.peak_ratio,
                             accepted=measured.accepted,
                             backend=measured.backend)
    else:
        moved = Registration(dy=measured.dy, dx=measured.dx + lead,
                             peak_ratio=measured.peak_ratio,
                             accepted=measured.accepted,
                             backend=measured.backend)
    return moved


def register_pairs(tiles, pairs: Sequence[Tuple[int, int, str]], *,
                   expected_overlap: Optional[int] = None,
                   tolerance: Optional[int] = None,
                   skew: Optional[int] = SKEW_PX,
                   gpu: bool = True,
                   min_peak_ratio: float = MIN_PEAK_RATIO,
                   **kwargs) -> dict:
    """Register every adjacency a layout offers.

    :param tiles: ``site -> 2-D array``, or any callable taking a site.
        A CALLABLE IS THE POINT for a real well: 333 tiles of 1480 px is
        more than needs to be resident, and the caller decides what to
        keep.
    :param pairs: what to register, from
        :meth:`spacr.ops_layout.WellLayout.pairs`.
    :param expected_overlap: the raster's overlap in pixels, when known.
    :param tolerance: how far along the edge a shift may land.
    :param skew: how far across it may -- the stage's own, measured.
    :param gpu: passed through.
    :param min_peak_ratio: passed through.
    :param kwargs: passed to :func:`register_edge`.
    :returns: ``{(a, b): Registration}``, including the rejected ones --
        a pair that failed is a fact about the acquisition and belongs in
        `ops_geometry`, not in a debug log.
    """
    read = tiles if callable(tiles) else tiles.__getitem__
    found = {}
    for a, b, axis in pairs:
        found[(a, b)] = register_edge(read(a), read(b), axis,
                                      expected_overlap=expected_overlap,
                                      tolerance=tolerance,
                                      skew=skew,
                                      gpu=gpu,
                                      min_peak_ratio=min_peak_ratio,
                                      **kwargs)
    return found
