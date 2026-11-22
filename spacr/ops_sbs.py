"""Decode in-situ sequencing barcodes out of an image stack.

Purpose and scope
=================
Optical pooled screening reads the guide barcode OUT OF THE IMAGES. Each
sequencing cycle stains four channels, one per base, and a spot's base in that
cycle is whichever channel is brightest -- once the channels have been put on
a common scale. Read one base per cycle at the same spot and the cycles spell
the barcode.

:mod:`spacr.sequencing` is NOT the back half of this and must not be pointed
at it. That module decodes FASTQ reads from an NGS run; there is no FASTQ here
and no sequencer, only pixels.

Rationale for each step
=======================
Three of these carry nearly all of the correctness risk, and all three are
counter-intuitive enough that implementing them from a description of the
pipeline rather than from a working one produces something that runs and is
wrong:

* **A read is not a bright spot.** It is a spot whose base CHANGES between
  cycles. So the location estimate is the standard deviation ACROSS CYCLES,
  averaged over channels -- a spot bright in every cycle is a piece of dirt or
  an autofluorescent blob and contributes no variance.

* **The brightest raw channel is not the base.** Every channel has its own
  gain and background, and on a real plate both drift from cycle to cycle,
  so the raw argmax is biased toward whichever channel is brightest overall.
  Each cycle's channels are divided by their median over the reads first.
  Unmixing the dyes' bleed-through is available on top of that; on the first
  real plate it lowered the library match.

* **An ambiguous read is worse than a missing one.** Error correction against
  the guide library only ever corrects to a UNIQUE closest match. A read that
  is equally close to two barcodes is discarded, because a wrongly assigned
  guide silently corrupts a screen's results while a dropped one only costs
  statistical power.

The method here follows brieflow (Cheeseman lab; github.com/cheeseman-lab/
brieflow, MIT, Copyright 2025 Massachusetts Institute of Technology), whose
source was read for the three points above rather than reconstructed from its
stage names.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The four bases, in the channel order the stack is expected to carry.
BASES = ("G", "T", "A", "C")


def estimate_read_locations(stack: np.ndarray) -> np.ndarray:
    """Return a map of likely read positions, scored by variance across cycles.

    A sequencing spot changes colour from cycle to cycle, which is exactly
    what a constant piece of debris does not do. Taking the standard deviation
    over the CYCLE axis and then the mean over channels therefore scores
    "something here changed" rather than "something here is bright", and the
    brightest object in a field is very often not a read.

    With a single cycle there is no cycle-to-cycle variance to measure, so the
    standard deviation is taken across channels instead -- a spot that is one
    colour rather than grey. That is a weaker signal and a one-cycle
    experiment is a weaker experiment; it is supported because the alternative
    is failing on a legitimate input.

    :param stack: ``(cycles, channels, Y, X)`` intensities.
    :returns: a ``(Y, X)`` float32 map, high where a read is likely.
    :raises ValueError: if ``stack`` is not four-dimensional.
    """
    array = np.asarray(stack, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(
            f"stack must be (cycles, channels, Y, X); got shape {array.shape}"
        )
    if array.shape[0] == 1:
        return array[0].std(axis=0).astype(np.float32, copy=False)
    return array.std(axis=0).mean(axis=0).astype(np.float32, copy=False)


def find_peaks(score: np.ndarray, *, min_distance: int = 3,
               threshold: Optional[float] = None,
               gpu: bool = True) -> np.ndarray:
    """Local maxima of ``score``, as an ``(N, 2)`` array of ``(y, x)``.

    A plain "is this pixel the largest in its window" test returns a clump of
    neighbouring pixels for one spot, so the window maximum is compared for
    EQUALITY with the pixel and ties are broken by taking the first -- which
    is what makes one spot yield one peak.

    :param score: the map from :func:`estimate_read_locations`.
    :param min_distance: half-width of the suppression window, in pixels. Two
        reads closer than this cannot both be found, which is a property of
        the optics rather than of this function.
    :param threshold: ignore maxima below this. ``None`` keeps every local
        maximum and leaves the decision to the caller, which is the right
        default because the useful cutoff depends on the stain.
    :param gpu: let the windowed maximum run on the card where there is a
        usable one. It is a max-pool, which is the single most expensive
        step in the decode chain on a well-sized field and the operation a
        GPU exists for. The result is identical either way -- see
        `tests/test_the_ops_primitives_agree_on_every_backend.py`.
    :returns: peak coordinates, strongest first.
    """
    from .ops_accel import maximum_filter

    field = np.asarray(score, dtype=np.float32)
    window = 2 * int(min_distance) + 1
    local_max = maximum_filter(field, window, gpu=gpu)
    hits = (field == local_max) & (field > 0)
    if threshold is not None:
        hits &= field >= float(threshold)
    ys, xs = np.nonzero(hits)
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(field[ys, xs])[::-1]
    return np.stack([ys[order], xs[order]], axis=1).astype(np.int64)


def extract_bases(stack: np.ndarray, peaks: np.ndarray, *,
                  window: int = 1) -> np.ndarray:
    """Per-cycle, per-channel intensity at each peak.

    The maximum over a small window is taken rather than the single pixel,
    because the cycles are registered to within a pixel or so and reading the
    exact centre would sample the shoulder of the spot in whichever cycle
    drifted. A window of one -- a 3x3 -- is enough for that and small enough
    not to swallow a neighbour.

    :param stack: ``(cycles, channels, Y, X)`` intensities.
    :param peaks: ``(N, 2)`` array of ``(y, x)`` from :func:`find_peaks`.
    :param window: half-width of the sampling window, in pixels.
    :returns: ``(N, cycles, channels)`` float32 intensities.
    """
    array = np.asarray(stack, dtype=np.float32)
    n_cycles, n_channels, height, width = array.shape
    coords = np.asarray(peaks, dtype=np.int64).reshape(-1, 2)
    out = np.zeros((coords.shape[0], n_cycles, n_channels), np.float32)
    for i, (y, x) in enumerate(coords):
        y0, y1 = max(0, y - window), min(height, y + window + 1)
        x0, x1 = max(0, x - window), min(width, x + window + 1)
        out[i] = array[:, :, y0:y1, x0:x1].max(axis=(2, 3))
    return out


#: The read modes of ``ops_read_mode`` (item 952), the first being the
#: default: ``log`` reads the -LoG(1 px) stack's 3 x 3 maximum and calls it
#: after the per-cycle median normalisation; ``raw`` reads the unfiltered
#: stack's maximum over a disk of :data:`_RAW_READ_RADIUS` px and calls the
#: brightest channel as it is; ``auto`` chooses one of the two per well from
#: decoy calls.
_READ_MODES = ("log", "raw", "auto")

#: The radius, in pixels, of the disk a raw read takes its maximum over:
#: PERISCOPE's read (item 952, BMC edge ablation round 3).
_RAW_READ_RADIUS = 3


def _read_mode_setting(settings) -> str:
    """The ``ops_read_mode`` of a run's settings, checked.

    :param settings: the run's settings.
    :returns: one of :data:`_READ_MODES`; empty, missing or ``'None'`` is
        ``log``, the OPS barcode decode's default (the spacer-prefix decode
        has its own, :func:`spacr.ops_prefix._prefix_read_mode`).
    :raises ValueError: for any other value.
    """
    value = settings.get("ops_read_mode") if settings is not None else None
    mode = str(value or _READ_MODES[0]).strip().lower()
    if mode == "none":
        mode = _READ_MODES[0]
    if mode not in _READ_MODES:
        raise ValueError(f"ops_read_mode must be one of {list(_READ_MODES)}; "
                         f"got {value!r}")
    return mode


def _reflected(index: np.ndarray, size: int) -> np.ndarray:
    """Pixel indices mirrored back into ``[0, size)`` as SciPy's 'reflect' mode does.

    :param index: indices at most one radius outside the axis.
    :param size: the axis length.
    :returns: ``-1`` -> ``0``, ``size`` -> ``size - 1``, and so on.
    """
    index = np.where(index < 0, -index - 1, index)
    return np.where(index >= size, 2 * size - 1 - index, index)


def _read_values(stack: np.ndarray, filtered: Optional[np.ndarray],
                 peaks: np.ndarray, mode: str, *,
                 radius: int = _RAW_READ_RADIUS) -> np.ndarray:
    """The per-cycle, per-channel intensity a read is called from (item 952).

    :param stack: ``(cycles, channels, Y, X)`` unfiltered intensities; read
        in ``raw`` mode only.
    :param filtered: the same stack after the -LoG(1 px) filter, clipped at
        0; read in ``log`` mode only.
    :param peaks: ``(N, 2)`` ``(y, x)`` spot positions.
    :param mode: ``log`` (:func:`extract_bases` on ``filtered``, a 3 x 3
        maximum) or ``raw`` (the maximum of ``stack`` over the disk of
        ``radius`` px around each peak, the pixels past the border mirrored
        as :func:`scipy.ndimage.maximum_filter` mirrors them).
    :param radius: the raw disk's radius.
    :returns: ``(N, cycles, channels)`` float32.
    :raises ValueError: for another mode.
    """
    if mode == "log":
        return extract_bases(filtered, peaks, window=1)
    if mode != "raw":
        raise ValueError(f"a read is taken in 'log' or 'raw' mode; got {mode!r}")
    array = np.asarray(stack)
    n_cycles, n_channels, height, width = array.shape
    coords = np.asarray(peaks, dtype=np.int64).reshape(-1, 2)
    if not len(coords):
        return np.zeros((0, n_cycles, n_channels), np.float32)
    reach = int(radius)
    dy, dx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    disk = dy ** 2 + dx ** 2 <= reach * reach
    out = np.full((n_cycles, n_channels, len(coords)), -np.inf, np.float32)
    for oy, ox in zip(dy[disk], dx[disk]):
        ys = _reflected(coords[:, 0] + oy, height)
        xs = _reflected(coords[:, 1] + ox, width)
        np.maximum(out, array[:, :, ys, xs], out=out)
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def _raw_called_bases(values: np.ndarray, bases: Sequence[str] = BASES
                      ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """:func:`called_bases` of a raw read: the brightest channel, as it is.

    :param values: ``(N, cycles, channels)`` from :func:`_read_values` in
        ``raw`` mode.
    :param bases: the letter for each channel, in channel order.
    :returns: :func:`called_bases`'s ``(data, barcodes, margin)`` with no
        per-cycle median normalisation, no unmixing and no library
        calibration.
    """
    return called_bases(values, bases=tuple(bases), compensate=False,
                        normalise=False, gpu=False)


def compensate_crosstalk(values: np.ndarray, *,
                         method: str = "percentile",
                         percentile: float = 95.0,
                         gpu: bool = True) -> np.ndarray:
    """Correct base intensities for dye bleed-through between channels.

    Without correction the raw argmax is biased: each dye emits into its
    neighbours' channels, so a channel that is bright overall wins
    comparisons it should lose, and the bias is systematic rather than noise:
    it mis-calls the same base everywhere.

    The correction is fitted FROM THE DATA rather than from a calibration
    file, because it depends on the stain, the filters and the exposure --
    which are properties of the run, not of the instrument. For each channel,
    the spots where that channel dominates are found, and their mean vector
    becomes that channel's axis; the matrix of those axes, inverted, maps
    observed intensity back onto base identity.

    :param values: ``(N, cycles, channels)`` from :func:`extract_bases`.
    :param method: ``"percentile"`` takes spots above ``percentile`` in a
        channel; ``"median"`` takes spots where the channel is the argmax.
        Percentile is the more robust of the two when one base is rare, which
        is the case a median call gets wrong. ``"argmax_median"`` is
        brieflow's ``transform_medians``: each channel is first divided by
        its median over the spots, each channel's axis is the MEDIAN of the
        spots where it is the argmax, and the axes are scaled to sum to one
        rather than to unit length, which keeps the corrected channels on
        the scale the argmax compares. On 116 fields of the Funk et al. 2022
        plates, after the per-cycle normalisation, it raised the library-exact
        read rate from 0.558 to 0.574 and lowered the decoy match rate.
    :param percentile: the cutoff for ``"percentile"``.
    :param gpu: let the unmixing multiply run on the card. One tiny matrix
        applied to every spot of every cycle is a large multiply by a small
        operand, which is the shape that goes to a card well.
    :returns: corrected intensities, same shape.
    :raises ValueError: on an unknown ``method``.
    """
    if method not in ("percentile", "median", "argmax_median"):
        raise ValueError(f"unknown method {method!r}")
    data = np.asarray(values, dtype=np.float32)
    n_channels = data.shape[-1]
    flat = data.reshape(-1, n_channels)
    if flat.shape[0] < n_channels:
        return data
    if method == "argmax_median":
        return _argmax_median_unmix(flat).reshape(data.shape)

    axes = np.zeros((n_channels, n_channels), np.float32)
    for channel in range(n_channels):
        if method == "percentile":
            cut = np.percentile(flat[:, channel], percentile)
            chosen = flat[flat[:, channel] >= cut]
        else:
            chosen = flat[flat.argmax(axis=1) == channel]
        if chosen.shape[0] == 0:
            axes[channel, channel] = 1.0
            continue
        vector = chosen.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        axes[channel] = vector / norm if norm > 0 else np.eye(
            n_channels, dtype=np.float32)[channel]

    try:
        correction = np.linalg.inv(axes)
    except np.linalg.LinAlgError:
        return data
    from .ops_accel import matmul

    return matmul(flat, correction, gpu=gpu).reshape(
        data.shape).astype(np.float32)


def _argmax_median_unmix(flat: np.ndarray) -> np.ndarray:
    """Brieflow's median crosstalk transform on ``(N, channels)`` rows.

    :param flat: finite intensities, one row per spot and cycle.
    :returns: the corrected rows, float32; the input unchanged when the
        axes cannot be inverted.
    """
    x = flat.astype(np.float64)
    median = np.median(x, axis=0)
    median[~(median > 0)] = 1.0
    x = x / median
    n_channels = x.shape[1]
    winner = x.argmax(axis=1)
    axes = np.empty((n_channels, n_channels))
    for channel in range(n_channels):
        chosen = x[winner == channel]
        axes[:, channel] = (np.median(chosen, axis=0) if len(chosen)
                            else np.eye(n_channels)[channel])
    total = axes.sum(axis=0)
    total[~(total > 0)] = 1.0
    try:
        unmix = np.linalg.inv(axes / total)
    except np.linalg.LinAlgError:
        return flat.astype(np.float32)
    return (x @ unmix.T).astype(np.float32)


#: Below this many reads a median across them is not a floor, so the
#: per-cycle normalisation is skipped and the raw brightest channel stands.
#:
#: The median of one channel in one cycle is that channel's floor only while
#: fewer than half the reads carry that base. A field of the real plate holds
#: about ten thousand reads from a 20,445-guide library, so no base comes near
#: half. A handful of cells does: a synthetic field of 7 nuclei, three reads
#: each, had base A in cycle 1 for 5 of the 7, the median of A was A's own
#: on-level, and dividing by it swapped A and T in a third of the reads. Reads
#: come several to a cell, so 200 reads is some 60 cells, where a base holding
#: half of one cycle is a binomial tail of about one in a million.
_MIN_READS_TO_NORMALISE = 200


def _median_normalised(data: np.ndarray) -> np.ndarray:
    """Divide each cycle's channels by their median over the reads.

    :param data: ``(N, cycles, channels)`` float32, NaN where a cycle was not
        measured.
    :returns: the normalised copy; returned unchanged when there are too few
        reads for a median to mean anything.

    372 PART 14-L measured this against the percentile compensation on the
    first real plate (well A1, 105 fields): 0.773 library-exact against
    0.581, and 0.779 against 0.593 over all 333 fields. The median across
    reads of one channel in one cycle is dominated by reads whose base is
    something else, so it estimates that channel's floor, and dividing by it
    removes the cycle-to-cycle gain drift that made the raw calls lean
    toward C.
    """
    if data.shape[0] < _MIN_READS_TO_NORMALISE:
        return data
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(data, axis=0, keepdims=True)
    median = np.where(np.isfinite(median) & (median > 0), median, 1.0)
    return (data / median).astype(np.float32)


def call_reads(values: np.ndarray, *, bases: Sequence[str] = BASES,
               compensate: bool = False,
               method: str = "percentile",
               normalise: bool = True,
               gpu: bool = True) -> Tuple[List[str], np.ndarray]:
    """Turn per-cycle intensities into a barcode string and a quality per read.

    Quality is the margin between the winning channel and the runner-up,
    divided by their sum, per cycle -- 0 when two bases are equally likely and
    1 when the call is unambiguous. Reported per read as the MINIMUM over
    cycles, because a barcode is only as trustworthy as its worst base.

    A cycle that was not measured is called ``N``. Where a read's values for
    one cycle are NaN -- the cycle's file could not be read, or its
    registration was refused -- that letter is ``N`` and the quality is taken
    over the cycles that were measured, so one lost cycle costs one base of
    each read and not the read.

    :param values: ``(N, cycles, channels)`` from :func:`extract_bases`.
    :param bases: the letter for each channel, in channel order.
    :param compensate: undo cross-talk with :func:`compensate_crosstalk`
        after normalising. Off by default: on the first real plate it
        lowered the library match.
    :param method: passed to :func:`compensate_crosstalk`.
    :param normalise: divide each cycle's channels by their median over the
        reads first, which removes the per-channel gain and background drift
        between cycles. Skipped when there are too few reads for a median.
    :param gpu: let the compensation's multiply run on the card. False keeps
        the whole call on the CPU.
    :returns: ``(barcodes, quality)`` -- one string and one float per read.
    """
    _data, barcodes, margin = called_bases(
        values, bases=bases, compensate=compensate, method=method,
        normalise=normalise, gpu=gpu)
    if not barcodes:
        return [], np.zeros((0,), np.float32)
    worst = np.where(np.isnan(margin), np.inf, margin).min(axis=1)
    return barcodes, np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)


def called_bases(values: np.ndarray, *, bases: Sequence[str] = BASES,
                 compensate: bool = False,
                 method: str = "percentile",
                 normalise: bool = True,
                 gpu: bool = True) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """The intensities a call was made on, the barcode, and the margin per cycle.

    :func:`call_reads` is this, with the per-cycle margins reduced to their
    minimum. The three pieces are separated because the ``ops_reads`` table of
    372's storage contract is one row per read PER CYCLE -- base, quality and
    the per-channel intensity -- and a caller that only gets the barcode and
    one number per read has to redo the normalisation to write it. Redoing it
    is what makes a stored intensity disagree with the call beside it.

    :param values: ``(N, cycles, channels)`` from :func:`extract_bases`.
    :param bases: the letter for each channel, in channel order.
    :param compensate: undo cross-talk after normalising.
    :param method: passed to :func:`compensate_crosstalk`.
    :param normalise: divide each cycle's channels by their median over the
        reads first.
    :param gpu: let the compensation's multiply run on the card.
    :returns: ``(data, barcodes, margin)`` -- ``(N, cycles, channels)``
        float32 intensities AFTER normalisation and any compensation, so
        they are the numbers the winner was picked from rather than the raw
        ones; one barcode string per read; and ``(N, cycles)`` float32
        margins, NaN for a cycle that was not measured and whose letter is
        therefore ``N``.
    """
    data = np.asarray(values, dtype=np.float32)
    if data.size == 0:
        return (np.zeros((0, 0, 0), np.float32), [],
                np.zeros((0, 0), np.float32))
    if normalise:
        data = _median_normalised(data)
    measured = np.isfinite(data).all(axis=-1)
    n_reads, n_cycles, n_channels = data.shape
    if compensate:
        flat = data.reshape(-1, n_channels)
        rows = measured.reshape(-1)
        corrected = np.full_like(flat, np.nan)
        if rows.any():
            corrected[rows] = compensate_crosstalk(
                flat[rows][:, None, :], method=method,
                gpu=gpu).reshape(-1, n_channels)
        data = corrected.reshape(data.shape)

    filled = np.where(np.isfinite(data), data, 0.0).astype(np.float32)
    ordered = np.sort(filled, axis=-1)
    best, second = ordered[..., -1], ordered[..., -2]
    total = best + second
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cycle = np.where(total > 0, (best - second) / total, 0.0)
    margin = np.where(measured, per_cycle, np.nan).astype(np.float32)

    winners = np.where(measured, filled.argmax(axis=-1), n_channels)
    letters = np.asarray([ord(str(letter)) for letter in bases] + [ord("N")],
                         dtype=np.uint8)
    codes = np.ascontiguousarray(letters[winners])
    barcodes = [code.decode("ascii")
                for code in codes.view(f"S{n_cycles}").reshape(n_reads)]
    return data, barcodes, margin


def _library_index(library: Sequence[str]):
    """Index a guide library for exact, one-mismatch and one-N lookups.

    :param library: the guide barcodes.
    :returns: ``(known, masked, arrays)`` -- the set of barcodes, a map from
        ``(position, barcode with that position removed)`` to the barcodes
        sharing it, and ``length -> (barcodes, uint8 array)`` for the rare
        read that needs a full comparison.

    ONE PASS OVER THE LIBRARY, NOT ONE PER READ. Two barcodes of one length
    differ at most at position ``i`` exactly when they agree once ``i`` is
    removed, so every candidate at distance one is found by eleven lookups
    rather than by comparing the read with 20,445 guides in Python -- which
    cost seconds per few hundred reads and a whole well's barcodes would
    have taken hours.
    """
    known = set(library)
    masked: Dict[Tuple[int, str], List[str]] = {}
    grouped: Dict[int, List[str]] = {}
    for entry in known:
        grouped.setdefault(len(entry), []).append(entry)
        for position in range(len(entry)):
            key = (position, entry[:position] + entry[position + 1:])
            masked.setdefault(key, []).append(entry)
    arrays = {}
    for length, entries in grouped.items():
        entries = sorted(entries)
        arrays[length] = (entries, np.frombuffer(
            "".join(entries).encode("ascii"), dtype=np.uint8).reshape(
                len(entries), length))
    return known, masked, arrays


def _closest_by_comparison(read: str, arrays, max_distance: int
                           ) -> Optional[str]:
    """The unique closest barcode by full comparison, N counting as unknown.

    :param read: the called read.
    :param arrays: from :func:`_library_index`.
    :param max_distance: the largest distance corrected.
    :returns: the barcode, or None when none is close enough or two tie.
    """
    found = arrays.get(len(read))
    if found is None:
        return None
    entries, table = found
    letters = np.frombuffer(read.encode("ascii"), dtype=np.uint8)
    distance = ((table != letters) & (letters != ord("N"))).sum(axis=1)
    best = int(distance.min())
    if best > max_distance or int((distance == best).sum()) > 1:
        return None
    return entries[int(np.argmin(distance))]


def correct_to_library(barcodes: Sequence[str], library: Sequence[str], *,
                       max_distance: int = 1) -> List[Optional[str]]:
    """Correct each read to its unique closest library barcode, or ``None``.

    A read equally close to two library barcodes returns ``None`` rather
    than one of them: a misassigned guide moves a cell's phenotype onto the
    wrong perturbation and silently corrupts every statistic downstream,
    while a dropped read only costs statistical power.

    An exact match short-circuits, so a clean run pays nothing for this.

    An ``N`` is a base that was not measured, not a mismatch. It matches any
    letter and is not counted against ``max_distance``; the uniqueness rule
    still applies, so a read whose lost base is the only thing separating two
    guides maps to neither.

    :param barcodes: the called reads.
    :param library: the guide barcodes the screen actually contains.
    :param max_distance: the largest Hamming distance that may be corrected.
    :returns: one entry per read -- the library barcode, or None.
    """
    known, masked, arrays = _library_index(library)
    out: List[Optional[str]] = []
    for read in barcodes:
        if read in known:
            out.append(read)
            continue
        unknown = [i for i, letter in enumerate(read) if letter == "N"]
        if not unknown and max_distance >= 1:
            candidates = set()
            for position in range(len(read)):
                candidates.update(masked.get(
                    (position, read[:position] + read[position + 1:]), ()))
            if len(candidates) == 1:
                out.append(candidates.pop())
                continue
            if candidates or max_distance == 1:
                out.append(None)
                continue
        elif len(unknown) == 1:
            position = unknown[0]
            candidates = set(masked.get(
                (position, read[:position] + read[position + 1:]), ()))
            if len(candidates) == 1:
                out.append(candidates.pop())
                continue
            if candidates or max_distance == 0:
                out.append(None)
                continue
        elif not unknown:
            out.append(None)
            continue
        out.append(_closest_by_comparison(read, arrays, max_distance))
    return out


def assign_reads_to_cells(peaks: np.ndarray, barcodes: Sequence[str],
                          labels: np.ndarray, *,
                          quality: Optional[np.ndarray] = None,
                          min_reads: int = 2,
                          min_fraction: float = 0.6) -> Dict[int, dict]:
    """Give each segmented cell the barcode its reads agree on.

    A cell contains several spots and they will not all decode identically:
    an out-of-focus spot, a spot shared with a neighbour, and a genuine second
    perturbation all look the same at this stage. So a cell is assigned only
    when its reads AGREE -- ``min_fraction`` of at least ``min_reads`` must
    carry the same barcode -- and is otherwise left unassigned.

    The defaults (``min_reads=2``, ``min_fraction=0.6``) are conservative. A
    single read cannot be checked against anything, and a single mis-called
    base would silently hand a cell the wrong perturbation. A cell with no
    barcode costs statistical power; a cell with the WRONG barcode moves a real
    phenotype onto another guide's average and is not recoverable downstream,
    because nothing later can tell it happened.

    Reads landing on label 0 -- background, between cells -- are discarded
    rather than attached to the nearest cell. A read that segmentation did not
    place inside anything is a read whose owner is unknown.

    :param peaks: ``(N, 2)`` ``(y, x)`` read positions.
    :param barcodes: one called barcode per read.
    :param labels: the segmentation, as an integer label image where 0 is
        background.
    :param quality: optional per-read quality from :func:`call_reads`; when
        given it is averaged over the reads that agreed.
    :param min_reads: how many reads a cell needs before it may be assigned.
    :param min_fraction: what share of them must agree.
    :returns: ``{label: {"barcode", "reads", "agreeing", "fraction",
        "quality"}}`` for every cell that met the bar.
    """
    label_image = np.asarray(labels)
    coords = np.asarray(peaks, dtype=np.int64).reshape(-1, 2)
    height, width = label_image.shape[-2:]

    per_cell: Dict[int, List[int]] = {}
    for index, (y, x) in enumerate(coords):
        if not (0 <= y < height and 0 <= x < width):
            continue
        cell = int(label_image[y, x])
        if cell == 0:
            continue
        per_cell.setdefault(cell, []).append(index)

    return _vote(per_cell, barcodes, quality, min_reads=min_reads,
                 min_fraction=min_fraction)


def _vote(per_owner: Dict[int, List[int]], barcodes: Sequence[str],
          quality: Optional[np.ndarray], *, min_reads: int,
          min_fraction: float) -> Dict[int, dict]:
    """The agreement rule shared by cells and plate objects.

    :param per_owner: ``owner -> indices of its reads``.
    :param barcodes: one called barcode per read.
    :param quality: optional per-read quality.
    :param min_reads: how many reads an owner needs.
    :param min_fraction: what share of them must agree.
    :returns: ``{owner: {"barcode", "reads", "agreeing", "fraction",
        "quality"}}`` for every owner that met the bar.
    """
    out: Dict[int, dict] = {}
    for owner, indices in per_owner.items():
        if len(indices) < min_reads:
            continue
        counts: Dict[str, int] = {}
        for i in indices:
            counts[barcodes[i]] = counts.get(barcodes[i], 0) + 1
        best = max(counts, key=counts.get)
        agreeing = counts[best]
        fraction = agreeing / len(indices)
        if fraction < min_fraction:
            continue
        if sum(1 for value in counts.values() if value == agreeing) > 1:
            continue
        mean_quality = None
        if quality is not None:
            agreed = [float(quality[i]) for i in indices
                      if barcodes[i] == best]
            mean_quality = float(np.mean(agreed)) if agreed else None
        out[owner] = {
            "barcode": best,
            "reads": len(indices),
            "agreeing": agreeing,
            "fraction": float(fraction),
            "quality": mean_quality,
        }
    return out


def attribute_reads(peaks: np.ndarray, centroids: np.ndarray,
                    areas: np.ndarray, *, footprint: float = 10.0,
                    tie: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Give each detected read to the object whose boundary is nearest.

    Reads are detected over the whole field first and attributed here, so a
    footprint that is too tight shows up as reads with no owner rather than
    staying invisible. The boundary is each object's equivalent disc, the
    circle of its area about its centroid, and a read whose two nearest
    boundaries lie within ``tie`` of each other is given to neither and
    flagged.

    :param peaks: ``(N, 2)`` read positions ``(y, x)``.
    :param centroids: ``(M, 2)`` object centroids ``(y, x)``, in the same
        frame as ``peaks``.
    :param areas: ``(M,)`` object areas in pixels.
    :param footprint: how far beyond an object's boundary a read may lie and
        still be its read, in pixels. On the first real plate 98 % of spots
        lay within 10 px of a nucleus and 67 % within 3 px.
    :param tie: how close the two nearest boundaries may be before a read is
        refused as ambiguous, in pixels.
    :returns: ``(owner, ambiguous)`` -- the index into ``centroids`` of each
        read's owner, or -1, and whether each read was refused as ambiguous.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(peaks, dtype=float).reshape(-1, 2)
    centres = np.asarray(centroids, dtype=float).reshape(-1, 2)
    radius = np.sqrt(np.maximum(np.asarray(areas, dtype=float).reshape(-1),
                                0.0) / np.pi)
    owner = np.full(points.shape[0], -1, dtype=np.int64)
    ambiguous = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0 or centres.shape[0] == 0:
        return owner, ambiguous

    tree = cKDTree(centres)
    reach = float(radius.max()) + float(footprint)
    for index, found in enumerate(tree.query_ball_point(points, r=reach)):
        if not found:
            continue
        found = np.asarray(found, dtype=np.int64)
        gap = np.hypot(centres[found, 0] - points[index, 0],
                       centres[found, 1] - points[index, 1]) - radius[found]
        order = np.argsort(gap, kind="stable")
        if gap[order[0]] > footprint:
            continue
        if order.size > 1 and gap[order[1]] - gap[order[0]] < tie:
            ambiguous[index] = True
            continue
        owner[index] = int(found[order[0]])
    return owner, ambiguous


def assign_reads_to_objects(owners: np.ndarray, barcodes: Sequence[str], *,
                            quality: Optional[np.ndarray] = None,
                            min_reads: int = 2,
                            min_fraction: float = 0.6) -> Dict[int, dict]:
    """Give each plate object the barcode its attributed reads agree on.

    The same rule as :func:`assign_reads_to_cells` -- at least ``min_reads``
    reads, ``min_fraction`` of them carrying one barcode, and no tie for first
    -- keyed by the object id each read was attributed to rather than by the
    label a read lands on. That is what lets reads collected from several
    fields vote once for the object they belong to.

    :param owners: one object id per read; 0 or a negative id means the read
        has no owner and is ignored.
    :param barcodes: one called barcode per read.
    :param quality: optional per-read quality from :func:`call_reads`.
    :param min_reads: how many reads an object needs before it may be
        assigned.
    :param min_fraction: what share of them must agree.
    :returns: ``{object_id: {"barcode", "reads", "agreeing", "fraction",
        "quality"}}`` for every object that met the bar.
    """
    ids = np.asarray(owners).reshape(-1)
    if ids.shape[0] != len(barcodes):
        raise ValueError(
            f"{ids.shape[0]} owners for {len(barcodes)} barcodes; every read "
            "needs exactly one owner entry, or the votes land on the wrong "
            "objects")
    per_owner: Dict[int, List[int]] = {}
    for index in np.flatnonzero(ids > 0):
        per_owner.setdefault(int(ids[index]), []).append(int(index))
    return _vote(per_owner, barcodes, quality, min_reads=min_reads,
                 min_fraction=min_fraction)


def decoy_library(library: Sequence[str], k: int = 4,
                  seed: int = 1) -> List[str]:
    """Base-shuffled decoy barcodes, ``k`` per guide, for a decoy FDR.

    Each decoy is a permutation of one guide's letters, so it keeps that
    guide's base composition, and it is accepted only when it lies at least
    two edits from every guide and every decoy taken before it -- the same
    separation the library was designed with. Matched alongside the real
    library, a decoy call is a call that cannot be true, and ``k`` decoys per
    guide measure a rare false match ``k`` times as precisely.

    :param library: the guide barcodes; duplicates are dropped and the rest
        sorted, so the decoys do not depend on the order they were listed in.
    :param k: decoys per guide.
    :param seed: the shuffle's seed. 1 is the seed of the decode-yield study
        (item 873), whose decoy FDRs this reproduces.
    :returns: the decoys, guide by guide and round by round. A guide whose 200
        shuffles all collide gets none in that round.
    """
    guides = sorted(set(str(entry) for entry in library))
    rng = np.random.default_rng(seed)
    taken = set(guides)
    masked = set()
    for entry in guides:
        for i in range(len(entry)):
            masked.add((i, entry[:i] + entry[i + 1:]))
    out: List[str] = []
    for _ in range(int(k)):
        for entry in guides:
            for _ in range(200):
                candidate = "".join(rng.permutation(list(entry)))
                if candidate in taken or any(
                        (i, candidate[:i] + candidate[i + 1:]) in masked
                        for i in range(len(candidate))):
                    continue
                out.append(candidate)
                taken.add(candidate)
                for i in range(len(candidate)):
                    masked.add((i, candidate[:i] + candidate[i + 1:]))
                break
    return out


@lru_cache(maxsize=8)
def _barcode_distance_profiles(guides: Tuple[str, ...], barcodes: Tuple[str, ...]) -> np.ndarray:
    """Count barcodes one, two and three substitutions from each sorted guide.

    The existing one-deletion index supplies the distance-one matches.
    Two- and three-deletion signatures count the other distances without
    constructing the guide-by-barcode distance matrix. One signature
    position set is held at a time. A pair at distance h contributes
    ``choose(length - h, d - h)`` signatures of order d; subtracting the
    lower-distance terms therefore gives exact Hamming counts.

    :param guides: sorted unique guide strings, including mixed lengths.
    :param barcodes: sorted unique strings whose distances are counted.
    :returns: read-only int64 ``(guides, 3)`` array, columns distances 1–3;
        exact matches, including a guide itself, are excluded.
    """
    from collections import Counter
    from itertools import combinations

    known, masked, arrays = _library_index(barcodes)
    result = np.zeros((len(guides), 3), np.int64)
    lengths = sorted({len(guide) for guide in guides})
    for length in lengths:
        indices = [i for i, guide in enumerate(guides) if len(guide) == length]
        sources = [guides[i] for i in indices]
        targets = arrays.get(length, ((), None))[0]
        if not targets:
            continue
        exact = np.array([guide in known for guide in sources], np.int64)
        sums = np.zeros((len(sources), 3), np.int64)
        sums[:, 0] = [sum(len(masked.get((p, guide[:p] + guide[p + 1:]), ()))
                          for p in range(length)) for guide in sources]
        for order in (2, 3):
            for positions in combinations(range(length), order):
                spans = tuple(zip((0,) + tuple(p + 1 for p in positions),
                                  positions + (length,)))
                counts = Counter(''.join(entry[a:b] for a, b in spans) for entry in targets)
                sums[:, order - 1] += [counts.get(''.join(guide[a:b] for a, b in spans), 0)
                                       for guide in sources]
        first = sums[:, 0] - length * exact
        second = sums[:, 1] - math.comb(length, 2) * exact - max(length - 1, 0) * first
        third = (sums[:, 2] - math.comb(length, 3) * exact
                 - math.comb(max(length - 1, 0), 2) * first - max(length - 2, 0) * second)
        result[indices] = np.column_stack((first, second, third))
    result.setflags(write=False)
    return result


@lru_cache(maxsize=8)
def _never_barcodes_cached(guides, distance, k, seed, excluded):
    """Generate and profile immutable controls in per-guide reference acceptance order.

    :param guides: sorted unique real guide strings.
    :param distance: nearest-guide Hamming distance, one or two.
    :param k: maximum accepted controls per guide.
    :param seed: the shared NumPy generator seed.
    :param excluded: previously occupied control strings.
    :returns: cached control strings and exact per-guide opportunity arrays.
    """
    known, guide_mask, _ = _library_index(guides)
    occupied, blocked_mask, _ = _library_index(excluded)
    blocked_mask = set(blocked_mask)
    taken = known | occupied
    rng = np.random.default_rng(seed)
    output = []
    for guide in guides:
        if len(guide) < distance:
            continue
        accepted = 0
        used_positions = set()
        for _ in range(200):
            if accepted >= k:
                break
            if distance == 1:
                free = [position for position in range(len(guide)) if position not in used_positions]
                if not free:
                    break
                changed = [int(rng.choice(free))]
            else:
                changed = list(rng.choice(len(guide), 2, replace=False))
            letters = list(guide)
            for position in changed:
                letters[position] = rng.choice([base for base in 'ACGT' if base != guide[position]])
            candidate = ''.join(letters)
            if candidate in taken:
                continue
            slots = [(position, candidate[:position] + candidate[position + 1:])
                     for position in range(len(candidate))]
            near = {entry for slot in slots for entry in guide_mask.get(slot, ())}
            if (distance == 1 and near != {guide}) or (distance == 2 and near):
                continue
            if any(slot in blocked_mask for slot in slots):
                continue
            output.append(candidate)
            taken.add(candidate)
            blocked_mask.update(slots)
            if distance == 1:
                used_positions.add(changed[0])
            accepted += 1
    barcodes = tuple(output)
    return {'barcodes': barcodes, 'guides': guides, 'distance': distance, 'k': k, 'seed': seed,
            'm_d': _barcode_distance_profiles(guides, tuple(sorted(barcodes))),
            'library_m_d': _barcode_distance_profiles(guides, guides)}


def never_barcodes(library, *, distance, k=4, seed=1, exclude=()):
    """Substitution-built barcodes that no cell in the guide library should carry.

    Distance-one controls have exactly one nearest guide and use a different
    substituted position for each accepted control of that guide.
    Distance-two controls have no guide at distance zero or one. Every
    control lies at least two substitutions from previously accepted and
    excluded controls. Sorted unique guides are visited in order, accepting
    up to k controls before moving to the next guide. One seeded generator
    makes up to 200 total attempts per guide, matching the reference order.
    The same builder serves full-barcode and sequenced-prefix calibration.

    :param library: real guide barcodes, using nonempty A/C/G/T strings.
        Duplicates and input order do not affect the result.
    :param distance: nearest-guide Hamming distance, 1 (N1) or 2 (N2).
    :param k: maximum controls per guide, default 4; zero gives an empty set.
    :param seed: nonnegative generator seed, default 1.
    :param exclude: controls already taken, such as matched decoys or N1.
    :returns: dict with ``barcodes`` in acceptance order, sorted ``guides``,
        ``distance``, ``k``, ``seed``, and read-only int64 ``m_d`` and
        ``library_m_d`` arrays. Both arrays have shape ``(guides, 3)``;
        columns count control or other-library barcodes at distances 1–3.
        Exact matches are excluded. Results are memoised per process.
    :raises ValueError: for an unsupported distance, negative or fractional
        k or seed, or a barcode containing letters outside A/C/G/T.
    """
    from .qt.i18n import tr

    if isinstance(distance, bool) or distance not in (1, 2):
        raise ValueError(tr('Never-barcode distance must be 1 or 2.'))
    if isinstance(k, bool) or int(k) != k or int(k) < 0:
        raise ValueError(tr('Never-barcode count must be a nonnegative integer.'))
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError(tr('Never-barcode seed must be a nonnegative integer.'))
    guides = tuple(sorted(set(str(entry) for entry in library)))
    excluded = tuple(sorted(set(str(entry) for entry in exclude)))
    if any(not entry or set(entry) - _ACGT for entry in (*guides, *excluded)):
        raise ValueError(tr('Never-barcode libraries must contain nonempty DNA strings using A, C, G and T.'))
    return dict(_never_barcodes_cached(guides, int(distance), int(k), int(seed), excluded))


#: The letters the one-mismatch index of :class:`LibraryMatcher` covers.
_ACGT = frozenset("ACGT")

#: The index's marker for a string one substitution from two barcodes.
_TIED = -2


class LibraryMatcher:
    """The unique closest barcode within a distance, with ``N`` as a wildcard.

    One matcher holds the guides and, for a decoy FDR, the decoys, so a read
    is matched against both under one uniqueness rule: a read equally close
    to a guide and a decoy maps to neither.

    :param entries: the barcodes to match against.
    """

    def __init__(self, entries: Sequence[str]):
        """Index the barcodes.

        :param entries: the barcodes; duplicates are dropped.
        """
        self.entries = sorted(set(str(entry) for entry in entries))
        width = len(self.entries[0]) if self.entries else 0
        self.width = width
        self.table = (np.frombuffer("".join(self.entries).encode("ascii"),
                                    dtype=np.uint8).reshape(len(self.entries), width)
                      if self.entries else np.zeros((0, 0), np.uint8))
        self.known = {entry: i for i, entry in enumerate(self.entries)}
        self.cache: Dict[Tuple[str, int], Tuple[Optional[str], int]] = {}
        self._neighbours: Optional[Dict[str, int]] = None

    def match(self, read: str, max_distance: int) -> Tuple[Optional[str], int]:
        """Match one read.

        :param read: the called read; ``N`` is a base that was not measured
            and matches any letter without counting as a mismatch.
        :param max_distance: the most mismatches accepted.
        :returns: ``(barcode, distance)``, or ``(None, -1)`` when no barcode
            is close enough or two tie for closest.
        """
        key = (read, max_distance)
        found = self.cache.get(key)
        if found is not None:
            return found
        if read in self.known:
            result: Tuple[Optional[str], int] = (read, 0)
        elif (max_distance == 0 and "N" not in read) or len(read) != self.width \
                or not self.entries:
            result = (None, -1)
        elif max_distance == 1 and _ACGT.issuperset(read):
            hit = self._one_mismatch().get(read)
            result = ((None, -1) if hit is None or hit == _TIED
                      else (self.entries[hit], 1))
        else:
            letters = np.frombuffer(read.encode("ascii"), dtype=np.uint8)
            distance = ((self.table != letters) & (letters != ord("N"))).sum(1)
            best = int(distance.min())
            if best > max_distance or int((distance == best).sum()) > 1:
                result = (None, -1)
            else:
                result = (self.entries[int(distance.argmin())], best)
        self.cache[key] = result
        return result

    def _one_mismatch(self) -> Dict[str, int]:
        """Every string one substitution from a barcode, built on first use.

        Maps the string to the barcode's index, or to :data:`_TIED` when it
        is one substitution from two barcodes. A read of A, C, G and T only
        that is not itself a barcode is at distance one from the barcodes
        this lists for it and from no other, so a lookup gives the scan's
        answer exactly (item 932: 78 % less time on the rule step of the
        decode-yield study, the same calls). A read with ``N`` or any other
        letter, or a wider distance, still goes through the scan.

        :returns: the neighbour index.
        """
        if self._neighbours is None:
            neighbours: Dict[str, int] = {}
            for index, entry in enumerate(self.entries):
                for position in range(len(entry)):
                    for base in "ACGT":
                        if base == entry[position]:
                            continue
                        near = entry[:position] + base + entry[position + 1:]
                        neighbours[near] = (index if near not in neighbours
                                            else _TIED)
            self._neighbours = neighbours
        return self._neighbours


def _argmax_median_unmix64(values: np.ndarray) -> np.ndarray:
    """Brieflow's median crosstalk transform, kept in float64.

    :param values: ``(N, cycles, channels)``; a row with any non-finite
        channel stays NaN.
    :returns: the unmixed intensities, float64, same shape. The library
        calibration fits its per-cycle axes on these, so they are not
        rounded to float32 first.
    """
    data = np.asarray(values)
    flat = data.reshape(-1, data.shape[-1]).astype(np.float64)
    ok = np.isfinite(flat).all(1)
    rows = flat[ok]
    median = np.median(rows, axis=0)
    median[~(median > 0)] = 1
    rows = rows / median
    axes = []
    for channel in range(rows.shape[1]):
        chosen = rows[rows.argmax(1) == channel]
        axes.append(np.median(chosen, axis=0) if len(chosen)
                    else np.eye(rows.shape[1])[channel])
    axes = np.array(axes).T
    axes = axes / axes.sum(0)
    try:
        unmix = np.linalg.inv(axes)
    except np.linalg.LinAlgError:
        unmix = np.eye(rows.shape[1])
    out = np.full_like(flat, np.nan)
    out[ok] = unmix.dot(rows.T).T
    return out.reshape(data.shape)


def library_calibrated_bases(values: np.ndarray, library: Sequence[str], *,
                             bases: Sequence[str] = BASES, passes: int = 2,
                             min_labels: int = 30, masked: Optional[dict] = None
                             ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Call bases with per-cycle axes learnt from the reads the library names.

    The reads are first normalised per cycle and unmixed with brieflow's
    median transform. Then, for each read and cycle, the read is matched on
    its OTHER cycles; when exactly one guide fits, that guide names the true
    base at the masked cycle without looking at it. Per cycle, the median
    intensity of the reads labelled with each base is that base's axis; the
    four axes, scaled to sum to one, are inverted and every read of the cycle
    is unmixed with them. Labels come from the real library only, so a decoy
    library matched later stays an independent false-match control. This is
    the "S1b" calibration of the decode-yield study (item 873).

    :param values: ``(N, cycles, channels)`` from :func:`extract_bases`.
    :param library: the guide barcodes.
    :param bases: the letter for each channel, in channel order.
    :param passes: how often the labels are re-derived from the last calls.
    :param min_labels: the fewest labelled reads per base a cycle needs; a
        cycle short of it keeps the median transform.
    :param masked: the library's masked index from :func:`_library_index`,
        when the caller keeps one; None builds it.
    :returns: ``(data, barcodes, margin)`` as :func:`called_bases` returns
        them.
    """
    data = np.asarray(values, dtype=np.float32)
    if data.size == 0:
        return called_bases(data, bases=bases, gpu=False)
    letters = "".join(str(b) for b in bases)
    first, _, _ = called_bases(data, bases=bases, normalise=True, gpu=False)
    first = _argmax_median_unmix64(first)
    _, calls, _ = called_bases(first, bases=bases, normalise=False, gpu=False)
    n_reads, n_cycles, n_channels = first.shape
    if masked is None:
        _, masked, _ = _library_index(library)
    unmixed = first.copy()
    for _ in range(int(passes)):
        labels = np.full((n_reads, n_cycles), -1, np.int8)
        for i, read in enumerate(calls):
            for c in range(n_cycles):
                candidates = masked.get((c, read[:c] + read[c + 1:]))
                if candidates and len(candidates) == 1:
                    labels[i, c] = letters.index(candidates[0][c])
        unmixed = first.copy()
        for c in range(n_cycles):
            cycle = first[:, c]
            ok = np.isfinite(cycle).all(1)
            axes = np.eye(n_channels)
            usable = True
            for k in range(n_channels):
                chosen = ok & (labels[:, c] == k)
                if chosen.sum() < min_labels:
                    usable = False
                    break
                axes[:, k] = np.median(cycle[chosen], axis=0)
            if not usable:
                continue
            axes = axes / axes.sum(0)
            try:
                weights = np.linalg.inv(axes)
            except np.linalg.LinAlgError:
                continue
            unmixed[ok, c] = cycle[ok] @ weights.T
        _, calls, _ = called_bases(unmixed, bases=bases, normalise=False,
                                   gpu=False)
    return called_bases(unmixed, bases=bases, normalise=False, gpu=False)


def runner_up_consistent(data_row: np.ndarray, margin_row: np.ndarray,
                         read: str, guide: str, max_margin: float, *,
                         bases: Sequence[str] = BASES) -> bool:
    """Whether a one-mismatch read may count for its guide.

    Every mismatched cycle must be one whose call was uncertain -- its margin
    below ``max_margin`` -- and whose RUNNER-UP channel is the guide's base,
    so the guide is what the read would have said had that cycle tipped the
    other way. An ``N`` is not a mismatch.

    :param data_row: ``(cycles, channels)``, the intensities the read was
        called from.
    :param margin_row: ``(cycles,)``, the per-cycle margins.
    :param read: the called read.
    :param guide: the guide it matched.
    :param max_margin: the largest margin a mismatched cycle may carry.
    :param bases: the letter for each channel, in channel order.
    :returns: True when every mismatch passes.
    """
    letters = "".join(str(b) for b in bases)
    for position, (called, wanted) in enumerate(zip(read, guide)):
        if called == wanted or called == "N":
            continue
        order = np.argsort(np.nan_to_num(data_row[position], nan=-np.inf))[::-1]
        if letters[order[1]] != wanted or not (margin_row[position] < max_margin):
            return False
    return True


def nearest_boundaries(points: np.ndarray, centroids: np.ndarray,
                       areas: np.ndarray, *, neighbours: int = 8
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """The two nearest object boundaries of each point, and their objects.

    Each object's boundary is its equivalent disc. The ``neighbours`` nearest
    centroids are ranked by the gap from the point to their boundary, so the
    rule that gives a read to the nearest boundary can also see how close the
    runner-up came, which is what the territory rule and a tie test read.

    :param points: ``(N, 2)`` positions ``(y, x)``.
    :param centroids: ``(M, 2)`` object centroids in the same frame.
    :param areas: ``(M,)`` object areas in pixels.
    :param neighbours: how many nearest centroids are ranked.
    :returns: ``(gap, index)`` -- ``(N, 2)`` float64 gaps, ``inf`` where
        there is no second object, and ``(N, 2)`` indices into
        ``centroids``, -1 where there is none.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    centres = np.asarray(centroids, dtype=float).reshape(-1, 2)
    radius = np.sqrt(np.maximum(np.asarray(areas, dtype=float).reshape(-1), 0)
                     / np.pi)
    n = len(pts)
    gap = np.full((n, 2), np.inf)
    index = np.full((n, 2), -1, np.int64)
    if n and len(centres):
        k = min(int(neighbours), len(centres))
        distance, near = cKDTree(centres).query(pts, k=k)
        distance, near = distance.reshape(n, k), near.reshape(n, k)
        gaps = distance - radius[near]
        order = np.argsort(gaps, axis=1, kind="stable")
        gaps = np.take_along_axis(gaps, order, 1)
        near = np.take_along_axis(near, order, 1)
        gap[:, :min(2, k)] = gaps[:, :2]
        index[:, :min(2, k)] = near[:, :2]
    return gap, index


def _distance_term(gap: float, decay_px: float, interior_px: float,
                   min_weight: float) -> float:
    """One read's distance term in the guide vote (item 988).

    :param gap: the read's distance to its object's boundary in pixels; a
        negative gap (inside the object's disc) counts as 0.
    :param decay_px: the decay length.
    :param interior_px: the distance with full weight.
    :param min_weight: the smallest term.
    :returns: ``max(exp(-(g - interior_px) / decay_px), min_weight)``, with
        ``g - interior_px`` floored at 0.
    """
    excess = max(max(float(gap), 0.0) - float(interior_px), 0.0)
    return max(math.exp(-excess / float(decay_px)), float(min_weight))


def _discount_neighbour_bleed(votes, exact, corrected, neighbours: Mapping[int, Any],
                              discount: float) -> None:
    """Weigh down the minority guides that are a neighbour's top guide (item 988).

    An object's guide is discounted when it has strictly fewer reads than
    the object's top guide and is a top guide, by reads with ties kept, of a
    neighbouring object. Decoy guides count like library guides.

    :param votes: object -> guide -> summed vote, changed in place.
    :param exact: object -> guide -> exact reads.
    :param corrected: object -> guide -> corrected reads.
    :param neighbours: object -> its neighbouring objects.
    :param discount: the factor a discounted vote is multiplied by.
    """
    reads = {owner: {g: exact[owner].get(g, 0) + corrected[owner].get(g, 0) for g in tally}
             for owner, tally in votes.items()}
    tops = {}
    for owner, per in reads.items():
        most = max(per.values())
        tops[owner] = {g for g, n in per.items() if n == most}
    for owner, tally in votes.items():
        near = set()
        for other in neighbours.get(owner, ()):
            near |= tops.get(int(other), set())
        if not near:
            continue
        most = max(reads[owner].values())
        for guide in list(tally):
            if guide in near and reads[owner][guide] < most:
                tally[guide] *= discount


def _disc_neighbours(ids: np.ndarray, centroids: np.ndarray, areas: np.ndarray,
                    reach: float) -> Dict[int, set]:
    """The objects whose equal-area discs come within ``reach`` of each other (item 988).

    :param ids: one id per object.
    :param centroids: ``(M, 2)`` object centroids.
    :param areas: ``(M,)`` object areas in pixels.
    :param reach: the largest gap between two disc edges, in pixels
        (``ops_footprint``).
    :returns: object id -> the ids of its neighbours.
    """
    from scipy.spatial import cKDTree

    ids = np.asarray(ids, np.int64).reshape(-1)
    centres = np.asarray(centroids, float).reshape(-1, 2)
    radius = np.sqrt(np.maximum(np.asarray(areas, float).reshape(-1), 0) / np.pi)
    out: Dict[int, set] = {}
    if len(ids) < 2:
        return out
    span = 2.0 * float(radius.max()) + float(reach)
    for i, j in cKDTree(centres).query_pairs(r=span):
        if np.hypot(*(centres[i] - centres[j])) - radius[i] - radius[j] <= reach:
            out.setdefault(int(ids[i]), set()).add(int(ids[j]))
            out.setdefault(int(ids[j]), set()).add(int(ids[i]))
    return out


def vote_mapped_reads(objects: np.ndarray, barcodes: Sequence[str],
                      quality: np.ndarray, data: np.ndarray,
                      margin: np.ndarray, matcher: LibraryMatcher, *,
                      max_distance: int = 0,
                      runner_up_margin: Optional[float] = None,
                      quality_weighted: bool = False,
                      weight_floor: float = 0.05,
                      min_fraction: float = 0.6,
                      single_read_quality: float = 0.03,
                      impute_missing: bool = True,
                      strength: Optional[np.ndarray] = None,
                      evidence: Optional[dict] = None,
                      bases: Sequence[str] = BASES,
                      gaps: Optional[np.ndarray] = None,
                      distance_weight: str = "off",
                      decay_px: float = 4.0,
                      interior_px: float = 0.0,
                      min_weight: float = 0.05,
                      neighbours: Optional[Mapping[int, Any]] = None,
                      neighbour_discount: float = 1.0) -> Dict[int, dict]:
    """Each object's guide from its reads, matched to the library first.

    Every read is matched (:class:`LibraryMatcher`); a read at distance one
    or more counts only when ``runner_up_margin`` is given and
    :func:`runner_up_consistent` passes. The matched reads vote, each with
    weight one, or with ``min(1, quality) + weight_floor`` when
    ``quality_weighted``. An object takes the top guide when it holds at
    least ``min_fraction`` of the votes with no tie, and, when one read is
    all that supports it, that read's quality reaches
    ``single_read_quality``.

    :param objects: one object id per read.
    :param barcodes: one called barcode per read.
    :param quality: one quality per read, the worst cycle's margin.
    :param data: ``(N, cycles, channels)``, the intensities called from.
    :param margin: ``(N, cycles)``, the per-cycle margins.
    :param matcher: the guides, and any decoys, to match against.
    :param max_distance: the most mismatches a read may carry.
    :param runner_up_margin: the runner-up rule's margin, or None to count
        every match within ``max_distance``.
    :param quality_weighted: weight votes by read quality.
    :param weight_floor: added to every quality weight.
    :param min_fraction: the share of the votes the guide must hold.
    :param single_read_quality: the quality a one-read guide needs.
    :param impute_missing: let an ``N`` -- a cycle that was not measured --
        match any base; False leaves such reads unmatched.
    :param strength: one spot strength per read, for the evidence; None
        records none.
    :param evidence: a dict filled with the per-object, per-guide tallies
        behind every vote -- ``votes``, ``exact``, ``corrected``, ``best``,
        ``qualities``, ``strength`` and ``matched`` -- for objects assigned
        or not; None keeps none.
    :param bases: the letter for each channel, in channel order.
    :param gaps: one distance per read, in pixels, from the read to its
        object's boundary, 0 inside (item 988). Only read when
        ``distance_weight`` is on.
    :param distance_weight: ``ops_read_distance_weight`` (item 988, alpha).
        ``'off'`` keeps the weights above. ``'decay'`` multiplies each
        read's weight by its distance term, and ``'distance_only'`` uses
        the distance term alone. The distance term is 1 up to
        ``interior_px`` and ``exp(-(g - interior_px) / decay_px)`` beyond
        it, at least ``min_weight``. A one-read guide then needs its
        quality times its distance term to reach ``single_read_quality``.
    :param decay_px: the decay length in pixels
        (``ops_read_distance_decay_px``).
    :param interior_px: the distance with full weight
        (``ops_read_distance_interior_px``).
    :param min_weight: the smallest distance term
        (``ops_read_distance_min_weight``).
    :param neighbours: object -> the objects whose boundaries come within
        the footprint of it, for the neighbour discount; None for none.
    :param neighbour_discount: ``ops_read_neighbour_discount`` (item 988,
        alpha). A guide with strictly fewer reads in the object than its
        top guide has its vote multiplied by this when it is a top guide
        of a neighbour, by reads with ties kept. 1 is off.
    :returns: ``{object: {"guide", "fraction", "n_support", "n_exact",
        "n_corrected", "n_matched", "n_guides", "vote_top", "vote_total",
        "runner_margin", "quality", "barcode"}}``; ``barcode`` is the
        commonest raw read behind the guide, ``runner_margin`` the top vote
        minus the second over the total.
    """
    from collections import Counter, defaultdict

    votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    exact: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    corrected: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    best: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    raw: Dict[int, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    qualities: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    brightest: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    matched: Counter = Counter()
    decay = distance_weight != "off"
    if distance_weight not in ("off", "decay", "distance_only"):
        raise ValueError("ops_read_distance_weight must be 'off', 'decay' or "
                         f"'distance_only'; got {distance_weight!r}")
    if decay and gaps is None:
        raise ValueError("ops_read_distance_weight needs each read's distance")
    for i, (owner, read) in enumerate(zip(objects, barcodes)):
        if not impute_missing and "N" in read:
            continue
        guide, distance = matcher.match(read, max_distance if max_distance else 0)
        if guide is None:
            continue
        owner = int(owner)
        if distance > 0:
            if runner_up_margin is not None and not runner_up_consistent(
                    data[i], margin[i], read, guide, runner_up_margin,
                    bases=bases):
                continue
            corrected[owner][guide] += 1
        else:
            exact[owner][guide] += 1
        weight = (float(min(1.0, max(quality[i], 0.0)) + weight_floor)
                  if quality_weighted and distance_weight != "distance_only" else 1.0)
        floor_quality = quality[i]
        if decay:
            term = _distance_term(float(gaps[i]), decay_px, interior_px, min_weight)
            weight *= term
            floor_quality = quality[i] * term
        votes[owner][guide] += weight
        best[owner][guide] = max(best[owner][guide], floor_quality)
        raw[owner][guide][read] += 1
        matched[owner] += 1
        if evidence is not None:
            qualities[owner][guide].append(float(quality[i]))
            if strength is not None:
                brightest[owner][guide] = max(brightest[owner][guide],
                                              float(strength[i]))
    if float(neighbour_discount) != 1.0:
        _discount_neighbour_bleed(votes, exact, corrected, neighbours or {},
                                  float(neighbour_discount))
    if evidence is not None:
        evidence.update({"votes": votes, "exact": exact, "corrected": corrected,
                         "best": best, "qualities": qualities,
                         "strength": brightest, "matched": matched})
    out: Dict[int, dict] = {}
    for owner, tally in votes.items():
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        guide, top = ranked[0]
        total = sum(tally.values())
        if len(ranked) > 1 and ranked[1][1] == top:
            continue
        if top / total < min_fraction:
            continue
        support = exact[owner].get(guide, 0) + corrected[owner].get(guide, 0)
        if support == 1 and best[owner][guide] < single_read_quality:
            continue
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        out[owner] = {"guide": guide, "fraction": float(top / total),
                      "n_support": int(support),
                      "n_exact": int(exact[owner].get(guide, 0)),
                      "n_corrected": int(corrected[owner].get(guide, 0)),
                      "n_matched": int(matched[owner]),
                      "n_guides": len(ranked), "vote_top": float(top),
                      "vote_total": float(total),
                      "runner_margin": float((top - second) / total),
                      "quality": float(best[owner][guide]),
                      "barcode": raw[owner][guide].most_common(1)[0][0]}
    return out


def vote_pixels(objects: np.ndarray, barcodes: Sequence[str],
                quality: np.ndarray, matcher: LibraryMatcher, *,
                min_quality: float = 0.1, min_pixels: int = 2,
                min_fraction: float = 0.6) -> Dict[int, dict]:
    """Each object's guide from its bright pixels, for objects no read named.

    Pixels are called like reads but are not independent -- the pixels of one
    spot repeat its call -- so the rule counts pixels, not reads: an object
    takes the top exact guide when at least ``min_pixels`` pixels of quality
    ``min_quality`` or better name it, it holds ``min_fraction`` of them, and
    no other guide ties it. This is the "N3" pixel fallback of the
    decode-yield study (item 873).

    :param objects: one object id per pixel.
    :param barcodes: one called barcode per pixel.
    :param quality: one quality per pixel.
    :param matcher: the guides, and any decoys, to match against exactly.
    :param min_quality: the weakest pixel that votes.
    :param min_pixels: the fewest agreeing pixels.
    :param min_fraction: the share of the matched pixels the guide must hold.
    :returns: ``{object: {"guide", "n_pixels", "n_matched", "fraction",
        "runner_margin", "barcode"}}``.
    """
    from collections import Counter, defaultdict

    votes: Dict[int, Counter] = defaultdict(Counter)
    raw: Dict[int, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for owner, read, q in zip(objects, barcodes, quality):
        if q < min_quality:
            continue
        guide, _ = matcher.match(read, 0)
        if guide is not None:
            votes[int(owner)][guide] += 1
            raw[int(owner)][guide][read] += 1
    out: Dict[int, dict] = {}
    for owner, tally in votes.items():
        (guide, n), *rest = tally.most_common()
        total = sum(tally.values())
        if n >= min_pixels and not (rest and rest[0][1] == n) \
                and n / total >= min_fraction:
            out[owner] = {"guide": guide, "n_pixels": int(n),
                          "n_matched": int(total), "fraction": float(n / total),
                          "runner_margin": float(
                              (n - (rest[0][1] if rest else 0)) / total),
                          "barcode": raw[owner][guide].most_common(1)[0][0]}
    return out
