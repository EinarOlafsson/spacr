"""Decode in-situ sequencing barcodes out of an image stack.

WHAT THIS IS FOR, AND WHAT IT IS NOT
====================================
Optical pooled screening reads the guide barcode OUT OF THE IMAGES. Each
sequencing cycle stains four channels, one per base, and a spot's base in that
cycle is whichever channel is brightest -- once the channels have been put on
a common scale. Read one base per cycle at the same spot and the cycles spell
the barcode.

:mod:`spacr.sequencing` is NOT the back half of this and must not be pointed
at it. That module decodes FASTQ reads from an NGS run; there is no FASTQ here
and no sequencer, only pixels.

WHY EACH STEP IS THE STEP IT IS
===============================
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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The four bases, in the channel order the stack is expected to carry.
BASES = ("G", "T", "A", "C")


def estimate_read_locations(stack: np.ndarray) -> np.ndarray:
    """Where the reads are: variance across cycles, not brightness.

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


def compensate_crosstalk(values: np.ndarray, *,
                         method: str = "percentile",
                         percentile: float = 95.0,
                         gpu: bool = True) -> np.ndarray:
    """Undo dye bleed-through, so the brightest channel IS the base.

    THE RAW ARGMAX IS NOT THE BASE. Each dye emits into its neighbours'
    channels, so a channel that is bright overall wins comparisons it should
    lose, and the bias is systematic rather than noise: it mis-calls the same
    base everywhere.

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
        is the case a median call gets wrong.
    :param percentile: the cutoff for ``"percentile"``.
    :param gpu: let the unmixing multiply run on the card. One tiny matrix
        applied to every spot of every cycle is a large multiply by a small
        operand, which is the shape that goes to a card well.
    :returns: corrected intensities, same shape.
    :raises ValueError: on an unknown ``method``.
    """
    if method not in ("percentile", "median"):
        raise ValueError(f"unknown method {method!r}")
    data = np.asarray(values, dtype=np.float32)
    n_channels = data.shape[-1]
    flat = data.reshape(-1, n_channels)
    if flat.shape[0] < n_channels:
        return data

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
    data = np.asarray(values, dtype=np.float32)
    if data.size == 0:
        return [], np.zeros((0,), np.float32)
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
    per_cycle = np.where(measured, per_cycle, np.inf)
    quality = per_cycle.min(axis=1)
    quality = np.where(np.isfinite(quality), quality, 0.0).astype(np.float32)

    winners = np.where(measured, filled.argmax(axis=-1), n_channels)
    letters = np.asarray([ord(str(letter)) for letter in bases] + [ord("N")],
                         dtype=np.uint8)
    codes = np.ascontiguousarray(letters[winners])
    barcodes = [code.decode("ascii")
                for code in codes.view(f"S{n_cycles}").reshape(n_reads)]
    return barcodes, quality


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
    """Snap each read to the library, but ONLY on a unique closest match.

    AMBIGUITY IS DISCARDED, NOT GUESSED. A read equally close to two library
    barcodes returns ``None``. That is deliberate and it is the whole point of
    the step: a misassigned guide moves a cell's phenotype onto the wrong
    perturbation and silently corrupts every statistic downstream, while a
    dropped read only costs statistical power that more cells can buy back.

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

    THE DEFAULTS REFUSE MORE THAN THEY ACCEPT, deliberately. One read is not
    evidence: it cannot be checked against anything, and a single mis-called
    base would silently hand a cell the wrong perturbation. A cell with no
    barcode costs statistical power; a cell with the WRONG barcode moves a
    real phenotype onto another guide's average and is not recoverable
    downstream, because nothing later can tell it happened.

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
    # SPOTS FIRST, OWNERS SECOND (372 PART 9 hole 2, measured in PART 14-L):
    # code that only looks inside a footprint can never discover that the
    # reads are elsewhere, and on the first real plate they were -- 25 % of
    # spots inside the nucleus, 67 % within 3 px, 98 % within 10 px -- while
    # sampling at the nuclear centroid decoded at chance. A read between two
    # objects is refused rather than split, and counted, so a footprint that
    # is too loose shows up as ambiguity instead of as wrong barcodes.
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
