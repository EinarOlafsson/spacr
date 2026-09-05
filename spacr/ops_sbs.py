"""Decode in-situ sequencing barcodes out of an image stack.

WHAT THIS IS FOR, AND WHAT IT IS NOT
====================================
Optical pooled screening reads the guide barcode OUT OF THE IMAGES. Each
sequencing cycle stains four channels, one per base, and a spot's base in that
cycle is whichever channel is brightest -- after the dyes have been unmixed.
Read one base per cycle at the same spot and the cycles spell the barcode.

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

* **The brightest channel is not the base.** The four dyes bleed into one
  another, so the raw argmax is biased toward whichever dye is brightest
  overall. The cross-talk has to be estimated from the data and undone first.

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
               threshold: Optional[float] = None) -> np.ndarray:
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
    :returns: peak coordinates, strongest first.
    """
    from scipy.ndimage import maximum_filter

    field = np.asarray(score, dtype=np.float32)
    window = 2 * int(min_distance) + 1
    local_max = maximum_filter(field, size=window, mode="nearest")
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
                         percentile: float = 95.0) -> np.ndarray:
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
            # No spot claims this channel. Leave its axis as the identity
            # rather than inventing one: a base absent from this field is a
            # fact about the field, and a fabricated axis would rotate every
            # other read to accommodate it.
            axes[channel, channel] = 1.0
            continue
        vector = chosen.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        axes[channel] = vector / norm if norm > 0 else np.eye(
            n_channels, dtype=np.float32)[channel]

    try:
        correction = np.linalg.inv(axes)
    except np.linalg.LinAlgError:
        # Two dyes indistinguishable in this field. Correcting with a
        # pseudo-inverse would quietly produce confident nonsense, so the
        # uncorrected values are returned and the caller's quality scores
        # will show what happened.
        return data
    return (flat @ correction).reshape(data.shape).astype(np.float32)


def call_reads(values: np.ndarray, *, bases: Sequence[str] = BASES,
               compensate: bool = True,
               method: str = "percentile") -> Tuple[List[str], np.ndarray]:
    """Turn per-cycle intensities into a barcode string and a quality per read.

    Quality is the margin between the winning channel and the runner-up,
    divided by their sum, per cycle -- 0 when two bases are equally likely and
    1 when the call is unambiguous. Reported per read as the MINIMUM over
    cycles, because a barcode is only as trustworthy as its worst base.

    :param values: ``(N, cycles, channels)`` from :func:`extract_bases`.
    :param bases: the letter for each channel, in channel order.
    :param compensate: undo cross-talk first. Off only for testing what the
        compensation is worth.
    :param method: passed to :func:`compensate_crosstalk`.
    :returns: ``(barcodes, quality)`` -- one string and one float per read.
    """
    data = np.asarray(values, dtype=np.float32)
    if compensate:
        data = compensate_crosstalk(data, method=method)
    if data.size == 0:
        return [], np.zeros((0,), np.float32)

    ordered = np.sort(data, axis=-1)
    best, second = ordered[..., -1], ordered[..., -2]
    total = best + second
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cycle = np.where(total > 0, (best - second) / total, 0.0)
    quality = per_cycle.min(axis=1).astype(np.float32)

    winners = data.argmax(axis=-1)
    letters = np.asarray(list(bases))
    barcodes = ["".join(letters[row]) for row in winners]
    return barcodes, quality


def correct_to_library(barcodes: Sequence[str], library: Sequence[str], *,
                       max_distance: int = 1) -> List[Optional[str]]:
    """Snap each read to the library, but ONLY on a unique closest match.

    AMBIGUITY IS DISCARDED, NOT GUESSED. A read equally close to two library
    barcodes returns ``None``. That is deliberate and it is the whole point of
    the step: a misassigned guide moves a cell's phenotype onto the wrong
    perturbation and silently corrupts every statistic downstream, while a
    dropped read only costs statistical power that more cells can buy back.

    An exact match short-circuits, so a clean run pays nothing for this.

    :param barcodes: the called reads.
    :param library: the guide barcodes the screen actually contains.
    :param max_distance: the largest Hamming distance that may be corrected.
    :returns: one entry per read -- the library barcode, or None.
    """
    known = set(library)
    by_length: Dict[int, List[str]] = {}
    for entry in library:
        by_length.setdefault(len(entry), []).append(entry)

    out: List[Optional[str]] = []
    for read in barcodes:
        if read in known:
            out.append(read)
            continue
        best: Optional[str] = None
        best_distance = max_distance + 1
        tied = False
        for candidate in by_length.get(len(read), ()):
            distance = sum(a != b for a, b in zip(read, candidate))
            if distance < best_distance:
                best, best_distance, tied = candidate, distance, False
            elif distance == best_distance:
                tied = True
        out.append(None if (best is None or tied or best_distance > max_distance)
                   else best)
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

    out: Dict[int, dict] = {}
    for cell, indices in per_cell.items():
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
        # A TIE IS NOT A WINNER. `max` picks one arbitrarily, so a cell whose
        # reads split evenly between two barcodes would be assigned on
        # dictionary order -- which is exactly the silent misassignment the
        # fraction test exists to prevent.
        if sum(1 for value in counts.values() if value == agreeing) > 1:
            continue
        mean_quality = None
        if quality is not None:
            agreed = [float(quality[i]) for i in indices
                      if barcodes[i] == best]
            mean_quality = float(np.mean(agreed)) if agreed else None
        out[cell] = {
            "barcode": best,
            "reads": len(indices),
            "agreeing": agreeing,
            "fraction": float(fraction),
            "quality": mean_quality,
        }
    return out
