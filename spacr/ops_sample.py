"""Every other channel, sampled at the object coordinates B4 assigned.

SAMPLING EVERY OTHER CHANNEL at the object coordinates the numbering step
assigned -- the part that can be built and checked without a
real cycle stack. B4 produced a plate-level object id and a well-frame
centroid for every nucleus; this is what reads the remaining channels AT
those coordinates and keys the result to those ids.

    C1  SEQUENCING CHANNELS, PER CYCLE ... NEVER averaged across cycles --
        that is the one operation that destroys a barcode while leaving it
        decodable.

THAT SENTENCE IS THE WHOLE DESIGN CONSTRAINT AND IT IS ENFORCED, NOT
DOCUMENTED. A mean across cycles of a four-channel readout still looks like a
four-channel readout: ``call_reads`` accepts it, returns a barcode of the
right length, and every downstream table fills in. The run completes, the
numbers are plausible, and the barcodes are noise. There is no later check
that can catch it, because nothing downstream knows what the per-cycle values
were. So :func:`decode_input` refuses an array whose cycle axis has been
collapsed, and this module offers no way to average one.

    THE PHENOTYPE PATH IS THE OPPOSITE CASE, which is why the two are
    separate functions rather than one with a flag. A phenotype channel
    measured in several cycles IS several samples of one quantity and
    averaging it is the right thing -- C4 emits "the schema `measure`
    already emits". Same plate, same objects, opposite correct answer; a
    shared flag would make that a caller's choice, and it is not one.

The samplers are injected, as they are in :mod:`spacr.ops_compose` and
:mod:`spacr.ops_objects`: nothing here opens a file or knows a transform's
provenance, so all of it is testable on planted truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.ops_sample")

__all__ = [
    "SampleError",
    "sample_objects", "decode_input", "average_over_cycles",
    "reads_rows", "barcode_rows",
]


class SampleError(ValueError):
    """A sampling that cannot mean anything, with the way out in the text."""


def sample_objects(objects: Sequence, image: np.ndarray, *,
                   transform: Optional[Callable[[float, float],
                                                Tuple[float, float]]] = None,
                   radius: int = 1) -> np.ndarray:
    """Read ``image`` at each object's centroid, through ``transform``.

    :param objects: :class:`spacr.ops_objects.PlateObject` values, or anything
        with ``centroid_y`` and ``centroid_x`` in the WELL frame.
    :param image: one channel of one cycle, in that cycle's own frame.
    :param transform: maps a well-frame ``(y, x)`` into this image's frame --
        the A2/A3 transforms of PART 7. ``None`` means the image is already in
        the well frame.
    :param radius: half-width of the square averaged around each point. The
        default 1 gives a 3x3, which is what a diffraction-limited spot
        occupies; 0 reads the single pixel.
    :returns: one float per object, NaN where the point falls outside.

    NaN RATHER THAN A CLAMPED EDGE PIXEL. An object whose coordinates land
    off this cycle's field was not measured in it, and the honest value for
    "not measured" is not the brightness of the nearest border pixel -- that
    would be a number, and a number gets averaged into a barcode.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise SampleError(
            f"a channel is one 2-D image; got shape {image.shape}. Pass one "
            f"cycle's one channel, not a stack -- the axes are what keeps "
            f"cycles separate.")
    height, width = image.shape
    radius = max(0, int(radius))

    out = np.full(len(objects), np.nan, dtype=float)
    for index, one in enumerate(objects):
        y, x = float(one.centroid_y), float(one.centroid_x)
        if transform is not None:
            y, x = transform(y, x)
        row, column = int(round(y)), int(round(x))
        if not (0 <= row < height and 0 <= column < width):
            continue
        top, bottom = max(0, row - radius), min(height, row + radius + 1)
        left, right = max(0, column - radius), min(width, column + radius + 1)
        patch = image[top:bottom, left:right]
        if patch.size:
            out[index] = float(patch.mean())
    return out


def decode_input(per_cycle: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Stack per-cycle, per-channel samples into ``call_reads``' input.

    :param per_cycle: ``per_cycle[c][k]`` is :func:`sample_objects`' output
        for cycle ``c``, channel ``k``. Every entry must be the same length.
    :returns: ``(objects, cycles, channels)`` -- the shape
        :func:`spacr.ops_sbs.call_reads` expects.
    :raises SampleError: when a cycle is missing channels, when the lengths
        disagree, or -- the one that matters -- when only ONE cycle is
        offered, which is what a caller who has already averaged them hands
        over.

    ONE CYCLE IS REFUSED AND THE REFUSAL IS THE POINT. A barcode is a sequence
    across cycles; a single "cycle" reaching here is either a run that lost
    its cycle axis or a caller who collapsed it. Both produce a decodable
    barcode made of nothing, and neither is visible downstream. If a
    single-cycle experiment ever needs decoding it should say so explicitly
    rather than arrive looking like this mistake.
    """
    cycles = [list(one) for one in per_cycle]
    if not cycles:
        raise SampleError(
            "no cycles were sampled; a barcode needs one letter per cycle")
    if len(cycles) == 1:
        raise SampleError(
            "only one cycle reached the decoder. A barcode is a sequence "
            "ACROSS cycles, so this is either a lost cycle axis or an "
            "average taken over them -- and 372 names that as 'the one "
            "operation that destroys a barcode while leaving it decodable'. "
            "The call would succeed and the barcodes would be noise.")

    widths = {len(one) for one in cycles}
    if len(widths) != 1:
        raise SampleError(
            f"the cycles offer different channel counts ({sorted(widths)}); "
            f"a base call compares the same channels in every cycle")

    lengths = {len(channel) for one in cycles for channel in one}
    if len(lengths) != 1:
        raise SampleError(
            f"the samples have different lengths ({sorted(lengths)}); every "
            f"cycle and channel must describe the same objects in the same "
            f"order, because the object id is the join key")

    return np.stack([np.stack(one, axis=-1) for one in cycles], axis=1)


def average_over_cycles(per_cycle: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of one PHENOTYPE channel across cycles. Correct here, only here.

    C4 samples phenotype channels "at the same object ids", and a phenotype
    measured in several cycles is several samples of one quantity -- so
    averaging raises its precision, exactly as averaging the Hoechst across
    tiles did in B1.

    SEPARATE FROM :func:`decode_input` ON PURPOSE. The same operation is
    right for a phenotype channel and catastrophic for a sequencing one, so
    it is two functions and not one with a flag: a flag would make that a
    caller's decision, and it is a property of the channel.

    NaN-aware, because :func:`sample_objects` returns NaN for an object this
    cycle did not cover, and an object measured in nine cycles of eleven
    should get the mean of the nine rather than NaN.
    """
    stack = np.stack([np.asarray(one, dtype=float) for one in per_cycle])
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def reads_rows(objects: Sequence, values: np.ndarray, *,
               channels: Sequence[str],
               bases: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """``ops_reads`` rows: one per object per cycle, per Phase D's columns.

    :param values: ``(objects, cycles, channels)`` from :func:`decode_input`.
    :param channels: a name per channel, for the column names.
    :param bases: the letter each channel votes for; when given, each row
        carries the base this cycle called and its margin.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise SampleError(
            f"ops_reads needs (objects, cycles, channels); got {values.shape}")
    if values.shape[0] != len(objects):
        raise SampleError(
            f"{values.shape[0]} sampled rows against {len(objects)} objects; "
            f"the object id is the join key and cannot be inferred")
    if values.shape[2] != len(channels):
        raise SampleError(
            f"{values.shape[2]} channels sampled but {len(channels)} names")

    rows: List[Dict[str, object]] = []
    for index, one in enumerate(objects):
        for cycle in range(values.shape[1]):
            intensities = values[index, cycle]
            row: Dict[str, object] = {
                "object_id": int(one.object_id),
                "cycle": int(cycle + 1),
            }
            for name, value in zip(channels, intensities):
                row[str(name)] = float(value)
            if bases is not None and np.isfinite(intensities).all():
                order = np.argsort(intensities)
                best, second = intensities[order[-1]], intensities[order[-2]]
                total = best + second
                row["base"] = str(bases[int(order[-1])])
                row["quality"] = float((best - second) / total) if total > 0 else 0.0
            rows.append(row)
    return rows


def barcode_rows(objects: Sequence, values: np.ndarray, *,
                 bases: Optional[Sequence[str]] = None,
                 library: Optional[Sequence[str]] = None,
                 ) -> List[Dict[str, object]]:
    """``ops_barcodes`` rows: one per object, assembled across cycles.

    Delegates the call itself to :func:`spacr.ops_sbs.call_reads`, which
    already compensates cross-talk and reports quality as the MINIMUM over
    cycles -- "a barcode is only as trustworthy as its worst base". Repeating
    that here would be a second decoder to keep in step.

    :param library: when given, each barcode is corrected to the nearest
        library member through :func:`spacr.ops_sbs.correct_to_library`, and
        the row carries what it mapped to.
    """
    from .ops_sbs import call_reads

    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise SampleError(
            f"ops_barcodes needs (objects, cycles, channels); got {values.shape}")

    kwargs = {"bases": bases} if bases is not None else {}
    barcodes, quality = call_reads(values, **kwargs)

    mapped: Optional[Sequence] = None
    if library is not None:
        from .ops_sbs import correct_to_library

        mapped = correct_to_library(barcodes, library)

    rows: List[Dict[str, object]] = []
    for index, one in enumerate(objects):
        row: Dict[str, object] = {
            "object_id": int(one.object_id),
            "barcode": str(barcodes[index]),
            "quality": float(quality[index]),
            "n_cycles": int(values.shape[1]),
        }
        if mapped is not None:
            # `correct_to_library` returns None for a read that is equally
            # close to two library barcodes -- "AMBIGUITY IS DISCARDED, NOT
            # GUESSED". That None must survive into the table as an empty
            # cell rather than the string "None", which would look like a
            # guide called None and join against nothing.
            entry = mapped[index]
            row["mapped_guide"] = "" if entry is None else str(entry)
        rows.append(row)
    return rows
