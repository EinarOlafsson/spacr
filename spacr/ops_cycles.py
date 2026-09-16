"""Registering one field across the sequencing cycles that imaged it.

THE CYCLE AXIS IS NOT THE TILE AXIS, and the difference decides everything
in this module. :mod:`spacr.ops_register` places two DIFFERENT fields that
overlap by a fraction of a tile; here the same field is imaged eleven times
and the stage returns to it, so the displacement is a handful of pixels
rather than a pitch.

AND IT CANNOT USE HOECHST. Measured on the reference acquisition: cycle 1
holds one five-channel ``DAPI-CY3-A594-CY5-CY7`` stack per site, and
cycles 2 to 11 hold four separate base-channel files with NO DAPI at all.
So the channel every tile-axis registration runs on does not exist on this
axis after the first cycle, and averaging Hoechst across cycles is not
available here either.

WHAT REPLACES IT IS THE REFERENCE IMPLEMENTATION'S OWN ANSWER.
``ops/firesnake.py:_align_SBS`` offers ``method='SBS_mean'`` for exactly
this case: the maximum across the base channels, normalised by a high
percentile and clipped, which turns four sparse spot channels into one
dense target. Measured on well A1 at two sites, cycles 2 to 11 against
cycle 1:

    site 100   peak/mean 169 to 236   shift (-6..-27,  0..10)
    site 200   peak/mean 218 to 281   shift (-4..-25, 11..21)

    20 of 20 cycles registered.

IT IS NOT A FALLBACK, IT IS THE BETTER CHANNEL. The tile axis on DAPI peaks
at 23 to 85; this peaks at 169 to 281. Spots are punctate and abundant and
nuclei are smooth and few, so losing the Hoechst costs nothing here.

AND THE OFFSET BELONGS TO THE SITE, NOT THE CYCLE. Site 100 sits at x = 0
to 10 across the eleven cycles and site 200 at x = 11 to 21. One rigid
transform per cycle for a whole well would carry that difference into every
field, so each site is registered against its own reference -- which is
also what the reference implementation does, one field's cycle stack at a
time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ops_register import Registration, phase_correlate

#: The percentile the target is divided by before clipping. The reference
#: uses 70; a spot channel is mostly background, so a high percentile still
#: lands inside the noise and the division brings the spots to order 1.
NORMALISE_PERCENTILE: float = 70.0

#: How far a cycle may move and still be believed, in pixels. The stage
#: returns to the same field, so this is a stage-repeatability number and
#: not a pitch: the measured spread on well A1 is 4 to 27 px, and 64 leaves
#: room for a worse stage without admitting a wrong field.
CYCLE_TOLERANCE: int = 64


def sbs_mean_target(planes: Sequence[np.ndarray], *,
                    percentile: float = NORMALISE_PERCENTILE,
                    cutoff: float = 1.0) -> np.ndarray:
    """Turn one cycle's base channels into a single registration target.

    THE MAXIMUM, NOT THE MEAN, over channels. A read is bright in exactly
    one channel per cycle -- that is what a base call is -- so averaging
    the four divides every spot by four and leaves the background where it
    was. The maximum keeps each spot at its own brightness whichever
    channel carried it.

    :param planes: the base channels for one cycle of one field.
    :param percentile: the percentile the result is divided by.
    :param cutoff: the value the result is clipped at afterwards, so a
        saturated spot cannot dominate the correlation on its own.
    :returns: a float32 image, the same shape as one plane.
    :raises ValueError: when no planes are given, or they disagree in shape.
    """
    stack = [np.asarray(plane, dtype=np.float32) for plane in planes]
    if not stack:
        raise ValueError("sbs_mean_target needs at least one channel")
    if any(plane.shape != stack[0].shape for plane in stack):
        raise ValueError(
            "the base channels of one cycle must share a shape: "
            f"{[plane.shape for plane in stack]}"
        )
    target = np.max(np.stack(stack), axis=0)
    scale = float(np.percentile(target, percentile))
    if scale > 0:
        target = target / scale
    return np.clip(target, 0.0, cutoff).astype(np.float32)


def register_cycles(reference: np.ndarray,
                    cycles: Mapping[int, np.ndarray], *,
                    tolerance: int = CYCLE_TOLERANCE,
                    **kwargs) -> Dict[int, Registration]:
    """Place every cycle of one field against that field's reference.

    EACH CYCLE AGAINST THE REFERENCE DIRECTLY, never against its
    predecessor. Chaining c1->c2->c3 saves one registration and makes every
    later cycle depend on every earlier one: a single bad link displaces
    the rest and the residual does not say which link failed. Registering
    each independently is what lets one cycle be dropped without costing
    the others -- PART 9's rule that a cycle which will not register must
    not cost the well.

    :param reference: the reference cycle's target, from
        :func:`sbs_mean_target`.
    :param cycles: cycle number to that cycle's target.
    :param tolerance: how far a cycle may land from the reference and still
        be accepted, in pixels.
    :param kwargs: passed to :func:`spacr.ops_register.phase_correlate`.
    :returns: cycle number to its :class:`Registration`, including the
        refused ones -- a caller reporting `n_cycles` needs to know a cycle
        was tried and refused, not merely that it is absent.
    """
    out: Dict[int, Registration] = {}
    for cycle, target in cycles.items():
        out[cycle] = phase_correlate(
            reference, target, expected=(0, 0), tolerance=tolerance, **kwargs)
    return out


def accepted_cycles(registrations: Mapping[int, Registration]) -> Tuple[int, ...]:
    """The cycles whose registration was believed, in order.

    :param registrations: the result of :func:`register_cycles`.
    :returns: the accepted cycle numbers, ascending.
    """
    return tuple(sorted(c for c, r in registrations.items() if r.accepted))


#: How far one base channel may sit from the first channel of its own cycle
#: and still be moved, in pixels (372 PART 14-L: shifts up to 14 px, 66 of 66
#: accepted at 16). Private: it is a property of this filter set and stage,
#: recorded here rather than offered as a knob nobody has measured.
_CHANNEL_TOLERANCE: int = 16

#: The percentile a channel is clipped at before channel-to-channel
#: registration. Spots differ between channels by definition -- a read is
#: bright in one of them -- so what the four share is the cell background,
#: and clipping the spots away is what lets the correlation see it.
_CHANNEL_CLIP_PERCENTILE: float = 90.0


def _translate(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move an image's content by whole pixels, filling the gap with zeros.

    :param image: an array whose last two axes are rows and columns.
    :param dy: rows to move down; negative moves up.
    :param dx: columns to move right; negative moves left.
    :returns: the moved copy, the same shape and dtype.

    Integer and zero-filled on purpose: a registration measured to the pixel
    is applied to the pixel, and an interpolated edge would invent intensity
    that a base call then reads.
    """
    dy, dx = int(dy), int(dx)
    out = np.zeros_like(image)
    height, width = image.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        return out
    out[..., max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = \
        image[..., max(-dy, 0):height + min(-dy, 0),
              max(-dx, 0):width + min(-dx, 0)]
    return out


def _clipped(plane: np.ndarray) -> np.ndarray:
    """A channel with its spots clipped off, for channel-to-channel registration.

    :param plane: one base channel.
    :returns: the plane clipped between its minimum and
        :data:`_CHANNEL_CLIP_PERCENTILE`, as float32.
    """
    plane = np.asarray(plane, dtype=np.float32)
    low, high = np.percentile(plane, [0.0, _CHANNEL_CLIP_PERCENTILE])
    return np.clip(plane, low, high)


def _align_channels(planes: Sequence[np.ndarray], *, tolerance: int,
                    gpu: bool):
    """Move every base channel of one cycle onto that cycle's first channel.

    :param planes: the cycle's base channels, in channel order.
    :param tolerance: the largest shift accepted, in pixels.
    :param gpu: passed to the phase correlation.
    :returns: ``(planes, shifts, refused)`` -- the moved float32 planes, one
        applied ``(dy, dx)`` per channel, and the indices of the channels
        whose registration was refused (left where they were).

    372 PART 14-L found this step missing on the first real plate: the four
    channels of one cycle sat up to 14 px apart, cycle-to-cycle registration
    cannot see that, and the decode ran end to end at 2.8 % library match
    instead of 77.9 %. The reference implementation does the same by default
    (``_align_SBS(..., align_within_cycle=True)``).
    """
    first = np.asarray(planes[0], dtype=np.float32)
    base = _clipped(first)
    moved = [first]
    shifts = [(0, 0)]
    refused = []
    for index, plane in enumerate(planes[1:], start=1):
        plane = np.asarray(plane, dtype=np.float32)
        found = phase_correlate(base, _clipped(plane), expected=(0, 0),
                                tolerance=tolerance, gpu=gpu)
        if found.accepted:
            shift = (int(found.dy), int(found.dx))
            moved.append(_translate(plane, *shift))
            shifts.append(shift)
        else:
            moved.append(plane)
            shifts.append((0, 0))
            refused.append(index)
    return moved, tuple(shifts), tuple(refused)


@dataclass(frozen=True)
class AlignedField:
    """One field's sequencing cycles, every channel moved into one frame.

    :ivar stack: ``(cycles, channels, Y, X)`` float32, one entry per cycle
        in :attr:`kept` and in that order.
    :ivar kept: the cycle numbers the stack holds, ascending. The reference
        cycle is always among them.
    :ivar refused: cycles whose registration against the reference was not
        accepted. They are left out of the stack, because base calls read at
        untrusted coordinates decode to a wrong barcode rather than to none.
    :ivar missing: cycles offered with no planes at all -- a file that could
        not be read, for instance.
    :ivar channel_shifts: ``cycle -> one (dy, dx) per channel``, the shift
        that moved each channel onto the first channel of its cycle.
    :ivar cycle_shifts: ``cycle -> (dy, dx)``, the shift that moved each kept
        cycle onto the reference; the reference's own is ``(0, 0)``.
    :ivar refused_channels: ``(cycle, channel)`` pairs whose within-cycle
        registration was refused and which were therefore not moved.
    """

    stack: np.ndarray
    kept: Tuple[int, ...]
    refused: Tuple[int, ...]
    missing: Tuple[int, ...]
    channel_shifts: Dict[int, Tuple[Tuple[int, int], ...]]
    cycle_shifts: Dict[int, Tuple[int, int]]
    refused_channels: Tuple[Tuple[int, int], ...] = ()


def align_field(planes: Mapping[int, Optional[Sequence[np.ndarray]]], *,
                reference: Optional[int] = None,
                tolerance: int = CYCLE_TOLERANCE,
                gpu: bool = True) -> AlignedField:
    """Register one field's channels within each cycle, then its cycles.

    Each base channel is first moved onto its cycle's first channel, on the
    clipped cell background the channels share -- on the measured plate they
    sat up to 14 px apart, which no cycle-to-cycle registration can see --
    and each cycle is then moved onto the reference with
    :func:`sbs_mean_target`. A cycle offered as ``None`` or with a missing
    plane is reported in :attr:`AlignedField.missing` and one whose
    registration is refused in :attr:`AlignedField.refused`; the rest are
    still aligned, so one lost cycle does not cost the field.

    :param planes: ``cycle -> the base channels of that cycle``, every plane
        one 2-D image of the same shape.
    :param reference: the cycle everything is moved onto. None takes the
        lowest cycle that has planes.
    :param tolerance: how far a cycle may land from the reference and still
        be accepted, in pixels.
    :param gpu: let the phase correlations run on the card.
    :returns: the aligned field.
    :raises ValueError: when no cycle has planes, or when the reference
        cycle has none.
    """
    usable = {}
    missing = []
    for cycle in sorted(planes):
        offered = planes[cycle]
        if offered is None or len(offered) == 0 or any(
                plane is None for plane in offered):
            missing.append(int(cycle))
        else:
            usable[int(cycle)] = offered
    if not usable:
        raise ValueError("no cycle of this field has planes to align")
    if reference is None:
        reference = min(usable)
    if reference not in usable:
        raise ValueError(
            f"the reference cycle {reference} has no planes; every other "
            "cycle is placed against it, so choose a cycle that was read")

    moved = {}
    channel_shifts = {}
    refused_channels = []
    for cycle, offered in usable.items():
        aligned, shifts, refused = _align_channels(
            offered, tolerance=_CHANNEL_TOLERANCE, gpu=gpu)
        moved[cycle] = aligned
        channel_shifts[cycle] = shifts
        refused_channels.extend((cycle, channel) for channel in refused)

    target = sbs_mean_target(moved[reference])
    others = {cycle: sbs_mean_target(moved[cycle])
              for cycle in usable if cycle != reference}
    registrations = register_cycles(target, others, tolerance=tolerance,
                                    gpu=gpu)
    kept = sorted([reference] + list(accepted_cycles(registrations)))
    cycle_shifts = {reference: (0, 0)}
    layers = []
    for cycle in kept:
        stack = np.stack(moved[cycle])
        if cycle != reference:
            found = registrations[cycle]
            cycle_shifts[cycle] = (int(found.dy), int(found.dx))
            stack = _translate(stack, *cycle_shifts[cycle])
        layers.append(stack)
    refused = tuple(sorted(c for c, r in registrations.items()
                           if not r.accepted))
    return AlignedField(
        stack=np.stack(layers).astype(np.float32, copy=False),
        kept=tuple(kept), refused=refused, missing=tuple(missing),
        channel_shifts=channel_shifts, cycle_shifts=cycle_shifts,
        refused_channels=tuple(refused_channels))
