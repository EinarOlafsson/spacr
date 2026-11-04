"""Registering one field across the sequencing cycles that imaged it.

Registration across cycles differs from registration across tiles.
:mod:`spacr.ops_register` places two different fields that overlap by a
fraction of a tile; here the same field is imaged eleven times and the stage
returns to it, so the displacement is a handful of pixels rather than a
pitch.

Cycle registration cannot use Hoechst. Measured on the reference
acquisition: cycle 1 holds one five-channel ``DAPI-CY3-A594-CY5-CY7`` stack
per site, and cycles 2 to 11 hold four separate base-channel files with no
DAPI channel.
So the channel every tile-axis registration runs on does not exist on this
axis after the first cycle, and averaging Hoechst across cycles is not
available here either.

The target used instead comes from the reference implementation.
``ops/firesnake.py:_align_SBS`` offers ``method='SBS_mean'`` for exactly
this case: the maximum across the base channels, normalised by a high
percentile and clipped, which turns four sparse spot channels into one
dense target. Measured on well A1 at two sites, cycles 2 to 11 against
cycle 1:

    site 100   peak/mean 169 to 236   shift (-6..-27,  0..10)
    site 200   peak/mean 218 to 281   shift (-4..-25, 11..21)

    20 of 20 cycles registered.

This target gives stronger correlation peaks than DAPI: tile registration
on DAPI peaks at 23 to 85; this target peaks at 169 to 281. Spots are
punctate and abundant while nuclei are smooth and few, so the missing
Hoechst channel does not reduce registration quality.

The offset depends on the site, not only on the cycle. Site 100 sits at
x = 0 to 10 across the eleven cycles and site 200 at x = 11 to 21. One rigid
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

    The maximum over channels is used, not the mean. A read is bright in exactly
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

    Each cycle is registered against the reference directly, never against
    its predecessor. Chaining c1->c2->c3 saves one registration and makes
    every later cycle depend on every earlier one: a single bad link
    displaces the rest and the residual does not say which link failed.
    Registering each cycle independently lets one cycle be dropped without
    affecting the others, so a cycle that does not register does not cost
    the well.

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
    """The cycles whose registration was accepted, in ascending order.

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
    :ivar spot_replaced: ``(cycle, channel)`` pairs whose background shift
        the spot registration overruled (``channel_mode='spots'`` only).
    """

    stack: np.ndarray
    kept: Tuple[int, ...]
    refused: Tuple[int, ...]
    missing: Tuple[int, ...]
    channel_shifts: Dict[int, Tuple[Tuple[int, int], ...]]
    cycle_shifts: Dict[int, Tuple[int, int]]
    refused_channels: Tuple[Tuple[int, int], ...] = ()
    spot_replaced: Tuple[Tuple[int, int], ...] = ()


#: How the channels are registered (``align_field(channel_mode=...)``).
#: ``background`` moves each base channel onto its cycle's first channel on
#: the clipped cell background, then each cycle onto the reference.
#: ``spots`` registers every channel of every cycle DIRECTLY onto the
#: reference cycle's spots: both images are Laplacian-of-Gaussian filtered,
#: so the cell background and the illumination -- which every channel of one
#: stage position shares at zero shift -- are gone and only the reads are left.
CHANNEL_MODES = ("background", "spots")

#: How far a channel's spot registration may disagree with its background
#: registration before the spot answer replaces it, in pixels. Within this the
#: two agree and the background answer stands, so a good field is unchanged.
_SPOT_AGREEMENT_PX = 2

#: The weakest spot-registration peak believed, against the surface mean.
#: Measured on the Funk et al. 2022 plates (2026-11-04): a true spot peak on a
#: 1,480 px field stands at hundreds; a field with no reads at a few.
_SPOT_MIN_PEAK = 20.0

#: The spots overrule the background only when the spot surface at the
#: background's shift is below this share of the spot peak. Where the
#: background is right the reads coincide there too and the surface is near
#: its peak; where it locked onto the zero-shift pattern the reads do not
#: coincide and the surface there is noise.
_SPOT_OVERRULE_SHARE = 0.5


def _spot_surface(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The phase-correlation surface of two spot images, CPU.

    :param first: the reference spots.
    :param second: the spots to place.
    :returns: the real surface; its peak at ``(dy, dx)`` means rolling
        ``second`` by that lands it on ``first``, as in
        :func:`spacr.ops_register.phase_correlate`.
    """
    a = np.fft.rfft2(np.asarray(first, dtype=np.float64))
    b = np.fft.rfft2(np.asarray(second, dtype=np.float64))
    cross = a * np.conj(b)
    cross /= np.abs(cross) + 1e-12
    return np.fft.irfft2(cross, s=np.shape(first))


def _surface_near(surface: np.ndarray, shift) -> float:
    """The surface's largest value within one pixel of ``shift``."""
    height, width = surface.shape
    return float(max(surface[(shift[0] + dy) % height, (shift[1] + dx) % width]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)))


def _spot_image(plane: np.ndarray) -> np.ndarray:
    """A plane's reads with its background removed: -LoG(1 px), clipped at 0.

    :param plane: one 2-D channel.
    :returns: float32, high only at spot centres.
    """
    from scipy import ndimage

    return np.clip(-ndimage.gaussian_laplace(
        np.asarray(plane, dtype=np.float32), 1.0), 0, None)


def _spot_shifts(usable, kept, background_total, reference, *, tolerance,
                 gpu):
    """Each channel's total shift onto the reference cycle, read off the spots.

    :param usable: ``cycle -> raw planes``.
    :param kept: the cycles whose background registration was accepted.
    :param background_total: ``cycle -> [(dy, dx) per channel]``, the
        background answer: cycle shift plus channel shift.
    :param reference: the reference cycle.
    :param tolerance: the largest total shift accepted, in pixels.
    :param gpu: passed to the phase correlation.
    :returns: ``(totals, replaced)`` -- the shift each channel is moved by,
        and the ``(cycle, channel)`` pairs whose background answer the spots
        overruled.

    WHY: on four of the Funk plates' weak wells one cycle's channels came
    back from the background registration at (0, 0) -- the peak every
    channel of one stage position shares through the illumination and the
    camera -- while the true drift was 10 to 25 px, and that one cycle then
    carried 70 to 84 % of the wells' base-call errors. A read is bright in
    exactly one channel of a cycle and in some channel of the reference, so a
    channel's spots are a subset of the reference's and register onto them
    directly, with nothing at zero shift to lock onto.
    """
    ref_spots = [_spot_image(plane) for plane in usable[reference]]
    target = np.max(np.stack([_translate(image, *background_total[reference][k])
                              for k, image in enumerate(ref_spots)]), axis=0)
    totals = {}
    replaced = []
    for cycle in kept:
        totals[cycle] = []
        for k, plane in enumerate(usable[cycle]):
            mine = background_total[cycle][k]
            spots = ref_spots[k] if cycle == reference else _spot_image(plane)
            surface = _spot_surface(target, spots)
            flat = int(np.argmax(surface))
            peak = float(surface.flat[flat])
            mean = float(np.abs(surface).mean())
            py, px = np.unravel_index(flat, surface.shape)
            dy = int(py) - surface.shape[0] if py > surface.shape[0] // 2 else int(py)
            dx = int(px) - surface.shape[1] if px > surface.shape[1] // 2 else int(px)
            agree = (abs(dy - mine[0]) <= _SPOT_AGREEMENT_PX
                     and abs(dx - mine[1]) <= _SPOT_AGREEMENT_PX)
            believed = (mean > 0 and peak / mean >= _SPOT_MIN_PEAK
                        and abs(dy) <= tolerance and abs(dx) <= tolerance
                        and _surface_near(surface, mine)
                        < _SPOT_OVERRULE_SHARE * peak)
            if believed and not agree:
                totals[cycle].append((dy, dx))
                replaced.append((int(cycle), int(k)))
            else:
                totals[cycle].append(tuple(int(v) for v in mine))
    return totals, replaced


def align_field(planes: Mapping[int, Optional[Sequence[np.ndarray]]], *,
                reference: Optional[int] = None,
                tolerance: int = CYCLE_TOLERANCE,
                gpu: bool = True,
                channel_mode: str = "background") -> AlignedField:
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
    :param gpu: allow the phase correlations to run on the GPU.
    :param channel_mode: one of :data:`CHANNEL_MODES`. ``spots`` runs the
        background registration first and then checks every channel against
        the reference cycle's spots, replacing the shifts the spots disagree
        with; :attr:`AlignedField.spot_replaced` lists those.
    :returns: the aligned field.
    :raises ValueError: when no cycle has planes, when the reference
        cycle has none, or for an unknown ``channel_mode``.
    """
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"channel_mode must be one of {list(CHANNEL_MODES)}; "
                         f"got {channel_mode!r}")
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
    for cycle in kept:
        if cycle != reference:
            found = registrations[cycle]
            cycle_shifts[cycle] = (int(found.dy), int(found.dx))
    replaced = []
    if channel_mode == "spots":
        background_total = {
            cycle: [(cycle_shifts[cycle][0] + dy, cycle_shifts[cycle][1] + dx)
                    for dy, dx in channel_shifts[cycle]] for cycle in kept}
        totals, replaced = _spot_shifts(
            usable, kept, background_total, reference,
            tolerance=tolerance + _CHANNEL_TOLERANCE, gpu=gpu)
        for cycle in kept:
            channel_shifts[cycle] = tuple(
                (dy - cycle_shifts[cycle][0], dx - cycle_shifts[cycle][1])
                for dy, dx in totals[cycle])
            moved[cycle] = [_translate(np.asarray(plane, dtype=np.float32), *shift)
                            for plane, shift in zip(usable[cycle], totals[cycle])]
    layers = []
    for cycle in kept:
        stack = np.stack(moved[cycle])
        if cycle != reference and channel_mode != "spots":
            stack = _translate(stack, *cycle_shifts[cycle])
        layers.append(stack)
    refused = tuple(sorted(c for c, r in registrations.items()
                           if not r.accepted))
    return AlignedField(
        stack=np.stack(layers).astype(np.float32, copy=False),
        kept=tuple(kept), refused=refused, missing=tuple(missing),
        channel_shifts=channel_shifts, cycle_shifts=cycle_shifts,
        refused_channels=tuple(refused_channels),
        spot_replaced=tuple(replaced))
