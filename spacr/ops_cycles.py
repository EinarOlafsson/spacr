"""Registering one field across the sequencing cycles that imaged it.

THE CYCLE AXIS IS NOT THE TILE AXIS, and the difference decides everything
in this module. :mod:`spacr.ops_register` places two DIFFERENT fields that
overlap by a fraction of a tile; here the same field is imaged eleven times
and the stage returns to it, so the displacement is a handful of pixels
rather than a pitch.

AND IT CANNOT USE HOECHST. Instruction 372 PART 10 measured this
acquisition: cycle 1 holds one five-channel ``DAPI-CY3-A594-CY5-CY7`` stack
per site and cycles 2 to 11 hold four separate base-channel files with NO
DAPI at all. So the channel every tile-axis registration runs on does not
exist on this axis after the first cycle, and PART 7's plan to average
Hoechst across cycles cannot run either.

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
