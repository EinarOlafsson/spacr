"""A wand click that changes nothing must still stop.

Instruction 310, A17. ``max_pixels`` counted only pixels that CHANGED
state, so a flood that changes nothing never incremented it and the
search walked the whole frame. Erasing where the mask is already empty,
or adding over ground the mask already owns, is the most ordinary wrong
click there is -- and it froze Make Masks for the length of a pure-Python
BFS over every pixel: 4.8 s on an 800x800 field, roughly half a minute at
2048x2048, with no way to cancel.

A second budget bounds VISITS. ``max_pixels`` still means what a user
thinks it means -- fill at most this many -- and the visit budget stops
the case where that number is never approached.

The tests below are about TIME as much as correctness, so each one that
measures says what it measured and why the threshold is where it is.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from spacr.qt.mask_engine import (VISIT_BUDGET_FACTOR, VISIT_BUDGET_FLOOR,
                                  magic_wand)

#: Big enough that an unbounded search is unmistakably slow, small enough
#: that the suite does not pay for it. The reported defect used this size.
SIDE = 800


@pytest.fixture
def uniform():
    """A field with no intensity structure, so tolerance rejects nothing.

    This is the shape that made the old budget useless: every pixel is
    in tolerance, so the BFS has somewhere to go until the frame runs
    out.
    """
    return np.zeros((SIDE, SIDE), np.uint8)


def test_erasing_an_already_empty_mask_returns_promptly(uniform):
    """THE REPORTED FREEZE. 4.8 s before; milliseconds now.

    The threshold is 1.5 s rather than something tighter because this is
    a pure-Python BFS on a shared machine and the point is the difference
    between a hitch and a hang, not a benchmark. Measured at 365 ms here
    against 4.8 s before, and the cap no longer grows with frame area --
    at 2048x2048 it is the same 365 ms rather than half a minute.
    """
    empty = np.zeros((SIDE, SIDE), np.uint8)
    started = time.perf_counter()
    out = magic_wand(uniform, empty, SIDE // 2, SIDE // 2, 10.0,
                     max_pixels=100, action="erase")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5, (
        f"a no-op erase took {elapsed:.2f} s; it is unbounded again")
    assert not out.any(), "an erase on an empty mask changed something"


def test_adding_over_ground_the_mask_already_owns_returns_promptly(uniform):
    """The other half of the same defect, and it is not symmetric.

    Add and erase increment the counter on opposite conditions, so a fix
    that only bounded one of them would leave this one walking the frame.
    """
    full = np.full((SIDE, SIDE), 255, np.uint8)
    started = time.perf_counter()
    out = magic_wand(uniform, full, SIDE // 2, SIDE // 2, 10.0,
                     max_pixels=100, action="add")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5, (
        f"a no-op add took {elapsed:.2f} s; it is unbounded again")
    assert out.all(), "an add on a full mask changed something"


def test_a_real_fill_still_gets_its_whole_budget(uniform):
    """THE HALF THAT MUST NOT REGRESS.

    Bounding work is only safe if a fill somebody meant still completes.
    A tolerance that accepts everything and a mask that is entirely set
    means every visited pixel also changes, so the change budget is what
    stops it -- exactly as before.
    """
    full = np.full((SIDE, SIDE), 255, np.uint8)
    out = magic_wand(uniform, full, SIDE // 2, SIDE // 2, 10.0,
                     max_pixels=100, action="erase")
    assert int((full > 0).sum() - (out > 0).sum()) == 100


@pytest.mark.parametrize("budget", [1, 10, 100, 5_000])
def test_no_budget_is_too_small_to_look_around_its_seed(uniform, budget):
    """The floor exists for this.

    Four visits per allowed change is generous for a large fill and
    absurd for a budget of one, so a floor keeps small deliberate fills
    workable. Without it a max_pixels of 1 would allow four visits, which
    is not enough to cross a single out-of-tolerance pixel.
    """
    full = np.full((SIDE, SIDE), 255, np.uint8)
    out = magic_wand(uniform, full, SIDE // 2, SIDE // 2, 10.0,
                     max_pixels=budget, action="erase")
    changed = int((full > 0).sum() - (out > 0).sum())
    assert changed == min(budget, SIDE * SIDE), (
        f"a budget of {budget} changed {changed}")


def test_re_wanding_an_owned_object_still_rewrites_all_of_it():
    """THE BEHAVIOUR THAT MUST NOT REGRESS, and it nearly did.

    Crossing pixels the mask already owns costs nothing against
    ``max_pixels`` ON PURPOSE: it is what lets a user re-run the wand
    with a wider tolerance on an object they already outlined and get the
    enlarged object rather than a partial one. A first version of the
    visit budget was tight enough to break this, and
    `test_cov_wf_qt_mask_engine.py` caught it -- which is why the floor is
    a hundred thousand rather than a multiple of the change budget.
    """
    image = np.full((40, 40), 100, dtype=np.uint16)
    owned = np.full((40, 40), 3, dtype=np.uint8)

    out = magic_wand(image, owned, 20, 20, tolerance=5, max_pixels=4,
                     action="add")
    assert int((out == 255).sum()) == 1600


def test_the_visit_budget_is_larger_than_any_object_and_smaller_than_a_frame():
    """The floor is the number that matters, and this says why.

    A pixel count cannot tell "usefully growing a large object" from
    "clicked on the background" -- on a uniform field neither is stopped
    by tolerance. So the floor is set above any single object somebody
    re-wands and below a large frame, which makes the wrong click a hitch
    rather than a hang without shortening any real fill.
    """
    assert VISIT_BUDGET_FLOOR >= 50_000, (
        "smaller than a large object, so re-wanding one would be cut short")
    assert VISIT_BUDGET_FLOOR <= 500_000, (
        "so large that a mis-click on a big frame hangs again")
    assert VISIT_BUDGET_FACTOR >= 3


def test_a_seed_outside_the_frame_is_refused_before_any_work(uniform):
    """Unchanged behaviour, asserted so the new budget did not disturb
    the early-out above it."""
    empty = np.zeros((SIDE, SIDE), np.uint8)
    assert magic_wand(uniform, empty, -1, 0, 10.0) is empty
    assert magic_wand(uniform, empty, 0, SIDE, 10.0) is empty
