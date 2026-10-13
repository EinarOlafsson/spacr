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

#: The second frame the no-op tests time, 6.55x the area of the first.
#: The pair is the measurement; neither number means anything alone.
BIG_SIDE = 2048

#: How much longer the 6.55x-larger frame may take. A bounded search does
#: the same work on both, so the honest expectation is 1.0 and this is
#: slack for a shared machine; an unbounded one tracks area and scores
#: ~6.3. Measured 2026-09-15 on a deliberately loaded box: bounded 1.03
#: (erase) and 1.02 (add), unbounded 6.26 against an area ratio of 6.25.
#: Anything between 2.5 and 6 would separate them equally well.
AREA_GROWTH_ALLOWED = 2.5

#: A hang detector, not a benchmark. CI has been seen at 2.57 s on a busy
#: runner while the local number was 0.31 s, which is what an absolute
#: second-count measures on shared hardware: the runner, not the code.
NOT_A_HANG = 20.0


def _no_op_wand_seconds(side, action):
    """Time one wand click that cannot change anything, on a ``side`` frame.

    Uniform field, so tolerance rejects nothing and the search has
    somewhere to go until something stops it. The mask is already in the
    state the action would put it in, so no visited pixel ever counts
    against ``max_pixels`` -- which is the defect this file is about.
    """
    import numpy as np

    image = np.zeros((side, side), np.uint8)
    mask = (np.zeros((side, side), np.uint8) if action == "erase"
            else np.full((side, side), 255, np.uint8))
    started = time.perf_counter()
    magic_wand(image, mask, side // 2, side // 2, 10.0,
               max_pixels=100, action=action)
    return time.perf_counter() - started


@pytest.fixture
def uniform():
    """A field with no intensity structure, so tolerance rejects nothing.

    This is the shape that made the old budget useless: every pixel is
    in tolerance, so the BFS has somewhere to go until the frame runs
    out.
    """
    return np.zeros((SIDE, SIDE), np.uint8)


@pytest.mark.parametrize("action", ["erase", "add"])
def test_a_no_op_wand_does_not_get_slower_on_a_bigger_frame(action):
    """THE REPORTED FREEZE, ASKED AS A RATIO RATHER THAN A STOPWATCH.

    Both halves of the defect at once -- add and erase increment the
    counter on opposite conditions, so a fix that bounded only one would
    leave the other walking the frame.

    WHY NOT `elapsed < 1.5`, WHICH IS WHAT THIS WAS. That assertion
    measured the machine. It passed at 365 ms locally and FAILED ON CI at
    2.39 s and 2.57 s -- not because the wand regressed, but because a
    pure-Python BFS on a shared runner is slow, and no absolute number is
    both tight enough to catch the bug and loose enough to survive a busy
    box. Raising the ceiling until CI went green would have kept the shape
    of the mistake and thrown away the signal.

    WHAT IS ACTUALLY BEING CLAIMED is in this file's own title: the search
    is bounded BY WORK. A bounded search does the same work on an 800x800
    frame and a 2048x2048 one, so it takes the same time on both however
    fast or busy the machine is. An unbounded one walks the frame and
    tracks its area. The ratio cancels the machine out, which an absolute
    threshold cannot do.

    Measured 2026-09-15 on a deliberately loaded box: erase 307 ms and
    316 ms, add 300 ms and 307 ms -- ratios 1.03 and 1.02 against an area
    ratio of 6.55. :func:`test_the_ratio_can_see_an_unbounded_search` runs
    the same probe with the budget lifted and gets 6.26, so the two cases
    are six standard-shaped multiples apart rather than a fine judgement.
    """
    small = _no_op_wand_seconds(SIDE, action)
    big = _no_op_wand_seconds(BIG_SIDE, action)

    assert big < NOT_A_HANG, (
        f"a no-op {action} on a {BIG_SIDE}x{BIG_SIDE} frame took "
        f"{big:.2f} s, which is a hang however loaded the machine is")
    assert big / small < AREA_GROWTH_ALLOWED, (
        f"a no-op {action} took {small:.3f} s on {SIDE}x{SIDE} and "
        f"{big:.3f} s on {BIG_SIDE}x{BIG_SIDE} -- {big / small:.2f}x for "
        f"{BIG_SIDE ** 2 / SIDE ** 2:.2f}x the area, so the search is "
        f"walking the frame again")


@pytest.mark.parametrize("action", ["erase", "add"])
def test_a_no_op_wand_still_changes_nothing(uniform, action):
    """The correctness half, kept apart from the timing half.

    A wand that returned instantly by doing the wrong thing would satisfy
    every assertion above.
    """
    if action == "erase":
        before = np.zeros((SIDE, SIDE), np.uint8)
        out = magic_wand(uniform, before, SIDE // 2, SIDE // 2, 10.0,
                         max_pixels=100, action="erase")
        assert not out.any(), "an erase on an empty mask changed something"
    else:
        before = np.full((SIDE, SIDE), 255, np.uint8)
        out = magic_wand(uniform, before, SIDE // 2, SIDE // 2, 10.0,
                         max_pixels=100, action="add")
        assert out.all(), "an add on a full mask changed something"


def test_the_ratio_can_see_an_unbounded_search(monkeypatch):
    """PROOF THE RATIO IS NOT A CONSTANT, and the reason it may be trusted.

    A ratio test that reads ~1.0 whatever the code does would be worse
    than the stopwatch it replaced, because it would look like coverage.
    So: lift the visit budget out of the way -- which is exactly the
    defect, a search bounded only by pixels that CHANGE -- and require the
    same probe to report the frame's area instead.

    Smaller frames than the test above, because this one deliberately
    does the slow thing and the suite should not pay full price to watch
    it fail.
    """
    from spacr.qt import mask_engine

    monkeypatch.setattr(mask_engine, "VISIT_BUDGET_FLOOR", 10 ** 9)
    small = _no_op_wand_seconds(300, "erase")
    big = _no_op_wand_seconds(750, "erase")

    assert big / small > AREA_GROWTH_ALLOWED, (
        f"with the visit budget lifted, a no-op erase went {small:.3f} s "
        f"-> {big:.3f} s ({big / small:.2f}x) for 6.25x the area. The "
        f"probe cannot tell a bounded search from an unbounded one, so "
        f"the tests above are asserting a constant")


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
