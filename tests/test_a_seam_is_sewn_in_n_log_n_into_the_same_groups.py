"""`sew` groups a well's observations in n log n, into the groups it always made.

MEASURED BEFORE THE FIX (372 PART 14-L, section 3): 0.18 s at 1,000
observations, 0.78 s at 2,000, 4.08 s at 4,000. Each seed scanned every
remaining observation, so a well's 1.57 million observations would have taken
days; the Phase B driver had to call it once per KD-tree component.

WHAT IT COMPUTED IS SINGLE LINKAGE: a group grows by any observation that
matches any member, so the groups are the connected components of the match
graph. The replacement finds candidate pairs with a KD-tree under the same
per-axis (Chebyshev) test and joins them with union-find. The implementation
it replaced is copied below and compared on random observations where the
seam rules added at the same time cannot apply: every observation from its
own window, none clipped, and no pair whose area ratio fails while its
centroids lie inside the smaller one's equivalent disc.

On realistic windows -- shared windows, overlap bands, clipped observations --
the test that remains is structural: every observation lands in exactly one
group, the grouping does not depend on input order, and no group holds two
different labels of one window.
"""
from __future__ import annotations

import math
import time

import numpy as np

from spacr.ops_compose import Window, windows_over
from spacr.ops_objects import (DEFAULT_AREA_RATIO, DEFAULT_CENTROID_TOLERANCE,
                               WindowObject, sew)


def _reference_same_object(a, b, *, tolerance, area_ratio):
    """The pairwise test before the fix, verbatim."""
    if abs(a.centroid_y - b.centroid_y) > tolerance:
        return False
    if abs(a.centroid_x - b.centroid_x) > tolerance:
        return False
    larger = max(a.area, b.area)
    if larger <= 0:
        return False
    return min(a.area, b.area) / larger >= area_ratio


def _reference_sew(observations, *, tolerance=DEFAULT_CENTROID_TOLERANCE,
                   area_ratio=DEFAULT_AREA_RATIO):
    """The grouping before the fix, verbatim: quadratic single linkage."""
    remaining = list(observations)
    groups = []
    while remaining:
        seed = remaining.pop()
        group = [seed]
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                if any(_reference_same_object(
                        member, other, tolerance=tolerance,
                        area_ratio=(0.0 if (member.clipped or other.clipped)
                                    else area_ratio))
                       for member in group):
                    group.append(other)
                    remaining.remove(other)
                    changed = True
        groups.append(tuple(group))
    return tuple(groups)


def _as_sets(groups):
    """Groups as a set of frozensets, which is what 'the same groups' means."""
    return {frozenset(group) for group in groups}


def _outside_the_new_rules(observations, tolerance, area_ratio):
    """Drop observations until no pair would be decided by the disc rule.

    The disc rule sews a cross-window pair whose area ratio fails when the
    centroids lie within the smaller one's equivalent radius. Where it would
    fire, the old and new functions are meant to disagree, so those pairs are
    removed from the comparison rather than from the implementation.
    """
    kept = list(observations)
    while True:
        ys = np.array([one.centroid_y for one in kept])
        xs = np.array([one.centroid_x for one in kept])
        areas = np.array([one.area for one in kept], dtype=float)
        dy = np.abs(ys[:, None] - ys[None, :])
        dx = np.abs(xs[:, None] - xs[None, :])
        smaller = np.minimum(areas[:, None], areas[None, :])
        larger = np.maximum(areas[:, None], areas[None, :])
        near = (dy <= tolerance) & (dx <= tolerance)
        fails_ratio = smaller / larger < area_ratio
        inside = np.hypot(dy, dx) <= np.sqrt(smaller / math.pi)
        offending = np.triu(near & fails_ratio & inside, k=1)
        if not offending.any():
            return kept
        drop = sorted({int(j) for _, j in zip(*np.nonzero(offending))},
                      reverse=True)
        for index in drop[: max(1, len(drop) // 4)]:
            kept.pop(index)


def _complete_one_window_each(rng, count, side):
    """Complete observations, one window each, dense enough to chain."""
    found = []
    for index in range(count):
        y, x = rng.uniform(0, side, 2)
        area = int(rng.choice([4, 5, 10, 11, 12]))
        found.append(WindowObject(
            window=Window(top=index, left=0, height=1, width=1), label=1,
            centroid_y=float(y), centroid_x=float(x), area=area,
            bbox=(int(y) - 2, int(x) - 2, int(y) + 2, int(x) + 2)))
    return found


def test_the_groups_are_the_ones_the_quadratic_version_made():
    """Union-find over KD-tree pairs is single linkage, on chaining data."""
    rng = np.random.default_rng(372)
    multi = 0
    largest = 0
    for _ in range(6):
        observations = _outside_the_new_rules(
            _complete_one_window_each(rng, 500, 110),
            DEFAULT_CENTROID_TOLERANCE, DEFAULT_AREA_RATIO)
        order = rng.permutation(len(observations))
        shuffled = [observations[i] for i in order]
        expected = _as_sets(_reference_sew(shuffled))
        assert _as_sets(sew(shuffled)) == expected
        multi += sum(1 for group in expected if len(group) > 1)
        largest = max(largest, max(len(group) for group in expected))
    assert multi > 100, "the fixture never exercised a merge"
    assert largest >= 5, "the fixture never exercised a chain"


def test_other_tolerances_and_ratios_agree_as_well():
    """The parameters are honoured, not just their defaults."""
    rng = np.random.default_rng(11)
    for tolerance, ratio in ((1.0, 0.9), (5.0, 0.5), (0.0, 0.0)):
        observations = _outside_the_new_rules(
            _complete_one_window_each(rng, 300, 80), tolerance, ratio)
        assert (_as_sets(sew(observations, tolerance=tolerance,
                             area_ratio=ratio))
                == _as_sets(_reference_sew(observations, tolerance=tolerance,
                                           area_ratio=ratio)))


def _well(nuclei, seed=0, size=256, overlap=32, density=2340e-6):
    """Observations of planted nuclei as a real window tiling produces them.

    Each nucleus is observed by every window its box touches: complete where
    the window holds all of it (centroid jitter and a context-dependent area,
    as Cellpose gives across windows), clipped to the window where it does
    not.
    """
    rng = np.random.default_rng(seed)
    side = int(math.sqrt(nuclei / density)) + 32
    windows = list(windows_over((side, side), size=size, overlap=overlap))
    labels = {}
    found = []
    for _ in range(nuclei):
        cy, cx = rng.uniform(10, side - 10, 2)
        radius = rng.uniform(4.5, 6.5)
        top, left = int(cy - radius), int(cx - radius)
        bottom, right = int(cy + radius), int(cx + radius)
        for window in windows:
            if (window.top > bottom or window.bottom <= top
                    or window.left > right or window.right <= left):
                continue
            labels[window] = labels.get(window, 0) + 1
            inside = (window.top <= top and bottom < window.bottom
                      and window.left <= left and right < window.right)
            if inside:
                found.append(WindowObject(
                    window=window, label=labels[window],
                    centroid_y=float(cy + rng.normal(0, 0.3)),
                    centroid_x=float(cx + rng.normal(0, 0.3)),
                    area=int(math.pi * radius ** 2 * rng.uniform(0.72, 1.0)),
                    bbox=(top, left, bottom, right)))
                continue
            it, il = max(top, window.top), max(left, window.left)
            ib, ir = min(bottom, window.bottom - 1), min(right, window.right - 1)
            found.append(WindowObject(
                window=window, label=labels[window],
                centroid_y=(it + ib) / 2.0, centroid_x=(il + ir) / 2.0,
                area=max(1, int((ib - it + 1) * (ir - il + 1) * 0.7)),
                bbox=(it, il, ib, ir), clipped=True))
    return found


def test_every_observation_lands_in_one_group_whatever_the_order():
    """A partition of the input, independent of arrival order."""
    observations = _well(1500, seed=3)
    groups = sew(observations)
    members = [one for group in groups for one in group]
    assert len(members) == len(observations)
    assert set(members) == set(observations)

    rng = np.random.default_rng(5)
    shuffled = [observations[i] for i in rng.permutation(len(observations))]
    assert _as_sets(sew(shuffled)) == _as_sets(groups)
    assert sew(shuffled) == groups


def test_no_group_holds_two_labels_of_one_window():
    """One label image never shows one nucleus under two labels."""
    for group in sew(_well(1500, seed=4)):
        seen = {}
        for one in group:
            if one.clipped:
                continue
            assert seen.setdefault(one.window, one.label) == one.label


def test_the_cost_per_observation_does_not_grow_with_the_well():
    """Eight times the nuclei at the same density is eight times the work.

    Before the fix this ratio was ~8 (quadratic): 2,000 -> 4,000 observations
    cost 0.78 -> 4.08 s. The sizes are timed alternately, best of two, so a
    busy machine slows both.
    """
    small = _well(1000, seed=1)
    large = _well(8000, seed=2)
    best = {False: float("inf"), True: float("inf")}
    for _ in range(2):
        for is_large, observations in ((False, small), (True, large)):
            start = time.perf_counter()
            sew(observations)
            best[is_large] = min(best[is_large], time.perf_counter() - start)
    per_small = best[False] / len(small)
    per_large = best[True] / len(large)
    assert per_large / per_small < 3.0, (
        f"{1e6 * per_small:.1f} us/observation at {len(small)}, "
        f"{1e6 * per_large:.1f} us/observation at {len(large)}")
