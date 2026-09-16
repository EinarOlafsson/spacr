"""`_advanced_family_members` must stay linear in the number of objects.

spaCR files 37,165 setting keys across 706 object roles, and the regroup that
lays out the Advanced panel matches three families against every one of them.
The duplicate check was `key in found` on a LIST, so filing one family did
about twelve million string comparisons -- 137 ms of pure Python on every
process that imports `spacr.settings`, which is every screen open, the CLI and
the queue validator.

THIS IS A RATIO TEST AND NOT A CLOCK. A ceiling in milliseconds measures the
machine as much as the code, and this suite has already had to repair one
timing test that did. Quadrupling the object count multiplies linear work by
about four and quadratic work by about sixteen, and those two are far enough
apart that the verdict survives a loaded runner.
"""

import time

import pytest

from spacr import settings as S


def _table(objects, suffixes):
    """A category table with ``objects`` roles each carrying every suffix.

    :param objects: how many object roles to invent.
    :param suffixes: the suffixes each role carries.
    :returns: a one-category table holding every combination.
    """
    return {"General": [f"obj{n}_{suffix}"
                        for n in range(objects)
                        for suffix in suffixes]}


def _cost(objects, suffixes, monkeypatch):
    """Seconds to file one family over ``objects`` roles, best of three.

    Best-of-three rather than a mean: a scheduler blip can only make a run
    slower, so the minimum is the closest estimate of the work itself.
    """
    table = _table(objects, suffixes)
    monkeypatch.setattr(S, "ADVANCED_OBJECT_ORDER",
                        [f"obj{n}" for n in range(objects)])
    best = None
    for _ in range(3):
        start = time.perf_counter()
        found = S._advanced_family_members(table, suffixes)
        taken = time.perf_counter() - start
        best = taken if best is None else min(best, taken)
    assert len(found) == objects * len(suffixes), "the table was not filed"
    return best


def test_quadrupling_the_objects_does_not_multiply_the_work_by_sixteen(
        monkeypatch):
    """Linear grows by ~4, the list scan this replaced grew by ~16."""
    suffixes = ("min_size", "max_size", "area", "intensity", "channel")

    small = _cost(400, suffixes, monkeypatch)
    large = _cost(1600, suffixes, monkeypatch)

    growth = large / small if small else float("inf")
    assert growth < 8.0, (
        f"filing a family grew {growth:.1f}x for 4x the objects, which is the "
        f"quadratic duplicate check coming back: {small*1000:.1f} ms -> "
        f"{large*1000:.1f} ms")


def test_the_order_is_still_object_then_suffix(monkeypatch):
    """The list is the answer; only the membership test moved to a set.

    A set alongside the list is safe ONLY because the list still decides the
    order, and the order is the contract -- the Advanced panel lays keys out
    in it. Swapping the list for a set would pass the test above and silently
    reshuffle every panel.
    """
    monkeypatch.setattr(S, "ADVANCED_OBJECT_ORDER", ["cell", "nucleus"])
    table = {"General": ["nucleus_area", "cell_area", "cell_size",
                         "nucleus_size"]}

    found = S._advanced_family_members(table, ("area", "size"))

    assert found == ["cell_area", "cell_size", "nucleus_area", "nucleus_size"]


def test_a_key_is_filed_once_however_many_ways_it_matches(monkeypatch):
    """The duplicate check is what the set replaced, so prove it still works.

    `remove_background_cell` matches as a PREFIX form while `cell_remove_
    background` would match as a suffix; a role that hits twice must still
    appear once, or the panel shows the same control twice.
    """
    monkeypatch.setattr(S, "ADVANCED_OBJECT_ORDER", ["cell"])
    table = {"General": ["cell_background", "remove_background_cell"]}

    found = S._advanced_family_members(
        table, ("background",), ("remove_background",))

    assert found == ["cell_background", "remove_background_cell"]
    assert len(found) == len(set(found))
