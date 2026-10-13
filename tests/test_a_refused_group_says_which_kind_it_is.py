"""A group no window saw whole is recorded, and sorted into its two kinds.

Well A1's validated run refused 54 groups and well A2 62. The report kept the
count and ONE box, so afterwards nobody could say whether those 54 were
nuclei the run lost or slivers of nuclei it had already counted from another
window -- and the two have different fixes. The observations are not stored,
so the only way to answer it later was to segment the well again.

`spacr.ops_objects.unseen_records` writes one record per refused group while
the observations are still in memory, and `spacr.ops_engine._explain_refusals`
asks of each record the question that separates the kinds: is there a
NUMBERED object over this box? If there is, the nucleus was counted from a
window that saw it whole and the sliver is a leftover of it. If there is not,
the sliver was the only evidence of anything at that spot.

Everything here is built by hand: the point is the bookkeeping, not a
segmenter.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

from spacr.ops_compose import Window
from spacr.ops_engine import _explain_refusals
from spacr.ops_objects import (ObjectsError, WindowObject, number,
                               objects_frame, unseen_records)

#: A window big enough that nothing in these fixtures reaches both its edges
#: by accident, which is a different refusal and is tested for separately.
WINDOW = Window(top=0, left=0, height=256, width=256)
NEIGHBOUR = Window(top=0, left=160, height=256, width=256)


def observation(window, label, y, x, area, box, clipped):
    """One window's view of one object.

    :param window: the window it was segmented in.
    :param label: its label in that window.
    :param y: well-frame centroid row.
    :param x: well-frame centroid column.
    :param area: the mask's pixel count as this window saw it.
    :param box: ``(top, left, bottom, right)`` in well-frame pixels.
    :param clipped: whether this window cut it off.
    :returns: the :class:`spacr.ops_objects.WindowObject`.
    """
    return WindowObject(window=window, label=label, centroid_y=y,
                        centroid_x=x, area=area, bbox=box, clipped=clipped)


def whole(label, y, x, radius=6):
    """A complete observation of a round object at ``(y, x)``."""
    box = (int(y - radius), int(x - radius), int(y + radius), int(x + radius))
    return [observation(WINDOW, label, y, x, int(3.14 * radius * radius),
                        box, False)]


def sliver(label, top, left, height, width, window=NEIGHBOUR):
    """A clipped observation only, of a thin object with its top-left given.

    The box is inclusive at every edge, as `objects_in_window` builds it, so
    a height of three is three rows.
    """
    box = (int(top), int(left), int(top + height - 1), int(left + width - 1))
    return [observation(window, label, top + (height - 1) / 2.0,
                        left + (width - 1) / 2.0, height * width, box, True)]


def test_only_the_groups_with_no_complete_observation_are_recorded():
    """A group with any complete observation is numbered, not recorded."""
    groups = [whole(1, 40.0, 40.0), sliver(2, 120, 200, 3, 19),
              whole(3, 60.0, 90.0)]
    records = unseen_records(groups)

    assert len(records) == 1
    assert records[0]["height"] == 3 and records[0]["width"] == 19
    assert records[0]["observations"] == 1 and records[0]["windows"] == 1
    assert records[0]["area_total"] == 3 * 19


def test_a_record_carries_the_box_the_refusal_message_only_gives_the_largest():
    """Every refusal gets its own box, not just the biggest one."""
    groups = [sliver(1, 30, 200, 3, 19), sliver(2, 90, 210, 2, 5),
              sliver(3, 150, 220, 19, 5)]
    records = unseen_records(groups)

    assert [(r["height"], r["width"]) for r in records] == [(3, 19), (2, 5),
                                                            (19, 5)]
    assert [r["top"] for r in records] == [30, 90, 150]

    with pytest.raises(ObjectsError) as refused:
        number(groups, strict=True)
    # The message names one box; the records name all three, which is the
    # whole difference between them.
    assert "3 group(s)" in str(refused.value)
    assert str(refused.value).count("px near") == 1


def test_a_sliver_under_a_numbered_object_is_told_from_one_over_nothing():
    """The question the 54 refusals raised, answered from the tables."""
    kept = number([whole(1, 100.0, 200.0, radius=10),
                   whole(2, 40.0, 40.0, radius=8)], strict=False)
    frame = objects_frame(kept)

    over = sliver(9, 104, 205, 3, 19)
    alone = sliver(10, 600, 600, 3, 19)
    found = _explain_refusals(unseen_records([over, alone]), frame, overlap=96)

    assert found["groups"] == 2
    assert found["covered_by_a_numbered_object"] == 1
    assert found["orphaned"] == 1
    covered, orphan = found["records"]
    assert covered["covered"] and covered["nearest_object_id"] == 2
    assert not orphan["covered"] and orphan["nearest_object_id"] == -1
    assert orphan["nearest_object_px"] is None, "json.dump has no Infinity"


def test_the_counts_say_whether_the_overlap_is_the_remedy():
    """A refusal smaller than the overlap cannot be fixed by raising it.

    This is what the validated well measured and what the old message got
    wrong: every one of A1's 54 was at most 3 x 19 px against a 96 px
    overlap, so "raise the overlap" named a remedy that could not apply.
    """
    kept = objects_frame(number([whole(1, 40.0, 40.0)], strict=False))
    small = [sliver(i, 100 + 20 * i, 200, 3, 19) for i in range(4)]
    big = [sliver(9, 400, 200, 120, 30)]
    found = _explain_refusals(unseen_records(small + big), kept, overlap=96)

    assert found["window_overlap_px"] == 96
    assert found["wider_than_the_overlap"] == 1
    assert found["largest_px"] == [120, 30]
    assert found["spanning_a_window_side"] == 0


def test_an_object_spanning_a_window_side_is_counted_as_that_kind():
    """No overlap fixes an object that runs the full length of a window."""
    across = [observation(WINDOW, 1, 128.0, 128.0, 256 * 4,
                          (126, 0, 130, 255), True)]
    found = _explain_refusals(unseen_records([across]),
                              objects_frame(number([whole(1, 700.0, 700.0)],
                                                   strict=False)),
                              overlap=96)

    assert found["spanning_a_window_side"] == 1
    assert found["records"][0]["spanned_side"] == WINDOW.width


def test_no_numbered_objects_at_all_still_answers():
    """An empty objects table means every refusal is orphaned, not a crash."""
    import pandas as pd

    empty = pd.DataFrame({name: [] for name in
                          ("object_id", "centroid_y", "centroid_x",
                           "bbox_top", "bbox_left", "bbox_bottom",
                           "bbox_right")})
    found = _explain_refusals(unseen_records([sliver(1, 10, 10, 3, 19)]),
                              empty, overlap=96)

    assert found["orphaned"] == 1 and found["covered_by_a_numbered_object"] == 0
