"""`cells_per_well` counted label VALUES, not cells.

`object_label` is assigned by the segmenter per FIELD and restarts at 1 in
each one, so `metadata.groupby('prc')['object_label'].nunique()` over a well
returned the size of that well's largest field, not the number of cells in
it. A 9-field well holding 360 cells reported roughly 40.

The number is a threshold, not a report. `cells_per_well` is documented as
the minimum a well must contribute and is used to drop under-populated
wells, so a user asking for at least 100 cells discarded every well on a
plate averaging 360 -- and the wells that survived were the ones with the
single most crowded field, which is the opposite of what was asked for.

The fix counts `prcfo`, the per-object key that already existed eleven lines
further down, and moves its construction above the count.
"""

import pandas as pd
import pytest


def well(fields=9, per_field=40):
    """One well, labels restarting at 1 in every field -- the real shape."""
    rows = [{"plateID": "p1", "rowID": "r1", "columnID": "c1",
             "fieldID": f"f{field}", "object_label": str(label)}
            for field in range(1, fields + 1)
            for label in range(1, per_field + 1)]
    frame = pd.DataFrame(rows)
    frame = frame.assign(prc=frame.plateID + "_" + frame.rowID + "_" + frame.columnID)
    frame = frame.assign(prcf=frame.prc + "_" + frame.fieldID)
    return frame.assign(prcfo=frame.prcf + "_" + frame.object_label)


@pytest.mark.parametrize(("fields", "per_field"), [
    (9, 40), (4, 60), (1, 250), (2, 3),
])
def test_the_count_is_the_number_of_cells(fields, per_field):
    frame = well(fields, per_field)
    counted = frame.groupby("prc")["prcfo"].nunique().iloc[0]
    assert counted == fields * per_field == len(frame)


def test_the_old_key_undercounts_by_the_field_count():
    """Pinned so nobody 'simplifies' prcfo back to object_label.

    One field is the only case where the two agree, which is why this went
    unnoticed on single-field test data.
    """
    frame = well(fields=9, per_field=40)
    old = frame.groupby("prc")["object_label"].nunique().iloc[0]
    new = frame.groupby("prc")["prcfo"].nunique().iloc[0]
    assert old == 40 and new == 360
    assert new == old * 9

    single = well(fields=1, per_field=40)
    assert (single.groupby("prc")["object_label"].nunique().iloc[0]
            == single.groupby("prc")["prcfo"].nunique().iloc[0])


def test_prcfo_is_built_before_the_count_in_the_source():
    """The fix is an ORDER change, so order is what has to be pinned."""
    import inspect
    from spacr import io

    source = inspect.getsource(io._read_and_merge_data)
    build = source.find("prcfo=lambda x: x['prcf']")
    count = source.find("groupby('prc')['prcfo'].nunique()")
    assert build != -1, "prcfo is no longer built here"
    assert count != -1, "the well count no longer uses prcfo"
    assert build < count, (
        "the well count runs before prcfo is built, so it is counting "
        "per-field labels again")
