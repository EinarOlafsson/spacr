"""`peaks_per_cell` divided by the largest field, not by the well's cells.

`summarize_per_well` grouped by ``well_ID`` -- row + column, spanning every
field -- and counted ``object_number``.nunique(). ``object_number`` is the
label the segmenter assigned WITHIN a field and it restarts at 1 in each one,
so the count came back as the size of the well's biggest field.

``peaks_per_well`` meanwhile counts every peak in the whole well. So the
ratio was too high by roughly the number of fields: a well imaged as 4 fields
of 60 cells holds 240 and produced 480 peaks, and the CSV reported
480 / 60 = 8 peaks per cell instead of 2.

This is the same root cause as io.py's ``cells_per_well``, fixed separately:
a per-field label used as if it identified an object across the plate.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.timelapse import summarize_per_well, summarize_per_well_inf_non_inf


def peaks(fields=4, per_field=60, peaks_each=2, infected=0):
    """One well, object labels restarting at 1 in every field."""
    return pd.DataFrame([
        {"ID": f"plate1_r1_c1_f{field}_{obj}", "amplitude": 1.0,
         "infected": infected}
        for field in range(1, fields + 1)
        for obj in range(1, per_field + 1)
        for _ in range(peaks_each)
    ])


@pytest.mark.parametrize(("fields", "per_field", "peaks_each"), [
    (4, 60, 2), (9, 40, 3), (2, 5, 1), (1, 30, 4),
])
def test_the_denominator_is_the_wells_cells(fields, per_field, peaks_each):
    out = summarize_per_well(peaks(fields, per_field, peaks_each))
    assert int(out["cells_per_well"][0]) == fields * per_field
    assert out["peaks_per_cell"][0] == pytest.approx(float(peaks_each))


def test_one_field_is_the_only_case_where_the_old_count_agreed():
    """Which is why single-field test data never showed this."""
    one = peaks(fields=1, per_field=30, peaks_each=2)
    many = peaks(fields=4, per_field=30, peaks_each=2)

    old_one = one["ID"].str.rsplit("_", n=1).str[-1].nunique()
    old_many = many["ID"].str.rsplit("_", n=1).str[-1].nunique()
    assert old_one == 30 and old_many == 30, "labels do restart per field"

    assert int(summarize_per_well(one)["cells_per_well"][0]) == 30
    assert int(summarize_per_well(many)["cells_per_well"][0]) == 120


def test_the_infected_split_counts_the_same_way():
    """The sibling function had the identical line."""
    out = summarize_per_well_inf_non_inf(peaks(fields=4, per_field=60,
                                               peaks_each=2))
    assert int(out["cells_per_well"][0]) == 240
    assert out["peaks_per_cell"][0] == pytest.approx(2.0)


def test_a_cell_with_no_measurable_peak_still_counts_in_the_denominator():
    """cells_per_well is built from the UNFILTERED frame on purpose: a cell
    that oscillated too weakly to register is still a cell the well holds,
    and dropping it would inflate peaks_per_cell again."""
    frame = peaks(fields=2, per_field=10, peaks_each=1)
    frame.loc[frame["ID"].str.endswith("_1"), "amplitude"] = np.nan

    out = summarize_per_well(frame)
    assert int(out["cells_per_well"][0]) == 20, (
        "cells with no amplitude were dropped from the denominator")
