"""A count CSV names its plate by WHICH FILE IT IS, and the montage must agree.

The real count tables of the TSG101 screen carry `row_name, column_name,
grna_name, count` and no plate column at all -- the plate is the file. Read
four of them without assigning it and every plate's r1/c1 becomes one well:
`fractions_from_counts` then divides each guide's reads by FOUR plates' total
and the result still sums to 1 per well, so nothing downstream can notice.

`ml.load_regression_input_pairs` resolves plate identity as own column, then
partner column, then pair-row order. These tests hold the montage to the last
of those, because it is handed the count paths in exactly that order.
"""
import pandas as pd
import pytest

from spacr.cell_montage import fractions_from_counts


def _counts(tmp_path, name, wells=("r1", "r2"), guides=("g1", "g2")):
    rows = [{"row_name": w, "column_name": "c1", "grna_name": g,
             "count": 10 if g == "g1" else 30}
            for w in wells for g in guides]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_four_plateless_count_files_stay_four_plates(tmp_path):
    paths = [_counts(tmp_path, f"plate_{i}.csv") for i in (1, 2, 3, 4)]
    frame = fractions_from_counts(paths)

    plates = sorted(frame["prc"].astype(str).str.split("_").str[0].unique())
    assert plates == ["plate1", "plate2", "plate3", "plate4"]
    # Two rows per plate x two guides: eight wells, not two.
    assert frame["prc"].nunique() == 8


def test_a_guide_fraction_is_its_share_of_ITS_OWN_well(tmp_path):
    paths = [_counts(tmp_path, f"plate_{i}.csv") for i in (1, 2, 3, 4)]
    frame = fractions_from_counts(paths)

    # 10 and 30 in every well, so 0.25 / 0.75 -- NOT 40/160 pooled.
    shares = sorted(frame["fraction"].round(6).unique())
    assert shares == [0.25, 0.75]
    totals = frame.groupby("prc")["fraction"].sum()
    assert totals.round(9).eq(1.0).all()


def test_a_file_that_names_its_plate_keeps_the_name_it_gave(tmp_path):
    named = tmp_path / "somewhere.csv"
    pd.DataFrame([{"plateID": "screenB_p7", "row_name": "r1",
                   "column_name": "c1", "grna_name": "g1", "count": 5}]
                 ).to_csv(named, index=False)

    frame = fractions_from_counts([str(named)])

    # The composer escapes an underscore INSIDE a component so the '_' that
    # separates plate from row stays unambiguous, so the plate reads back as
    # `screenB%5Fp7` -- what matters is that it is the plate the file named
    # and not a `plate1` invented from its position.
    plate = frame["prc"].iloc[0].split("_")[0]
    assert plate.replace("%5F", "_") == "screenB_p7"


def test_an_unreadable_file_does_not_shift_the_plates_after_it(tmp_path):
    # The second path is missing. The third file is still plate3, because a
    # skipped row must consume its number -- otherwise every later plate is
    # mislabelled by one and the fractions are attributed to the wrong plate.
    first = _counts(tmp_path, "plate_1.csv")
    third = _counts(tmp_path, "plate_3.csv")
    frame = fractions_from_counts([first, str(tmp_path / "gone.csv"), third])

    plates = sorted(frame["prc"].astype(str).str.split("_").str[0].unique())
    assert plates == ["plate1", "plate3"]
