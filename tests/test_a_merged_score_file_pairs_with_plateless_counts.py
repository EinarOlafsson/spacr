"""One score file holding every plate, paired with counts that name none.

This is the Measurements tab's own shape. `column_run_settings` points EVERY
pair row's score at the single merged frame -- which carries all four plates
-- while a real count CSV of the TSG101 screen carries `row_name`,
`column_name`, `grna_name`, `count` and no plate column whatsoever.

`load_regression_input_pairs` documents its resolution order as own column,
partner column, then pair-row order. The third rule was only ever reached
when NEITHER side declared a plate, so this shape refused with "cannot copy
['plate1'...] onto a partner with no plateID" and every measurement-column
regression failed before it fitted anything.
"""
import pandas as pd
import pytest

from spacr.ml import load_regression_input_pairs


def _merged_score(tmp_path, plates=("plate1", "plate2", "plate3", "plate4")):
    rows = [{"plateID": p, "rowID": "r1", "columnID": "c1",
             "cell_area": 100.0 + i} for i, p in enumerate(plates)]
    path = tmp_path / "merged_measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _plateless_counts(tmp_path, name):
    path = tmp_path / name
    pd.DataFrame([{"rowID": "r1", "columnID": "c1", "grna": "g1",
                   "count": 7}]).to_csv(path, index=False)
    return str(path)


def test_each_pair_row_takes_the_plate_its_position_names(tmp_path):
    score = _merged_score(tmp_path)
    pairs = [{"score": score, "count": _plateless_counts(tmp_path, f"c{i}.csv")}
             for i in range(1, 5)]

    counts, scores, audit = load_regression_input_pairs(pairs)

    assert [row["plate"] for row in audit] == ["plate1", "plate2", "plate3",
                                               "plate4"]
    assert sorted(counts["plateID"].unique()) == ["plate1", "plate2",
                                                  "plate3", "plate4"]


def test_the_score_side_is_cut_down_to_that_plate_not_repeated_whole(tmp_path):
    # Four pair rows against one four-plate score file must yield four score
    # rows, not sixteen -- otherwise every well is fitted four times.
    score = _merged_score(tmp_path)
    pairs = [{"score": score, "count": _plateless_counts(tmp_path, f"c{i}.csv")}
             for i in range(1, 5)]

    _counts, scores, _audit = load_regression_input_pairs(pairs)

    assert len(scores) == 4
    assert sorted(scores["plateID"].unique()) == ["plate1", "plate2",
                                                  "plate3", "plate4"]


def test_the_rule_used_is_named_in_the_audit(tmp_path):
    score = _merged_score(tmp_path)
    pairs = [{"score": score, "count": _plateless_counts(tmp_path, "c1.csv")}]

    _counts, _scores, audit = load_regression_input_pairs(pairs)

    assert "pair row order" in audit[0]["rule"]


def test_a_plate_the_score_file_does_not_hold_is_still_refused(tmp_path):
    # The safety half: positional resolution is only allowed to name a plate
    # the partner actually has. A screen whose plates are called anything
    # else must refuse rather than invent a match.
    score = _merged_score(tmp_path, plates=("screenA", "screenB"))
    pairs = [{"score": score, "count": _plateless_counts(tmp_path, "c1.csv")}]

    with pytest.raises(ValueError, match="cannot copy"):
        load_regression_input_pairs(pairs)


def test_a_count_file_that_names_its_own_plate_is_untouched(tmp_path):
    score = _merged_score(tmp_path)
    named = tmp_path / "named.csv"
    pd.DataFrame([{"plateID": "plate3", "rowID": "r1", "columnID": "c1",
                   "grna": "g1", "count": 7}]).to_csv(named, index=False)

    counts, _scores, audit = load_regression_input_pairs(
        [{"score": score, "count": str(named)}])

    # Pair-row order would have said plate1; the file's own column wins.
    assert list(counts["plateID"].unique()) == ["plate3"]
    assert audit[0]["plate"] == "plate3"
