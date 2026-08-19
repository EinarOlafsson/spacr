"""A file handed to several pair rows is parsed once, not once per row.

Reported 2026-08-19: "it looks like it is not running while it does say that
it is running. it is in any cas taking longer than it should".

It WAS running. Measured on the live process: the regression worker at 82%
CPU, `write_bytes` sitting at exactly 2,752,598,016 -- the merged frame it had
just written -- and `read_bytes` static at 1.4 MB, because it was reading that
2.75 GB file back out of the PAGE CACHE. Four times: the Measurements tab
points every pair row's score at the one merged frame, and this parsed it once
per row.
"""
import os
import tempfile

import pandas as pd
import pytest

from spacr import tabular
from spacr.ml import load_regression_input_pairs


@pytest.fixture()
def one_score_four_counts(tmp_path):
    rows = [{"plateID": f"plate{p}", "rowID": "r1", "columnID": "c1",
             "cell_area": 100.0 + p} for p in (1, 2, 3, 4)]
    score = tmp_path / "merged_measurements.csv"
    pd.DataFrame(rows).to_csv(score, index=False)
    pairs = []
    for i in (1, 2, 3, 4):
        count = tmp_path / f"count{i}.csv"
        pd.DataFrame([{"rowID": "r1", "columnID": "c1", "grna": "g1",
                       "count": 7}]).to_csv(count, index=False)
        pairs.append({"score": str(score), "count": str(count)})
    return pairs


def test_the_shared_score_file_is_read_once(one_score_four_counts, monkeypatch):
    seen = []
    real = tabular.read_table

    def counted(path, *args, **kwargs):
        seen.append(os.path.basename(str(path)))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(tabular, "read_table", counted)

    load_regression_input_pairs(one_score_four_counts)

    merged = [name for name in seen if name == "merged_measurements.csv"]
    assert len(merged) == 1, f"parsed the 2.75 GB file {len(merged)} times"


def test_every_distinct_file_is_still_read(one_score_four_counts, monkeypatch):
    seen = []
    real = tabular.read_table
    monkeypatch.setattr(
        tabular, "read_table",
        lambda p, *a, **k: (seen.append(os.path.basename(str(p))),
                            real(p, *a, **k))[1])

    load_regression_input_pairs(one_score_four_counts)

    assert sorted(set(seen)) == ["count1.csv", "count2.csv", "count3.csv",
                                 "count4.csv", "merged_measurements.csv"]


def test_each_pair_gets_its_own_copy_to_mutate(one_score_four_counts):
    """The caller assigns plateID onto what it gets and filters it down to one
    plate. Handing out the cached frame itself would let the first pair row's
    edits reach the second."""
    counts, scores, audit = load_regression_input_pairs(one_score_four_counts)

    assert [row["plate"] for row in audit] == ["plate1", "plate2", "plate3",
                                               "plate4"]
    # Four rows, one per plate -- not the same plate four times, and not one
    # plate's rows repeated.
    assert sorted(scores["plateID"].unique()) == ["plate1", "plate2",
                                                  "plate3", "plate4"]
    assert len(scores) == 4
