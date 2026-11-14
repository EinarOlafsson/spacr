"""The sweep must survive trials whose output is not the shape it hoped for.

Every path here is a place where the sweep meets a partial answer: a
coefficient table with no q column, a fit that reports only its significant
rows, a runner that hands back something that is not a mapping at all, a
results table where the positive control was never recovered, and a row a user
clicked without ever choosing a folder for the rerun. Each of those is a real
condition -- different regression families and corrections emit different
keys -- and each one used to be reachable only through a real fit, so nothing
exercised it. If any of them regressed the user would see a crash in the
middle of a sweep, or a summary claiming a control rank nothing measured,
rather than a row that says what it knows.

The regressions are never actually fitted: runners and
``spacr.ml.perform_regression`` are injected, and the parallel pool runs its
jobs in this process.
"""
from __future__ import annotations

import pandas as pd

from spacr import parameter_sweep as ps


def test_a_trial_that_only_reports_significant_rows_has_no_hit_count():
    """The hit count comes from the full table, not from the surviving rows.

    ``n_below_alpha`` is what ``summarise_sweep`` compares across settings, and
    it is only meaningful when the trial handed back every coefficient with its
    q value. A fit that reports just its significant rows must therefore
    contribute the count it does have and stay silent about the one it does
    not, rather than reporting "3 hits below alpha" from a table that was
    already filtered.
    """
    significant_only = ps._count_hits(
        {"significant": pd.DataFrame({"grna": ["a", "b"]})})

    full_table = ps._count_hits({"results": pd.DataFrame(
        {"grna": ["a", "b", "c"], "q_value": [0.01, 0.2, 0.9]})})

    assert significant_only == {"n_significant": 2}
    assert "n_below_alpha" not in significant_only
    assert full_table == {"n_results": 3, "n_below_alpha": 1}


def test_reopening_a_row_with_no_folder_creates_no_directory(tmp_path,
                                                             monkeypatch):
    """Clicking a row is a look, and a look must not litter the disk.

    A row read back from a CSV that was moved, or one built in memory before
    the sweep chose a destination, names no folder. ``rerun_trial`` has to fit
    it and hand back the figures anyway rather than calling ``makedirs`` on
    ``None``; and when a destination IS given, that folder has to appear,
    because the rerun writes into it.
    """
    import matplotlib.pyplot as plt

    made = []

    def fake(settings):
        made.append(plt.figure())
        return {"results": pd.DataFrame({"grna": ["a"], "q_value": [0.01]})}

    import spacr.ml as ml
    monkeypatch.setattr(ml, "perform_regression", fake)
    row = {"trial_id": 7, "regression_type": "ridge", "status": "ok"}

    try:
        homeless = ps.rerun_trial({}, row)
        assert "src" not in homeless["settings"]
        assert list(tmp_path.iterdir()) == [], \
            "a rerun with no folder still wrote to disk"

        housed = ps.rerun_trial({}, row, destination=str(tmp_path / "reopened"))
    finally:
        for figure in made:
            plt.close(figure)

    assert homeless["settings"]["regression_type"] == "ridge"
    assert homeless["settings"]["verbose"] is True
    assert len(homeless["figures"]) == 1
    assert homeless["output"]["results"].iloc[0]["grna"] == "a"
    assert housed["settings"]["src"] == str(tmp_path / "reopened")
    assert (tmp_path / "reopened").is_dir()
