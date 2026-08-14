"""Three ways a reported number was quietly not the number that was measured.

All three come from instruction 77's survey. Each changes a figure or a count
on real data, none of them raises, and all three are invisible in the output.

  (a) `na='drop'` removed a cell for having an unmeasurable child, not for
      having no child.
  (c) `_merge_grouped` joined inner unconditionally, deleting uninfected
      cells from the denominator of every analysis that went through it.
  (d) `remove_outliers` trimmed the tails BEFORE the statistical tests, which
      shrinks the standard deviation and grows the t-statistic.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import MergePolicy, merge_tables
from spacr.object_roles import JOIN_HOW, join_how

BASE = dict(plateID="p1", rowID="r1", columnID="c1", fieldID="f1",
            prcf="p1_r1_c1_f1")


# ---------------------------------------------------------------------------
# (a) dropped for having no child, not for having an unmeasurable one
# ---------------------------------------------------------------------------

def plate(tmp_path):
    """Three cells: a good pathogen, a pathogen with a NaN measurement, none.

    The middle one is the case. It HAS a pathogen. Its area and count are
    real numbers. Only its correlation is NaN -- which happens whenever a
    channel is flat inside the object, and says nothing at all about whether
    the object exists.
    """
    path = str(tmp_path / "measurements.db")
    con = sqlite3.connect(path)
    try:
        pd.DataFrame([{**BASE, "object_label": i, "area": 100.0 * i}
                      for i in (1, 2, 3)]).to_sql("cell", con, index=False)
        pd.DataFrame([
            {**BASE, "object_label": 1, "cell_id": 1,
             "pathogen_area": 20.0, "pathogen_corr": 0.8},
            {**BASE, "object_label": 1, "cell_id": 2,
             "pathogen_area": 30.0, "pathogen_corr": np.nan},
        ]).to_sql("pathogen", con, index=False)
        con.commit()
    finally:
        con.close()
    return path


def test_a_cell_with_a_child_survives_a_nan_measurement(tmp_path):
    """The regression: only cell 1 used to survive."""
    out = merge_tables(plate(tmp_path), ["cell", "pathogen"],
                       policy=MergePolicy(na="drop"))
    assert sorted(out["object_label"]) == [1, 2]


def test_a_cell_with_no_child_is_still_dropped(tmp_path):
    """The policy must keep doing what it says: cell 3 has no pathogen."""
    out = merge_tables(plate(tmp_path), ["cell", "pathogen"],
                       policy=MergePolicy(na="drop"))
    assert 3 not in set(out["object_label"])


def test_the_nan_measurement_is_carried_not_invented(tmp_path):
    out = merge_tables(plate(tmp_path), ["cell", "pathogen"],
                       policy=MergePolicy(na="drop")
                       ).set_index("object_label")
    assert out.loc[2, "pathogen_count"] == 1
    assert pd.isna(out.loc[2, "pathogen_corr"])


def test_a_merge_with_no_children_at_all_is_not_emptied(tmp_path):
    """No roll-up means no childlessness to test.

    Dropping on the measurements here would silently narrow the population
    on a merge that has no children in it.
    """
    path = str(tmp_path / "only_cells.db")
    con = sqlite3.connect(path)
    frame = pd.DataFrame([{**BASE, "object_label": i, "area": 1.0 * i,
                           "some_corr": np.nan} for i in (1, 2, 3)])
    frame.to_sql("cell", con, index=False)
    con.commit()
    con.close()

    out = merge_tables(path, ["cell"], policy=MergePolicy(na="drop"))
    assert len(out) == 3


def test_keep_and_zero_are_unchanged(tmp_path):
    db = plate(tmp_path)
    kept = merge_tables(db, ["cell", "pathogen"], policy=MergePolicy(na="keep"))
    assert sorted(kept["object_label"]) == [1, 2, 3]
    assert kept.set_index("object_label").loc[3, "pathogen_count"] == 0


# ---------------------------------------------------------------------------
# (c) the two readers agree about which objects exist
# ---------------------------------------------------------------------------

def test_merge_grouped_reads_the_registry():
    """It joined inner unconditionally while `_read_and_join_tables` -- the
    other reader of the same tables -- read `join_how`. The two disagreed
    about which objects exist."""
    import inspect

    from spacr.io import _read_and_merge_data

    source = inspect.getsource(_read_and_merge_data)
    assert "join_how(right_name" in source


def test_the_uninfected_cell_is_kept_by_default():
    """Inner for pathogen silently conditioned every result on infection,
    deleting the control population from the denominator."""
    assert join_how("pathogen") == "left"
    assert join_how("organelle") == "left"


def test_asking_for_infected_only_still_works():
    assert join_how("pathogen", keep_uninfected=False) == "inner"


def test_the_joins_that_are_not_about_infection_stay_inner():
    for table in ("nucleus", "png_list"):
        assert join_how(table) == "inner"
        assert join_how(table, keep_uninfected=False) == "inner"


def test_a_non_object_table_keeps_the_historical_inner_join():
    """The metadata and stamp merges go through the same helper and are not
    object tables. Guessing at them would be a behaviour change nobody
    asked for."""
    for name in ("grouped object data", "metadata"):
        assert name.strip().lower() not in JOIN_HOW


def test_the_reader_exposes_the_setting():
    import inspect

    from spacr.io import _read_and_merge_data

    parameters = inspect.signature(_read_and_merge_data).parameters
    assert "keep_uninfected" in parameters
    assert parameters["keep_uninfected"].default is True


# ---------------------------------------------------------------------------
# (d) the trim is for the picture, not for the test
# ---------------------------------------------------------------------------

def loud_tails(seed=0, n=60):
    """Two groups that overlap only because of their tails.

    Trimming 1.5*IQR per group removes exactly the points that make them
    overlap, which is why doing it before a test is not a neutral tidy-up.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, centre in (("nc", 0.0), ("pc", 0.35)):
        values = list(rng.normal(centre, 1.0, n))
        values += [centre + 6.0, centre - 6.0]      # the tails
        rows.extend({"condition": name, "value": float(v)} for v in values)
    return pd.DataFrame(rows)


def test_trimming_shrinks_the_spread_which_is_why_it_moved():
    """The mechanism, measured, so the fix is not taken on faith."""
    from scipy.stats import ttest_ind

    df = loud_tails()
    whole = [df.loc[df["condition"] == g, "value"].to_numpy()
             for g in ("nc", "pc")]

    def trim(values):
        q1, q3 = np.percentile(values, [25, 75])
        span = 1.5 * (q3 - q1)
        return values[(values >= q1 - span) & (values <= q3 + span)]

    trimmed = [trim(v) for v in whole]
    assert trimmed[0].std() < whole[0].std()
    assert ttest_ind(*trimmed)[1] < ttest_ind(*whole)[1], (
        "pick data where trimming actually moves the p-value")


def test_the_statistics_use_every_point(monkeypatch):
    from spacr.plot import spacrGraph

    df = loud_tails()
    plain = spacrGraph(df, grouping_column="condition", data_column=["value"])
    trimmed = spacrGraph(df, grouping_column="condition",
                         data_column=["value"], remove_outliers=True)

    groups = ["nc", "pc"]
    a = plain.perform_statistical_tests(groups, True)[0]
    b = trimmed.perform_statistical_tests(groups, True)[0]
    assert a["p-value"] == pytest.approx(b["p-value"]), (
        "remove_outliers changed the p-value; the trim is back in front of "
        "the test")


def test_the_trim_happens_after_the_tests_in_the_source():
    import inspect

    from spacr.plot import spacrGraph

    source = inspect.getsource(spacrGraph.create_plot)
    trim_at = source.index("self.remove_outliers_from_plot()")
    tests_at = source.index("perform_statistical_tests(")
    assert trim_at > tests_at, (
        "the outlier trim runs before the statistical tests again")


def test_the_results_table_says_the_plot_was_trimmed():
    """A reader looking at a trimmed plot beside a p-value has to know which
    one used what."""
    import matplotlib
    matplotlib.use("Agg")

    from spacr.plot import spacrGraph

    graph = spacrGraph(loud_tails(), grouping_column="condition",
                       data_column=["value"], remove_outliers=True)
    graph.create_plot()
    results = graph.results_df
    assert "outliers_removed_from_plot_only" in results.columns
    assert bool(results["outliers_removed_from_plot_only"].iloc[0]) is True


def test_nothing_is_annotated_when_nothing_was_trimmed():
    import matplotlib
    matplotlib.use("Agg")

    from spacr.plot import spacrGraph

    graph = spacrGraph(loud_tails(), grouping_column="condition",
                       data_column=["value"])
    graph.create_plot()
    assert "outliers_removed_from_plot_only" not in graph.results_df.columns
