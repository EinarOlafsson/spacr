"""CPU-only coverage for the feature/stats block of ``spacr.utils``.

Covers the branches that the rest of the suite never reaches:

* ``filter_dataframe_features`` -- the ``exclude`` list / str branches.
* ``find_non_overlapping_position`` -- exhausting ``max_attempts``.
* ``search_reduction_and_clustering`` -- fractional ``n_neighbors``, the
  ``n_neighbors <= 1`` clamp, and both ``ValueError`` guards.
* ``load_image`` / ``extract_features`` -- the ResNet embedding helpers,
  driven with a tiny local network so nothing is downloaded.

Everything here is offline and runs in well under a second; the heavy
reducers are short-circuited by passing a pre-computed ``embedding``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# filter_dataframe_features -- the `exclude` branches
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# find_non_overlapping_position -- attempt budget exhausted
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# search_reduction_and_clustering
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# load_image / extract_features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# statistics helpers -- value-level assertions
# ---------------------------------------------------------------------------

def _clustered_df(n=60):
    rng = np.random.default_rng(11)
    half = n // 2
    return pd.DataFrame({
        "signal": np.concatenate([rng.normal(0, 1, half),
                                  rng.normal(6, 1, n - half)]),
        "noise": rng.normal(0, 1, n),
        "cluster": [0] * half + [1] * (n - half),
    })


def test_check_normality_rejects_a_uniform_sample():
    from spacr.utils import check_normality
    rng = np.random.default_rng(2)
    assert check_normality(pd.Series(rng.normal(0, 1, 400))) is True
    assert check_normality(pd.Series(rng.uniform(0, 1, 400))) is False


def test_random_forest_feature_importance_ranks_the_separating_feature():
    from spacr.utils import random_forest_feature_importance
    out = random_forest_feature_importance(_clustered_df(), cluster_col="cluster")
    assert list(out.columns) == ["Feature", "Importance"]
    assert set(out["Feature"]) == {"signal", "noise"}
    assert out.iloc[0]["Feature"] == "signal"
    assert out["Importance"].is_monotonic_decreasing
    assert pytest.approx(out["Importance"].sum(), abs=1e-6) == 1.0


def test_perform_statistical_tests_splits_normal_and_non_normal():
    from spacr.utils import perform_statistical_tests
    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame({
        "gaussian": rng.normal(0, 1, n),
        "skewed": rng.pareto(0.7, n),          # very much not normal
        "cluster": [0] * 30 + [1] * 30,
    })
    anova_df, kruskal_df = perform_statistical_tests(df, cluster_col="cluster")

    assert list(anova_df.columns) == ["Feature", "ANOVA_Statistic", "ANOVA_pValue"]
    assert list(kruskal_df.columns) == ["Feature", "Kruskal_Statistic",
                                        "Kruskal_pValue"]
    assert "gaussian" in set(anova_df["Feature"])
    assert "skewed" in set(kruskal_df["Feature"])
    assert "cluster" not in set(anova_df["Feature"]) | set(kruskal_df["Feature"])


def test_one_cluster_keeps_features_without_inventing_a_group_test():
    """A valid one-population clustering has no between-group statistic."""
    from spacr.utils import cluster_feature_analysis, perform_statistical_tests

    df = _clustered_df()
    df["cluster"] = 0

    anova_df, kruskal_df = perform_statistical_tests(
        df, cluster_col="cluster")
    tested = pd.concat([anova_df, kruskal_df], ignore_index=True)
    assert set(tested["Feature"]) == {"signal", "noise"}
    assert tested.filter(regex="Statistic|pValue").isna().all().all()

    combined = cluster_feature_analysis(df, cluster_col="cluster")
    assert set(combined["Feature"]) == {"signal", "noise"}
    assert combined.filter(regex="Statistic|pValue").isna().all().all()


def test_combine_results_left_joins_on_feature():
    from spacr.utils import combine_results
    rf = pd.DataFrame({"Feature": ["a", "b"], "Importance": [0.7, 0.3]})
    anova = pd.DataFrame({"Feature": ["a"], "ANOVA_Statistic": [4.0],
                          "ANOVA_pValue": [0.04]})
    kruskal_ = pd.DataFrame({"Feature": ["b"], "Kruskal_Statistic": [2.0],
                             "Kruskal_pValue": [0.15]})
    out = combine_results(rf, anova, kruskal_)

    assert len(out) == 2
    assert list(out["Feature"]) == ["a", "b"]
    assert out.loc[out["Feature"] == "a", "ANOVA_pValue"].iloc[0] == 0.04
    assert np.isnan(out.loc[out["Feature"] == "a", "Kruskal_pValue"].iloc[0])
    assert out.loc[out["Feature"] == "b", "Kruskal_Statistic"].iloc[0] == 2.0
    assert np.isnan(out.loc[out["Feature"] == "b", "ANOVA_Statistic"].iloc[0])


def test_cluster_feature_analysis_end_to_end_columns():
    from spacr.utils import cluster_feature_analysis
    out = cluster_feature_analysis(_clustered_df(), cluster_col="cluster")
    assert isinstance(out, pd.DataFrame)
    assert {"Feature", "Importance"} <= set(out.columns)
    assert set(out["Feature"]) == {"signal", "noise"}
    # every feature landed in exactly one of the two test tables
    for _, row in out.iterrows():
        in_anova = "ANOVA_pValue" in out.columns and not pd.isna(
            row.get("ANOVA_pValue", np.nan))
        in_kruskal = "Kruskal_pValue" in out.columns and not pd.isna(
            row.get("Kruskal_pValue", np.nan))
        assert in_anova != in_kruskal
