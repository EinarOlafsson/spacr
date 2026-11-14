"""Coverage-fill batch 2 for spacr.ml mid-size functions (real-data fixtures)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import ml as ML


# ---------------------------------------------------------------------------
# calculate_p_values
# ---------------------------------------------------------------------------

def case_calculate_p_values_normal():
    _seen = []
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 60)
    model = LinearRegression().fit(X, y)
    p = ML.calculate_p_values(X, y, model)
    _seen.append(bool(
        len(p) == 3 and np.all(np.isfinite(p))))
    return _seen


def case_calculate_p_values_underdetermined():
    _seen = []
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (3, 5))    # more features than obs → dof<=0 → NaN
    y = rng.normal(0, 1, 3)
    model = LinearRegression().fit(X, y)
    p = ML.calculate_p_values(X, y, model)
    _seen.append(bool(
        len(p) == 5 and np.all(np.isnan(p))))
    return _seen


# ---------------------------------------------------------------------------
# regression_model — several backends
# ---------------------------------------------------------------------------

def _Xy(n=80, p=3):
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (n, p))
    y = X @ np.array([0.8, -0.5, 0.3]) + rng.normal(0, 0.2, n)
    return X, y


#: The coefficients `_Xy` actually generates. Asserting `hasattr(model,
#: "predict")` cannot tell a fitted model from a sign-inverted one, and a
#: sign inversion is the mistake that turns a protective hit into a
#: sensitising one. The noise is 0.2 against effects of 0.3 to 0.8 over 80
#: rows, so a correct fit lands well inside this tolerance.
TRUE_COEFFICIENTS = (0.8, -0.5, 0.3)
COEFFICIENT_TOLERANCE = 0.15


def _assert_it_recovered_the_truth(model):
    """The fit found the coefficients the fixture put in."""
    import numpy as np

    params = getattr(model, "params", None)
    if params is None:
        return                      # a backend that reports no coefficients
    values = np.asarray(params, dtype=float).ravel()
    # An intercept, when the design carries one, is the extra leading term.
    if values.size == len(TRUE_COEFFICIENTS) + 1:
        values = values[1:]
    if values.size != len(TRUE_COEFFICIENTS):
        return
    for got, want in zip(values, TRUE_COEFFICIENTS):
        assert abs(got - want) < COEFFICIENT_TOLERANCE, (
            f"recovered {list(np.round(values, 3))}, "
            f"expected about {list(TRUE_COEFFICIENTS)}")


def case_regression_model_ols():
    _seen = []
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ols")
    _seen.append(bool(
        hasattr(model, "predict")))
    _assert_it_recovered_the_truth(model)
    return _seen


def case_regression_model_ols_cov_type():
    _seen = []
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ols", cov_type="HC3")
    _seen.append(bool(
        hasattr(model, "predict")))
    # A robust covariance changes the standard errors, never the estimates.
    _assert_it_recovered_the_truth(model)
    return _seen


def case_regression_model_lasso_fixed_and_auto():
    _seen = []
    X, y = _Xy()
    m_fixed = ML.regression_model(X, y, regression_type="lasso", alpha=0.1)
    _seen.append(bool(
        hasattr(m_fixed, "coef_")))
    m_auto = ML.regression_model(X, y, regression_type="lasso", alpha="auto")
    _seen.append(bool(
        hasattr(m_auto, "coef_")))
    return _seen


def case_regression_model_ridge():
    _seen = []
    X, y = _Xy()
    model = ML.regression_model(X, y, regression_type="ridge", alpha=1.0)
    _seen.append(bool(
        hasattr(model, "coef_")))
    return _seen


def case_regression_model_glm():
    _seen = []
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (80, 2))
    y = np.abs(X @ np.array([0.5, 0.3]) + rng.normal(2, 0.3, 80))  # positive
    model = ML.regression_model(X, y, regression_type="glm")
    _seen.append(bool(
        hasattr(model, "predict")))
    return _seen


def case_regression_model_unsupported():
    _seen = []
    X, y = _Xy()
    with pytest.raises(ValueError) as _exc1:
        ML.regression_model(X, y, regression_type="bogus")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# _calculate_similarity
# ---------------------------------------------------------------------------

def case_calculate_similarity():
    _seen = []
    rng = np.random.default_rng(4)
    n = 40
    df = pd.DataFrame({
        "columnID": ["c1"] * 20 + ["c2"] * 20,
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.normal(0, 1, n),
    })
    out = ML._calculate_similarity(
        df, ["f1", "f2", "f3"], "columnID", "c2", "c1")
    _seen.append(('similarity_to_pos_euclidean' if 'similarity_to_pos_euclidean' in (out.columns) else ("missing", 'similarity_to_pos_euclidean')))
    _seen.append(('similarity_to_neg_mahalanobis' if 'similarity_to_neg_mahalanobis' in (out.columns) else ("missing", 'similarity_to_neg_mahalanobis')))
    return _seen


def case_calculate_similarity_list_controls():
    _seen = []
    rng = np.random.default_rng(5)
    n = 30
    df = pd.DataFrame({
        "columnID": (["c1", "c2", "c3"] * 10)[:n],
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
    })
    out = ML._calculate_similarity(
        df, ["f1", "f2"], "columnID", ["c2", "c3"], ["c1"])
    _seen.append(('similarity_to_pos_cosine' if 'similarity_to_pos_cosine' in (out.columns) else ("missing", 'similarity_to_pos_cosine')))
    return _seen


# ---------------------------------------------------------------------------
# check_and_clean_data
# ---------------------------------------------------------------------------

def _clean_data_frame(n=40, seed=6):
    """One row per (well, gRNA) - the shape perform_regression's merge makes."""
    rng = np.random.default_rng(seed)
    wells = ("p1_r1_c1", "p1_r2_c2")
    genes = ("geneA", "geneB")
    rows = []
    for i in range(n):
        well = wells[i % 2]
        gene = genes[(i // 2) % 2]
        rows.append({
            "fraction": float(rng.uniform(0, 1)),
            "prediction": float(rng.uniform(0, 1)),
            "grna": f"{gene}_g{i}",
            "gene": gene,
            "plateID": "p1",
            "rowID": well.split("_")[1],
            "columnID": well.split("_")[2],
            "prc": well,
        })
    return pd.DataFrame(rows)


def case_check_and_clean_data():
    _seen = []
    df = _clean_data_frame()
    out = ML.check_and_clean_data(df, dependent_variable="prediction")
    _seen.append(('gene_fraction' if 'gene_fraction' in (out.columns) else ("missing", 'gene_fraction')))
    _seen.append(('gene' if 'gene' in (out.columns) else ("missing", 'gene')))
    # gene_fraction is the gene's share of the well: the sum of its gRNAs'
    # fractions, each counted once.
    expect = df.groupby(["prc", "gene"])["fraction"].sum()
    for (well, gene), total in expect.items():
        got = out.loc[(out["prc"] == well) & (out["gene"] == gene),
                      "gene_fraction"].unique()
        assert len(got) == 1 and abs(got[0] - total) < 1e-12
    return _seen


def case_check_and_clean_data_refuses_two_fractions_for_one_well_grna():
    """The same gRNA cannot hold two shares of the same well's library.

    Left to itself the sum would silently double-count, and a merge that
    duplicated the count table is exactly how that happens.
    """
    _seen = []
    df = _clean_data_frame(n=8)
    duplicate = df.iloc[[0]].copy()
    duplicate["fraction"] = float(df.iloc[0]["fraction"]) + 0.1
    df = pd.concat([df, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="carry more than one 'fraction'") as _exc1:
        ML.check_and_clean_data(df, dependent_variable="prediction")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# process_scores
# ---------------------------------------------------------------------------

def _scores_df(n_wells=4, per_well=10):
    rng = np.random.default_rng(7)
    rows = []
    for w in range(n_wells):
        for o in range(per_well):
            rows.append({
                "prcfo": f"plate1_r{w+1}_c1_f1_o{o+1}",
                "pred": float(rng.uniform(0.1, 0.9)),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("agg", ["mean", "median", "quantile"])
def test_process_scores_agg_types(agg):
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2, agg_type=agg)
    assert "cell_count" in out.columns and dv == "pred"
    assert (out["cell_count"] >= 2).all()


def case_process_scores_none_agg():
    _seen = []
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2, agg_type=None)
    _seen.append(('cell_count' if 'cell_count' in (out.columns) else ("missing", 'cell_count')))
    return _seen


def case_process_scores_poisson():
    _seen = []
    df = _scores_df()
    # Poisson models per-well positive-cell counts, so their object-level
    # response must be a binary call rather than an arbitrary score.
    df["pred"] = (df["pred"] >= 0.5).astype(int)
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2,
        regression_type="poisson")
    _seen.append(('cell_count' if 'cell_count' in (out.columns) else ("missing", 'cell_count')))
    return _seen


def case_process_scores_invert_complement():
    _seen = []
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2,
        invert_dependent_variable=True)
    _seen.append(('cell_count' if 'cell_count' in (out.columns) else ("missing", 'cell_count')))
    return _seen


def case_process_scores_invert_reciprocal():
    _seen = []
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2,
        invert_dependent_variable=-1)
    _seen.append(('cell_count' if 'cell_count' in (out.columns) else ("missing", 'cell_count')))
    return _seen


def case_process_scores_transform():
    _seen = []
    df = _scores_df()
    out, dv = ML.process_scores(
        df, "pred", plate="plate1", min_cells_per_well=2,
        agg_type="mean", transform="log")
    _seen.append(keep(dv))
    return _seen


def case_process_scores_bad_agg():
    _seen = []
    df = _scores_df()
    with pytest.raises(ValueError) as _exc1:
        ML.process_scores(df, "pred", plate="plate1",
                          min_cells_per_well=2, agg_type="bogus")
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_process_scores_bad_invert():
    _seen = []
    df = _scores_df()
    with pytest.raises(ValueError) as _exc1:
        ML.process_scores(df, "pred", plate="plate1",
                          min_cells_per_well=2, invert_dependent_variable=99)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# ml_analysis (end-to-end classifier on control wells)
# ---------------------------------------------------------------------------

def _feature_df(per_class=60):
    rng = np.random.default_rng(8)
    feats = [
        "cell_channel_3_mean_intensity",
        "cell_channel_3_percentile_75",
        "nucleus_channel_3_mean_intensity",
        "cytoplasm_channel_3_mean_intensity",
        "pathogen_channel_3_mean_intensity",
        "cell_channel_3_std_intensity",
    ]
    rows = []
    index = []
    counter = 0
    for cls, col, lo in [("neg", "c1", 0.3), ("pos", "c2", 0.7),
                          ("other", "c3", 0.5)]:
        for j in range(per_class):
            row = {"columnID": col}
            for f in feats:
                row[f] = float(rng.normal(lo, 0.12) + rng.normal(0, 0.05))
            rows.append(row)
            # ml_analysis requires a prcfo-style index (5 underscore parts)
            index.append(f"plate1_r1_{col}_f1_o{counter}")
            counter += 1
    return pd.DataFrame(rows, index=index)


def case_ml_analysis_random_forest():
    _seen = []
    df = _feature_df()
    output, figs = ML.ml_analysis(
        df, channel_of_interest=3, location_column="columnID",
        positive_control="c2", negative_control="c1",
        model_type="random_forest", n_repeats=2, test_size=0.25,
        split_by="cell",
        remove_highly_correlated_features=False, n_jobs=1)
    scored_df = output[0]
    _seen.append(bool(
        "prediction_probability" in scored_df.columns or len(output) >= 8))
    _seen.append(bool(
        figs is not None))
    return _seen


def case_ml_analysis_bad_model_type():
    _seen = []
    df = _feature_df(per_class=30)
    with pytest.raises(ValueError) as _exc1:
        ML.ml_analysis(
            df, channel_of_interest=3, location_column="columnID",
            positive_control="c2", negative_control="c1",
            model_type="bogus", n_repeats=1,
            split_by="cell",
            remove_highly_correlated_features=False, n_jobs=1)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_ml_analysis_cross_validation():
    _seen = []
    df = _feature_df()
    output, figs = ML.ml_analysis(
        df, channel_of_interest=3, location_column="columnID",
        positive_control="c2", negative_control="c1",
        model_type="random_forest", n_repeats=2, cross_validation=True,
        split_by="cell",
        remove_highly_correlated_features=False, n_jobs=1)
    _seen.append(bool(
        output[0] is not None))
    return _seen


def case_ml_analysis_logistic_regression():
    # logistic_regression has no feature_importances_ → else branch, which
    # now FILLS the panel from the permutation importance rather than
    # handing back an empty frame and no figure.
    # (regression test: previously raised UnboundLocalError on the return)
    _seen = []
    df = _feature_df()
    output, figs = ML.ml_analysis(
        df, channel_of_interest=3, location_column="columnID",
        positive_control="c2", negative_control="c1",
        model_type="logistic_regression", n_repeats=2, test_size=0.25,
        split_by="cell",
        remove_highly_correlated_features=False, n_jobs=1)
    _seen.append(bool(
        not output[2].empty))
    _seen.append(bool(
        figs[1] is not None))
    _seen.append(('permutation' if 'permutation' in (figs[1].axes[0].get_title().lower()) else ("missing", 'permutation')))
    return _seen


def case_ml_analysis_gradient_boosting():
    # HistGradientBoostingClassifier lacks feature_importances_ → else branch
    # (regression test: previously raised AttributeError)
    _seen = []
    df = _feature_df()
    output, figs = ML.ml_analysis(
        df, channel_of_interest=3, location_column="columnID",
        positive_control="c2", negative_control="c1",
        model_type="gradient_boosting", n_repeats=2, test_size=0.25,
        split_by="cell",
        n_estimators=20,
        remove_highly_correlated_features=False, n_jobs=1)
    _seen.append(bool(
        output[0] is not None))
    # HistGradientBoostingClassifier exposes no `feature_importances_`, so
    # the panel is drawn from the permutation importance and says so.
    _seen.append(bool(
        figs[1] is not None))
    _seen.append(('permutation' if 'permutation' in (figs[1].axes[0].get_title().lower()) else ("missing", 'permutation')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_process_scores_none_agg, ['cell_count'], id="process_scores_none_agg"),
    pytest.param(case_process_scores_poisson, ['cell_count'], id="process_scores_poisson"),
    pytest.param(case_process_scores_invert_complement, ['cell_count'], id="process_scores_invert_complement"),
    pytest.param(case_process_scores_invert_reciprocal, ['cell_count'], id="process_scores_invert_reciprocal"),
    pytest.param(case_process_scores_transform, ['log_pred'], id="process_scores_transform"),
    pytest.param(case_process_scores_bad_agg, ['ValueError'], id="process_scores_bad_agg"),
    pytest.param(case_process_scores_bad_invert, ['ValueError'], id="process_scores_bad_invert"),
    pytest.param(case_regression_model_ols, [True], id="regression_model_ols"),
    pytest.param(case_regression_model_ols_cov_type, [True], id="regression_model_ols_cov_type"),
    pytest.param(case_regression_model_lasso_fixed_and_auto, [True, True], id="regression_model_lasso_fixed_and_auto"),
    pytest.param(case_regression_model_ridge, [True], id="regression_model_ridge"),
    pytest.param(case_regression_model_glm, [True], id="regression_model_glm"),
    pytest.param(case_regression_model_unsupported, ['ValueError'], id="regression_model_unsupported"),
    pytest.param(case_calculate_similarity, ['similarity_to_pos_euclidean', 'similarity_to_neg_mahalanobis'], id="calculate_similarity"),
    pytest.param(case_calculate_similarity_list_controls, ['similarity_to_pos_cosine'], id="calculate_similarity_list_controls"),
    pytest.param(case_calculate_p_values_normal, [True], id="calculate_p_values_normal"),
    pytest.param(case_calculate_p_values_underdetermined, [True], id="calculate_p_values_underdetermined"),
    pytest.param(case_check_and_clean_data, ['gene_fraction', 'gene'], id="check_and_clean_data"),
    pytest.param(case_check_and_clean_data_refuses_two_fractions_for_one_well_grna, ['ValueError'], id="check_and_clean_data_refuses_two_fractions_for_one_well_grna"),
    pytest.param(case_ml_analysis_random_forest, [True, True], id="ml_analysis_random_forest"),
    pytest.param(case_ml_analysis_bad_model_type, ['ValueError'], id="ml_analysis_bad_model_type"),
    pytest.param(case_ml_analysis_cross_validation, [True], id="ml_analysis_cross_validation"),
    pytest.param(case_ml_analysis_logistic_regression, [True, True, 'permutation'], id="ml_analysis_logistic_regression"),
    pytest.param(case_ml_analysis_gradient_boosting, [True, True, 'permutation'], id="ml_analysis_gradient_boosting"),
])
def test_fill_ml2_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected


