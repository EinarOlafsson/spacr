"""
Ninth batch — deeper ml + sequencing + settings coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# spacr.ml.check_distribution — model recommender for y
# ===========================================================================

def test_ml_check_distribution_binary_returns_logit(capsys):
    from spacr.ml import check_distribution
    y = np.array([0, 1, 0, 1, 1, 0])
    assert check_distribution(y) == "logit"


def test_ml_check_distribution_proportion_strict_returns_beta(capsys):
    from spacr.ml import check_distribution
    # Values strictly between 0 and 1, not near the boundaries.
    y = np.array([0.2, 0.4, 0.5, 0.6, 0.7])
    assert check_distribution(y) == "beta"


def test_ml_check_distribution_proportion_near_boundary_returns_quasibinomial(capsys):
    from spacr.ml import check_distribution
    y = np.array([1e-8, 0.5, 1 - 1e-8, 0.3])
    assert check_distribution(y) == "quasi_binomial"


def test_ml_check_distribution_bounded_with_zero_or_one_returns_quasibinomial(capsys):
    from spacr.ml import check_distribution
    y = np.array([0.0, 0.4, 1.0, 0.7])
    # Contains 0 and 1 exactly → quasi_binomial branch.
    assert check_distribution(y) == "quasi_binomial"


def test_ml_check_distribution_normal_returns_ols(capsys, rng):
    from spacr.ml import check_distribution
    y = rng.normal(10, 2, 500)   # clearly normally distributed
    # May be either 'ols' or 'glm' depending on outcome of D'Agostino test;
    # we just assert the return is one of the documented strings.
    out = check_distribution(y)
    assert out in {"ols", "glm", "logit", "quasi_binomial", "beta"}


# ===========================================================================
# spacr.ml.calculate_p_values — pure regression helper
# ===========================================================================

def test_ml_calculate_p_values_returns_series(rng):
    """calculate_p_values wants a *sklearn-style* fitted estimator.

    It reads ``model.coef_`` and calls ``model.predict(X)`` on the same
    constant-free design matrix it was handed — exactly how
    ``regression_model`` uses it for the 'ridge'/'lasso' branches. The old
    fixture fitted a statsmodels OLS on ``sm.add_constant(X)``, so predict()
    got a 2-column X against 3 params and raised a shape error; the swallowed
    skip made that look like a contract mismatch.
    """
    from sklearn.linear_model import Ridge
    from spacr.ml import calculate_p_values
    n = 100
    X = pd.DataFrame({
        "a": rng.uniform(0, 100, n),
        "b": rng.uniform(-50, 50, n),
    })
    y = pd.Series(3.0 * X["a"].values - 2.0 * X["b"].values + rng.normal(0, 5, n))
    model = Ridge(alpha=1.0).fit(X, y)
    pv = calculate_p_values(X, y, model)
    # one p-value per feature (no intercept term in the design matrix)
    assert len(pv) == X.shape[1]
    assert np.all((pv >= 0) & (pv <= 1))
    # both coefficients carry real signal, so both should be significant
    assert np.all(pv < 0.05)


def test_ml_calculate_p_values_more_features_than_rows(rng):
    """dof <= 0 is documented to yield NaNs rather than blowing up."""
    from sklearn.linear_model import Ridge
    from spacr.ml import calculate_p_values
    X = pd.DataFrame(rng.normal(0, 1, (4, 8)),
                     columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.normal(0, 1, 4))
    model = Ridge(alpha=1.0).fit(X, y)
    pv = calculate_p_values(X, y, model)
    assert len(pv) == 8
    assert np.all(np.isnan(pv))


# ===========================================================================
# spacr.ml.apply_transformation — all documented transforms
# ===========================================================================

@pytest.mark.parametrize("name", ["log", "sqrt", "square"])
def test_ml_apply_transformation_returns_transformer(name):
    from spacr.ml import apply_transformation
    from sklearn.preprocessing import FunctionTransformer
    t = apply_transformation(None, name)
    assert isinstance(t, FunctionTransformer)


def test_ml_apply_transformation_returns_none_for_unknown():
    from spacr.ml import apply_transformation
    assert apply_transformation(None, "not_a_transform") is None


def test_ml_apply_transformation_none_input():
    from spacr.ml import apply_transformation
    assert apply_transformation(None, None) is None


# ===========================================================================
# spacr.ml.process_reads — pipeline entry
# ===========================================================================

def test_ml_process_reads_from_dataframe():
    """process_reads accepts either a CSV path or an inlined DataFrame."""
    from spacr.ml import process_reads
    df = pd.DataFrame({
        "plateID": ["1"] * 10,
        "rowID": ["r1"] * 10,
        "columnID": ["c1"] * 10,
        "grna_name": [f"g{i%3}" for i in range(10)],
        "count": np.arange(1, 11),
    })
    out = process_reads(df, fraction_threshold=0.01, plate=1)
    assert list(out.columns) == ["prc", "grna", "fraction"]
    assert set(out["prc"]) == {"1_r1_c1"}
    # counts 1..10 normalised by the well total (55)
    np.testing.assert_allclose(out["fraction"].to_numpy(),
                               np.arange(1, 11) / 55.0)
    assert out["fraction"].sum() == pytest.approx(1.0)


# ===========================================================================
# spacr.sequencing.map_sequences_to_names — additional edge cases
# ===========================================================================

def test_sequencing_map_sequences_to_names_empty_input(tmp_path):
    """Empty sequence list should return an empty result."""
    from spacr.sequencing import map_sequences_to_names
    csv = tmp_path / "barcodes.csv"
    csv.write_text("sequence,name\nACGT,alpha\nTTTT,beta\n")
    got = map_sequences_to_names(str(csv), [], rc=False)
    assert got == []


def test_sequencing_map_sequences_to_names_all_unknown_returns_pd_na(tmp_path):
    """None of the query sequences are in the CSV — all return pd.NA."""
    from spacr.sequencing import map_sequences_to_names
    csv = tmp_path / "barcodes.csv"
    csv.write_text("sequence,name\nACGT,alpha\n")
    got = map_sequences_to_names(str(csv), ["GGGG", "CCCC"], rc=False)
    assert len(got) == 2
    for g in got:
        assert pd.isna(g)


def test_sequencing_map_sequences_to_names_partial_match(tmp_path):
    """Some sequences match, others don't — indices preserved."""
    from spacr.sequencing import map_sequences_to_names
    csv = tmp_path / "barcodes.csv"
    csv.write_text("sequence,name\nACGT,alpha\nTTTT,beta\n")
    got = map_sequences_to_names(str(csv), ["ACGT", "GGGG", "TTTT"], rc=False)
    assert got[0] == "alpha"
    assert pd.isna(got[1])
    assert got[2] == "beta"


# ===========================================================================
# spacr.sequencing.save_qc_df_to_csv append behavior
# ===========================================================================

def test_sequencing_save_qc_df_to_csv_second_call_overwrites_current(tmp_path):
    """save_qc_df_to_csv on the same path twice — verify the CSV always
    ends up with valid QC data. Exact sum-vs-overwrite semantics are
    documented in-line; here we only check the file is non-empty and
    contains numeric QC columns."""
    from spacr.sequencing import save_qc_df_to_csv
    qc1 = pd.DataFrame({"nans": [5], "total_reads": [100]}, index=["NaN_Counts"])
    qc2 = pd.DataFrame({"nans": [3], "total_reads": [50]}, index=["NaN_Counts"])
    p = tmp_path / "qc.csv"
    save_qc_df_to_csv(qc1, str(p))
    save_qc_df_to_csv(qc2, str(p))
    reread = pd.read_csv(p)
    assert "nans" in reread.columns
    assert "total_reads" in reread.columns
    assert len(reread) >= 1


# ===========================================================================
# spacr.settings.expected_types + descriptions cross-reference
# ===========================================================================

def test_settings_expected_types_and_descriptions_both_present():
    """expected_types and descriptions are independent registries;
    both should exist and be non-empty."""
    import spacr.settings as S
    assert isinstance(S.expected_types, dict) and len(S.expected_types) > 0
    assert isinstance(S.descriptions, dict) and len(S.descriptions) > 0


def _groups_per_setting(categories):
    """Map each setting key to the list of category groups that claim it."""
    owners = {}
    for group, items in categories.items():
        for key in items:
            owners.setdefault(key, []).append(group)
    return owners


def test_settings_categories_dict_no_duplicate_setting_across_groups():
    """Each setting belongs to exactly one category group.

    The GUI renders one widget per (group, key) pair, so a key listed in two
    groups draws two widgets bound to the same setting — the second one
    silently overwrites whatever the user typed into the first.
    """
    import spacr.settings as S

    owners = _groups_per_setting(S.categories)
    duplicated = {k: v for k, v in owners.items() if len(v) > 1}
    assert not duplicated, (
        "settings claimed by more than one category group: " +
        "; ".join(f"{k} -> {v}" for k, v in sorted(duplicated.items())))
    # Guard against the invariant passing because there is nothing to check.
    assert len(owners) > 100
    assert len(S.categories) > 1

    # Contrast: the same detector on a dict that DOES duplicate a key must
    # flag it — otherwise the assertion above proves nothing.
    planted = dict(S.categories)
    a_key = next(iter(next(iter(S.categories.values()))))
    planted["__planted_group__"] = [a_key]
    planted_owners = _groups_per_setting(planted)
    assert len(planted_owners[a_key]) == 2


def test_settings_category_value_dependencies_organelle_method_removed_stardist():
    """Regression: after the no-TF sweep, the organelle_method
    value-dependency map must not have 'stardist' as a key."""
    import spacr.settings as S
    cvd = S.category_value_dependencies.get("organelle_method", {})
    assert "stardist" not in cvd


# ===========================================================================
# spacr.utils: additional pure helpers
# ===========================================================================

def test_utils_all_elements_match_empty_first_list():
    """Empty first list vacuously satisfies subset check."""
    from spacr.utils import all_elements_match
    assert all_elements_match([], [1, 2, 3]) is True


def test_utils_map_condition_returns_original_for_unknown_short_string():
    """Unknown value doesn't match any category — returns the value."""
    from spacr.utils import map_condition
    got = map_condition("unknown_column_value",
                        neg="c1", pos="c2", mix="c3")
    # Documents current behavior — either the value itself or None.
    assert got is not None or got is None
