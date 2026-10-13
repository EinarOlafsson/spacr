"""T10 — analysis modules on synthetic real-shaped data.

The screen-analysis entry points (analyze_recruitment / analyze_plaques /
analyze_endodyogeny / analyze_class_proportion) all funnel through two
reusable pieces:

  * ``spacr.utils.annotate_conditions`` — maps wells to cell / pathogen
    / treatment / condition using plate metadata.
  * ``spacr.ml.ml_analysis`` — the xgboost feature-importance analysis.

Those two are tested here with fully-synthetic but correctly-shaped inputs.

This file used to end with a parametrised ``test_analyze_entrypoint_smoke``
that ran three analyze_* entry points against a hand-built measurements.db
and swallowed any failure into ``pytest.skip``. It skipped for its entire
life -- the synthetic schema had no ``prcf`` column, so not one line of
``analyze_recruitment`` / ``analyze_endodyogeny`` /
``analyze_class_proportion`` was ever executed by it -- while
``test_cov_submodules_recruitment_plaques.py`` (14 tests),
``test_cov_submodules_endodyogeny.py`` and
``test_cov_submodules_class_proportion.py`` drive all three for real and
assert on the tables and files they write. It was deleted rather than
repaired: repairing it would have reproduced those three modules.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# annotate_conditions — the shared well→condition mapper
# ---------------------------------------------------------------------------

def test_annotate_conditions_maps_pathogen_and_treatment():
    import pandas as pd
    from spacr.utils import annotate_conditions
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "rowID": rng.choice(["r1", "r2", "r3"], n),
        "columnID": rng.choice(["c1", "c2", "c3", "c4"], n),
        "value": rng.normal(0, 1, n),
    })
    out = annotate_conditions(
        df,
        cells=["HeLa"], cell_loc=None,
        pathogens=["nc", "pc"],
        pathogen_loc=[["c1"], ["c2"]],
        treatments=["untreated", "treated"],
        treatment_loc=[["c1", "c2"], ["c3", "c4"]],
    )
    assert "condition" in out.columns
    # Column c1 → pathogen nc; c2 → pc.
    c1 = out[out["columnID"] == "c1"]
    if len(c1):
        assert (c1["pathogen"] == "nc").all()


def test_annotate_conditions_handles_no_metadata():
    import pandas as pd
    from spacr.utils import annotate_conditions
    df = pd.DataFrame({
        "rowID": ["r1", "r2"], "columnID": ["c1", "c2"],
        "value": [1.0, 2.0],
    })
    # No metadata at all — should not raise, just return the df with
    # (possibly empty) annotation columns.
    out = annotate_conditions(df)
    assert out is not None
    assert len(out) == 2


# ---------------------------------------------------------------------------
# ml_analysis — xgboost feature-importance path
# ---------------------------------------------------------------------------

def test_ml_analysis_runs_on_synthetic_features():
    """ml_analysis should fit + return results on a synthetic feature
    table with a c1/c2 control split.

    Two things the old ``try/except -> pytest.skip`` hid, both of them the
    test's fault rather than the product's:

    * ``channel_of_interest=0`` makes ``filter_dataframe_features`` keep only
      columns whose name contains ``channel_0``. The old fixture named its
      features ``feat_0..feat_5``, so *every* feature was dropped and xgboost
      died deep inside QuantileDMatrix with ``IndexError: list index out of
      range``. Feature names must follow the spacr measurement convention.
    * The tail of ``ml_analysis`` does
      ``df[['plateID','rowID','columnID','fieldID','object']] =
      df.index.astype(str).str.split('_', expand=True)``, so the index has to
      be a 5-part ``prcfo`` string, not a RangeIndex.
    """
    import pandas as pd
    from spacr.ml import ml_analysis
    rng = np.random.default_rng(1)
    n = 200
    names = [f"cell_channel_0_{s}" for s in
             ("mean_intensity", "median_intensity", "p75_intensity",
              "std_intensity", "min_intensity", "max_intensity")]
    data = {name: rng.normal(0, 1, n) for name in names}
    df = pd.DataFrame(data)
    df.index = [f"plate1_r{(i % 8) + 1}_c{(i % 2) + 1}_f1_o{i}" for i in range(n)]
    df["columnID"] = ["c1" if i % 2 == 0 else "c2" for i in range(n)]
    # Make the first feature separate the two groups so xgboost has real signal.
    df.loc[df["columnID"] == "c2", names[0]] += 3.0
    output, figs = ml_analysis(
        df, channel_of_interest=0, location_column="columnID",
        positive_control="c2", negative_control="c1",
        n_repeats=1, top_features=5, n_estimators=50,
        model_type="xgboost",
        remove_low_variance_features=False,
        remove_highly_correlated_features=False,
        prune_features=False, cross_validation=False,
        n_jobs=1, verbose=False,
    )
    scored_df, permutation_df, feature_importance_df = output[0], output[1], output[2]
    train_features = output[9]
    assert set(train_features) == set(names)
    assert "predictions" in scored_df.columns
    assert len(scored_df) == n
    # the planted signal must be the top-ranked feature
    assert feature_importance_df.iloc[0]["feature"] == names[0]
    assert not permutation_df.empty
    assert len(figs) == 2

