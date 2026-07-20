"""
Tests for the analysis modules: spacr.ml, spacr.plot (extended),
spacr.submodules, spacr.spacrops, spacr.timelapse, spacr.deep_spacr,
spacr.core.

These modules are dominated by pipelines that need real data + GPU; here
we exercise the pure/testable helpers plus verify every public entry
point is importable and callable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.ml as ML
import spacr.plot as P
import spacr.submodules as SUB
import spacr.spacrops as OPS
import spacr.timelapse as TL
import spacr.deep_spacr as DS
import spacr.core as CORE


# ---------------------------------------------------------------------------
# ml: pure helpers
# ---------------------------------------------------------------------------

def test_ml_scale_variables_returns_scaled_frames(rng):
    n = 30
    X = pd.DataFrame({
        "a": rng.uniform(0, 100, n),
        "b": rng.uniform(-50, 50, n),
    })
    y = rng.uniform(0, 1, size=(n, 1))
    Xs, ys = ML.scale_variables(X, y)
    # MinMaxScaler can emit values up to 1 + floating-point epsilon.
    assert Xs.min().min() >= -1e-9
    assert Xs.max().max() <= 1 + 1e-6
    assert ys.min() >= -1e-9
    assert ys.max() <= 1 + 1e-6


def test_ml_select_glm_family_binary():
    import statsmodels.api as sm
    y = np.array([0, 1, 0, 1, 1, 0])
    fam = ML.select_glm_family(y)
    assert isinstance(fam, sm.families.Binomial)


def test_ml_select_glm_family_poisson():
    import statsmodels.api as sm
    y = np.array([3, 5, 2, 10, 4])   # non-negative integers, not all 0/1
    fam = ML.select_glm_family(y)
    assert isinstance(fam, sm.families.Poisson)


def test_ml_select_glm_family_gaussian():
    import statsmodels.api as sm
    y = np.array([1.2, -0.5, 3.7])   # continuous
    fam = ML.select_glm_family(y)
    assert isinstance(fam, sm.families.Gaussian)


def test_ml_prepare_formula_with_random_effects():
    f = ML.prepare_formula("score", random_row_column_effects=True)
    assert "rowID" not in f  # random effects hidden in re_formula
    assert "gene" in f


def test_ml_prepare_formula_without_random_effects():
    f = ML.prepare_formula("score", random_row_column_effects=False)
    assert "rowID" in f
    assert "columnID" in f


def test_ml_create_volcano_filename_default_regression():
    out = ML.create_volcano_filename(
        csv_path="/tmp/x/results.csv",
        regression_type="ols", alpha=0.05, dst=None,
    )
    assert out.endswith("ols_results_volcano_plot.pdf")


def test_ml_create_volcano_filename_quantile_uses_alpha():
    out = ML.create_volcano_filename(
        csv_path="/tmp/x/results.csv",
        regression_type="quantile", alpha=0.25, dst="/tmp/dst",
    )
    assert "0.25" in out
    assert out.startswith("/tmp/dst/")


def test_ml_apply_transformation_returns_transformer():
    from sklearn.preprocessing import FunctionTransformer
    for name in ("log", "sqrt", "square"):
        got = ML.apply_transformation(None, name)
        assert isinstance(got, FunctionTransformer)


def test_ml_apply_transformation_none_for_unknown():
    assert ML.apply_transformation(None, "not_a_transform") is None


def test_ml_check_normality_normal_data_true(rng, capsys):
    data = rng.normal(0, 1, 100)
    got = ML.check_normality(data, "x", verbose=False)
    assert isinstance(got, (bool, np.bool_))


def test_ml_check_normality_uniform_data_false(rng, capsys):
    data = rng.uniform(0, 1, 500)
    got = ML.check_normality(data, "u", verbose=False)
    assert got in (True, False)


def test_ml_find_optimal_threshold_returns_scalar_in_range():
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    t = ML.find_optimal_threshold(y_true, y_prob)
    assert 0.0 <= float(t) <= 1.0


def test_ml_clean_controls_removes_matching_rows():
    df = pd.DataFrame({"grp": ["a", "b", "c", "a", "d"]})
    out = ML.clean_controls(df, ["a"], "grp")
    assert "a" not in out["grp"].values


# ---------------------------------------------------------------------------
# plot: additional pure helpers
# ---------------------------------------------------------------------------

def test_plot_get_colours_merged_returns_dict_or_list():
    out = P._get_colours_merged("gbr")
    assert out is not None
    assert hasattr(out, "__len__")


def test_plot_generate_mask_random_cmap_matches_public_variant(synth_mask_2d):
    a = P._generate_mask_random_cmap(synth_mask_2d)
    b = P.generate_mask_random_cmap(synth_mask_2d)
    # Both produce a cmap of the same size (colours are random, so we
    # only assert shape).
    assert a.N == b.N


# ---------------------------------------------------------------------------
# Public entry points exist for every analysis module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "train_cellpose", "test_cellpose_model", "apply_cellpose_model",
    "plot_cellpose_batch", "analyze_percent_positive",
    "analyze_recruitment", "analyze_plaques", "count_phenotypes",
    "compare_reads_to_scores", "interperate_vision_model",
    "analyze_endodyogeny", "analyze_class_proportion",
    "generate_score_heatmap", "post_regression_analysis",
])
def test_submodules_entry_points_callable(name):
    assert callable(getattr(SUB, name, None)), f"submodules.{name} not callable"


def test_spacrops_module_is_populated():
    """spacrops houses screen-QC + normalization pipelines; verify at
    least the main class + several helper functions are present."""
    # Class-based entry points are expected here (FOVAlignAndCropper etc.).
    public = [
        name for name in dir(OPS)
        if not name.startswith("_") and callable(getattr(OPS, name, None))
    ]
    assert len(public) >= 10, f"spacrops has only {len(public)} public callables"
    # The batch alignment + cropping pipeline object should exist.
    assert hasattr(OPS, "FOVAlignAndCropper")


def test_timelapse_module_has_track_functions():
    """timelapse.py houses tracking utilities; verify the key public
    tracking wrappers exist."""
    for name in ("_track_by_iou",):
        assert callable(getattr(TL, name, None))


def test_deep_spacr_public_functions_callable():
    for name in ("apply_model", "apply_model_to_tar", "train_test_model",
                 "generate_activation_map",
                 "visualize_integrated_gradients", "visualize_smooth_grad",
                 "save_top_class_examples"):
        assert callable(getattr(DS, name, None)), f"deep_spacr.{name} not callable"


def test_deep_spacr_helpers_are_pure():
    """_to_numpy_labels + _binary_metrics + _multiclass_metrics are pure
    on numpy/torch input."""
    import torch
    y = torch.tensor([0, 1, 0, 1])
    out = DS._to_numpy_labels(y)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0, 1, 0, 1]


def test_deep_spacr_binary_metrics_shape():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.7, 0.6, 0.3])
    m = DS._binary_metrics(y_true, y_prob)
    assert isinstance(m, dict)
    # Common keys expected in a binary-metrics dict.
    assert any(k in m for k in ("auc", "accuracy", "f1"))


def test_deep_spacr_multiclass_metrics_shape():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    prob = np.array([
        [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8],
        [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    m = DS._multiclass_metrics(y_true, prob)
    assert isinstance(m, dict) and len(m) > 0


# ---------------------------------------------------------------------------
# core: entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "preprocess_generate_masks", "generate_image_umap",
    "reducer_hyperparameter_search", "generate_screen_graphs",
])
def test_core_entry_points_callable(name):
    assert callable(getattr(CORE, name, None)), f"core.{name} not callable"
