"""
Third batch of behavioral tests targeting:
  * spacr.plot     — plot_histogram, plot_feature_importance, create_venn_diagram
  * spacr.sim      — validate_and_adjust_beta_params, remove_constant_columns
  * spacr.ml       — check_and_clean_data, prepare_formula variants,
                     find_optimal_threshold edge cases
  * spacr.timelapse — _sort_key regex mismatch paths, additional filtering
  * spacr.utils    — additional pure helpers
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# spacr.plot: more helpers
# ===========================================================================

def test_plot_histogram_writes_pdf_to_dst(tmp_path, rng):
    """plot_histogram(df, column, dst=...) writes {column}_histogram.pdf."""
    from spacr.plot import plot_histogram
    df = pd.DataFrame({"score": rng.normal(0, 1, 100)})
    plot_histogram(df, "score", dst=str(tmp_path))
    pdf = tmp_path / "score_histogram.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0
    plt.close("all")


def test_plot_histogram_dst_none_still_returns(tmp_path, monkeypatch):
    """``dst=None`` writes nothing but still draws the histogram.

    The sibling test above pins the file-writing half of the contract; this
    one pins the other half, so the pair discriminates: with ``dst`` a PDF
    appears, without it *no* file does — and either way the bars are drawn.
    """
    from spacr.plot import plot_histogram
    df = pd.DataFrame({"x": np.arange(50, dtype=float)})
    # A relative savefig path would land in the cwd, so watch the cwd too.
    monkeypatch.chdir(tmp_path)
    plt.close("all")

    plot_histogram(df, "x", dst=None)

    # Nothing was written anywhere it could reach.
    assert list(tmp_path.rglob("*")) == []

    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.patches) > 0
    # Every value landed in a bar — a blank figure totals 0.
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(len(df))
    # The bars span the data range 0..49.
    left = min(p.get_x() for p in ax.patches)
    right = max(p.get_x() + p.get_width() for p in ax.patches)
    assert left <= 0.0 and right >= 49.0
    assert ax.get_title() == "Histogram of x"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "Frequency"
    plt.close("all")


def test_plot_feature_importance_returns_figure():
    from spacr.plot import plot_feature_importance
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(5)],
        "importance": [0.5, 0.3, 0.2, 0.1, 0.05],
    })
    fig = plot_feature_importance(df)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_feature_importance_handles_many_features():
    """The function scales figsize with the number of features. Large N
    should still produce a figure without error."""
    from spacr.plot import plot_feature_importance
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(100)],
        "importance": np.linspace(0, 1, 100),
    })
    fig = plot_feature_importance(df)
    assert fig is not None
    plt.close(fig)


# ===========================================================================
# spacr.sim: validate_and_adjust_beta_params
# ===========================================================================

def test_sim_validate_and_adjust_beta_params_valid_range():
    """The function takes a *list* of per-run dicts (see generate_paramiters),
    not a single dict — the old call passed one dict, so iterating it yielded
    the string keys and ``params['positive_mean']`` raised
    ``TypeError: string indices must be integers``. The swallowed skip made
    that look like a contract mismatch."""
    from spacr.sim import validate_and_adjust_beta_params
    feasible = {
        "positive_mean": 0.7,
        "positive_variance": 0.02,
        "negative_mean": 0.3,
        "negative_variance": 0.02,
    }
    # max variance for mean 0.7 is 0.7*0.3 = 0.21, so 0.5 must be clamped
    infeasible = {
        "positive_mean": 0.7,
        "positive_variance": 0.5,
        "negative_mean": 0.3,
        "negative_variance": 0.5,
    }
    out = validate_and_adjust_beta_params([feasible, infeasible])
    assert len(out) == 2
    # feasible variances are left alone, means are never touched
    assert out[0]["positive_variance"] == 0.02
    assert out[0]["negative_variance"] == 0.02
    assert [p["positive_mean"] for p in out] == [0.7, 0.7]
    assert [p["negative_mean"] for p in out] == [0.3, 0.3]
    # infeasible ones are capped at 99% of mean*(1-mean)
    assert out[1]["positive_variance"] == pytest.approx(0.7 * 0.3 * 0.99)
    assert out[1]["negative_variance"] == pytest.approx(0.3 * 0.7 * 0.99)


# ===========================================================================
# spacr.ml: check_and_clean_data + additional variants
# ===========================================================================

def test_ml_prepare_formula_uses_dependent_variable():
    from spacr.ml import prepare_formula
    f = prepare_formula("hit_score", random_row_column_effects=False)
    # The dependent variable must appear as the LHS.
    assert f.startswith("hit_score ~")


def test_ml_find_optimal_threshold_at_perfect_separation():
    """When the predictions perfectly separate the classes, the optimal
    F1 threshold should lie strictly between the two clusters."""
    from spacr.ml import find_optimal_threshold
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = float(find_optimal_threshold(y_true, y_prob))
    assert 0.3 <= t <= 0.7


def test_ml_scale_variables_zero_variance_column_stays_valid(rng):
    """A constant column has zero range — MinMaxScaler collapses it to 0
    (or 1). The function should still return without raising."""
    from spacr.ml import scale_variables
    X = pd.DataFrame({
        "constant": [1.0] * 20,
        "varying":  rng.uniform(0, 100, 20),
    })
    y = rng.uniform(0, 1, size=(20, 1))
    Xs, ys = scale_variables(X, y)
    assert Xs.shape == X.shape


def test_ml_select_glm_family_proportion_data_is_quasibinomial():
    """Values in [0, 1] but not exactly binary should get spacr's
    QuasiBinomial family (a Binomial subclass with pseudo-likelihood)."""
    from spacr.ml import select_glm_family, QuasiBinomial
    y = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    fam = select_glm_family(y)
    assert isinstance(fam, QuasiBinomial)


def test_ml_clean_controls_missing_column_returns_unchanged():
    """If the column doesn't exist, clean_controls should return the df
    unchanged (or at least not raise)."""
    from spacr.ml import clean_controls
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = clean_controls(df, ["x"], "nonexistent_column")
    assert len(out) == 3


# ===========================================================================
# spacr.timelapse: additional filtering / linking
# ===========================================================================

def test_timelapse_link_by_iou_partial_overlap_below_threshold():
    """When the IoU is below threshold, no match is emitted."""
    from spacr.timelapse import link_by_iou
    prev = np.zeros((20, 20), dtype=np.int32)
    prev[0:10, 0:10] = 1     # 100 px
    curr = np.zeros((20, 20), dtype=np.int32)
    curr[8:18, 8:18] = 2     # 100 px, overlap = 2x2 = 4 → IoU = 4/196 ≈ 0.02
    matches = link_by_iou(prev, curr, iou_threshold=0.5)
    assert matches == []


def test_timelapse_sort_key_multiple_wells_sort_stable():
    from spacr.timelapse import _sort_key
    files = ["1_A01_1_5.npy", "1_A02_1_1.npy", "1_A01_1_1.npy"]
    ordered = sorted(files, key=_sort_key)
    # A01 (well) sorts before A02; within A01, time 1 < time 5.
    assert ordered[0].endswith("A01_1_1.npy")
    assert ordered[1].endswith("A01_1_5.npy")
    assert ordered[2].endswith("A02_1_1.npy")


def test_timelapse_filter_short_tracks_empty_df():
    from spacr.timelapse import _filter_short_tracks
    df = pd.DataFrame({"track_id": [], "frame": []})
    out = _filter_short_tracks(df, min_length=5)
    assert len(out) == 0


# ===========================================================================
# spacr.utils: additional pure helpers
# ===========================================================================

def test_utils_generate_cytoplasm_mask_excludes_nucleus():
    """generate_cytoplasm_mask returns the cell minus the nucleus.

    This test previously PINNED a bug: the implementation called
    `np.logical_or(nucleus_mask != 0)` with a single argument, so every call
    raised TypeError, and the old docstring said "the eventual fix has to
    update the test". Fixed in 1.4.8.3 to
    `np.where(nucleus_mask != 0, 0, cell_mask)`; the test now asserts the
    documented behaviour instead of the breakage.
    """
    from spacr.utils import generate_cytoplasm_mask
    cell = np.zeros((20, 20), dtype=np.int32)
    cell[5:15, 5:15] = 1
    nuc = np.zeros((20, 20), dtype=np.int32)
    nuc[8:12, 8:12] = 1
    cyto = generate_cytoplasm_mask(nuc, cell)
    assert cyto[10, 10] == 0          # nucleus removed
    assert cyto[6, 6] == 1            # cytoplasmic rim retained
    assert cyto[0, 0] == 0            # background untouched


def test_utils_normalize_src_path_resolves_realpath(tmp_path):
    from spacr.utils import normalize_src_path
    out = normalize_src_path(str(tmp_path))
    assert isinstance(out, (str, list))


def test_utils_format_path_for_system_preserves_extension(tmp_path):
    from spacr.utils import format_path_for_system
    p = str(tmp_path / "sub" / "img.tif")
    got = format_path_for_system(p)
    assert got.endswith("img.tif")


# ===========================================================================
# spacr.io: more helpers
# ===========================================================================

def test_io_create_database_creates_empty_sqlite(tmp_path):
    """_create_database on a fresh path leaves an openable sqlite file."""
    from spacr.io import _create_database
    db = tmp_path / "empty.db"
    _create_database(str(db))
    import sqlite3
    with sqlite3.connect(db) as con:
        cur = con.execute("SELECT name FROM sqlite_master")
        # No tables in a fresh DB.
        assert cur.fetchall() == []


# ===========================================================================
# spacr.spacr_cellpose: parser edge cases
# ===========================================================================

def test_spacr_cellpose_parse_batch_matches_num_images():
    """Batched format: masks list length must match flow-array batch dim."""
    from spacr.spacr_cellpose import parse_cellpose4_output
    n = 5
    masks = [np.zeros((4, 4), dtype=np.int32) for _ in range(n)]
    flow0 = np.zeros((n, 4, 4, 3), dtype=np.float32)
    flow1 = np.zeros((2, n, 4, 4), dtype=np.float32)
    flow2 = np.zeros((n, 4, 4), dtype=np.float32)
    flow3 = np.zeros((n, 4, 4), dtype=np.float32)
    out = parse_cellpose4_output([masks, [flow0, flow1, flow2, flow3]])
    _, flows0, flows1, flows2, flows3 = out
    assert len(flows0) == len(flows1) == len(flows2) == len(flows3) == n


# ===========================================================================
# Extra invariants
# ===========================================================================

def test_utils_calculate_iou_symmetric():
    """IoU is a set-theoretic ratio and thus symmetric."""
    from spacr.utils import calculate_iou
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), dtype=bool)
    b[4:8, 4:8] = True
    assert abs(calculate_iou(a, b) - calculate_iou(b, a)) < 1e-9


def test_utils_dice_symmetric():
    from spacr.utils import dice_coefficient
    a = np.zeros((10, 10), dtype=bool)
    a[1:5, 1:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[3:7, 3:7] = True
    assert abs(dice_coefficient(a, b) - dice_coefficient(b, a)) < 1e-9


def test_utils_jaccard_index_matches_iou():
    """jaccard_index and calculate_iou are the same measure."""
    from spacr.utils import calculate_iou, jaccard_index
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[2:7, 2:7] = True
    assert abs(calculate_iou(a, b) - jaccard_index(a, b)) < 1e-9
