"""
Tests for spacr.toxo (Toxo screen analysis helpers) + spacr.spacr_cellpose
(the small pure-python surface — CellposeSAM eval-output parser).

Full pipelines in these modules (volcano plotting, GO enrichment,
identify_masks_finetune) are expensive / interactive; here we focus on
the pure helpers that don't need GPU or the network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import spacr.toxo as T
import spacr.spacr_cellpose as SC


# ---------------------------------------------------------------------------
# toxo._normalize_y_lims — coerces the y-axis shape used by volcano plots
# ---------------------------------------------------------------------------

def test_normalize_y_lims_none_auto_fits():
    neg_log_p = np.array([1.0, 2.0, 3.5, 5.0])
    broken, lo, hi = T._normalize_y_lims(None, neg_log_p)
    assert broken is False
    assert lo[0] == 0.0
    # Upper bound is 5*1.05 (or at least 1.0).
    assert lo[1] >= 5.0


def test_normalize_y_lims_none_empty_input():
    """All-inf neg_log_p (no finite values) should still return a
    sensible default of [0, 1]."""
    broken, lo, hi = T._normalize_y_lims(None, np.array([np.inf, np.nan]))
    assert broken is False
    assert lo == [0.0, 1.0]


def test_normalize_y_lims_simple_pair():
    broken, lo, hi = T._normalize_y_lims([0.0, 6.0], np.array([1.0, 2.0]))
    assert broken is False
    assert lo == [0.0, 6.0]
    assert hi is None


def test_normalize_y_lims_broken_axis_pair_of_pairs():
    broken, lo, hi = T._normalize_y_lims(
        [[0.0, 6.0], [9.0, 20.0]],
        np.array([1.0, 15.0]),
    )
    assert broken is True
    assert lo == [0.0, 6.0]
    assert hi == [9.0, 20.0]


def test_normalize_y_lims_rejects_wrong_shape():
    with pytest.raises(ValueError):
        T._normalize_y_lims([1, 2, 3], np.array([1.0]))
    with pytest.raises(ValueError):
        T._normalize_y_lims("nope", np.array([1.0]))


def test_normalize_y_lims_rejects_mixed_pair_shape():
    """One scalar + one pair is not a valid form."""
    with pytest.raises(ValueError):
        T._normalize_y_lims([0.0, [9.0, 20.0]], np.array([1.0]))


# ---------------------------------------------------------------------------
# toxo public entry points are importable + accept the documented signatures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "custom_volcano_plot", "go_term_enrichment_by_column",
    "plot_gene_phenotypes", "plot_gene_heatmaps",
    "generate_score_heatmap",
])
def test_toxo_public_functions_are_callable(name):
    assert callable(getattr(T, name, None)), f"toxo.{name} should be callable"


def test_toxo_generate_score_heatmap_mixed_condition_fractions(tmp_path):
    """The mixed-condition fraction is computed inside generate_score_heatmap.

    ``calculate_fraction_mixed_condition`` is a *nested* helper of
    :func:`spacr.toxo.generate_score_heatmap`, not a module-level symbol, so
    the previous version of this test raised AttributeError on every run and
    the swallowed skip made that look like a contract mismatch. Drive it
    through the real entry point and check the arithmetic instead: for each
    well the fraction of ``fraction_grna`` reads among the two control sgRNAs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["r1", "r2", "r3"]
    col, other_col = "c3", "c2"

    def _counts(i):
        # sgA rises, sgB falls, sgC is not a control sgRNA and must be ignored
        return {"sgA": 10 + 3 * i, "sgB": 40 - 2 * i, "sgC": 1000}

    # per-sgRNA read counts for the mixed control condition
    recs = []
    for i, r in enumerate(rows):
        for name, n in _counts(i).items():
            recs.append({"column_name": col, "rowID": r,
                         "grna_name": name, "count": n})
        # a well in another column that the c3 filter must drop
        recs.append({"column_name": other_col, "rowID": r,
                     "grna_name": "sgA", "count": 99999})
    mixed = tmp_path / "mixed.csv"
    pd.DataFrame(recs).to_csv(mixed, index=False)

    def _write_scores(path, seed, value_column="pred"):
        rng = np.random.default_rng(seed)
        pd.DataFrame([
            {"column_name": c, "rowID": r,
             value_column: float(rng.uniform(0, 1))}
            for r in rows for c in (col, other_col)
        ]).to_csv(path, index=False)

    folder = tmp_path / "models"
    for i, m in enumerate(("modelA", "modelB")):
        (folder / m).mkdir(parents=True)
        _write_scores(folder / m / "scores.csv", seed=17 + i)
    cv = tmp_path / "cv.csv"
    _write_scores(cv, seed=3, value_column="pred_cv")
    dst = tmp_path / "out"
    dst.mkdir()

    out = T.generate_score_heatmap({
        "folders": [str(folder)], "csv_name": "scores.csv",
        "data_column": "pred", "csv": str(mixed), "cv_csv": str(cv),
        "data_column_cv": "pred_cv", "plateID": 1, "columnID": col,
        "control_sgrnas": ["sgA", "sgB"], "fraction_grna": "sgA",
        "dst": str(dst),
    })
    plt.close("all")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(rows)
    expected = {
        f"plate1_{r}_{col}": _counts(i)["sgA"] / (_counts(i)["sgA"] + _counts(i)["sgB"])
        for i, r in enumerate(rows)
    }
    assert dict(zip(out["prc"], out["fraction"])) == pytest.approx(expected)
    assert (dst / "scores_comparison_plate_1.pdf").is_file()
    assert (dst / "mae_scores_comparison_plate_1.csv").is_file()


# ---------------------------------------------------------------------------
# spacr_cellpose.parse_cellpose4_output — the eval-output normaliser
# ---------------------------------------------------------------------------

def _dummy_batched_output(n_images=3, h=8, w=8):
    """Emulate cellpose 4 batched format: masks list + 4 stacked flow arrays."""
    masks = [np.ones((h, w), dtype=np.int32) * (i + 1) for i in range(n_images)]
    # 4 flow arrays, each shaped (n_images, ...) or (..., n_images).
    flow0 = np.zeros((n_images, h, w, 3), dtype=np.float32)
    flow1 = np.zeros((2, n_images, h, w), dtype=np.float32)  # (dy, dx)
    flow2 = np.zeros((n_images, h, w), dtype=np.float32)
    flow3 = np.zeros((n_images, h, w), dtype=np.float32)
    return [masks, [flow0, flow1, flow2, flow3]]


def test_parse_cellpose4_output_batched_format():
    out = _dummy_batched_output()
    masks, flows0, flows1, flows2, flows3 = SC.parse_cellpose4_output(out)
    assert len(masks) == 3
    assert len(flows0) == 3
    assert len(flows1) == 3
    assert len(flows2) == 3
    assert len(flows3) == 3


def _dummy_per_image_output(n_images=2, h=8, w=8):
    """Emulate cellpose 4 per-image format: flows is a list where each
    element is a 4-tuple (or a single ndarray)."""
    masks = [np.ones((h, w), dtype=np.int32) * (i + 1) for i in range(n_images)]
    flows = [
        [
            np.zeros((h, w, 3), dtype=np.float32),   # rgb flow
            np.zeros((2, h, w), dtype=np.float32),   # dy, dx
            np.zeros((h, w), dtype=np.float32),      # cellprob
            np.zeros((h, w), dtype=np.float32),      # styles
        ]
        for _ in range(n_images)
    ]
    return [masks, flows]


def test_parse_cellpose4_output_per_image_format():
    out = _dummy_per_image_output()
    masks, flows0, flows1, flows2, flows3 = SC.parse_cellpose4_output(out)
    assert len(masks) == 2
    for coll in (flows0, flows1, flows2, flows3):
        assert len(coll) == 2


def test_parse_cellpose4_output_per_image_ndarray_variant():
    """Per-image flows where each element is a plain ndarray (fewer than 4
    per-image outputs) — must not crash."""
    masks = [np.ones((4, 4), dtype=np.int32), np.ones((4, 4), dtype=np.int32)]
    flows = [np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
    out = SC.parse_cellpose4_output([masks, flows])
    assert len(out) == 5


def test_parse_cellpose4_output_rejects_unrecognized_flows_type():
    with pytest.raises(ValueError):
        SC.parse_cellpose4_output([[np.zeros((4, 4))], "not-a-list"])


# ---------------------------------------------------------------------------
# spacr_cellpose entry points are importable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "identify_masks_finetune", "generate_masks_from_imgs",
    "check_cellpose_models", "save_results_and_figure",
    "compare_mask", "compare_cellpose_masks",
])
def test_spacr_cellpose_entry_points_callable(name):
    assert callable(getattr(SC, name, None)), f"spacr_cellpose.{name} should be callable"
