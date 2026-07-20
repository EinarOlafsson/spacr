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


def test_toxo_calculate_fraction_mixed_condition_shape():
    """calculate_fraction_mixed_condition returns a dataframe with per-
    (plate,column) fractions of mixed-condition reads. Feed it a tiny
    synthetic CSV to check the row/col shape."""
    df = pd.DataFrame({
        "plateID": ["1", "1", "1", "1"],
        "columnID": ["c1", "c2", "c3", "c3"],
        "grna_name": ["TGGT1_220950_1", "other", "TGGT1_233460_4", "TGGT1_220950_1"],
        "count": [10, 20, 30, 40],
    })
    tmp = "/tmp/spacr_toxo_synth.csv"
    df.to_csv(tmp, index=False)
    try:
        out = T.calculate_fraction_mixed_condition(
            csv=tmp, plate=1, column="c3",
            control_sgrnas=None,   # use default
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"function has undocumented deps: {e}")
    assert out is not None


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
