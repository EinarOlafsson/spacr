"""
Fifth batch: focused tests targeting still-cold areas —
  * spacr.plot: create_venn_diagram, plot_lorenz_curves
  * spacr.timelapse: _compute_parent_child_overlaps + _summarise_child_features
  * spacr.utils: fill_holes_in_mask on more shapes
  * spacr.deep_spacr: _to_numpy_labels + metrics variants
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
# spacr.plot.create_venn_diagram — CSV in, dict out
# ===========================================================================

def test_plot_create_venn_diagram_returns_overlap_dict(tmp_path):
    from spacr.plot import create_venn_diagram
    df1 = pd.DataFrame({"gene": ["g1", "g2", "g3"], "coefficient": [0.5, 0.6, 0.7]})
    df2 = pd.DataFrame({"gene": ["g2", "g3", "g4"], "coefficient": [0.4, 0.5, 0.6]})
    p1 = tmp_path / "f1.csv"; p2 = tmp_path / "f2.csv"
    df1.to_csv(p1, index=False); df2.to_csv(p2, index=False)
    save_path = tmp_path / "venn.pdf"
    result = create_venn_diagram(str(p1), str(p2), gene_column="gene",
                                 filter_coeff=0.1, save=True,
                                 save_path=str(save_path))
    assert set(result.keys()) == {"overlap", "unique_to_file1", "unique_to_file2"}
    # g2, g3 are shared; g1 unique to file1; g4 unique to file2.
    assert set(result["overlap"]) == {"g2", "g3"}
    assert set(result["unique_to_file1"]) == {"g1"}
    assert set(result["unique_to_file2"]) == {"g4"}
    assert save_path.exists()
    plt.close("all")


def test_plot_create_venn_diagram_filter_coeff_drops_low():
    """filter_coeff=0.55 with coefficient column drops genes below."""
    import tempfile
    from spacr.plot import create_venn_diagram
    tmp = Path(tempfile.mkdtemp())
    df = pd.DataFrame({"gene": ["a", "b", "c"], "coefficient": [0.4, 0.6, 0.8]})
    p = tmp / "x.csv"
    df.to_csv(p, index=False)
    save = tmp / "v.pdf"
    r = create_venn_diagram(str(p), str(p), gene_column="gene",
                            filter_coeff=0.55, save=True, save_path=str(save))
    # After filter: only b and c remain in both sets → all overlap; nothing unique.
    assert set(r["overlap"]) == {"b", "c"}
    assert r["unique_to_file1"] == [] and r["unique_to_file2"] == []
    plt.close("all")


def test_plot_create_venn_diagram_requires_save_path_when_save_true(tmp_path):
    """Documented contract: save=True with save_path=None raises."""
    from spacr.plot import create_venn_diagram
    df = pd.DataFrame({"gene": ["a"], "coefficient": [0.5]})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="save_path"):
        create_venn_diagram(str(p), str(p), gene_column="gene",
                            filter_coeff=None, save=True, save_path=None)
    plt.close("all")


# ===========================================================================
# spacr.timelapse: parent/child overlaps
# ===========================================================================

def test_timelapse_compute_parent_child_overlaps_basic():
    """Two frames, each with one cell containing one nucleus. Should
    produce 2 records — one per (frame, parent, child) tuple."""
    from spacr.timelapse import _compute_parent_child_overlaps
    parent = np.zeros((2, 20, 20), dtype=np.int32)
    child  = np.zeros((2, 20, 20), dtype=np.int32)
    # Frame 0: parent=1 covers a big region, child=7 overlaps it.
    parent[0, 5:15, 5:15] = 1
    child[0, 8:12, 8:12] = 7
    # Frame 1: parent=2, child=8.
    parent[1, 5:15, 5:15] = 2
    child[1, 8:12, 8:12] = 8

    df = _compute_parent_child_overlaps(parent, child, "cell_id", "obj_id")
    assert set(df.columns) == {"frame", "cell_id", "obj_id"}
    assert len(df) == 2
    row0 = df.iloc[0]
    assert row0["frame"] == 0
    assert row0["cell_id"] == 1
    assert row0["obj_id"] == 7


def test_timelapse_compute_parent_child_overlaps_empty_frames_return_empty_df():
    from spacr.timelapse import _compute_parent_child_overlaps
    parent = np.zeros((3, 10, 10), dtype=np.int32)   # no parents
    child  = np.zeros((3, 10, 10), dtype=np.int32)
    df = _compute_parent_child_overlaps(parent, child, "cell_id", "obj_id")
    # Empty DataFrame with the documented columns.
    assert len(df) == 0
    assert set(df.columns) == {"frame", "cell_id", "obj_id"}


def test_timelapse_compute_parent_child_overlaps_multiple_children_per_parent():
    """A single parent containing 3 disjoint children — 3 rows returned."""
    from spacr.timelapse import _compute_parent_child_overlaps
    parent = np.zeros((1, 20, 20), dtype=np.int32)
    child  = np.zeros((1, 20, 20), dtype=np.int32)
    parent[0, 2:18, 2:18] = 1
    child[0, 3:5, 3:5] = 1
    child[0, 7:9, 7:9] = 2
    child[0, 12:14, 12:14] = 3

    df = _compute_parent_child_overlaps(parent, child, "p", "c")
    assert len(df) == 3
    assert set(df["c"]) == {1, 2, 3}
    assert (df["p"] == 1).all()


# ===========================================================================
# spacr.utils.fill_holes_in_mask — more shapes
# ===========================================================================

def test_utils_fill_holes_in_mask_no_holes_unchanged():
    """A solid mask with no holes is unchanged (up to relabeling)."""
    from spacr.utils import fill_holes_in_mask
    m = np.zeros((20, 20), dtype=np.int32)
    m[5:15, 5:15] = 1
    filled = fill_holes_in_mask(m)
    # Total area unchanged.
    assert (filled != 0).sum() == (m != 0).sum()


def test_utils_fill_holes_in_mask_empty_input():
    from spacr.utils import fill_holes_in_mask
    m = np.zeros((10, 10), dtype=np.int32)
    out = fill_holes_in_mask(m)
    assert out.shape == m.shape
    assert (out == 0).all()


def test_utils_fill_holes_in_mask_multiple_objects_with_holes():
    """Two ring-shaped objects — both should have their holes filled."""
    from spacr.utils import fill_holes_in_mask
    m = np.zeros((30, 30), dtype=np.int32)
    # Object 1
    m[2:10, 2:10] = 1
    m[4:8, 4:8] = 0
    # Object 2
    m[15:25, 15:25] = 2
    m[18:22, 18:22] = 0

    filled = fill_holes_in_mask(m)
    # Both holes should now be labeled (either 1 or 2, whichever the
    # implementation propagates).
    assert (filled[4:8, 4:8] != 0).all()
    assert (filled[18:22, 18:22] != 0).all()


# ===========================================================================
# spacr.deep_spacr: metrics + tensor helpers
# ===========================================================================

def test_deep_spacr_to_numpy_labels_multiclass_tensor():
    """When the input is a (batch, num_classes) logit tensor,
    _to_numpy_labels returns the argmax per row."""
    import torch
    from spacr.deep_spacr import _to_numpy_labels
    logits = torch.tensor([
        [1.0, 5.0, 2.0],
        [4.0, 2.0, 1.0],
        [0.5, 0.4, 3.0],
    ])
    out = _to_numpy_labels(logits)
    # argmax by row: 1, 0, 2.
    assert out.tolist() == [1, 0, 2]


def test_deep_spacr_binary_metrics_perfect_classification():
    """Perfectly separable predictions → accuracy 1.0 across all splits."""
    import numpy as np
    from spacr.deep_spacr import _binary_metrics
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    m = _binary_metrics(y_true, y_prob)
    assert isinstance(m, dict) and len(m) > 0
    # Documented keys per the implementation output:
    for key in ("accuracy", "neg_accuracy", "pos_accuracy"):
        assert key in m
        assert m[key] == pytest.approx(1.0)


def test_deep_spacr_multiclass_metrics_returns_dict_with_keys():
    import numpy as np
    from spacr.deep_spacr import _multiclass_metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    prob = np.array([
        [0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8],
        [0.7, 0.2, 0.1],   [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    m = _multiclass_metrics(y_true, prob)
    assert isinstance(m, dict)
    assert len(m) > 0


# ===========================================================================
# spacr.sequencing: more helpers
# ===========================================================================

def test_sequencing_get_consensus_base_prefers_higher_quality():
    from spacr.sequencing import get_consensus_base
    # ASCII: 'I' (73) > '!' (33) → the 'I'-quality base wins.
    assert get_consensus_base([("A", "!"), ("G", "I")]) == "G"
    assert get_consensus_base([("A", "I"), ("G", "!")]) == "A"


def test_sequencing_extract_sequence_and_quality_bounds():
    from spacr.sequencing import extract_sequence_and_quality
    seq = "ACGTACGT"
    qual = "IIIIIIII"
    # Beyond-end indices should clamp gracefully via Python slicing.
    s, q = extract_sequence_and_quality(seq, qual, 5, 100)
    assert s == "CGT"
    assert q == "III"


def test_sequencing_reverse_complement_gc_pairing():
    from spacr.sequencing import reverse_complement
    assert reverse_complement("GC") == "GC"    # palindrome
    assert reverse_complement("AT") == "AT"    # palindrome
    assert reverse_complement("GATC") == "GATC"  # palindrome
