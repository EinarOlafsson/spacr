"""
Eighth batch of behavioral tests — plot helpers, visualization functions,
additional core algorithms.
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
# spacr.plot: plot_lorenz_curves — nested lorenz_curve function is pure math
# ===========================================================================

def test_plot_lorenz_curves_smoke_synthetic_csvs(tmp_path):
    """One curve per CSV plus the combined curve, and the more unequal plate
    bows further from the diagonal.

    df2's counts are df1's squared, so plate 2 *must* be more unequal than
    plate 1: a bigger Gini and a smaller area under its Lorenz curve. Two
    identical (or absent) curves cannot satisfy that.
    """
    from spacr.plot import plot_lorenz_curves
    df1 = pd.DataFrame({
        "grna_name": [f"g{i}" for i in range(20)],
        "count": np.arange(1, 21),
    })
    df2 = pd.DataFrame({
        "grna_name": [f"g{i}" for i in range(20)],
        "count": np.arange(1, 21) ** 2,   # more unequal
    })
    p1 = tmp_path / "d1.csv"; p2 = tmp_path / "d2.csv"
    df1.to_csv(p1, index=False); df2.to_csv(p2, index=False)
    plt.close("all")
    # Deliberately NOT wrapped in try/pytest.skip. Swallowing the exception is
    # exactly what hid the remove_keys=None crash: this call raised TypeError
    # on every invocation and the suite reported a tidy "skipped".
    plot_lorenz_curves(csv_files=[str(p1), str(p2)], save=False)

    fig = plt.gcf()
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # plate 1, plate 2, combined.
    assert len(ax.lines) == 3
    labels = [ln.get_label() for ln in ax.lines]
    assert labels[0].startswith("plate 1 (Gini: ")
    assert labels[1].startswith("plate 2 (Gini: ")
    assert labels[2].startswith("Combined (Gini: ")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == labels

    areas, ginis = [], []
    for line, label in zip(ax.lines, labels):
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        # A Lorenz curve runs (0,0) -> (1,1) and never decreases.
        assert (x[0], y[0]) == (0.0, 0.0)
        assert (x[-1], y[-1]) == pytest.approx((1.0, 1.0))
        assert np.all(np.diff(y) >= -1e-12)
        assert np.all(y <= x + 1e-12)          # bows below the diagonal
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        areas.append(float(trapezoid(y, x)))
        ginis.append(float(label.split("Gini: ")[1].rstrip(")")))

    # The whole point of the fixture: plate 2 is the unequal one.
    assert ginis[1] > ginis[0] > 0
    assert areas[1] < areas[0]
    # Printed Gini and drawn area are two views of the same number.
    for area, gini in zip(areas, ginis):
        assert gini == pytest.approx(1 - 2 * area, abs=1e-3)
    # save=False must not leave a results directory behind.
    assert not (tmp_path / "results").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d1.csv", "d2.csv"]
    plt.close("all")


# ===========================================================================
# spacr.plot: plot_permutation returns a figure
# ===========================================================================

def test_plot_permutation_returns_figure():
    """plot_permutation draws error bars, so it needs the permutation frame
    ml_analysis builds: feature / importance_mean / importance_std. The old
    fixture passed a plain ``importance`` column (that is plot_feature_
    importance's contract) and the swallowed skip hid the KeyError."""
    from spacr.plot import plot_permutation
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(5)],
        "importance_mean": np.linspace(0, 1, 5),
        "importance_std": np.full(5, 0.05),
    })
    fig = plot_permutation(df)
    assert fig is not None
    assert len(fig.axes[0].patches) == 5
    plt.close("all")


# ===========================================================================
# spacr.plot: visualize_masks with three synthetic masks
# ===========================================================================

def test_plot_visualize_masks_runs_on_three_masks():
    """Three panels, each showing the mask it was handed."""
    from spacr.plot import visualize_masks
    m = np.zeros((10, 10), dtype=np.int32)
    m[2:8, 2:8] = 1
    plt.close("all")
    visualize_masks(m, m.copy(), m.copy(), title="test")

    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert fig._suptitle.get_text() == "test"
    for ax, panel in zip(fig.axes, ["Mask 1", "Mask 2", "Mask 3"]):
        assert ax.get_title() == panel
        assert len(ax.images) == 1
        # The array handed to imshow is the mask itself, not an empty canvas.
        drawn = np.asarray(ax.images[0].get_array())
        assert drawn.shape == m.shape
        assert np.array_equal(drawn, m)
        assert int((drawn > 0).sum()) == 36      # the 6x6 object really is drawn
        assert not ax.axison
    plt.close("all")


def test_plot_visualize_masks_binary_and_multilabel():
    """Binary and multi-label panels are coloured on their own label counts.

    The contrast is the test: panels 1 and 3 hold a single object, panel 2
    holds three. The panels must therefore differ in the array handed to
    ``imshow``, in the normalisation and in the size of the random colormap —
    a figure that drew the same thing three times fails on every one of them.
    """
    from spacr.plot import visualize_masks
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[2:5, 2:5] = 1
    multi = np.zeros((10, 10), dtype=np.int32)
    multi[2:4, 2:4] = 1
    multi[5:7, 5:7] = 2
    multi[7:9, 7:9] = 3
    plt.close("all")
    visualize_masks(binary, multi, binary.copy(), title="mixed")

    fig = plt.gcf()
    assert len(fig.axes) == 3
    assert fig._suptitle.get_text() == "mixed"
    assert all(len(ax.images) == 1 for ax in fig.axes)
    drawn = [np.asarray(ax.images[0].get_array()) for ax in fig.axes]

    for panel, source in zip(drawn, [binary, multi, binary]):
        assert np.array_equal(panel, source)

    # Panel 2 carries three distinct non-zero labels; panels 1 and 3 carry one.
    assert [len(np.unique(p[p > 0])) for p in drawn] == [1, 3, 1]
    assert sorted(np.unique(drawn[1])) == [0, 1, 2, 3]
    assert sorted(np.unique(drawn[0])) == [0, 1]
    assert not np.array_equal(drawn[0], drawn[1])
    assert np.array_equal(drawn[0], drawn[2])

    # The colormap is sized off the label count, and only the non-binary panel
    # is normalised against its own maximum.
    assert [ax.images[0].cmap.N for ax in fig.axes] == [2, 4, 2]
    assert fig.axes[1].images[0].norm.vmax == 3
    plt.close("all")


# ===========================================================================
# spacr.plot: generate_plate_heatmap with prc parsing
# ===========================================================================

def test_plot_generate_plate_heatmap_derives_metadata_from_prc(rng):
    """generate_plate_heatmap derives plateID/rowID/columnID from the
    3-part `prc` column when they're absent."""
    from spacr.plot import generate_plate_heatmap
    n = 30
    df = pd.DataFrame({
        "prc": [f"p1_r{i%16+1}_c{i%24+1}" for i in range(n)],
        "value": rng.uniform(0, 100, n),
    })
    plate_map, (vmin, vmax) = generate_plate_heatmap(
        df, plate_number="p1", variable="value", grouping="mean",
        min_max="all", min_count=0,
    )
    assert isinstance(plate_map, pd.DataFrame)
    assert vmin <= vmax


def test_plot_generate_plate_heatmap_min_max_allq_uses_quantiles(rng):
    from spacr.plot import generate_plate_heatmap
    n = 60
    df = pd.DataFrame({
        "prc": [f"p1_r{i%16+1}_c{i%24+1}" for i in range(n)],
        "value": np.concatenate([rng.uniform(0, 100, n - 2), [1e6, -1e6]]),  # outliers
    })
    _, (vmin, vmax) = generate_plate_heatmap(
        df, "p1", "value", "mean", "allq", min_count=0,
    )
    # 'allq' uses the 2nd/98th percentiles → the outliers shouldn't set the range.
    assert vmax < 1e6


# ===========================================================================
# spacr.plot: random_cmap first slot invariant
# ===========================================================================

def test_plot_random_cmap_all_slots_valid_rgba():
    from spacr.plot import random_cmap
    cmap = random_cmap(num_objects=15)
    for i in range(cmap.N):
        r, g, b, a = cmap(i)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        assert a == 1.0


# ===========================================================================
# spacr.utils: additional coverage
# ===========================================================================

def test_utils_all_elements_match_subset_returns_true():
    """all_elements_match tests set-subset (every el of list1 in list2).
    Different-length lists where list1 is a subset of list2 → True."""
    from spacr.utils import all_elements_match
    assert all_elements_match([1, 2, 3], [1, 2, 3, 4]) is True


def test_utils_all_elements_match_disjoint_lists_returns_false():
    from spacr.utils import all_elements_match
    assert all_elements_match([1, 2, 3], [4, 5, 6]) is False


def test_utils_is_list_of_lists_nested_two_deep():
    from spacr.utils import is_list_of_lists
    assert is_list_of_lists([[1, 2], [3, 4], [5, 6]]) is True


def test_utils_map_condition_case_sensitive():
    """Different strings that only differ in case should not match."""
    from spacr.utils import map_condition
    # 'C1' should not match neg='c1' (case-sensitive equality).
    got = map_condition("C1", neg="c1", pos="c2", mix="c3")
    assert got != "neg"


def test_utils_convert_cq1_well_id_row_boundary():
    """CQ1 encoding: well IDs at row boundaries (24, 48, 72, ..., 384)."""
    from spacr.utils import _convert_cq1_well_id
    for well_id, row_letter, col in [(24, "A", 24), (48, "B", 24),
                                       (72, "C", 24), (240, "J", 24),
                                       (360, "O", 24), (384, "P", 24)]:
        assert _convert_cq1_well_id(well_id) == f"{row_letter}{col:02d}"


# ===========================================================================
# spacr.io additional
# ===========================================================================

def test_io_get_avg_object_size_average_across_many_masks():
    """Three masks with 1, 3, and 5 objects → avg count = 3."""
    from spacr.io import _get_avg_object_size
    m1 = np.zeros((10, 10), dtype=np.int32); m1[0:2, 0:2] = 1
    m2 = np.zeros((10, 10), dtype=np.int32)
    m2[0:2, 0:2] = 1; m2[4:6, 4:6] = 2; m2[7:9, 7:9] = 3
    m3 = np.zeros((10, 10), dtype=np.int32)
    for i in range(5):
        m3[i, i] = i + 1
    n, avg = _get_avg_object_size([m1, m2, m3])
    assert n == pytest.approx(3.0)   # (1 + 3 + 5) / 3
    assert avg > 0


# ===========================================================================
# spacr.sim: additional coverage
# ===========================================================================

def test_sim_generate_plate_map_columns_populated():
    from spacr.sim import generate_plate_map
    pm = generate_plate_map(nr_plates=1)
    # 16 rows × 24 cols = 384.
    assert len(pm) == 384
    for col in ("plate_id", "row_id", "column_id", "plate_row_column"):
        assert col in pm.columns
    # plate_row_column is the underscore-joined composite.
    row0 = pm.iloc[0]
    assert row0["plate_row_column"] == "1_1_1"


def test_sim_generate_plate_map_multiple_plates_stack():
    from spacr.sim import generate_plate_map
    pm = generate_plate_map(nr_plates=3)
    assert len(pm) == 3 * 384
    assert set(pm["plate_id"].unique()) == {"1", "2", "3"}


# ===========================================================================
# spacr.timelapse edge cases
# ===========================================================================

def test_timelapse_link_by_iou_empty_previous_frame():
    """If the previous frame has no objects, there's nothing to match."""
    from spacr.timelapse import link_by_iou
    prev = np.zeros((10, 10), dtype=np.int32)   # no objects
    curr = np.zeros((10, 10), dtype=np.int32); curr[2:5, 2:5] = 1
    matches = link_by_iou(prev, curr, iou_threshold=0.5)
    assert matches == []


def test_timelapse_link_by_iou_empty_current_frame():
    from spacr.timelapse import link_by_iou
    prev = np.zeros((10, 10), dtype=np.int32); prev[2:5, 2:5] = 1
    curr = np.zeros((10, 10), dtype=np.int32)   # no objects
    matches = link_by_iou(prev, curr, iou_threshold=0.5)
    assert matches == []


def test_timelapse_track_by_iou_single_frame():
    """A single-frame stack should produce one track per object."""
    from spacr.timelapse import _track_by_iou
    masks = np.zeros((1, 20, 20), dtype=np.int32)
    masks[0, 5:8, 5:8] = 1
    masks[0, 12:15, 12:15] = 2
    df = _track_by_iou(masks)
    assert set(df["track_id"]) == {1, 2}
    assert len(df) == 2
