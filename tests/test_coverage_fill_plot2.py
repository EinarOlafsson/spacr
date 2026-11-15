"""Coverage-fill batch 2 for spacr.plot pure/small helpers (Agg).

Colormap and colour helpers, the object filter and the per-condition
histogram figure.  Each helper below sets up its input, runs the function,
keeps the secondary assertions and returns the observed quantity that the
parametrised tests compare with the expected value of the row.
"""
from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n, p in names.items()
                     if p.default is inspect.Parameter.empty})


# ---------------------------------------------------------------------------
# colormap / colour helpers
# ---------------------------------------------------------------------------

def _two_object_mask():
    m = np.zeros((16, 16), dtype=np.int32)
    m[2:6, 2:6] = 1
    m[8:12, 8:12] = 2
    return m


def _case_generate_mask_random_cmap_private():
    cmap = P._generate_mask_random_cmap(_two_object_mask())
    return cmap.N   # background + 2 objects


def _case_random_cmap_has_at_least_n_colours():
    return P.random_cmap(30).N >= 30


def _case_generate_mask_random_cmap_public():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:4, 1:4] = 1
    cmap = P.generate_mask_random_cmap(m)
    return type(cmap).__name__, cmap.N


def _case_get_colours_merged(fmt):
    out = P._get_colours_merged(fmt)
    return out[0], len(out)


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_generate_mask_random_cmap_private, 3,
                 id="snap-private-mask-cmap-has-background-plus-objects"),
    pytest.param(_case_random_cmap_has_at_least_n_colours, True,
                 id="random-cmap-has-at-least-n-colours"),
    pytest.param(_case_generate_mask_random_cmap_public, ("ListedColormap", 2),
                 id="snap-public-mask-cmap-is-listed-colormap"),
    pytest.param(functools.partial(_case_get_colours_merged, fmt="rgb"),
                 ([1, 0, 0], 3), id="colours-merged-rgb"),
    pytest.param(functools.partial(_case_get_colours_merged, fmt="bgr"),
                 ([0, 0, 1], 3), id="colours-merged-bgr"),
    pytest.param(functools.partial(_case_get_colours_merged, fmt="gbr"),
                 ([0, 1, 0], 3), id="colours-merged-gbr"),
    pytest.param(functools.partial(_case_get_colours_merged, fmt="rbg"),
                 ([1, 0, 0], 3), id="colours-merged-rbg"),
    pytest.param(functools.partial(_case_get_colours_merged, fmt="other"),
                 ([1, 0, 0], 3), id="colours-merged-unknown-falls-back-to-rgb"),
])
def test_colormap_and_colour_helpers(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


# ---------------------------------------------------------------------------
# _filter_objects_in_plot / _plot_histograms_and_stats
# ---------------------------------------------------------------------------

def _case_filter_objects_in_plot():
    size = 24
    cell = np.zeros((size, size), dtype=np.int32)
    cell[2:10, 2:10] = 1
    cell[14:22, 14:22] = 2
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[3:6, 3:6] = 1
    nucleus[15:18, 15:18] = 2
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[6:9, 6:9] = 1
    stack = np.stack([cell, nucleus, pathogen], axis=-1)
    out = P._filter_objects_in_plot(
        stack.copy(), cell_mask_dim=0, nucleus_mask_dim=1,
        pathogen_mask_dim=2, mask_dims=[0, 1, 2],
        filter_min_max=[[1, 100000], [1, 100000], [1, 100000]],
        nuclei_limit=True, pathogen_limit=True)
    return out.shape, stack.shape


def _case_plot_histograms_and_stats(capsys):
    """One 30-bin histogram per condition, each holding that condition rows.

    The two conditions deliberately carry *different* row counts, so the bar
    totals cannot both come out right by accident, and a figure that drew
    nothing totals 0 for either of them.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "condition": rng.choice(["ctrl", "trt"], 100),
        "pred": rng.uniform(0, 1, 100),
    })
    plt.close("all")
    P._plot_histograms_and_stats(df)

    figs = [plt.figure(n) for n in plt.get_fignums()]

    by_condition = {}
    for fig in figs:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        by_condition[ax.get_title().rsplit(": ", 1)[1]] = ax

    counts = {}
    bars = {}
    for condition, ax in by_condition.items():
        subset = df[df["condition"] == condition]
        # 30 bins were asked for and 30 bars must be drawn.
        bars[condition] = len(ax.patches)
        # Every row of this condition, and only this condition, is binned.
        total = sum(p.get_height() for p in ax.patches)
        assert total == pytest.approx(len(subset))
        counts[condition] = total
        # The dashed mean line sits exactly on the subset mean.
        assert len(ax.lines) == 1
        assert np.unique(ax.lines[0].get_xdata()) == pytest.approx(
            subset["pred"].mean())
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            f"Mean = {subset['pred'].mean():.2f}"]
        assert ax.get_xlabel() == "Pred Value"
        assert ax.get_ylabel() == "Count"

    # Contrast: the split is uneven, so the two panels must disagree.
    assert counts["ctrl"] != counts["trt"]

    out = capsys.readouterr().out
    for condition in ("ctrl", "trt"):
        subset = df[df["condition"] == condition]
        assert f"Number of rows: {len(subset)}" in out
        assert (f"Count of pred values over 0.5: "
                f"{int((subset['pred'] > 0.5).sum())}") in out
    return (len(figs), sorted(by_condition), bars,
            counts["ctrl"] + counts["trt"], len(df))


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_filter_objects_in_plot, ((24, 24, 3), (24, 24, 3)),
                 id="filter-objects-keeps-stack-shape"),
    pytest.param(_case_plot_histograms_and_stats,
                 (2, ["ctrl", "trt"], {"ctrl": 30, "trt": 30}, 100, 100),
                 id="histograms-one-30-bin-figure-per-condition"),
])
def test_object_filter_and_condition_histograms(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected
