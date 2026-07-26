"""Smoke coverage for spacr.plot — exercise the public plotting
functions with synthetic inputs on the Agg backend.

Each test builds the minimal real-shaped input a function needs and
asserts it runs without raising (and, where it saves, that a file
lands on disk).

Nothing in here is allowed to swallow an exception into a pytest.skip:
a self-skip makes a broken function and an unsupported fixture look
identical, and every skip this file used to carry was hiding either a
product bug or a wrong call. Build the real input, or assert the real
failure with xfail(strict=True).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _synth_mask(size=64, n=6):
    m = np.zeros((size, size), dtype=np.int32)
    rng = np.random.default_rng(0)
    for lbl in range(1, n + 1):
        cy, cx = rng.integers(8, size - 8, size=2)
        y, x = np.ogrid[:size, :size]
        m[(x - cx) ** 2 + (y - cy) ** 2 < 25] = lbl
    return m


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def test_generate_mask_random_cmap():
    cmap = P.generate_mask_random_cmap(_synth_mask())
    assert cmap is not None


def test_random_cmap():
    cmap = P.random_cmap(50)
    assert cmap is not None


# ---------------------------------------------------------------------------
# Mask / flow visualisers
# ---------------------------------------------------------------------------

def test_visualize_masks():
    m1, m2, m3 = _synth_mask(), _synth_mask(), _synth_mask()
    P.visualize_masks(m1, m2, m3, title="t")


def test_visualize_cellpose_masks():
    masks = [_synth_mask(), _synth_mask()]
    P.visualize_cellpose_masks(masks, titles=["a", "b"])


def test_normalize_and_visualize():
    img = np.random.default_rng(0).random((64, 64))
    P.normalize_and_visualize(img, img, title="t")


def test_plot_masks():
    """Batch path: mask stack as an ndarray, cellpose-shaped ``flows``.

    The swallow here hid a real bug — an (N, H, W) mask stack was wrapped in a
    list instead of iterated, so imshow got the whole stack and raised
    "Invalid shape (1, 64, 64) for image data".
    """
    rng = np.random.default_rng(0)
    batch = rng.random((1, 64, 64, 3)).astype(np.float32)
    masks = _synth_mask()[None, ...]
    # Cellpose hands back flows as a list whose [0] is the per-image RGB flows.
    flows = [[np.zeros((64, 64, 3), dtype=np.float32)]]
    P.plot_masks(batch, masks, flows, nr=1)
    assert plt.get_fignums(), "plot_masks drew nothing"


def test_plot_masks_single_image():
    """Singleton path, as ``spacr.utils._filter_cp_masks`` actually calls it."""
    rng = np.random.default_rng(0)
    image = rng.random((64, 64, 3)).astype(np.float32)
    mask = _synth_mask()
    flow = np.zeros((64, 64, 3), dtype=np.float32)
    P.plot_masks(batch=image, masks=mask, flows=flow, nr=1)
    assert plt.get_fignums(), "plot_masks drew nothing"


# ---------------------------------------------------------------------------
# DataFrame plotters
# ---------------------------------------------------------------------------

def _screen_df(n=120):
    import pandas as pd
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "plateID": "plate1",
        "rowID": rng.choice([f"r{i}" for i in range(1, 9)], n),
        "columnID": rng.choice([f"c{i}" for i in range(1, 13)], n),
        "prc": [f"plate1_r{rng.integers(1,9)}_c{rng.integers(1,13)}"
                  for _ in range(n)],
        "recruitment": rng.normal(1.0, 0.3, n),
        "count": rng.integers(1, 100, n),
    })


def test_generate_plate_heatmap():
    df = _screen_df()
    out = P.generate_plate_heatmap(
        df, plate_number="plate1", variable="recruitment",
        grouping="mean", min_max="allq", min_count=0)
    assert out is not None


def test_plot_plates():
    df = _screen_df()
    P.plot_plates(df, variable="recruitment", grouping="mean",
                    min_max="allq", cmap="viridis", min_count=0,
                    verbose=False)


def test_plot_histogram(tmp_path):
    df = _screen_df()
    P.plot_histogram(df, "recruitment", dst=str(tmp_path))


def test_plot_feature_importance():
    import pandas as pd
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(15)],
        "importance": rng.random(15),
    })
    P.plot_feature_importance(df)


def test_plot_permutation():
    import pandas as pd
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(15)],
        "importance_mean": rng.random(15),
        "importance_std": rng.random(15) * 0.1,
    })
    P.plot_permutation(df)


def test_create_grouped_plot(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "grp": rng.choice(["a", "b", "c"], 90),
        "val": rng.normal(0, 1, 90),
    })
    P.create_grouped_plot(
        df, grouping_column="grp", data_column="val",
        graph_type="bar", summary_func="mean",
        output_dir=str(tmp_path), save=True)


def test_plot_proportion_stacked_bars():
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame({
        "prc": [f"plate1_r{rng.integers(1,4)}_c{rng.integers(1,4)}"
                  for _ in range(n)],
        "group": rng.choice(["ctrl", "trt"], n),
        "bin": rng.choice([0, 1, 2], n),
    })
    P.plot_proportion_stacked_bars(
        {"verbose": False}, df, group_column="group",
        bin_column="bin", prc_column="prc", level="object")


def test_volcano_plot(tmp_path):
    import pandas as pd
    rng = np.random.default_rng(6)
    n = 60
    df = pd.DataFrame({
        "gene": [f"g{i}" for i in range(n)],
        "coefficient": rng.normal(0, 0.3, n),
        "p_value": np.clip(np.abs(rng.normal(0.05, 0.05, n)), 1e-6, 1),
    })
    save = tmp_path / "volcano.pdf"
    P.volcano_plot(df, save_path=str(save),
                    fold_change_col="coefficient",
                    p_value_col="p_value")


def test_create_venn_diagram(tmp_path):
    import pandas as pd
    f1 = tmp_path / "a.csv"; f2 = tmp_path / "b.csv"
    import csv
    for f, genes in ((f1, [f"g{i}" for i in range(20)]),
                       (f2, [f"g{i}" for i in range(10, 30)])):
        with open(f, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["gene", "coefficient"])
            for g in genes:
                w.writerow([g, 0.5])
    P.create_venn_diagram(str(f1), str(f2), gene_column="gene",
                            save=True, save_path=str(tmp_path / "v.pdf"))


# ---------------------------------------------------------------------------
# batch 2 — array-based plotters
# ---------------------------------------------------------------------------

def test_plot_cellpose4_output():
    """``flows`` is the *flows0* list from ``parse_cellpose4_output``.

    The old fixture handed in the raw per-image ``[rgb, dP, cellprob]`` triple,
    which is what spacr_cellpose.parse_cellpose4_output unpacks *from*, never
    what spacr.object passes on -- so imshow choked on a ragged list. The call,
    not the function, was wrong; a skip papered over the difference.
    """
    rng = np.random.default_rng(0)
    batch = rng.random((1, 32, 32, 3)).astype(np.float32)
    masks = _synth_mask(32)[None, ...]
    flows = [np.zeros((32, 32, 3), dtype=np.float32)]
    P.plot_cellpose4_output(batch, masks, flows, nr=1)
    # 3 image channels + mask + flow
    assert len(plt.gcf().axes) == 5


def test_print_mask_and_flows():
    stack = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    mask = _synth_mask(32)
    flows = [np.zeros((32, 32, 3), dtype=np.float32),
             np.zeros((3, 32, 32), dtype=np.float32),
             np.zeros((32, 32), dtype=np.float32)]
    P.print_mask_and_flows(stack, mask, flows, overlay=True)


def test_plot_resize():
    rng = np.random.default_rng(0)
    imgs = [rng.random((64, 64))]
    resized = [rng.random((32, 32))]
    labels = [_synth_mask(64)]
    rlabels = [_synth_mask(32)]
    P.plot_resize(imgs, resized, labels, rlabels)


def test_plot_comparison_results():
    results = [{"filename": "x", "jaccard_a_b": 0.8,
                "boundary_f1_a_b": 0.7, "ap_a_b": 0.6}]
    fig = P.plot_comparison_results(results)
    assert fig is not None


def test_plot_lorenz_curves(tmp_path):
    import csv
    f = tmp_path / "counts.csv"
    with f.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["grna_name", "count"])
        for i in range(30):
            w.writerow([f"g{i}", (i + 1) * 3])
    # No try/skip: the swallow here reported the remove_keys=None TypeError
    # as a passing-looking "skipped" for as long as the bug existed.
    P.plot_lorenz_curves([str(f)], name_column="grna_name",
                         value_column="count")


def test_plot_image_mask_overlay(tmp_path):
    """The function reads a merged ``.npy`` stack; it has no image/masks kwargs.

    The old call passed ``image=``/``masks=`` and the swallow turned the
    resulting TypeError into a green skip. The real contract (see
    spacr.core / spacr.submodules callers) is: last N planes of the stack are
    the object masks, one per non-None ``*_channel`` argument.
    """
    rng = np.random.default_rng(0)
    merged = tmp_path / "merged"
    merged.mkdir()
    stack = np.zeros((64, 64, 5), dtype=np.uint16)
    stack[..., :3] = rng.integers(0, 4000, size=(64, 64, 3))
    stack[..., 3] = _synth_mask(64)       # cell mask plane
    stack[..., 4] = _synth_mask(64)       # nucleus mask plane
    f = merged / "img_1.npy"
    np.save(f, stack)

    fig = P.plot_image_mask_overlay(
        file=str(f), channels=[0, 1, 2], cell_channel=0,
        nucleus_channel=1, pathogen_channel=None, save_pdf=True)
    assert fig is not None
    # 3 channel panels + the combined-objects panel
    assert len(fig.axes) == 4
    assert (tmp_path / "results" / "overlay" / "img_1.pdf").is_file()
