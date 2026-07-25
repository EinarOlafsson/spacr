"""Branch coverage for the tail of :mod:`spacr.plot`.

Covers ``overlay_masks_on_images``, ``graph_importance``,
``plot_proportion_stacked_bars`` (well/plate level), ``create_venn_diagram``
(show branch) and every transform / threshold / annotation branch of
``volcano_plot``.

Everything here is CPU-only, offline and synthetic: TIFF pairs written with
tifffile, small pandas frames, and matplotlib driven through the Agg backend.
Figures are asserted on (offsets, colours, line positions, labels) rather than
merely produced.
"""
from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    yield
    plt.close("all")


@pytest.fixture
def no_show(monkeypatch):
    """Replace plt.show with a recorder so nothing ever blocks."""
    calls = []

    def _fake_show(*args, **kwargs):
        fig = plt.gcf()
        rec = {"title": None, "shape": None}
        if fig.axes:
            ax = fig.axes[0]
            rec["title"] = ax.get_title()
            if ax.images:
                rec["shape"] = np.asarray(ax.images[0].get_array()).shape
        calls.append(rec)

    monkeypatch.setattr(plt, "show", _fake_show)
    return calls


def _write_tif(path, arr):
    import tifffile
    tifffile.imwrite(str(path), arr)


def _overlay_dirs(tmp_path, images, masks):
    """Write {name: array} images + masks into an img_folder/masks layout."""
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    mask_dir = img_dir / "masks"
    mask_dir.mkdir()
    for name, arr in images.items():
        _write_tif(img_dir / name, arr)
    for name, arr in masks.items():
        _write_tif(mask_dir / name, arr)
    return img_dir


def _square_mask(shape, box=(20, 60)):
    m = np.zeros(shape, dtype=np.uint16)
    lo, hi = box
    m[lo:hi, lo:hi] = 1
    return m


def _volcano_df():
    """Deterministic 6-row volcano frame (x already in logFC units)."""
    return pd.DataFrame(
        {
            "name": list("ABCDEF"),
            "fc": [2.0, -3.0, 0.5, 1.5, -0.2, 4.0],
            "p": [1e-5, 1e-9, 1e-8, 0.5, 1e-3, 0.04],
        }
    )


def _positive_df():
    """Fold changes strictly > 0 so the log x-transforms are legal."""
    return pd.DataFrame(
        {
            "name": list("ABCD"),
            "fc": [0.5, 2.0, 4.0, 1.0],
            "p": [1e-4, 1e-6, 1e-2, 0.5],
        }
    )


# ===========================================================================
# overlay_masks_on_images
# ===========================================================================

def test_overlay_masks_on_images_saves_resized_rgb_overlays(tmp_path):
    """save=True + resize=True writes 1000x1000 RGB PNG-ish TIFFs with red outlines."""
    import cv2
    from spacr.plot import overlay_masks_on_images

    rng = np.random.default_rng(0)
    imgs = {
        "f01.tif": (rng.random((128, 128)) * 20000 + 1000).astype(np.uint16),
        "f02.tif": (rng.random((128, 128)) * 20000 + 1000).astype(np.uint16),
    }
    masks = {name: _square_mask((128, 128)) for name in imgs}
    img_dir = _overlay_dirs(tmp_path, imgs, masks)

    assert overlay_masks_on_images(str(img_dir), save=True, plot=False) is None

    out_dir = img_dir / "overlay"
    assert out_dir.is_dir()
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == ["f01.tif", "f02.tif"]

    saved = cv2.imread(str(out_dir / "f01.tif"), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (1000, 1000, 3)          # resize=True
    assert saved.dtype == np.uint8                 # normalize=True -> uint8
    # File is written BGR, so the red contour shows up as channel 2 > channel 0.
    assert np.any(saved[..., 2].astype(int) - saved[..., 0].astype(int) > 50)


def test_overlay_masks_on_images_plot_branch_no_resize_and_mask_rescale(tmp_path, no_show):
    """plot=True, normalize=False, resize=False; mask of a different size is resized."""
    from spacr.plot import overlay_masks_on_images

    img = (np.linspace(0, 60000, 128 * 128).reshape(128, 128)).astype(np.uint16)
    imgs = {"a.tif": img}
    masks = {"a.tif": _square_mask((64, 64), box=(10, 30))}   # half-size mask
    img_dir = _overlay_dirs(tmp_path, imgs, masks)

    overlay_masks_on_images(
        str(img_dir), normalize=False, resize=False, save=False, plot=True, thickness=1
    )

    # One figure shown, at native resolution, with the per-file title.
    assert len(no_show) == 1
    assert no_show[0]["shape"] == (128, 128, 3)
    assert no_show[0]["title"] == "Overlay: a.tif"
    # save=False -> no overlay folder at all.
    assert not (img_dir / "overlay").exists()


def test_overlay_masks_on_images_rgb_input_keeps_three_channels(tmp_path, no_show):
    """An already-RGB image takes the image.copy() branch instead of cvtColor."""
    from spacr.plot import overlay_masks_on_images

    rng = np.random.default_rng(3)
    rgb = (rng.random((80, 80, 3)) * 200).astype(np.uint8)
    imgs = {"rgb.tif": rgb}
    masks = {"rgb.tif": _square_mask((80, 80), box=(15, 55))}
    img_dir = _overlay_dirs(tmp_path, imgs, masks)

    overlay_masks_on_images(str(img_dir), normalize=True, resize=False,
                            save=False, plot=True)

    assert len(no_show) == 1
    assert no_show[0]["shape"] == (80, 80, 3)


def test_overlay_masks_on_images_no_common_filenames(tmp_path, capsys, no_show):
    """Disjoint filenames -> early return, nothing plotted or written."""
    from spacr.plot import overlay_masks_on_images

    img_dir = _overlay_dirs(
        tmp_path,
        {"only_image.tif": np.zeros((32, 32), np.uint16)},
        {"only_mask.tif": np.zeros((32, 32), np.uint16)},
    )

    assert overlay_masks_on_images(str(img_dir), save=True, plot=True) is None
    assert "No matching filenames found" in capsys.readouterr().out
    assert no_show == []
    # The overlay folder is created before the check, but stays empty.
    assert list((img_dir / "overlay").iterdir()) == []


# ===========================================================================
# graph_importance
# ===========================================================================

def _importance_csv(path, seed=0, data_col="compartment_importance_sum"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "compartment": np.repeat(["cell", "nucleus", "pathogen"], 8),
            data_col: rng.uniform(0.0, 1.0, 24),
            "prc": [f"plate1_A0{i % 3 + 1}_1" for i in range(24)],
        }
    )
    df.to_csv(path, index=False)
    return df


def test_graph_importance_concatenates_csvs_and_saves(tmp_path, no_show):
    """Two CSVs are concatenated, defaults filled in, and artefacts saved."""
    from spacr.plot import graph_importance

    p1 = tmp_path / "imp0.csv"
    p2 = tmp_path / "imp1.csv"
    df1 = _importance_csv(p1, seed=1)
    df2 = _importance_csv(p2, seed=2)

    settings = {"csvs": [str(p1), str(p2)], "save": True}
    assert graph_importance(settings) is None

    # defaults were injected in place
    assert settings["src"] == str(tmp_path)
    assert settings["grouping_column"] == "compartment"
    assert settings["data_column"] == "compartment_importance_sum"
    assert settings["graph_type"] == "jitter_bar"

    # settings snapshot written by save_settings
    saved = pd.read_csv(tmp_path / "settings" / "graph_importance.csv")
    assert set(saved["Key"]) >= {"csvs", "grouping_column", "data_column"}

    stem = "compartment_compartment_importance_sum_compartment_jitter_bar"
    assert (tmp_path / f"{stem}.pdf").is_file()
    data_csv = pd.read_csv(tmp_path / f"{stem}_data.csv")
    # concat of both inputs -> 48 rows
    assert len(data_csv) == len(df1) + len(df2) == 48
    assert (tmp_path / f"{stem}_stats.csv").is_file()
    # plt.show() was reached at the end of graph_importance
    assert len(no_show) >= 1


def test_graph_importance_missing_columns_returns_early(tmp_path, capsys, no_show):
    """A CSV without the requested data column prints and bails out before plotting."""
    from spacr.plot import graph_importance

    p = tmp_path / "imp.csv"
    pd.DataFrame({"compartment": ["cell", "nucleus"], "other": [1.0, 2.0]}).to_csv(
        p, index=False
    )

    settings = {"csvs": [str(p)], "save": True}
    assert graph_importance(settings) is None

    out = capsys.readouterr().out
    assert "must be in" in out
    assert "compartment_importance_sum" in out
    # bailed out before spacrGraph ran
    assert not any(f.suffix == ".pdf" for f in tmp_path.iterdir())
    assert no_show == []
    # ...but the settings snapshot was still written
    assert (tmp_path / "settings" / "graph_importance.csv").is_file()


def test_graph_importance_accepts_a_single_path_string(tmp_path, monkeypatch, no_show):
    """A bare string path should behave like a one-element list."""
    from spacr.plot import graph_importance

    monkeypatch.chdir(tmp_path)
    _importance_csv(tmp_path / "imp0.csv", seed=4)

    # Absolute path: os.path.dirname of a bare relative name is '', which
    # breaks the (separate, pre-existing) output_dir handling for the list
    # form too, so it would not isolate the string-vs-list behaviour.
    settings = {"csvs": str(tmp_path / "imp0.csv"), "save": True}
    graph_importance(settings)

    assert settings["csvs"] == [str(tmp_path / "imp0.csv")]
    stem = "compartment_compartment_importance_sum_compartment_jitter_bar"
    assert (tmp_path / f"{stem}.pdf").is_file()


# ===========================================================================
# plot_proportion_stacked_bars — well / plateID aggregation branch
# ===========================================================================

def _proportion_df():
    rows = []
    for group, n_low in (("ctrl", 10), ("trt", 3)):
        for well in ("plate1_A01", "plate1_A02", "plate1_A03"):
            for _ in range(n_low):
                rows.append({"group": group, "prc": well, "bin": "low"})
            for _ in range(13 - n_low):
                rows.append({"group": group, "prc": well, "bin": "high"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("level", ["well", "plateID"])
def test_plot_proportion_stacked_bars_well_level(level):
    """Well/plate level aggregates per-well proportions and adds SD error bars."""
    from spacr.plot import plot_proportion_stacked_bars

    df = _proportion_df()
    results_df, pairwise, fig = plot_proportion_stacked_bars(
        {"verbose": False}, df, group_column="group", bin_column="bin",
        prc_column="prc", level=level,
    )

    assert list(results_df.columns) == [
        "chi_squared_stat", "p_value", "degrees_of_freedom"
    ]
    assert results_df.loc[0, "degrees_of_freedom"] == 1
    assert 0.0 < results_df.loc[0, "p_value"] < 0.05      # groups really differ
    assert results_df.loc[0, "chi_squared_stat"] > 0

    assert isinstance(pairwise, pd.DataFrame)
    assert len(pairwise) == 1                              # ctrl vs trt

    ax = fig.axes[0]
    assert ax.get_title().endswith("(Mean ± SD across wells)")
    assert ax.get_xlabel() == "Group"
    assert ax.get_ylabel() == "Proportion"
    assert ax.get_ylim() == (0.0, 1.0)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ctrl", "trt"]
    # stacked bars: 2 groups x 2 bins
    bars = [p for c in ax.containers for p in getattr(c, "patches", [])]
    assert len(bars) == 4
    # every stack sums to 1
    heights = np.array([p.get_height() for p in bars])
    assert np.isclose(heights.sum(), 2.0)
    # error bars were drawn (yerr=std)
    assert any(getattr(c, "errorbar", None) is not None for c in ax.containers)


def test_plot_proportion_stacked_bars_object_level_matches_raw_proportions():
    """The default object level plots raw count proportions (no aggregation)."""
    from spacr.plot import plot_proportion_stacked_bars

    df = _proportion_df()
    results_df, pairwise, fig = plot_proportion_stacked_bars(
        {"verbose": True}, df, group_column="group", bin_column="bin",
        prc_column="prc", level="object",
    )
    ax = fig.axes[0]
    assert ax.get_title() == "Proportion of Volume Bins by Group"
    bars = [p for c in ax.containers for p in getattr(c, "patches", [])]
    heights = sorted(round(p.get_height(), 6) for p in bars)
    # ctrl: 30 low / 9 high ; trt: 9 low / 30 high
    assert heights == sorted(
        [round(30 / 39, 6), round(9 / 39, 6), round(9 / 39, 6), round(30 / 39, 6)]
    )
    assert results_df.loc[0, "p_value"] < 0.05
    assert len(pairwise) == 1


# ===========================================================================
# create_venn_diagram — the show (save=False) branch
# ===========================================================================

def test_create_venn_diagram_show_branch_returns_overlap(tmp_path, no_show):
    """save=False shows the figure and still returns the three gene groups."""
    from spacr.plot import create_venn_diagram

    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    pd.DataFrame({"gene": ["g1", "g2", "g3"], "coefficient": [0.5, 0.6, 0.7]}).to_csv(
        f1, index=False
    )
    pd.DataFrame({"gene": ["g2", "g3", "g4"], "coefficient": [0.5, 0.6, 0.7]}).to_csv(
        f2, index=False
    )

    res = create_venn_diagram(str(f1), str(f2), filter_coeff=0.1, save=False)

    assert sorted(res["overlap"]) == ["g2", "g3"]
    assert res["unique_to_file1"] == ["g1"]
    assert res["unique_to_file2"] == ["g4"]
    assert len(no_show) == 1
    assert no_show[0]["title"] == "Venn Diagram of Overlapping Genes"


def test_create_venn_diagram_negative_filter_coeff_keeps_downregulated(tmp_path, no_show):
    """A negative filter_coeff selects rows below the threshold instead of above."""
    from spacr.plot import create_venn_diagram

    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    pd.DataFrame({"gene": ["down1", "up1"], "coefficient": [-0.9, 0.9]}).to_csv(
        f1, index=False
    )
    pd.DataFrame({"gene": ["down1", "down2"], "coefficient": [-0.8, -0.7]}).to_csv(
        f2, index=False
    )

    res = create_venn_diagram(str(f1), str(f2), filter_coeff=-0.1, save=False)
    assert res["overlap"] == ["down1"]
    assert res["unique_to_file1"] == []          # up1 was filtered out
    assert res["unique_to_file2"] == ["down2"]


# ===========================================================================
# volcano_plot — I/O helpers
# ===========================================================================

@pytest.mark.parametrize(
    "suffix,writer",
    [
        (".csv", lambda df, p: df.to_csv(p, index=False)),
        (".tsv", lambda df, p: df.to_csv(p, sep="\t", index=False)),
        (".tab", lambda df, p: df.to_csv(p, sep="\t", index=False)),
        (".txt", lambda df, p: df.to_csv(p, index=False)),          # sniffed comma
        (".dat", lambda df, p: df.to_csv(p, sep="\t", index=False)),  # sniffed tab
    ],
)
def test_volcano_plot_reads_every_delimited_format(tmp_path, suffix, writer):
    """Extension dispatch + delimiter sniffing all reach the same plotted points."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    path = tmp_path / f"table{suffix}"
    writer(df, path)

    fig, ax, hits = volcano_plot(
        str(path), fold_change_col="fc", p_value_col="p", show=False
    )
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape == (6, 2)
    assert np.allclose(offsets[:, 0], df["fc"].to_numpy())
    assert np.allclose(offsets[:, 1], -np.log10(df["p"].to_numpy()))
    assert hits == []


def test_volcano_plot_reads_excel_sheet_by_name(tmp_path):
    """.xlsx dispatch honours the sheet_name argument."""
    pytest.importorskip("openpyxl")
    from spacr.plot import volcano_plot

    df = _volcano_df()
    other = pd.DataFrame({"fc": [0.0], "p": [1.0]})
    path = tmp_path / "table.xlsx"
    with pd.ExcelWriter(path) as xl:
        other.to_excel(xl, sheet_name="first", index=False)
        df.to_excel(xl, sheet_name="second", index=False)

    fig, ax, hits = volcano_plot(
        str(path), fold_change_col="fc", p_value_col="p",
        sheet_name="second", show=False,
    )
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape == (6, 2)
    assert np.allclose(sorted(offsets[:, 0]), sorted(df["fc"].to_numpy()))


def test_volcano_plot_excel_without_engine_raises_helpful_importerror(tmp_path, monkeypatch):
    """A missing Excel engine is re-raised with install instructions."""
    from spacr.plot import volcano_plot

    path = tmp_path / "table.xlsx"
    path.write_bytes(b"not really excel")

    def _boom(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd, "read_excel", _boom)

    with pytest.raises(ImportError) as exc:
        volcano_plot(str(path), fold_change_col="fc", p_value_col="p", show=False)
    assert "openpyxl" in str(exc.value)
    assert "xlrd" in str(exc.value)
    assert isinstance(exc.value.__cause__, ImportError)


def test_volcano_plot_non_numeric_column_raises(tmp_path):
    """A wholly non-numeric column cannot be coerced and is rejected by name."""
    from spacr.plot import volcano_plot

    df = pd.DataFrame({"fc": ["a", "b", "c"], "p": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError) as exc:
        volcano_plot(df, fold_change_col="fc", p_value_col="p", show=False)
    assert "Column 'fc' could not be converted to numeric." in str(exc.value)


@pytest.mark.parametrize("missing", ["fold_change_col", "p_value_col", "name_col"])
def test_volcano_plot_missing_columns_raise_keyerror(missing):
    """Each of the three column arguments is validated separately."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    kwargs = dict(fold_change_col="fc", p_value_col="p", name_col="name", show=False)
    kwargs[missing] = "nope"
    with pytest.raises(KeyError) as exc:
        volcano_plot(df, **kwargs)
    assert missing in str(exc.value)
    assert "nope" in str(exc.value)


def test_volcano_plot_drops_nan_rows():
    """Rows with NaN in either numeric column are dropped before plotting."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    df.loc[0, "fc"] = np.nan
    df.loc[1, "p"] = np.nan

    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", show=False
    )
    assert ax.collections[0].get_offsets().shape == (4, 2)


# ===========================================================================
# volcano_plot — transforms
# ===========================================================================

@pytest.mark.parametrize(
    "mode,fn",
    [("log2", np.log2), ("log10", np.log10), ("ln", np.log), ("log", np.log)],
)
def test_volcano_plot_x_transforms(mode, fn):
    """Each log x-transform is applied to the plotted x values and the label."""
    from spacr.plot import volcano_plot

    df = _positive_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", x_transform=mode, show=False
    )
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], fn(df["fc"].to_numpy()))
    assert ax.get_xlabel() == f"{mode}(fc)"


def test_volcano_plot_x_transform_rejects_non_positive():
    """A log x-transform on negative fold changes points at x_transform='none'."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="log2", show=False)
    assert "requires all fold changes > 0" in str(exc.value)
    assert "x_transform='none'" in str(exc.value)


def test_volcano_plot_unknown_x_transform_raises():
    """An unrecognised (but positive-safe) x_transform is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_positive_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="sqrt", show=False)
    assert "Unknown x_transform: sqrt" in str(exc.value)


@pytest.mark.parametrize(
    "mode,fn",
    [
        ("none", lambda p: p),
        ("-log10", lambda p: -np.log10(p)),
        ("-ln", lambda p: -np.log(p)),
        ("log10", np.log10),
        ("ln", np.log),
        ("log", np.log),
    ],
)
def test_volcano_plot_y_transforms(mode, fn):
    """Each y-transform maps p-values onto the plotted y axis and label."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", y_transform=mode, show=False
    )
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 1], fn(df["p"].to_numpy()))
    assert ax.get_ylabel() == ("p" if mode == "none" else f"{mode}(p)")


def test_volcano_plot_y_transform_clips_zero_p_values():
    """p=0 is clipped to the smallest positive float instead of producing inf."""
    from spacr.plot import volcano_plot

    df = pd.DataFrame({"fc": [1.0, -1.0], "p": [0.0, 0.5]})
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", show=False
    )
    y = ax.collections[0].get_offsets()[:, 1]
    assert np.isfinite(y).all()
    assert y[0] == pytest.approx(-np.log10(np.finfo(float).tiny))


def test_volcano_plot_unknown_y_transform_raises():
    """An unrecognised y_transform is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     y_transform="sqrt", show=False)
    assert "Unknown y_transform: sqrt" in str(exc.value)


# ===========================================================================
# volcano_plot — thresholds
# ===========================================================================

def test_volcano_plot_thresholds_none_transform_draws_lines_and_colors():
    """x_transform='none': the FC threshold is used verbatim (as |t|)."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p",
        fold_change_threshold=-1.0,          # abs() is taken
        p_value_threshold=0.01, show=False,
    )
    # 2 vertical FC lines + 1 horizontal p line + the cosmetic x=0 line
    assert len(ax.lines) == 4
    assert list(ax.lines[0].get_xdata()) == [-1.0, -1.0]
    assert list(ax.lines[1].get_xdata()) == [1.0, 1.0]
    assert ax.lines[2].get_ydata()[0] == pytest.approx(2.0)   # -log10(0.01)
    assert list(ax.lines[3].get_xdata()) == [0, 0]

    # A (fc=+2) is crimson, B (fc=-3) royalblue, the rest lightgray.
    fc_colors = ax.collections[0].get_facecolors()[:, :3]
    assert np.allclose(fc_colors[0], to_rgb("crimson"))
    assert np.allclose(fc_colors[1], to_rgb("royalblue"))
    assert np.allclose(fc_colors[2], to_rgb("lightgray"))


@pytest.mark.parametrize(
    "mode,fn", [("log2", np.log2), ("log10", np.log10), ("ln", np.log)]
)
def test_volcano_plot_fc_threshold_converted_to_plot_units(mode, fn):
    """With a log x-transform the FC threshold is converted to plot units."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _positive_df(), fold_change_col="fc", p_value_col="p",
        x_transform=mode, fold_change_threshold=2.0, show=False,
    )
    expected = abs(fn(2.0))
    assert ax.lines[0].get_xdata()[0] == pytest.approx(-expected)
    assert ax.lines[1].get_xdata()[0] == pytest.approx(expected)


def test_volcano_plot_fc_threshold_must_be_positive_for_log_transform():
    """A non-positive FC threshold makes no sense in log space."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_positive_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="log2", fold_change_threshold=0.0, show=False)
    assert "fold_change_threshold must be > 0" in str(exc.value)


def test_volcano_plot_p_threshold_must_be_positive():
    """p_value_threshold <= 0 is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     p_value_threshold=0.0, show=False)
    assert "p_value_threshold must be > 0" in str(exc.value)


def test_volcano_plot_p_threshold_untransformed_axis():
    """y_transform='none' compares raw p-values against the raw threshold."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", name_col="name",
        y_transform="none", p_value_threshold=0.05, show=False,
    )
    # threshold line sits at the raw p value
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.05)
    # hits = every row with p <= 0.05 (A, B, C, F) -- E has p=1e-3 too
    assert hits == ["A", "B", "C", "E", "F"]


def test_volcano_plot_p_threshold_positive_log_axis():
    """A non-negated log y-transform flips the significance comparison."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", name_col="name",
        y_transform="log10", p_value_threshold=1e-4, show=False,
    )
    assert ax.lines[0].get_ydata()[0] == pytest.approx(np.log10(1e-4))
    # log10(p) <= log10(1e-4) -> p <= 1e-4 -> A, B, C
    assert hits == ["A", "B", "C"]


# ===========================================================================
# volcano_plot — figure plumbing
# ===========================================================================

def test_volcano_plot_uses_supplied_axes_and_style_kwargs(tmp_path):
    """ax=..., scatter/threshold/text kwargs, title, xlim/ylim and save_path."""
    from spacr.plot import volcano_plot

    fig0, ax0 = plt.subplots(figsize=(4, 3))
    out = tmp_path / "volcano.pdf"

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        fold_change_threshold=1.0, p_value_threshold=0.01,
        title="my volcano", xlim=(-5.0, 5.0), ylim=(0.0, 12.0),
        point_size=11.0, alpha=0.5,
        scatter_kwargs={"marker": "s", "s": 55.0},
        threshold_line_kwargs={"color": "green", "linestyle": ":"},
        text_kwargs={"fontsize": 14},
        save_path=str(out), show=False, ax=ax0,
    )

    assert ax is ax0
    assert fig is fig0                      # fig = ax.figure branch
    assert ax.get_title() == "my volcano"
    assert ax.get_xlim() == (-5.0, 5.0)
    assert ax.get_ylim() == (0.0, 12.0)
    assert ax.collections[0].get_sizes()[0] == pytest.approx(55.0)
    assert ax.collections[0].get_alpha() == pytest.approx(0.5)
    # threshold lines picked up the override, the cosmetic x=0 line did not
    assert [tuple(np.round(l.get_color() if isinstance(l.get_color(), tuple)
                           else to_rgb(l.get_color()), 3)) for l in ax.lines[:3]] == \
        [tuple(np.round(to_rgb("green"), 3))] * 3
    assert ax.lines[0].get_linestyle() == ":"
    assert to_rgb(ax.lines[3].get_color()) == to_rgb("black")
    # text kwargs reached the annotations
    assert [t.get_text() for t in ax.texts] == hits == ["A", "B"]
    assert all(t.get_fontsize() == 14 for t in ax.texts)
    # spines hidden
    assert not ax.spines["right"].get_visible()
    assert not ax.spines["top"].get_visible()
    assert out.is_file() and out.stat().st_size > 0


def test_volcano_plot_show_true_calls_plt_show(no_show):
    """show=True routes through plt.show exactly once."""
    from spacr.plot import volcano_plot

    volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p", show=True)
    assert len(no_show) == 1


def test_volcano_plot_without_thresholds_is_all_gray():
    """No thresholds -> a single gray colour and no threshold lines."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", show=False
    )
    colors = ax.collections[0].get_facecolors()
    assert colors.shape == (1, 4)                       # scalar colour spec
    assert np.allclose(colors[0, :3], to_rgb("lightgray"))
    assert len(ax.lines) == 1                            # only the x=0 line


# ===========================================================================
# volcano_plot — annotation
# ===========================================================================

def test_volcano_plot_annotate_needs_thresholds_or_annotate_max():
    """With no thresholds and no cap, nothing is annotated."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        show=False,
    )
    assert hits == []
    assert len(ax.texts) == 0


def test_volcano_plot_annotate_max_picks_highest_y():
    """annotate_max keeps only the top-y points, in descending y order."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        annotate_max=2, show=False,
    )
    # B (p=1e-9) then C (p=1e-8)
    assert hits == ["B", "C"]
    assert sorted(t.get_text() for t in ax.texts) == ["B", "C"]


def test_volcano_plot_annotate_disabled():
    """annotate=False leaves the axes text-free even with thresholds set."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        fold_change_threshold=1.0, p_value_threshold=0.01,
        annotate=False, show=False,
    )
    assert hits == []
    assert len(ax.texts) == 0


def test_volcano_plot_annotation_requires_adjusttext(monkeypatch):
    """Missing adjustText raises a helpful ImportError from the annotation block."""
    from spacr.plot import volcano_plot

    real_import = builtins.__import__

    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "adjustText" or name.startswith("adjustText."):
            raise ImportError("No module named 'adjustText'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(ImportError) as exc:
        volcano_plot(
            _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
            fold_change_threshold=1.0, p_value_threshold=0.01, show=False,
        )
    assert "adjustText" in str(exc.value)
    assert "pip install adjustText" in str(exc.value)
    assert isinstance(exc.value.__cause__, ImportError)
