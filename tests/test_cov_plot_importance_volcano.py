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
import functools
import inspect
import os

import numpy as np
import pandas as pd
import pytest

from spacr.figures.style import ROLES

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


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n, p in names.items()
                     if p.default is inspect.Parameter.empty})


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

def _case_overlay_masks_on_images_saves_resized_rgb_overlays(tmp_path):
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

    # Returns a report rather than None: the per-image loop now survives a
    # field it cannot read, so the caller has to be able to tell which
    # ones are missing. Nothing is missing here.
    report = overlay_masks_on_images(str(img_dir), save=True, plot=False)

    out_dir = img_dir / "overlay"
    assert out_dir.is_dir()
    written = sorted(p.name for p in out_dir.iterdir())
    assert written == ["f01.tif", "f02.tif"]

    saved = cv2.imread(str(out_dir / "f01.tif"), cv2.IMREAD_UNCHANGED)
    assert saved.shape == (1000, 1000, 3)          # resize=True
    assert saved.dtype == np.uint8                 # normalize=True -> uint8
    # File is written BGR, so the red contour shows up as channel 2 > channel 0.
    assert np.any(saved[..., 2].astype(int) - saved[..., 0].astype(int) > 50)
    return report


def _case_overlay_masks_on_images_plot_branch_no_resize_and_mask_rescale(tmp_path, no_show):
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
    # save=False -> no overlay folder at all.
    assert not (img_dir / "overlay").exists()
    return len(no_show), no_show[0]["shape"], no_show[0]["title"]


def _case_overlay_masks_on_images_rgb_input_keeps_three_channels(tmp_path, no_show):
    """An already-RGB image takes the image.copy() branch instead of cvtColor."""
    from spacr.plot import overlay_masks_on_images

    rng = np.random.default_rng(3)
    rgb = (rng.random((80, 80, 3)) * 200).astype(np.uint8)
    imgs = {"rgb.tif": rgb}
    masks = {"rgb.tif": _square_mask((80, 80), box=(15, 55))}
    img_dir = _overlay_dirs(tmp_path, imgs, masks)

    overlay_masks_on_images(str(img_dir), normalize=True, resize=False,
                            save=False, plot=True)

    return len(no_show), no_show[0]["shape"]


def _case_overlay_masks_on_images_no_common_filenames(tmp_path, capsys, no_show):
    """Disjoint filenames -> early return, nothing plotted or written."""
    from spacr.plot import overlay_masks_on_images

    img_dir = _overlay_dirs(
        tmp_path,
        {"only_image.tif": np.zeros((32, 32), np.uint16)},
        {"only_mask.tif": np.zeros((32, 32), np.uint16)},
    )

    result = overlay_masks_on_images(str(img_dir), save=True, plot=True)
    assert "No matching filenames found" in capsys.readouterr().out
    # The overlay folder is created before the check, but stays empty.
    assert list((img_dir / "overlay").iterdir()) == []
    return result, list(no_show)


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


def _case_graph_importance_concatenates_csvs_and_saves(tmp_path, no_show):
    """Two CSVs are concatenated, defaults filled in, and artefacts saved."""
    from spacr.plot import graph_importance

    p1 = tmp_path / "imp0.csv"
    p2 = tmp_path / "imp1.csv"
    df1 = _importance_csv(p1, seed=1)
    df2 = _importance_csv(p2, seed=2)

    settings = {"csvs": [str(p1), str(p2)], "save": True}
    result = graph_importance(settings)

    # defaults were injected in place
    assert settings["src"] == str(tmp_path)
    assert settings["grouping_column"] == "compartment"
    assert settings["data_column"] == "compartment_importance_sum"
    assert settings["graph_type"] == "jitter_box"

    # settings snapshot written by save_settings
    saved = pd.read_csv(tmp_path / "settings" / "graph_importance.csv")
    assert set(saved["Key"]) >= {"csvs", "grouping_column", "data_column"}

    stem = "compartment_compartment_importance_sum_compartment_jitter_box"
    assert (tmp_path / f"{stem}.pdf").is_file()
    data_csv = pd.read_csv(tmp_path / f"{stem}_data.csv")
    # concat of both inputs -> 48 rows
    assert len(data_csv) == len(df1) + len(df2)
    assert (tmp_path / f"{stem}_stats.csv").is_file()
    # plt.show() was reached at the end of graph_importance
    assert len(no_show) >= 1
    return result, len(data_csv)


def _case_graph_importance_missing_columns_returns_early(tmp_path, capsys, no_show):
    """A CSV without the requested data column prints and bails out before plotting."""
    from spacr.plot import graph_importance

    p = tmp_path / "imp.csv"
    pd.DataFrame({"compartment": ["cell", "nucleus"], "other": [1.0, 2.0]}).to_csv(
        p, index=False
    )

    settings = {"csvs": [str(p)], "save": True}
    result = graph_importance(settings)

    out = capsys.readouterr().out
    assert "must be in" in out
    assert "compartment_importance_sum" in out
    # bailed out before spacrGraph ran
    assert not any(f.suffix == ".pdf" for f in tmp_path.iterdir())
    # ...but the settings snapshot was still written
    assert (tmp_path / "settings" / "graph_importance.csv").is_file()
    return result, list(no_show)


def _case_graph_importance_accepts_a_single_path_string(tmp_path, monkeypatch, no_show):
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
    stem = "compartment_compartment_importance_sum_compartment_jitter_box"
    assert (tmp_path / f"{stem}.pdf").is_file()
    return [os.path.basename(p) for p in settings["csvs"]]


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


def _case_plot_proportion_stacked_bars_well_level():
    """Well level aggregates per-well proportions and adds SD error bars.

    "plateID" used to be parametrized here too, and it passed because the
    function grouped BY `prc` -- the well column -- whatever the level said.
    That is the defect: a plate-level request averaged wells and labelled the
    result plates. Plate level now groups by `plateID`, so it needs a plate
    column, and this fixture has none. See test_proportion_bars_level.py.
    """
    from spacr.plot import plot_proportion_stacked_bars

    df = _proportion_df()
    results_df, pairwise, fig = plot_proportion_stacked_bars(
        {"verbose": False}, df, group_column="group", bin_column="bin",
        prc_column="prc", level="well",
    )

    # The historical three columns are still the FIRST row and still mean
    # what they meant -- every published figure came from that number.
    # Instruction 80 added the columns that say WHICH TEST and WHICH UNIT,
    # and rows for the level-appropriate test and the mixed model beside it,
    # because a chi-squared over objects and a t-test over wells answer
    # different questions and were previously reported as one number.
    for column in ("chi_squared_stat", "p_value", "degrees_of_freedom",
                   "test", "unit", "n"):
        assert column in results_df.columns, column
    assert 0.0 < results_df.loc[0, "p_value"] < 0.05      # groups really differ
    assert results_df.loc[0, "chi_squared_stat"] > 0

    assert isinstance(pairwise, pd.DataFrame)
    ax = fig.axes[0]
    # stacked bars: 2 groups x 2 bins
    bars = [p for c in ax.containers for p in getattr(c, "patches", [])]
    # every stack sums to 1
    heights = np.array([p.get_height() for p in bars])
    # error bars were drawn (yerr=std)
    assert any(getattr(c, "errorbar", None) is not None for c in ax.containers)
    return {
        "unit": results_df.loc[0, "unit"],
        "degrees_of_freedom": results_df.loc[0, "degrees_of_freedom"],
        "pairwise_rows": len(pairwise),                    # ctrl vs trt
        "title_suffix": ax.get_title().rsplit("(", 1)[-1],
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "ylim": ax.get_ylim(),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "n_bars": len(bars),
        "height_total": float(heights.sum()),
    }


def _case_plot_proportion_stacked_bars_object_level_matches_raw_proportions():
    """The default object level plots raw count proportions (no aggregation)."""
    from spacr.plot import plot_proportion_stacked_bars

    df = _proportion_df()
    results_df, pairwise, fig = plot_proportion_stacked_bars(
        {"verbose": True}, df, group_column="group", bin_column="bin",
        prc_column="prc", level="object",
    )
    ax = fig.axes[0]
    bars = [p for c in ax.containers for p in getattr(c, "patches", [])]
    heights = sorted(round(p.get_height(), 6) for p in bars)
    # ctrl: 30 low / 9 high ; trt: 9 low / 30 high
    assert results_df.loc[0, "p_value"] < 0.05
    return ax.get_title(), heights, len(pairwise)


# ===========================================================================
# create_venn_diagram — the show (save=False) branch
# ===========================================================================

def _case_venn_show_branch_returns_overlap(tmp_path, no_show):
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

    return (sorted(res["overlap"]), res["unique_to_file1"], res["unique_to_file2"],
            len(no_show), no_show[0]["title"])


def _case_venn_negative_filter_coeff_keeps_downregulated(tmp_path, no_show):
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
    # up1 was filtered out of file 1
    return res["overlap"], res["unique_to_file1"], res["unique_to_file2"]


# ===========================================================================
# volcano_plot — I/O helpers
# ===========================================================================

def _write_comma(df, path):
    df.to_csv(path, index=False)


def _write_tab(df, path):
    df.to_csv(path, sep="\t", index=False)


def _case_volcano_plot_reads_every_delimited_format(tmp_path, suffix, writer):
    """Extension dispatch + delimiter sniffing all reach the same plotted points."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    path = tmp_path / f"table{suffix}"
    writer(df, path)

    fig, ax, hits = volcano_plot(
        str(path), fold_change_col="fc", p_value_col="p", show=False
    )
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], df["fc"].to_numpy())
    assert np.allclose(offsets[:, 1], -np.log10(df["p"].to_numpy()))
    return offsets.shape, hits


def _case_volcano_plot_reads_excel_sheet_by_name(tmp_path):
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
    assert np.allclose(sorted(offsets[:, 0]), sorted(df["fc"].to_numpy()))
    return offsets.shape


def _case_volcano_plot_excel_without_engine_raises_helpful_importerror(tmp_path, monkeypatch):
    """A missing Excel engine is re-raised with install instructions."""
    from spacr.plot import volcano_plot

    path = tmp_path / "table.xlsx"
    path.write_bytes(b"not really excel")

    def _boom(*args, **kwargs):
        raise ImportError("Missing optional dependency 'openpyxl'")

    monkeypatch.setattr(pd, "read_excel", _boom)

    with pytest.raises(ImportError) as exc:
        volcano_plot(str(path), fold_change_col="fc", p_value_col="p", show=False)
    assert isinstance(exc.value.__cause__, ImportError)
    return exc.value


def _case_volcano_plot_non_numeric_column_raises(tmp_path):
    """A wholly non-numeric column cannot be coerced and is rejected by name."""
    from spacr.plot import volcano_plot

    df = pd.DataFrame({"fc": ["a", "b", "c"], "p": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError) as exc:
        volcano_plot(df, fold_change_col="fc", p_value_col="p", show=False)
    return exc.value


def _case_volcano_plot_missing_columns_raise_keyerror(missing):
    """Each of the three column arguments is validated separately."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    kwargs = dict(fold_change_col="fc", p_value_col="p", name_col="name", show=False)
    kwargs[missing] = "nope"
    with pytest.raises(KeyError) as exc:
        volcano_plot(df, **kwargs)
    return exc.value


def _case_volcano_plot_drops_nan_rows():
    """Rows with NaN in either numeric column are dropped before plotting."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    df.loc[0, "fc"] = np.nan
    df.loc[1, "p"] = np.nan

    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", show=False
    )
    return ax.collections[0].get_offsets().shape


# ===========================================================================
# volcano_plot — transforms
# ===========================================================================

def _case_volcano_plot_x_transforms(mode):
    """Each log x-transform is applied to the plotted x values and the label."""
    from spacr.plot import volcano_plot

    df = _positive_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", x_transform=mode, show=False
    )
    offsets = ax.collections[0].get_offsets()
    return offsets[:, 0].tolist(), ax.get_xlabel()


def _case_volcano_plot_x_transform_rejects_non_positive():
    """A log x-transform on negative fold changes points at x_transform='none'."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="log2", show=False)
    return exc.value


def _case_volcano_plot_unknown_x_transform_raises():
    """An unrecognised (but positive-safe) x_transform is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_positive_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="sqrt", show=False)
    return exc.value


def _case_volcano_plot_y_transforms(mode):
    """Each y-transform maps p-values onto the plotted y axis and label."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", y_transform=mode, show=False
    )
    offsets = ax.collections[0].get_offsets()
    return offsets[:, 1].tolist(), ax.get_ylabel()


def _case_volcano_plot_y_transform_clips_zero_p_values():
    """p=0 is clipped to the smallest positive float instead of producing inf."""
    from spacr.plot import volcano_plot

    df = pd.DataFrame({"fc": [1.0, -1.0], "p": [0.0, 0.5]})
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", show=False
    )
    y = ax.collections[0].get_offsets()[:, 1]
    assert np.isfinite(y).all()
    return float(y[0])


def _case_volcano_plot_unknown_y_transform_raises():
    """An unrecognised y_transform is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     y_transform="sqrt", show=False)
    return exc.value


# ===========================================================================
# volcano_plot — thresholds
# ===========================================================================

def _case_volcano_plot_thresholds_none_transform_draws_lines_and_colors():
    """x_transform='none': the FC threshold is used verbatim (as |t|)."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p",
        fold_change_threshold=-1.0,          # abs() is taken
        p_value_threshold=0.01, show=False,
    )
    # 2 vertical FC lines + 1 horizontal p line + the cosmetic x=0 line

    # A (fc=+2) is crimson, B (fc=-3) royalblue, the rest lightgray.
    fc_colors = ax.collections[0].get_facecolors()[:, :3]
    return {
        "n_lines": len(ax.lines),
        "fc_line_x": [list(ax.lines[0].get_xdata()), list(ax.lines[1].get_xdata())],
        "p_line_y": float(ax.lines[2].get_ydata()[0]),   # -log10(0.01)
        "zero_line_x": list(ax.lines[3].get_xdata()),
        "colours": [tuple(float(v) for v in fc_colors[i]) for i in range(3)],
    }


def _case_volcano_plot_fc_threshold_converted_to_plot_units(mode):
    """With a log x-transform the FC threshold is converted to plot units."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _positive_df(), fold_change_col="fc", p_value_col="p",
        x_transform=mode, fold_change_threshold=2.0, show=False,
    )
    return float(ax.lines[0].get_xdata()[0]), float(ax.lines[1].get_xdata()[0])


def _case_volcano_plot_fc_threshold_must_be_positive_for_log_transform():
    """A non-positive FC threshold makes no sense in log space."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_positive_df(), fold_change_col="fc", p_value_col="p",
                     x_transform="log2", fold_change_threshold=0.0, show=False)
    return exc.value


def _case_volcano_plot_p_threshold_must_be_positive():
    """p_value_threshold <= 0 is rejected."""
    from spacr.plot import volcano_plot

    with pytest.raises(ValueError) as exc:
        volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p",
                     p_value_threshold=0.0, show=False)
    return exc.value


def _case_volcano_plot_p_threshold_untransformed_axis():
    """y_transform='none' compares raw p-values against the raw threshold."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", name_col="name",
        y_transform="none", p_value_threshold=0.05, show=False,
    )
    # threshold line sits at the raw p value
    # hits = every row with p <= 0.05 (A, B, C, F) -- E has p=1e-3 too
    return float(ax.lines[0].get_ydata()[0]), hits


def _case_volcano_plot_p_threshold_positive_log_axis():
    """A non-negated log y-transform flips the significance comparison."""
    from spacr.plot import volcano_plot

    df = _volcano_df()
    fig, ax, hits = volcano_plot(
        df, fold_change_col="fc", p_value_col="p", name_col="name",
        y_transform="log10", p_value_threshold=1e-4, show=False,
    )
    # log10(p) <= log10(1e-4) -> p <= 1e-4 -> A, B, C
    return float(ax.lines[0].get_ydata()[0]), hits


# ===========================================================================
# volcano_plot — figure plumbing
# ===========================================================================

def _case_volcano_plot_uses_supplied_axes_and_style_kwargs(tmp_path):
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

    # text kwargs reached the annotations
    assert [t.get_text() for t in ax.texts] == hits
    return {
        "same_ax": ax is ax0,
        "same_fig": fig is fig0,                      # fig = ax.figure branch
        "title": ax.get_title(),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "size": float(ax.collections[0].get_sizes()[0]),
        "alpha": float(ax.collections[0].get_alpha()),
        # threshold lines picked up the override, the cosmetic x=0 line did not
        "threshold_colours": [tuple(np.round(l.get_color() if isinstance(l.get_color(), tuple)
                                             else to_rgb(l.get_color()), 3))
                              for l in ax.lines[:3]],
        "linestyle": ax.lines[0].get_linestyle(),
        "zero_line_colour": to_rgb(ax.lines[3].get_color()),
        "hits": hits,
        "annotation_fontsizes": [t.get_fontsize() for t in ax.texts],
        "spines_visible": (ax.spines["right"].get_visible(),
                           ax.spines["top"].get_visible()),
        "saved": out.is_file() and out.stat().st_size > 0,
    }


def _case_volcano_plot_show_true_calls_plt_show(no_show):
    """show=True routes through plt.show exactly once."""
    from spacr.plot import volcano_plot

    volcano_plot(_volcano_df(), fold_change_col="fc", p_value_col="p", show=True)
    return len(no_show)


def _case_volcano_plot_without_thresholds_is_all_gray():
    """No thresholds -> a single gray colour and no threshold lines."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", show=False
    )
    colors = ax.collections[0].get_facecolors()
    # shape (1, 4) is a scalar colour spec; only the x=0 line exists
    return colors.shape, tuple(float(v) for v in colors[0, :3]), len(ax.lines)


# ===========================================================================
# volcano_plot — annotation
# ===========================================================================

def _case_volcano_plot_annotate_needs_thresholds_or_annotate_max():
    """With no thresholds and no cap, nothing is annotated."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        show=False,
    )
    return hits, len(ax.texts)


def _case_volcano_plot_annotate_max_picks_highest_y():
    """annotate_max keeps only the top-y points, in descending y order."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        annotate_max=2, show=False,
    )
    # B (p=1e-9) then C (p=1e-8)
    return hits, sorted(t.get_text() for t in ax.texts)


def _case_volcano_plot_annotate_disabled():
    """annotate=False leaves the axes text-free even with thresholds set."""
    from spacr.plot import volcano_plot

    fig, ax, hits = volcano_plot(
        _volcano_df(), fold_change_col="fc", p_value_col="p", name_col="name",
        fold_change_threshold=1.0, p_value_threshold=0.01,
        annotate=False, show=False,
    )
    return hits, len(ax.texts)


def _case_volcano_plot_annotation_requires_adjusttext(monkeypatch):
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
    assert isinstance(exc.value.__cause__, ImportError)
    return exc.value


def _fc():
    return _positive_df()["fc"].to_numpy()


def _p():
    return _volcano_df()["p"].to_numpy()


def _x_row(mode, fn):
    return pytest.param(
        functools.partial(_case_volcano_plot_x_transforms, mode=mode),
        (pytest.approx(fn(_fc()).tolist()), f"{mode}(fc)"), id=f"x-transform-{mode}")


def _y_row(mode, fn):
    return pytest.param(
        functools.partial(_case_volcano_plot_y_transforms, mode=mode),
        (pytest.approx(fn(_p()).tolist()), "p" if mode == "none" else f"{mode}(p)"),
        id=f"y-transform-{mode}")


def _fc_row(mode, fn):
    return pytest.param(
        functools.partial(_case_volcano_plot_fc_threshold_converted_to_plot_units, mode=mode),
        (pytest.approx(-abs(fn(2.0))), pytest.approx(abs(fn(2.0)))),
        id=f"fc-threshold-in-{mode}-units")


def _format_row(suffix, writer):
    return pytest.param(
        functools.partial(_case_volcano_plot_reads_every_delimited_format,
                          suffix=suffix, writer=writer),
        ((6, 2), []), id=f"reads-{suffix.lstrip('.')}")


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_overlay_masks_on_images_saves_resized_rgb_overlays,
                 {"written": 2, "failed": []}, id="overlay-saves-resized-rgb-overlays"),
    pytest.param(_case_overlay_masks_on_images_plot_branch_no_resize_and_mask_rescale,
                 (1, (128, 128, 3), "Overlay: a.tif"),
                 id="overlay-plot-branch-native-size-and-mask-rescale"),
    pytest.param(_case_overlay_masks_on_images_rgb_input_keeps_three_channels,
                 (1, (80, 80, 3)), id="overlay-rgb-input-keeps-three-channels"),
    pytest.param(_case_overlay_masks_on_images_no_common_filenames, (None, []),
                 id="overlay-no-common-filenames-returns-early"),
    pytest.param(_case_graph_importance_concatenates_csvs_and_saves, (None, 48),
                 id="graph-importance-concatenates-csvs-and-saves"),
    pytest.param(_case_graph_importance_missing_columns_returns_early, (None, []),
                 id="graph-importance-missing-columns-returns-early"),
    pytest.param(_case_graph_importance_accepts_a_single_path_string, ["imp0.csv"],
                 id="graph-importance-accepts-single-path-string"),
    pytest.param(
        _case_plot_proportion_stacked_bars_well_level,
        {"unit": "object", "degrees_of_freedom": 1, "pairwise_rows": 1,
         "title_suffix": "Mean \u00b1 SD across wells)", "xlabel": "Group",
         "ylabel": "Proportion", "ylim": (0.0, 1.0), "xticklabels": ["ctrl", "trt"],
         "n_bars": 4, "height_total": pytest.approx(2.0)},
        id="proportion-bars-well-level-mean-sd"),
    pytest.param(
        _case_plot_proportion_stacked_bars_object_level_matches_raw_proportions,
        ("Proportion of Volume Bins by Group",
         sorted([round(30 / 39, 6), round(9 / 39, 6), round(9 / 39, 6),
                 round(30 / 39, 6)]), 1),
        id="proportion-bars-object-level-raw-proportions"),
    pytest.param(
        _case_venn_show_branch_returns_overlap,
        (["g2", "g3"], ["g1"], ["g4"], 1, "Venn Diagram of Overlapping Genes"),
        id="venn-show-branch-returns-overlap"),
    pytest.param(
        _case_venn_negative_filter_coeff_keeps_downregulated,
        (["down1"], [], ["down2"]), id="venn-negative-filter-keeps-downregulated"),
])
def test_overlay_importance_proportion_and_venn_outputs(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    _format_row(".csv", _write_comma),
    _format_row(".tsv", _write_tab),
    _format_row(".tab", _write_tab),
    _format_row(".txt", _write_comma),     # delimiter sniffed: comma
    _format_row(".dat", _write_tab),       # delimiter sniffed: tab
    pytest.param(_case_volcano_plot_reads_excel_sheet_by_name, (6, 2),
                 id="reads-excel-sheet-by-name"),
    pytest.param(_case_volcano_plot_drops_nan_rows, (4, 2), id="drops-nan-rows"),
    _x_row("log2", np.log2),
    _x_row("log10", np.log10),
    _x_row("ln", np.log),
    _x_row("log", np.log),
    _y_row("none", lambda p: p),
    _y_row("-log10", lambda p: -np.log10(p)),
    _y_row("-ln", lambda p: -np.log(p)),
    _y_row("log10", np.log10),
    _y_row("ln", np.log),
    _y_row("log", np.log),
    pytest.param(_case_volcano_plot_y_transform_clips_zero_p_values,
                 pytest.approx(-np.log10(np.finfo(float).tiny)),
                 id="y-transform-clips-zero-p-values"),
    pytest.param(
        _case_volcano_plot_thresholds_none_transform_draws_lines_and_colors,
        {"n_lines": 4, "fc_line_x": [[-1.0, -1.0], [1.0, 1.0]],
         "p_line_y": pytest.approx(2.0), "zero_line_x": [0, 0],
         "colours": [pytest.approx(tuple(to_rgb(ROLES[role])))
                     for role in ("up", "down", "data")]},
        id="thresholds-draw-lines-and-colour-hits"),
    _fc_row("log2", np.log2),
    _fc_row("log10", np.log10),
    _fc_row("ln", np.log),
    pytest.param(_case_volcano_plot_p_threshold_untransformed_axis,
                 (pytest.approx(0.05), ["A", "B", "C", "E", "F"]),
                 id="p-threshold-untransformed-axis"),
    pytest.param(_case_volcano_plot_p_threshold_positive_log_axis,
                 (pytest.approx(np.log10(1e-4)), ["A", "B", "C"]),
                 id="p-threshold-positive-log-axis"),
    pytest.param(
        _case_volcano_plot_uses_supplied_axes_and_style_kwargs,
        {"same_ax": True, "same_fig": True, "title": "my volcano",
         "xlim": (-5.0, 5.0), "ylim": (0.0, 12.0), "size": pytest.approx(55.0),
         "alpha": pytest.approx(0.5),
         "threshold_colours": [tuple(np.round(to_rgb("green"), 3))] * 3,
         "linestyle": ":", "zero_line_colour": to_rgb(ROLES["reference"]),
         "hits": ["A", "B"], "annotation_fontsizes": [14, 14],
         "spines_visible": (False, False), "saved": True},
        id="uses-supplied-axes-and-style-kwargs"),
    pytest.param(_case_volcano_plot_show_true_calls_plt_show, 1,
                 id="show-true-calls-plt-show-once"),
    pytest.param(_case_volcano_plot_without_thresholds_is_all_gray,
                 ((1, 4), pytest.approx(to_rgb(ROLES["data"]), abs=1e-6), 1),
                 id="without-thresholds-is-all-one-colour"),
    pytest.param(_case_volcano_plot_annotate_needs_thresholds_or_annotate_max,
                 ([], 0), id="annotate-needs-thresholds-or-annotate-max"),
    pytest.param(_case_volcano_plot_annotate_max_picks_highest_y,
                 (["B", "C"], ["B", "C"]), id="annotate-max-picks-highest-y"),
    pytest.param(_case_volcano_plot_annotate_disabled, ([], 0),
                 id="annotate-false-leaves-axes-text-free"),
])
def test_volcano_plot_draws(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


def _missing_row(missing):
    return pytest.param(
        functools.partial(_case_volcano_plot_missing_columns_raise_keyerror,
                          missing=missing),
        KeyError, (missing, "nope"), id=f"refuses-missing-{missing}")


@pytest.mark.parametrize("helper, error, fragments", [
    pytest.param(_case_volcano_plot_excel_without_engine_raises_helpful_importerror,
                 ImportError, ("openpyxl", "xlrd"), id="refuses-excel-without-engine"),
    pytest.param(_case_volcano_plot_non_numeric_column_raises, ValueError,
                 ("Column 'fc' could not be converted to numeric.",),
                 id="refuses-non-numeric-column"),
    _missing_row("fold_change_col"),
    _missing_row("p_value_col"),
    _missing_row("name_col"),
    pytest.param(_case_volcano_plot_x_transform_rejects_non_positive, ValueError,
                 ("requires all fold changes > 0", "x_transform='none'"),
                 id="refuses-log-x-transform-of-non-positive"),
    pytest.param(_case_volcano_plot_unknown_x_transform_raises, ValueError,
                 ("Unknown x_transform: sqrt",), id="refuses-unknown-x-transform"),
    pytest.param(_case_volcano_plot_unknown_y_transform_raises, ValueError,
                 ("Unknown y_transform: sqrt",), id="refuses-unknown-y-transform"),
    pytest.param(_case_volcano_plot_fc_threshold_must_be_positive_for_log_transform,
                 ValueError, ("fold_change_threshold must be > 0",),
                 id="refuses-non-positive-fc-threshold-in-log-space"),
    pytest.param(_case_volcano_plot_p_threshold_must_be_positive, ValueError,
                 ("p_value_threshold must be > 0",),
                 id="refuses-non-positive-p-threshold"),
    pytest.param(_case_volcano_plot_annotation_requires_adjusttext, ImportError,
                 ("adjustText", "pip install adjustText"),
                 id="refuses-annotation-without-adjusttext"),
])
def test_volcano_plot_refuses_by_name(request, helper, error, fragments):
    raised = _call(request, helper)
    observed = (type(raised), tuple(f for f in fragments if f in str(raised)))
    assert observed == (error, fragments)
