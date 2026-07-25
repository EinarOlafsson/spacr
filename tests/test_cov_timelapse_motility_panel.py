"""Coverage for spacr.timelapse figure helpers around the motility panel.

Covers:
  * ``_generate_mask_random_cmap``   — random label colormap w/ black background
  * ``create_results_figure``        — 3-panel QC figure layout
  * ``_make_intensity_motility_panel`` — the large per-well figure builder,
    including every QC strategy branch (histogram / PCA / UMAP / t-SNE /
    XGBoost), the embedded QC PNG path, the early-return guards and the
    per-axis "nothing to draw" fallbacks.

Everything is CPU-only and offline: the panel builder is fed hand-made
DataFrames / track dicts and writes small PDFs into ``tmp_path``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def panels(monkeypatch):
    """Capture every ``(fig, axes)`` produced by ``plt.subplots``.

    ``_make_intensity_motility_panel`` closes each figure once it has been
    saved, but the Figure/Axes objects stay introspectable, so this lets the
    tests assert on titles / visibility / artists per subplot.
    """
    captured = []
    orig = plt.subplots

    def _spy(*args, **kwargs):
        fig, axes = orig(*args, **kwargs)
        captured.append((fig, np.atleast_1d(np.asarray(axes)).ravel()))
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _spy)
    return captured


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------

PLATE = "plate1"
WELL = "A01"
KEY_COLS = ["plateID", "wellID", "fieldID", "cellID"]


def _all_df(
    plate=PLATE,
    well=WELL,
    n_cells=4,
    n_frames=3,
    n_channels=2,
    pathogen_chan=None,
    infection_col="infected",
    all_infected=False,
    intensity_prefix="cell_mean_intensity_ch",
):
    """Per-frame, per-cell measurement table with the columns the panel wants."""
    rows = []
    for c in range(n_cells):
        inf = True if all_infected else bool(c % 2 == 0)
        for t in range(n_frames):
            row = {
                "plateID": plate,
                "wellID": well,
                "fieldID": 1,
                "cellID": c + 1,
                infection_col: inf,
            }
            for ch in range(n_channels):
                row[f"{intensity_prefix}{ch}"] = 100.0 + 10.0 * c + 5.0 * ch + t
            rows.append(row)
    df = pd.DataFrame(rows)
    if pathogen_chan is not None:
        base = df[f"{intensity_prefix}{pathogen_chan}"]
        df[f"cell_p75_intensity_ch{pathogen_chan}"] = base * 1.2
        df[f"pathogen_mean_intensity_ch{pathogen_chan}"] = base * 2.0
        df[f"cytoplasm_mean_intensity_ch{pathogen_chan}"] = base * 0.5
    return df


def _track_df(plate=PLATE, well=WELL, infected=(True, False, True, False)):
    n = len(infected)
    return pd.DataFrame(
        {
            "plateID": [plate] * n,
            "wellID": [well] * n,
            "infected": list(infected),
            "velocity": np.linspace(0.5, 2.0, n),
        }
    )


def _tracks(plate=PLATE, well=WELL, infected=(True, False, True, False), length=5):
    out = []
    for i, inf in enumerate(infected):
        n = length
        x = 10.0 + np.arange(n, dtype=float) * (i + 1)
        y = 20.0 + np.arange(n, dtype=float) * 0.5 * (i + 1)
        out.append(
            {"plateID": plate, "wellID": well, "x_px": x, "y_px": y, "infected": bool(inf)}
        )
    return out


def _call(tmp_path, **over):
    """Invoke the panel builder with sensible defaults, return the out dir."""
    from spacr.timelapse import _make_intensity_motility_panel

    settings = {"pathogen_channel": 1}
    settings.update(over.pop("settings", {}) or {})
    motility_dir = over.pop("motility_dir", None) or str(tmp_path / "motility")

    kwargs = dict(
        all_df=_all_df(n_channels=2, pathogen_chan=1),
        infection_col="infected",
        track_df=_track_df(),
        per_well_tracks={"plate1_A01": _tracks()},
        n_channels=2,
        motility_dir=motility_dir,
        pixels_per_um=2.0,
        seconds_per_frame=30.0,
        vel_unit="um/s",
        settings=settings,
        label_tag="mask_labels",
    )
    kwargs.update(over)
    _make_intensity_motility_panel(**kwargs)
    return motility_dir


def _pdfs(motility_dir):
    if not os.path.isdir(motility_dir):
        return []
    return sorted(f for f in os.listdir(motility_dir) if f.endswith(".pdf"))


# ---------------------------------------------------------------------------
# _generate_mask_random_cmap
# ---------------------------------------------------------------------------

def test_generate_mask_random_cmap_one_colour_per_label_plus_black_background():
    from spacr.timelapse import _generate_mask_random_cmap
    import matplotlib as mpl

    mask = np.zeros((8, 8), dtype=np.int32)
    mask[0:2, 0:2] = 1
    mask[3:5, 3:5] = 4
    mask[6:8, 6:8] = 9  # non-contiguous labels: 3 objects
    cmap = _generate_mask_random_cmap(mask)

    assert isinstance(cmap, mpl.colors.ListedColormap)
    assert cmap.N == 4  # 3 objects + background
    colours = np.asarray(cmap.colors)
    assert colours.shape == (4, 4)
    assert np.array_equal(colours[0], np.array([0.0, 0.0, 0.0, 1.0]))
    assert np.all(colours[:, 3] == 1.0)
    # objects must not be black (probability ~0 with random floats)
    assert np.all(colours[1:, :3].sum(axis=1) > 0)


def test_generate_mask_random_cmap_background_only_mask():
    from spacr.timelapse import _generate_mask_random_cmap

    cmap = _generate_mask_random_cmap(np.zeros((5, 5), dtype=np.int32))
    assert cmap.N == 1
    assert np.array_equal(np.asarray(cmap.colors)[0], np.array([0.0, 0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# create_results_figure
# ---------------------------------------------------------------------------

def test_create_results_figure_layout():
    from spacr.timelapse import create_results_figure
    from matplotlib.figure import Figure

    fig, ax_pca, ax_xgb, ax_hist = create_results_figure()
    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == (7.0, 6.0)
        assert fig.dpi == 100
        assert len(fig.axes) == 3
        # PCA top-left, XGB top-right, histogram spanning the bottom row
        assert list(ax_pca.get_subplotspec().colspan) == [0]
        assert list(ax_xgb.get_subplotspec().colspan) == [1]
        assert list(ax_hist.get_subplotspec().colspan) == [0, 1]
        assert list(ax_hist.get_subplotspec().rowspan) == [1]
        # height_ratios=[2, 1] -> the top row is taller than the bottom one
        assert ax_pca.get_position().height > ax_hist.get_position().height
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# _make_intensity_motility_panel — early-return guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "over",
    [
        {"all_df": pd.DataFrame()},
        {"track_df": pd.DataFrame()},
        {"per_well_tracks": {}},
    ],
    ids=["empty_all_df", "empty_track_df", "no_tracks"],
)
def test_panel_returns_early_on_missing_inputs(tmp_path, capsys, over):
    out = _call(tmp_path, **over)
    msg = capsys.readouterr().out
    assert "No data for panel 'mask_labels'" in msg
    # early return happens before os.makedirs
    assert not os.path.isdir(out)


def test_panel_requires_plate_and_well_columns(tmp_path, capsys):
    df = _all_df()
    out = _call(tmp_path, all_df=df.drop(columns=["wellID"]))
    msg = capsys.readouterr().out
    assert "Missing 'plateID'/'wellID' columns" in msg
    # the directory *is* created before the column check
    assert os.path.isdir(out)
    assert _pdfs(out) == []


def test_panel_skips_well_without_tracks(tmp_path, capsys):
    """A well present in all_df but absent from per_well_tracks is skipped."""
    out = _call(
        tmp_path,
        per_well_tracks={"other": _tracks(well="B02")},
        track_df=_track_df(well="B02"),
    )
    msg = capsys.readouterr().out
    assert "No data for plate=plate1, well=A01; skipping." in msg
    assert _pdfs(out) == []


def test_panel_skips_well_without_intensity_columns(tmp_path, capsys):
    out = _call(
        tmp_path,
        all_df=_all_df(n_channels=1, intensity_prefix="bogus_intensity_ch"),
        settings={"pathogen_channel": None},
        n_channels=1,
    )
    msg = capsys.readouterr().out
    assert "No cell_mean_intensity_ch* columns" in msg
    assert _pdfs(out) == []


# ---------------------------------------------------------------------------
# mask panel — happy path
# ---------------------------------------------------------------------------

def test_mask_panel_full_layout_and_filename(tmp_path, panels):
    out = _call(
        tmp_path,
        settings={
            "pathogen_channel": 1,
            "motility_xlim": (-50, 50),
            "motility_ylim": (-40, 40),
        },
    )
    assert _pdfs(out) == ["plate1_A01.pdf"]
    assert os.path.getsize(os.path.join(out, "plate1_A01.pdf")) > 0

    assert len(panels) == 1
    fig, axes = panels[0]
    # 2 channels + p75 + ratio + all-tracks + 2 origin plots
    assert len(axes) == 7
    assert [axes[i].get_title() for i in range(4)] == [
        "Ch 0 mean",
        "Ch 1 mean",
        "Ch 1 p75",
        "Ch 1 pathogen/cytoplasm",
    ]
    assert axes[0].get_ylabel() == "Mean cell intensity"
    assert axes[2].get_ylabel() == "Cell p75 intensity"
    assert axes[3].get_ylabel() == "Intensity ratio"
    # violins: infected + uninfected -> 2 bodies, means scattered on top
    assert len(axes[0].collections) >= 2
    # all values positive -> y anchored at 0
    assert axes[0].get_ylim()[0] == 0

    ax_all = axes[4]
    assert ax_all.get_xlabel() == "x (µm)"
    assert ax_all.get_ylabel() == "y (µm)"
    assert len(ax_all.lines) == 4  # one polyline per track
    assert len(ax_all.texts) == 1
    legend_txt = ax_all.texts[0].get_text()
    assert "Infected (1.00 um/s)" in legend_txt
    assert "Uninfected (1.50 um/s)" in legend_txt
    assert "1 µm = 2.00 px" in legend_txt
    assert "1 frame = 30 s" in legend_txt

    ax_inf, ax_uninf = axes[5], axes[6]
    assert ax_inf.get_title() == "Infected\n(n=2, v=1.00 um/s)"
    assert ax_uninf.get_title() == "Uninfected\n(n=2, v=1.50 um/s)"
    # motility_xlim / motility_ylim honoured
    assert ax_inf.get_xlim() == (-50, 50)
    assert ax_inf.get_ylim() == (-40, 40)
    assert fig._suptitle.get_text().startswith("Infection panel – mask labels – method=none")


def test_mask_panel_pixel_units_when_no_calibration(tmp_path, panels):
    out = _call(
        tmp_path,
        all_df=_all_df(n_channels=2, pathogen_chan=None),
        settings={"pathogen_channel": None},
        pixels_per_um=None,
        seconds_per_frame=None,
    )
    assert _pdfs(out) == ["plate1_A01.pdf"]
    fig, axes = panels[0]
    assert len(axes) == 5  # 2 channels + 3 motility axes
    ax_all = axes[2]
    assert ax_all.get_xlabel() == "x (pixels)"
    assert ax_all.get_ylabel() == "y (pixels)"
    txt = ax_all.texts[0].get_text()
    assert "px" not in txt and "frame" not in txt
    # unscaled coordinates -> raw pixel range of the tracks
    assert ax_all.get_xlim()[1] > 20


def test_mask_panel_two_wells_emits_two_pdfs(tmp_path, panels):
    all_df = pd.concat(
        [_all_df(well="A01", pathogen_chan=1), _all_df(well="B02", pathogen_chan=1)],
        ignore_index=True,
    )
    track_df = pd.concat([_track_df(well="A01"), _track_df(well="B02")], ignore_index=True)
    out = _call(
        tmp_path,
        all_df=all_df,
        track_df=track_df,
        per_well_tracks={"a": _tracks(well="A01"), "b": _tracks(well="B02")},
    )
    assert _pdfs(out) == ["plate1_A01.pdf", "plate1_B02.pdf"]
    assert len(panels) == 2


def test_mask_panel_unknown_label_tag_uses_fallback_filename(tmp_path, panels):
    out = _call(
        tmp_path,
        label_tag="weird_tag",
        settings={"pathogen_channel": 1, "infection_intensity_strategy": "histogram"},
    )
    assert _pdfs(out) == ["plate1_A01_weird_tag_histogram.pdf"]
    fig, _axes = panels[0]
    # panel_label falls back to the raw tag
    assert "– weird_tag labels – method=histogram" in fig._suptitle.get_text()


def test_mask_panel_blank_strategy_reports_method_none(tmp_path, panels):
    out = _call(tmp_path, settings={"pathogen_channel": 1, "infection_intensity_strategy": ""})
    assert _pdfs(out) == ["plate1_A01.pdf"]
    fig, _axes = panels[0]
    assert "method=none" in fig._suptitle.get_text()


# ---------------------------------------------------------------------------
# mask panel — embedded QC PNG
# ---------------------------------------------------------------------------

def _write_png(path):
    plt.imsave(str(path), np.linspace(0, 1, 64).reshape(8, 8), cmap="gray")
    return str(path)


@pytest.mark.parametrize(
    "panel_type,expected",
    [
        ("histogram", "Intensity histogram"),
        ("pca", "PCA/UMAP clustering"),
        ("xgboost", "XGBoost feature importance"),
        ("something_else", "Infection QC"),
    ],
)
def test_mask_panel_embeds_qc_png(tmp_path, panels, panel_type, expected):
    png = _write_png(tmp_path / "qc.png")
    out = _call(
        tmp_path,
        settings={
            "pathogen_channel": 1,
            "infection_intensity_qc_panel_path": png,
            "infection_intensity_qc_panel_type": panel_type,
        },
    )
    assert _pdfs(out) == ["plate1_A01.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 8  # extra QC axis appended
    ax_qc = axes[7]
    assert ax_qc.get_title() == expected
    assert len(ax_qc.images) == 1
    assert not ax_qc.axison


def test_mask_panel_hides_qc_axis_when_png_unreadable(tmp_path, panels, capsys):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"definitely not a png")
    out = _call(
        tmp_path,
        settings={
            "pathogen_channel": 1,
            "infection_intensity_qc_panel_path": str(bad),
            "infection_intensity_qc_panel_type": "histogram",
        },
    )
    assert _pdfs(out) == ["plate1_A01.pdf"]
    assert "Could not embed QC plot" in capsys.readouterr().out
    _fig, axes = panels[0]
    assert len(axes) == 8
    assert axes[7].get_visible() is False


def test_mask_panel_ignores_qc_png_when_graphs_disabled(tmp_path, panels):
    png = _write_png(tmp_path / "qc.png")
    _call(
        tmp_path,
        settings={
            "pathogen_channel": 1,
            "infection_intensity_qc_graphs": False,
            "infection_intensity_qc_panel_path": png,
        },
    )
    _fig, axes = panels[0]
    assert len(axes) == 7  # no QC axis allocated


def test_mask_panel_ignores_missing_qc_png_path(tmp_path, panels):
    _call(
        tmp_path,
        settings={
            "pathogen_channel": 1,
            "infection_intensity_qc_panel_path": str(tmp_path / "nope.png"),
        },
    )
    _fig, axes = panels[0]
    assert len(axes) == 7


# ---------------------------------------------------------------------------
# adjusted panel — histogram strategy
# ---------------------------------------------------------------------------

def test_adjusted_histogram_computed_from_well_dataframe(tmp_path, panels):
    out = _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "histogram",
            "infection_intensity_n_bins": 8,
            "infection_intensity_threshold": 115.0,
        },
    )
    assert _pdfs(out) == ["plate1_A01_histogram_adjusted.pdf"]
    fig, axes = panels[0]
    assert len(axes) == 8
    ax_hist = axes[7]
    assert ax_hist.get_visible() is True
    assert ax_hist.get_title() == "Pathogen-channel intensity\n(adjusted labels)"
    assert ax_hist.get_xlabel() == "cell_mean_intensity_ch1"
    assert ax_hist.get_ylabel() == "Count"
    # two hist stacks (8 bins each) plus the threshold line
    assert len(ax_hist.patches) == 16
    assert len(ax_hist.lines) == 1
    assert ax_hist.lines[0].get_xdata()[0] == 115.0
    assert "adjusted labels – method=histogram" in fig._suptitle.get_text()


def test_adjusted_histogram_from_settings_payload(tmp_path, panels):
    payload = {
        "intensities_inf": [1.0, 2.0, 3.0],
        "intensities_uninf": [0.1, 0.2],
        "bin_edges": [0.0, 1.0, 2.0, 3.0],
        "thr_val": 1.5,
        "intensity_col": "my_intensity",
    }
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "histogram",
            "infection_hist_data": payload,
        },
    )
    _fig, axes = panels[0]
    ax_hist = axes[7]
    assert ax_hist.get_xlabel() == "my_intensity"
    assert len(ax_hist.patches) == 6  # 3 bins x 2 histograms
    assert len(ax_hist.lines) == 1  # threshold


def test_adjusted_histogram_payload_invalid_hides_axis(tmp_path, panels, capsys):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "histogram",
            "infection_hist_data": {"intensities_inf": [1.0]},  # missing keys
        },
    )
    assert "Histogram payload invalid" in capsys.readouterr().out
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False


def test_adjusted_histogram_all_nan_intensity_hides_axis(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["nan_intensity"] = np.nan
    _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "histogram",
            "infection_hist_intensity_col": "nan_intensity",
        },
    )
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def test_adjusted_histogram_dataframe_payload_without_intensity_column(tmp_path, panels):
    """A non-dict payload is treated as a frame; no usable column -> axis hidden."""
    payload = pd.DataFrame({"plateID": [PLATE] * 2, "unrelated": [1.0, 2.0]})
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "histogram",
            "infection_hist_data": payload,
        },
    )
    _fig, axes = panels[0]
    assert len(axes) == 8
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def test_adjusted_histogram_without_pathogen_channel_uses_first_channel(tmp_path, panels):
    _call(
        tmp_path,
        all_df=_all_df(n_channels=2, pathogen_chan=None),
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": None,
            "infection_intensity_strategy": "histogram",
            "infection_intensity_n_bins": 4,
        },
    )
    _fig, axes = panels[0]
    assert len(axes) == 6  # 2 channels + 3 motility + 1 QC
    assert axes[5].get_xlabel() == "cell_mean_intensity_ch0"
    assert len(axes[5].patches) == 8
    # no threshold available -> no axvline
    assert len(axes[5].lines) == 0


# ---------------------------------------------------------------------------
# adjusted panel — PCA / UMAP / t-SNE strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy,method", [("pca", "PCA"), ("umap", "UMAP"), ("tsne", "t-SNE")])
def test_adjusted_embedding_qc_axis(tmp_path, panels, strategy, method):
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    labels = np.arange(20) % 2 == 0
    out = _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": strategy,
            "infection_pca_data": {
                "coords": coords,
                "labels": labels,
                "method_label": method,
            },
        },
    )
    assert _pdfs(out) == [f"plate1_A01_{strategy}_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 8
    ax = axes[7]
    assert ax.get_xlabel() == f"{method} 1"
    assert ax.get_ylabel() == f"{method} 2"
    assert ax.get_title() == f"{method} of features\n(adjusted labels)"
    # one scatter per class
    assert len(ax.collections) == 2
    assert ax.collections[0].get_offsets().shape == (10, 2)
    assert ax.collections[1].get_offsets().shape == (10, 2)


def test_adjusted_embedding_defaults_to_pca_label(tmp_path, panels):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "pca",
            "infection_pca_data": {
                "coords": [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                "labels": [True, False, False],
            },
        },
    )
    _fig, axes = panels[0]
    assert axes[7].get_xlabel() == "PCA 1"


def test_adjusted_embedding_invalid_payload_hides_axis(tmp_path, panels, capsys):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "pca",
            "infection_pca_data": {"coords": [[0.0, 1.0]]},  # no 'labels'
        },
    )
    assert "PCA/embedding payload invalid" in capsys.readouterr().out
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False


def test_adjusted_embedding_one_dimensional_coords_hides_axis(tmp_path, panels):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={
            "pathogen_channel": 1,
            "infection_intensity_strategy": "umap",
            "infection_pca_data": {"coords": [0.0, 1.0, 2.0], "labels": [True, False, True]},
        },
    )
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].collections) == 0


def test_adjusted_embedding_without_payload_allocates_no_qc_axis(tmp_path, panels):
    out = _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={"pathogen_channel": 1, "infection_intensity_strategy": "tsne"},
    )
    assert _pdfs(out) == ["plate1_A01_tsne_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 7


# ---------------------------------------------------------------------------
# adjusted panel — XGBoost strategy
# ---------------------------------------------------------------------------

def _xgb_settings(**extra):
    s = {
        "pathogen_channel": 1,
        "infection_intensity_strategy": "xgboost",
        "infection_xgb_importance": {
            "feature_names": ["feat_a", "feat_b", "feat_c"],
            "feature_importances": [0.5, 0.3, 0.2],
        },
    }
    s.update(extra)
    return s


def test_adjusted_xgboost_prob_and_importance_axes(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["infection_prob"] = np.linspace(0.05, 0.95, len(df))
    out = _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings=_xgb_settings(),
    )
    assert _pdfs(out) == ["plate1_A01_xgboost_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 9  # 4 intensity + 3 motility + prob + importance

    ax_prob = axes[7]
    assert ax_prob.get_title() == "Probability separation (adjusted labels)"
    assert ax_prob.get_xlabel() == "XGBoost infection probability"
    assert ax_prob.get_ylabel() == "Cells"
    assert len(ax_prob.patches) == 40  # 20 bins x 2 histograms

    ax_imp = axes[8]
    assert ax_imp.get_title() == "XGBoost feature importance"
    assert ax_imp.get_xlabel() == "Importance (gain)"
    assert [t.get_text() for t in ax_imp.get_yticklabels()] == ["feat_a", "feat_b", "feat_c"]
    assert len(ax_imp.patches) == 3
    # invert_yaxis()
    assert ax_imp.get_ylim()[0] > ax_imp.get_ylim()[1]


def test_adjusted_xgboost_uses_configured_probability_column(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["my_proba"] = np.linspace(0.1, 0.9, len(df))
    _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings=_xgb_settings(infection_xgb_proba_column="my_proba"),
    )
    _fig, axes = panels[0]
    assert axes[7].get_visible() is True
    assert len(axes[7].patches) == 40


def test_adjusted_xgboost_hides_prob_axis_without_probability_column(tmp_path, panels):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings=_xgb_settings(infection_xgb_proba_column="absent_col"),
    )
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert axes[8].get_visible() is True  # importance axis still drawn


def test_adjusted_xgboost_hides_prob_axis_when_all_probabilities_nan(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["infection_prob"] = np.nan
    _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings=_xgb_settings(),
    )
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def test_adjusted_xgboost_single_class_probabilities(tmp_path, panels):
    """All cells infected -> only the 'Infected' histogram is drawn."""
    df = _all_df(n_channels=2, pathogen_chan=1, all_infected=True)
    df["infection_prob"] = np.linspace(0.2, 0.8, len(df))
    _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings=_xgb_settings(),
    )
    _fig, axes = panels[0]
    assert len(axes[7].patches) == 20  # a single 20-bin histogram
    # violin plots collapse to the infected group only
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Inf"]


def test_adjusted_xgboost_importance_payload_invalid_hides_axis(tmp_path, panels, capsys):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["infection_prob"] = np.linspace(0.05, 0.95, len(df))
    _call(
        tmp_path,
        all_df=df,
        label_tag="adjusted_labels",
        settings=_xgb_settings(infection_xgb_importance={"wrong_key": 1}),
    )
    assert "XGB importance payload invalid" in capsys.readouterr().out
    _fig, axes = panels[0]
    assert axes[8].get_visible() is False


def test_adjusted_xgboost_empty_feature_list_hides_axis(tmp_path, panels):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings=_xgb_settings(
            infection_xgb_importance={"feature_names": [], "feature_importances": []}
        ),
    )
    _fig, axes = panels[0]
    assert axes[8].get_visible() is False
    assert len(axes[8].patches) == 0


def test_adjusted_xgboost_without_payload_allocates_no_qc_axes(tmp_path, panels):
    out = _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings={"pathogen_channel": 1, "infection_intensity_strategy": "xgboost"},
    )
    assert _pdfs(out) == ["plate1_A01_xgboost_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 7


def test_adjusted_panel_with_qc_graphs_disabled(tmp_path, panels):
    _call(
        tmp_path,
        label_tag="adjusted_labels",
        settings=_xgb_settings(infection_intensity_qc_graphs=False),
    )
    _fig, axes = panels[0]
    assert len(axes) == 7  # QC axes suppressed entirely


# ---------------------------------------------------------------------------
# per-axis fallbacks inside the panel
# ---------------------------------------------------------------------------

def test_violin_axis_hidden_when_value_column_is_all_nan(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["cell_p75_intensity_ch1"] = np.nan
    _call(tmp_path, all_df=df)
    _fig, axes = panels[0]
    assert axes[2].get_visible() is False  # the p75 axis
    assert axes[1].get_visible() is True


def test_violin_axis_not_anchored_at_zero_for_negative_values(tmp_path, panels):
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["pathogen_mean_intensity_ch1"] = -df["pathogen_mean_intensity_ch1"]
    _call(tmp_path, all_df=df)
    _fig, axes = panels[0]
    ax_ratio = axes[3]
    assert ax_ratio.get_visible() is True
    assert ax_ratio.get_ylim()[0] < 0


def test_all_tracks_axis_hidden_when_every_track_is_too_short(tmp_path, panels):
    short = _tracks(length=1)
    out = _call(tmp_path, per_well_tracks={"w": short})
    assert _pdfs(out) == ["plate1_A01.pdf"]
    _fig, axes = panels[0]
    ax_all = axes[4]
    assert ax_all.get_visible() is False
    assert len(ax_all.lines) == 0
    # the origin plots still render, but with zero tracks
    assert axes[5].get_title() == "Infected\n(n=0, v=1.00 um/s)"
    assert axes[6].get_title() == "Uninfected\n(n=0, v=1.50 um/s)"


def test_tracks_shorter_than_two_points_are_skipped(tmp_path, panels):
    tracks = _tracks()
    tracks[0]["x_px"] = np.array([5.0])
    tracks[0]["y_px"] = np.array([6.0])
    _call(tmp_path, per_well_tracks={"w": tracks})
    _fig, axes = panels[0]
    assert len(axes[4].lines) == 3  # the 1-point track is dropped
    assert axes[5].get_title().startswith("Infected\n(n=1,")


def test_all_uninfected_tracks_report_nan_infected_velocity(tmp_path, panels):
    infected = (False, False, False, False)
    _call(
        tmp_path,
        all_df=_all_df(n_channels=2, pathogen_chan=1),
        track_df=_track_df(infected=infected),
        per_well_tracks={"w": _tracks(infected=infected)},
    )
    _fig, axes = panels[0]
    assert "Infected (nan um/s)" in axes[4].texts[0].get_text()
    assert axes[5].get_title() == "Infected\n(n=0, v=nan um/s)"
    assert axes[6].get_title().startswith("Uninfected\n(n=4,")


def test_all_infected_cells_produce_single_violin_group(tmp_path, panels):
    _call(tmp_path, all_df=_all_df(n_channels=2, pathogen_chan=1, all_infected=True))
    _fig, axes = panels[0]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Inf"]
    assert list(axes[0].get_xticks()) == [0]
