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



def _drive(tmp_path, capsys, over):
    """Run the panel builder and hand back (out_dir, printed_text)."""
    out = _call(tmp_path, **over)
    return out, capsys.readouterr().out


def _write_png(path):
    plt.imsave(str(path), np.linspace(0, 1, 64).reshape(8, 8), cmap="gray")
    return str(path)


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


def _with_prob(**kw):
    df = _all_df(n_channels=2, pathogen_chan=1, **kw)
    df["infection_prob"] = np.linspace(0.05, 0.95, len(df))
    return df


# ---------------------------------------------------------------------------
# helpers: colour map and the three-axis results figure
# ---------------------------------------------------------------------------

def _check_cmap_has_one_colour_per_label_plus_black_background():
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
    return cmap.N


def _check_cmap_of_a_background_only_mask_is_one_black_entry():
    from spacr.timelapse import _generate_mask_random_cmap

    cmap = _generate_mask_random_cmap(np.zeros((5, 5), dtype=np.int32))
    assert cmap.N == 1
    assert np.array_equal(np.asarray(cmap.colors)[0], np.array([0.0, 0.0, 0.0, 1.0]))
    return cmap.N


def _check_results_figure_layout():
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
        return len(fig.axes), tuple(fig.get_size_inches())
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "check, expected",
    [
        pytest.param(_check_cmap_has_one_colour_per_label_plus_black_background, 4, id="cmap-one-colour-per-label"),
        pytest.param(_check_cmap_of_a_background_only_mask_is_one_black_entry, 1, id="cmap-background-only"),
        pytest.param(_check_results_figure_layout, (3, (7.0, 6.0)), id="results-figure-layout"),
    ],
)
def test_colour_map_and_results_figure_helpers_have_the_documented_shape(check, expected):
    assert check() == expected


# ---------------------------------------------------------------------------
# the panel builder prints a reason and writes nothing when it cannot draw
# ---------------------------------------------------------------------------

_EMPTY_MSG = "No data for panel 'mask_labels'"

_REFUSAL_CASES = [
    # (override builder, expected message, directory created?)
    pytest.param(lambda: {"all_df": pd.DataFrame()}, _EMPTY_MSG, False, id="empty-all-df"),
    pytest.param(lambda: {"track_df": pd.DataFrame()}, _EMPTY_MSG, False, id="empty-track-df"),
    pytest.param(lambda: {"per_well_tracks": {}}, _EMPTY_MSG, False, id="no-tracks"),
    pytest.param(lambda: {"all_df": _all_df().drop(columns=["wellID"])},
                 "Missing 'plateID'/'wellID' columns", True, id="missing-plate-well-columns"),
    pytest.param(lambda: {"per_well_tracks": {"other": _tracks(well="B02")},
                          "track_df": _track_df(well="B02")},
                 "No data for plate=plate1, well=A01; skipping.", True, id="well-without-tracks"),
    pytest.param(lambda: {"all_df": _all_df(n_channels=1, intensity_prefix="bogus_intensity_ch"),
                          "settings": {"pathogen_channel": None}, "n_channels": 1},
                 "No cell_mean_intensity_ch* columns", True, id="well-without-intensity-columns"),
]


@pytest.mark.parametrize("build_over, message, dir_made", _REFUSAL_CASES)
def test_panel_prints_a_reason_and_writes_no_pdf_when_it_cannot_draw(
        tmp_path, capsys, build_over, message, dir_made):
    out, msg = _drive(tmp_path, capsys, build_over())
    assert message in msg
    # the early-return guards fire before os.makedirs; the later ones after it
    assert os.path.isdir(out) is dir_made
    assert _pdfs(out) == []


# ---------------------------------------------------------------------------
# mask panel: layout, units, file names, per-axis fallbacks
# ---------------------------------------------------------------------------

def _check_full_layout_and_filename(out, panels):
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


def _check_pixel_units_when_no_calibration(out, panels):
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


def _check_two_wells_emit_two_pdfs(out, panels):
    assert _pdfs(out) == ["plate1_A01.pdf", "plate1_B02.pdf"]
    assert len(panels) == 2


def _check_unknown_label_tag_uses_fallback_filename(out, panels):
    assert _pdfs(out) == ["plate1_A01_weird_tag_histogram.pdf"]
    fig, _axes = panels[0]
    # panel_label falls back to the raw tag
    assert "– weird_tag labels – method=histogram" in fig._suptitle.get_text()


def _check_blank_strategy_reports_method_none(out, panels):
    assert _pdfs(out) == ["plate1_A01.pdf"]
    fig, _axes = panels[0]
    assert "method=none" in fig._suptitle.get_text()


def _check_violin_axis_hidden_when_value_column_is_all_nan(out, panels):
    _fig, axes = panels[0]
    assert axes[2].get_visible() is False  # the p75 axis
    assert axes[1].get_visible() is True


def _check_violin_axis_not_anchored_at_zero_for_negative_values(out, panels):
    _fig, axes = panels[0]
    ax_ratio = axes[3]
    assert ax_ratio.get_visible() is True
    assert ax_ratio.get_ylim()[0] < 0


def _check_all_tracks_axis_hidden_when_every_track_is_too_short(out, panels):
    assert _pdfs(out) == ["plate1_A01.pdf"]
    _fig, axes = panels[0]
    ax_all = axes[4]
    assert ax_all.get_visible() is False
    assert len(ax_all.lines) == 0
    # the origin plots still render, but with zero tracks
    assert axes[5].get_title() == "Infected\n(n=0, v=1.00 um/s)"
    assert axes[6].get_title() == "Uninfected\n(n=0, v=1.50 um/s)"


def _check_tracks_shorter_than_two_points_are_skipped(out, panels):
    _fig, axes = panels[0]
    assert len(axes[4].lines) == 3  # the 1-point track is dropped
    assert axes[5].get_title().startswith("Infected\n(n=1,")


def _check_all_uninfected_tracks_report_nan_infected_velocity(out, panels):
    _fig, axes = panels[0]
    assert "Infected (nan um/s)" in axes[4].texts[0].get_text()
    assert axes[5].get_title() == "Infected\n(n=0, v=nan um/s)"
    assert axes[6].get_title().startswith("Uninfected\n(n=4,")


def _check_all_infected_cells_produce_single_violin_group(out, panels):
    _fig, axes = panels[0]
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Inf"]
    assert list(axes[0].get_xticks()) == [0]


def _short_tracks():
    return {"per_well_tracks": {"w": _tracks(length=1)}}


def _one_point_track():
    tracks = _tracks()
    tracks[0]["x_px"] = np.array([5.0])
    tracks[0]["y_px"] = np.array([6.0])
    return {"per_well_tracks": {"w": tracks}}


def _all_uninfected():
    infected = (False, False, False, False)
    return {"all_df": _all_df(n_channels=2, pathogen_chan=1),
            "track_df": _track_df(infected=infected),
            "per_well_tracks": {"w": _tracks(infected=infected)}}


def _nan_p75():
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["cell_p75_intensity_ch1"] = np.nan
    return {"all_df": df}


def _negative_ratio():
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["pathogen_mean_intensity_ch1"] = -df["pathogen_mean_intensity_ch1"]
    return {"all_df": df}


def _two_wells():
    all_df = pd.concat(
        [_all_df(well="A01", pathogen_chan=1), _all_df(well="B02", pathogen_chan=1)],
        ignore_index=True,
    )
    track_df = pd.concat([_track_df(well="A01"), _track_df(well="B02")], ignore_index=True)
    return {"all_df": all_df, "track_df": track_df,
            "per_well_tracks": {"a": _tracks(well="A01"), "b": _tracks(well="B02")}}


_MASK_PANEL_CASES = [
    pytest.param(lambda: {"settings": {"pathogen_channel": 1, "motility_xlim": (-50, 50),
                                       "motility_ylim": (-40, 40)}},
                 _check_full_layout_and_filename, (["plate1_A01.pdf"], [7]), id="full-layout-and-filename"),
    pytest.param(lambda: {"all_df": _all_df(n_channels=2, pathogen_chan=None),
                          "settings": {"pathogen_channel": None},
                          "pixels_per_um": None, "seconds_per_frame": None},
                 _check_pixel_units_when_no_calibration, (["plate1_A01.pdf"], [5]), id="pixel-units-without-calibration"),
    pytest.param(_two_wells, _check_two_wells_emit_two_pdfs, (["plate1_A01.pdf", "plate1_B02.pdf"], [7, 7]), id="two-wells-two-pdfs"),
    pytest.param(lambda: {"label_tag": "weird_tag",
                          "settings": {"pathogen_channel": 1,
                                       "infection_intensity_strategy": "histogram"}},
                 _check_unknown_label_tag_uses_fallback_filename, (["plate1_A01_weird_tag_histogram.pdf"], [7]), id="unknown-label-tag-fallback-name"),
    pytest.param(lambda: {"settings": {"pathogen_channel": 1, "infection_intensity_strategy": ""}},
                 _check_blank_strategy_reports_method_none, (["plate1_A01.pdf"], [7]), id="blank-strategy-is-method-none"),
    pytest.param(_nan_p75, _check_violin_axis_hidden_when_value_column_is_all_nan, (["plate1_A01.pdf"], [7]), id="violin-hidden-all-nan"),
    pytest.param(_negative_ratio, _check_violin_axis_not_anchored_at_zero_for_negative_values, (["plate1_A01.pdf"], [7]),
                 id="violin-not-anchored-for-negative-values"),
    pytest.param(_short_tracks, _check_all_tracks_axis_hidden_when_every_track_is_too_short, (["plate1_A01.pdf"], [7]),
                 id="all-tracks-axis-hidden-when-too-short"),
    pytest.param(_one_point_track, _check_tracks_shorter_than_two_points_are_skipped, (["plate1_A01.pdf"], [7]), id="one-point-track-skipped"),
    pytest.param(_all_uninfected, _check_all_uninfected_tracks_report_nan_infected_velocity, (["plate1_A01.pdf"], [7]),
                 id="all-uninfected-nan-infected-velocity"),
    pytest.param(lambda: {"all_df": _all_df(n_channels=2, pathogen_chan=1, all_infected=True)},
                 _check_all_infected_cells_produce_single_violin_group, (["plate1_A01.pdf"], [7]), id="all-infected-single-violin"),
]


@pytest.mark.parametrize("build_over, check, expected", _MASK_PANEL_CASES)
def test_mask_panel_lays_out_axes_units_titles_and_file_names_from_its_inputs(
        tmp_path, panels, capsys, build_over, check, expected):
    out, _msg = _drive(tmp_path, capsys, build_over())
    check(out, panels)
    assert (_pdfs(out), [len(a) for _f, a in panels]) == expected


# ---------------------------------------------------------------------------
# mask panel: the embedded QC picture
# ---------------------------------------------------------------------------

def _qc_over(tmp_path, **settings):
    base = {"pathogen_channel": 1}
    base.update(settings)
    return {"settings": base}


def _embed_case(panel_type):
    def build(tmp_path):
        png = _write_png(tmp_path / "qc.png")
        return _qc_over(tmp_path, infection_intensity_qc_panel_path=png,
                        infection_intensity_qc_panel_type=panel_type)
    return build


def _check_embedded(expected):
    def check(out, panels, msg):
        assert _pdfs(out) == ["plate1_A01.pdf"]
        _fig, axes = panels[0]
        assert len(axes) == 8  # extra QC axis appended
        ax_qc = axes[7]
        assert ax_qc.get_title() == expected
        assert len(ax_qc.images) == 1
        assert not ax_qc.axison
    return check


def _unreadable_png(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"definitely not a png")
    return _qc_over(tmp_path, infection_intensity_qc_panel_path=str(bad),
                    infection_intensity_qc_panel_type="histogram")


def _check_unreadable_png_hides_axis(out, panels, msg):
    assert _pdfs(out) == ["plate1_A01.pdf"]
    assert "Could not embed QC plot" in msg
    _fig, axes = panels[0]
    assert len(axes) == 8
    assert axes[7].get_visible() is False


def _graphs_disabled(tmp_path):
    png = _write_png(tmp_path / "qc.png")
    return _qc_over(tmp_path, infection_intensity_qc_graphs=False,
                    infection_intensity_qc_panel_path=png)


def _missing_png(tmp_path):
    return _qc_over(tmp_path, infection_intensity_qc_panel_path=str(tmp_path / "nope.png"))


def _check_no_qc_axis(out, panels, msg):
    _fig, axes = panels[0]
    assert len(axes) == 7  # no QC axis allocated


_EMBED_CASES = [
    pytest.param(_embed_case("histogram"), _check_embedded("Intensity histogram"), (["plate1_A01.pdf"], [8]), id="histogram-title"),
    pytest.param(_embed_case("pca"), _check_embedded("PCA/UMAP clustering"), (["plate1_A01.pdf"], [8]), id="pca-title"),
    pytest.param(_embed_case("xgboost"), _check_embedded("XGBoost feature importance"), (["plate1_A01.pdf"], [8]), id="xgboost-title"),
    pytest.param(_embed_case("something_else"), _check_embedded("Infection QC"), (["plate1_A01.pdf"], [8]), id="other-type-generic-title"),
    pytest.param(_unreadable_png, _check_unreadable_png_hides_axis, (["plate1_A01.pdf"], [8]), id="unreadable-png-hides-axis"),
    pytest.param(_graphs_disabled, _check_no_qc_axis, (["plate1_A01.pdf"], [7]), id="graphs-disabled-no-axis"),
    pytest.param(_missing_png, _check_no_qc_axis, (["plate1_A01.pdf"], [7]), id="missing-png-no-axis"),
]


@pytest.mark.parametrize("build_over, check, expected", _EMBED_CASES)
def test_mask_panel_embeds_the_qc_picture_or_hides_its_axis(
        tmp_path, panels, capsys, build_over, check, expected):
    out, msg = _drive(tmp_path, capsys, build_over(tmp_path))
    check(out, panels, msg)
    assert (_pdfs(out), [len(a) for _f, a in panels]) == expected


# ---------------------------------------------------------------------------
# adjusted panel: histogram and embedding strategies
# ---------------------------------------------------------------------------

def _adj(strategy, **extra):
    s = {"pathogen_channel": 1, "infection_intensity_strategy": strategy}
    s.update(extra)
    return s


def _check_hist_from_well_dataframe(out, panels, msg):
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


def _check_hist_from_payload(out, panels, msg):
    _fig, axes = panels[0]
    ax_hist = axes[7]
    assert ax_hist.get_xlabel() == "my_intensity"
    assert len(ax_hist.patches) == 6  # 3 bins x 2 histograms
    assert len(ax_hist.lines) == 1  # threshold


def _check_hist_invalid_payload(out, panels, msg):
    assert "Histogram payload invalid" in msg
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False


def _check_hidden_and_empty(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def _check_hist_frame_payload_without_column(out, panels, msg):
    _fig, axes = panels[0]
    assert len(axes) == 8
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def _check_hist_first_channel_fallback(out, panels, msg):
    _fig, axes = panels[0]
    assert len(axes) == 6  # 2 channels + 3 motility + 1 QC
    assert axes[5].get_xlabel() == "cell_mean_intensity_ch0"
    assert len(axes[5].patches) == 8
    # no threshold available -> no axvline
    assert len(axes[5].lines) == 0


def _embedding_check(strategy, method):
    def check(out, panels, msg):
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
    return check


def _embedding_over(strategy, method):
    rng = np.random.default_rng(0)
    return {"label_tag": "adjusted_labels",
            "settings": _adj(strategy, infection_pca_data={
                "coords": rng.normal(size=(20, 2)),
                "labels": np.arange(20) % 2 == 0,
                "method_label": method})}


def _check_embedding_defaults_to_pca_label(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_xlabel() == "PCA 1"


def _check_embedding_invalid_payload(out, panels, msg):
    assert "PCA/embedding payload invalid" in msg
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False


def _check_embedding_one_dimensional(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].collections) == 0


def _check_tsne_without_payload(out, panels, msg):
    assert _pdfs(out) == ["plate1_A01_tsne_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 7


def _nan_hist_over():
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["nan_intensity"] = np.nan
    return {"all_df": df, "label_tag": "adjusted_labels",
            "settings": _adj("histogram", infection_hist_intensity_col="nan_intensity")}


_ADJUSTED_CASES = [
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("histogram", infection_intensity_n_bins=8,
                                           infection_intensity_threshold=115.0)},
                 _check_hist_from_well_dataframe, (["plate1_A01_histogram_adjusted.pdf"], [8]), id="histogram-from-well-dataframe"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("histogram", infection_hist_data={
                              "intensities_inf": [1.0, 2.0, 3.0],
                              "intensities_uninf": [0.1, 0.2],
                              "bin_edges": [0.0, 1.0, 2.0, 3.0],
                              "thr_val": 1.5,
                              "intensity_col": "my_intensity"})},
                 _check_hist_from_payload, (["plate1_A01_histogram_adjusted.pdf"], [8]), id="histogram-from-settings-payload"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("histogram", infection_hist_data={"intensities_inf": [1.0]})},
                 _check_hist_invalid_payload, (["plate1_A01_histogram_adjusted.pdf"], [8]), id="histogram-invalid-payload-hides-axis"),
    pytest.param(_nan_hist_over, _check_hidden_and_empty, (["plate1_A01_histogram_adjusted.pdf"], [8]), id="histogram-all-nan-hides-axis"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("histogram", infection_hist_data=pd.DataFrame(
                              {"plateID": [PLATE] * 2, "unrelated": [1.0, 2.0]}))},
                 _check_hist_frame_payload_without_column, (["plate1_A01_histogram_adjusted.pdf"], [8]), id="histogram-frame-payload-without-column"),
    pytest.param(lambda: {"all_df": _all_df(n_channels=2, pathogen_chan=None),
                          "label_tag": "adjusted_labels",
                          "settings": {"pathogen_channel": None,
                                       "infection_intensity_strategy": "histogram",
                                       "infection_intensity_n_bins": 4}},
                 _check_hist_first_channel_fallback, (["plate1_A01_histogram_adjusted.pdf"], [6]), id="histogram-without-pathogen-channel"),
    pytest.param(lambda: _embedding_over("pca", "PCA"), _embedding_check("pca", "PCA"), (["plate1_A01_pca_adjusted.pdf"], [8]), id="embedding-pca"),
    pytest.param(lambda: _embedding_over("umap", "UMAP"), _embedding_check("umap", "UMAP"), (["plate1_A01_umap_adjusted.pdf"], [8]), id="embedding-umap"),
    pytest.param(lambda: _embedding_over("tsne", "t-SNE"), _embedding_check("tsne", "t-SNE"), (["plate1_A01_tsne_adjusted.pdf"], [8]), id="embedding-tsne"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("pca", infection_pca_data={
                              "coords": [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                              "labels": [True, False, False]})},
                 _check_embedding_defaults_to_pca_label, (["plate1_A01_pca_adjusted.pdf"], [8]), id="embedding-default-pca-label"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("pca", infection_pca_data={"coords": [[0.0, 1.0]]})},
                 _check_embedding_invalid_payload, (["plate1_A01_pca_adjusted.pdf"], [8]), id="embedding-invalid-payload-hides-axis"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _adj("umap", infection_pca_data={
                              "coords": [0.0, 1.0, 2.0], "labels": [True, False, True]})},
                 _check_embedding_one_dimensional, (["plate1_A01_umap_adjusted.pdf"], [8]), id="embedding-one-dimensional-coords"),
    pytest.param(lambda: {"label_tag": "adjusted_labels", "settings": _adj("tsne")},
                 _check_tsne_without_payload, (["plate1_A01_tsne_adjusted.pdf"], [7]), id="embedding-without-payload-no-axis"),
]


@pytest.mark.parametrize("build_over, check, expected", _ADJUSTED_CASES)
def test_adjusted_panel_draws_the_histogram_or_embedding_axis_or_hides_it(
        tmp_path, panels, capsys, build_over, check, expected):
    out, msg = _drive(tmp_path, capsys, build_over())
    check(out, panels, msg)
    assert (_pdfs(out), [len(a) for _f, a in panels]) == expected


# ---------------------------------------------------------------------------
# adjusted panel: XGBoost strategy
# ---------------------------------------------------------------------------

def _check_xgb_prob_and_importance(out, panels, msg):
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


def _check_xgb_configured_column(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_visible() is True
    assert len(axes[7].patches) == 40


def _check_xgb_prob_hidden_importance_drawn(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert axes[8].get_visible() is True  # importance axis still drawn


def _check_xgb_nan_probabilities(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[7].get_visible() is False
    assert len(axes[7].patches) == 0


def _check_xgb_single_class(out, panels, msg):
    _fig, axes = panels[0]
    assert len(axes[7].patches) == 20  # a single 20-bin histogram
    # violin plots collapse to the infected group only
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Inf"]


def _check_xgb_invalid_importance(out, panels, msg):
    assert "XGB importance payload invalid" in msg
    _fig, axes = panels[0]
    assert axes[8].get_visible() is False


def _check_xgb_empty_feature_list(out, panels, msg):
    _fig, axes = panels[0]
    assert axes[8].get_visible() is False
    assert len(axes[8].patches) == 0


def _check_xgb_without_payload(out, panels, msg):
    assert _pdfs(out) == ["plate1_A01_xgboost_adjusted.pdf"]
    _fig, axes = panels[0]
    assert len(axes) == 7


def _check_xgb_graphs_disabled(out, panels, msg):
    _fig, axes = panels[0]
    assert len(axes) == 7  # QC axes suppressed entirely


def _my_proba():
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["my_proba"] = np.linspace(0.1, 0.9, len(df))
    return {"all_df": df, "label_tag": "adjusted_labels",
            "settings": _xgb_settings(infection_xgb_proba_column="my_proba")}


def _nan_proba():
    df = _all_df(n_channels=2, pathogen_chan=1)
    df["infection_prob"] = np.nan
    return {"all_df": df, "label_tag": "adjusted_labels", "settings": _xgb_settings()}


_XGB_CASES = [
    pytest.param(lambda: {"all_df": _with_prob(), "label_tag": "adjusted_labels",
                          "settings": _xgb_settings()},
                 _check_xgb_prob_and_importance, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="probability-and-importance-axes"),
    pytest.param(_my_proba, _check_xgb_configured_column, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="configured-probability-column"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _xgb_settings(infection_xgb_proba_column="absent_col")},
                 _check_xgb_prob_hidden_importance_drawn, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="absent-probability-column"),
    pytest.param(_nan_proba, _check_xgb_nan_probabilities, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="all-nan-probabilities"),
    pytest.param(lambda: {"all_df": _with_prob(all_infected=True), "label_tag": "adjusted_labels",
                          "settings": _xgb_settings()},
                 _check_xgb_single_class, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="single-class-probabilities"),
    pytest.param(lambda: {"all_df": _with_prob(), "label_tag": "adjusted_labels",
                          "settings": _xgb_settings(infection_xgb_importance={"wrong_key": 1})},
                 _check_xgb_invalid_importance, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="invalid-importance-payload"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _xgb_settings(infection_xgb_importance={
                              "feature_names": [], "feature_importances": []})},
                 _check_xgb_empty_feature_list, (["plate1_A01_xgboost_adjusted.pdf"], [9]), id="empty-feature-list"),
    pytest.param(lambda: {"label_tag": "adjusted_labels", "settings": _adj("xgboost")},
                 _check_xgb_without_payload, (["plate1_A01_xgboost_adjusted.pdf"], [7]), id="no-payload-no-qc-axes"),
    pytest.param(lambda: {"label_tag": "adjusted_labels",
                          "settings": _xgb_settings(infection_intensity_qc_graphs=False)},
                 _check_xgb_graphs_disabled, (["plate1_A01_xgboost_adjusted.pdf"], [7]), id="qc-graphs-disabled"),
]


@pytest.mark.parametrize("build_over, check, expected", _XGB_CASES)
def test_adjusted_panel_draws_the_xgboost_axes_or_hides_them(
        tmp_path, panels, capsys, build_over, check, expected):
    out, msg = _drive(tmp_path, capsys, build_over())
    check(out, panels, msg)
    assert (_pdfs(out), [len(a) for _f, a in panels]) == expected
