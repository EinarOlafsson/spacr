"""QC presentation preserves measured values, independent losses and field normalization."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from spacr import illumination, sequencing_qc
from spacr.figures import style


@pytest.fixture(params=["print", "screen"])
def target(request, monkeypatch):
    """Exercise both declared output themes without consulting user preferences."""
    monkeypatch.setattr(style, "theme_target", lambda: request.param)
    monkeypatch.setattr(sequencing_qc, "theme_target", lambda: request.param)
    original_rc = style.rc
    ground, ink = (("#FFFFFF", "#000000") if request.param == "print"
                   else ("#181818", "#F0F0F0"))

    def themed_rc(target, **kwargs):
        """Supply an actual contrasting ground and ink through the shared style API."""
        return original_rc(target, **dict(kwargs, ground=ground, ink=ink, line=ink))

    monkeypatch.setattr(style, "rc", themed_rc)
    with style.pipeline_figures():
        yield request.param
    plt.close("all")


def barcode_inputs():
    """Provide unequal depths, natural position order and overlapping barcode failures."""
    counts = pd.DataFrame({"count": [1, 2, 3, 4, 8, 20]})
    per_well = pd.DataFrame({"reads": [1., 2., 3., 4., 8., 20.]})
    per_well.attrs["source"] = "synthetic unchanged input"
    starved = per_well.iloc[:1].copy()
    starved.attrs["cutoff"] = 1.5
    positions = pd.DataFrame({"axis": ["column"] * 3, "label": ["c10", "c2", "c1"],
                              "ratio_to_plate": [1.4, .8, 1.], "flagged": [True, False, False]})
    depth = {"reads_per_grna": pd.Series([0., 2., 4., 8.]), "gini": .45,
             "skew_ratio": 4., "dropout_fraction": .25}
    unmapped = {"per_field": {"columnID": .2, "rowID": .3, "grna_name": .1},
                "unmapped_fraction": .4, "total_reads": 100}
    return counts, dict(per_well=per_well, starved=starved, positions=positions,
                        depth=depth, unmapped=unmapped)


def test_barcode_qc_keeps_bins_lorenz_order_and_overlapping_losses(target):
    """No category count is a replicate count and joint losses are never summed or stacked."""
    counts, arguments = barcode_inputs()
    original = deepcopy(arguments)
    figure = sequencing_qc.plot_barcode_qc(counts, **arguments)
    reads, positions, coverage, fate = figure.axes
    assert figure.get_facecolor() == reads.get_facecolor()
    assert reads.get_position().y1 == pytest.approx(positions.get_position().y1)
    assert coverage.get_position().y1 == pytest.approx(fate.get_position().y1)
    assert coverage.get_position().y1 < reads.get_position().y0
    expected_counts, expected_edges = np.histogram(arguments["per_well"]["reads"], bins=5)
    np.testing.assert_array_equal([patch.get_height() for patch in reads.patches], expected_counts)
    oracle_figure, oracle_axis = plt.subplots()
    try:
        actual_counts, actual_edges, oracle_patches = oracle_axis.hist(
            arguments["per_well"]["reads"].to_numpy(float), bins=5)
        np.testing.assert_array_equal(actual_counts, expected_counts)
        np.testing.assert_array_equal(actual_edges, expected_edges)
        np.testing.assert_array_equal(
            [(patch.get_x(), patch.get_width(), patch.get_height()) for patch in reads.patches],
            [(patch.get_x(), patch.get_width(), patch.get_height()) for patch in oracle_patches])
    finally:
        plt.close(oracle_figure)
    assert [label.get_text() for label in positions.get_xticklabels()] == ["c1", "c2", "c10"]
    np.testing.assert_array_equal([patch.get_height() for patch in positions.patches], [1., .8, 1.4])
    np.testing.assert_array_equal(coverage.lines[0].get_xdata(), [0., .25, .5, .75, 1.])
    np.testing.assert_array_equal(coverage.lines[0].get_ydata(), [0., 0., 2/14, 6/14, 1.])
    np.testing.assert_array_equal([patch.get_height() for patch in fate.patches], [20., 30., 10., 40.])
    assert all(patch.get_y() == 0 for patch in fate.patches)
    for axis in (positions, fate):
        spec = axis._spacr_style_spec
        assert dict(spec.comparison)["representation"] == "estimates"
        assert spec.sample_counts == ()
        assert axis.get_position().width * figure.get_figwidth() == pytest.approx(spec.conditions * .22)
        assert all(patch.get_width() == pytest.approx(.6) for patch in axis.patches)
    for axis in (reads, coverage):
        assert axis.get_position().width * figure.get_figwidth() == pytest.approx(
            axis.get_position().height * figure.get_figheight())
    for axis in figure.axes:
        assert axis._spacr_style_spec.target == target
        assert all(artist.get_alpha() == .6 for artist in (*axis.lines, *axis.patches))
        for line in axis.lines:
            if line.get_gid() == "spacr-reference":
                assert line.get_color() == "#000000"
                assert bool(line.get_path_effects()) == (target == "screen")
        legend = axis.get_legend()
        if legend is not None:
            for handle, label in zip(legend.legend_handles, legend.get_texts()):
                if any(line.get_gid() == "spacr-reference" and line.get_label() == label.get_text()
                       for line in axis.lines):
                    assert bool(handle.get_path_effects()) == (target == "screen")
    for key in ("per_well", "starved", "positions"):
        pd.testing.assert_frame_equal(arguments[key], original[key])
        assert arguments[key].attrs == original[key].attrs
    pd.testing.assert_series_equal(arguments["depth"]["reads_per_grna"], original["depth"]["reads_per_grna"])
    assert arguments["unmapped"] == original["unmapped"]


def test_barcode_qc_missing_evidence_does_not_invent_a_joint_loss_or_distribution(target):
    """Empty positions and coverage stay empty, and an upper bound is not a joint fraction."""
    counts, arguments = barcode_inputs()
    arguments["positions"] = arguments["positions"].iloc[:0]
    arguments["depth"]["reads_per_grna"] = pd.Series([], dtype=float)
    arguments["starved"].attrs.clear()
    arguments["unmapped"].pop("unmapped_fraction")
    arguments["unmapped"]["unmapped_upper_bound"] = .6
    figure = sequencing_qc.plot_barcode_qc(counts, **arguments)
    assert not figure.axes[1].axison and not figure.axes[2].axison
    assert not figure.axes[0].lines
    assert figure.axes[0].get_legend() is None
    assert not figure.axes[2].lines
    np.testing.assert_array_equal([patch.get_height() for patch in figure.axes[3].patches], [20., 30., 10.])


def test_threshold_sweep_preserves_all_four_series_and_original_scale_limits(target):
    """Styling cannot change the threshold, retention/collision conversion or logarithmic scales."""
    sweep = pd.DataFrame({"threshold": [.001, .01, .1], "grnas_per_well": [30., 5., 1.],
                          "grnas_per_well_retained": [31., 6., 2.],
                          "well_retention": [1., .8, .5], "collision_rate": [.9, .2, 0.]})
    sweep.attrs["units"] = "fractions before display conversion"
    original = sweep.copy(deep=True)
    choice = sequencing_qc.ThresholdChoice(.01, 5., 5., "median", 6, True, 3, .005, .02)
    figure = sequencing_qc.plot_threshold_sweep(sweep, choice)
    top, bottom = figure.axes
    assert figure.get_facecolor() == top.get_facecolor() == bottom.get_facecolor()
    for axis, columns, multiplier in ((top, ["grnas_per_well", "grnas_per_well_retained"], 1),
                                       (bottom, ["well_retention", "collision_rate"], 100)):
        for line, column in zip(axis.lines[:2], columns):
            np.testing.assert_array_equal(line.get_xdata(), sweep["threshold"])
            np.testing.assert_array_equal(line.get_ydata(), multiplier * sweep[column])
        np.testing.assert_array_equal(axis.lines[-1].get_xdata(), [.01, .01])
        assert not axis.texts
        assert axis.get_xscale() == "log"
    assert top.get_yscale() == "symlog" and top.get_ylim()[0] == 0
    assert bottom.get_ylim() == (-2, 102)
    pd.testing.assert_frame_equal(sweep, original)
    assert sweep.attrs == original.attrs


@pytest.mark.parametrize("creator", ["barcode", "sweep"])
@pytest.mark.parametrize("ground,ink", [("#FFFFFF", "#000000"), ("#181818", "#F0F0F0")])
def test_saved_qc_canvas_matches_actual_styled_ground(creator, ground, ink, monkeypatch, tmp_path):
    """The normal screen writer must save opaque panel-matching outer pixels after style exit."""
    from matplotlib.colors import to_rgba
    from PIL import Image
    from spacr import plot

    def saved_choices(kind=None):
        """Supply explicit saved theme roles without touching user preferences."""
        return {"background": ground, "foreground": ink}

    monkeypatch.setattr(style, "_preference_deltas", saved_choices)
    monkeypatch.setattr(style, "chosen_ink", lambda: ink)
    monkeypatch.setattr(style, "chosen_line_ink", lambda: ink)
    monkeypatch.setattr(sequencing_qc, "theme_target", lambda: "screen")
    normal_writer = plot.save_figure
    written = []

    def save(figure, path, **kwargs):
        """Exercise the real writer with a fixed bounded PNG format and screen appearance."""
        result = normal_writer(figure, path, **dict(kwargs, fmt="png", dpi=80,
            save_mode="screen", announce_colours=False))
        written.append(result)
        return result

    monkeypatch.setattr(plot, "save_figure", save)
    with plt.rc_context({"savefig.facecolor": "auto", "savefig.edgecolor": "auto"}), style.pipeline_figures():
        if creator == "barcode":
            counts, arguments = barcode_inputs()
            figure = sequencing_qc.plot_barcode_qc(counts, **arguments, dst=str(tmp_path))
        else:
            sweep = pd.DataFrame({"threshold": [.001, .01, .1], "grnas_per_well": [30., 5., 1.],
                "grnas_per_well_retained": [31., 6., 2.], "well_retention": [1., .8, .5],
                "collision_rate": [.9, .2, 0.]})
            choice = sequencing_qc.ThresholdChoice(.01, 5., 5., "median", 6, True, 3, .005, .02)
            figure = sequencing_qc.plot_threshold_sweep(sweep, choice, dst=str(tmp_path))
    try:
        assert len(written) == 1
        assert figure.get_facecolor() == to_rgba(ground)
        with Image.open(written[0]) as image:
            pixels = np.asarray(image.convert("RGBA"))
        expected = np.rint(np.asarray(to_rgba(ground)) * 255).astype(np.uint8)
        for rows in (slice(0, 3), slice(-3, None)):
            for columns in (slice(0, 3), slice(-3, None)):
                np.testing.assert_array_equal(pixels[rows, columns], np.broadcast_to(expected, (3, 3, 4)))
    finally:
        plt.close(figure)


def test_illumination_qc_keeps_array_dtypes_radial_normalization_and_display_limits(target, monkeypatch, tmp_path):
    """Flat-field extrema are labelled truthfully while the source correction and residual scale stay exact."""
    from spacr import plot

    captured = []
    monkeypatch.setattr(plot, "save_figure", lambda figure, path, **kwargs: captured.append(figure) or path)
    observed = np.linspace(.6, 1.4, 48, dtype=np.float32).reshape(6, 8)
    plane = np.linspace(.8, 1.2, 48, dtype=np.float32).reshape(6, 8)
    corrected = observed / plane
    panels = {0: (plane, observed, corrected)}
    originals = [value.copy() for value in panels[0]]
    metrics = {0: {"gain_min": float(plane.min()), "gain_max": float(plane.max()),
                   "nonuniformity_pct": 30., "slope_before": .3, "slope_after": .1,
                   "bias_removed_pct": 200/3, "n_fields": 4}}
    original_metrics = deepcopy(metrics)
    result = illumination._write_qc_figure("p1", SimpleNamespace(shape=(6, 8)), [0],
        panels, metrics, str(tmp_path), 1, stage="measurement_qc")
    assert result.endswith("illumination_qc_measurement_qc_p1.png")
    figure = captured[0]
    maps = [axis for axis in figure.axes if axis.images]
    trend = next(axis for axis in figure.axes if len(axis.lines) == 3)
    assert figure.get_facecolor() == maps[0].get_facecolor()
    assert "Slope before" in trend.get_legend().get_title().get_text()
    assert "Slope after" in trend.get_legend().get_title().get_text()
    assert "→" not in trend.get_legend().get_title().get_text()
    np.testing.assert_array_equal(maps[0].images[0].get_array(), plane)
    np.testing.assert_array_equal(maps[1].images[0].get_array(), corrected)
    assert maps[0].images[0].get_array().dtype == plane.dtype
    assert maps[1].images[0].get_array().dtype == corrected.dtype
    assert maps[0].images[0].get_clim() == (float(plane.min()), float(plane.max()))
    span = float(np.max(np.abs(observed - 1.)))
    assert maps[1].images[0].get_clim() == (1-span, 1+span)
    assert maps[1].images[0].norm(1.) == .5
    radius, before = illumination._radial_profile(observed, 1, (6, 8))
    _, after = illumination._radial_profile(corrected, 1, (6, 8))
    np.testing.assert_array_equal(trend.lines[0].get_xdata(), radius)
    np.testing.assert_array_equal(trend.lines[0].get_ydata(), before)
    np.testing.assert_array_equal(trend.lines[1].get_ydata(), after)
    assert trend.lines[0].get_color() == style.ROLES["control"]
    assert trend.lines[2].get_color() == "#000000"
    assert trend.lines[2].get_alpha() == .6
    assert bool(trend.lines[2].get_path_effects()) == (target == "screen")
    assert "Flat field" in maps[0].get_legend().get_title().get_text()
    assert "gain" not in maps[0].get_legend().get_title().get_text()
    for axis in maps:
        assert dict(axis._spacr_style_spec.comparison)["gamma"] == 1.
        assert dict(axis._spacr_style_spec.comparison)["processing"]
        colourbar_axes = style._axis_colourbars(axis)
        assert len(colourbar_axes) == 1
        colourbar = colourbar_axes[0]._colorbar
        assert colourbar.ax.get_facecolor() == axis.get_facecolor()
        from matplotlib.colors import to_rgba
        foreground = to_rgba(axis.xaxis.label.get_color())
        for text in (colourbar.ax.yaxis.label, *colourbar.ax.get_yticklabels()):
            assert to_rgba(text.get_color()) == foreground
            assert text.get_alpha() == 1.
    for value, original in zip(panels[0], originals):
        assert value.dtype == original.dtype and value.tobytes() == original.tobytes()
    assert metrics == original_metrics
