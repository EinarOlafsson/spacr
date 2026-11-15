"""Coverage for the recruitment / control / grid-display / residual-stats
helpers in ``spacr.plot`` (lines ~1724-2040).

Everything here is CPU-only and headless: matplotlib runs on Agg, so the
``plt.show()`` calls inside the product code are no-ops and the figures stay
alive in ``plt.get_fignums()``, which is exactly what lets these tests assert
on the *contents* of the figures (bar heights, axis labels, canvas pixels,
text annotations) instead of merely asserting "did not raise".

Symbols covered:
    _plot_recruitment, _plot_controls, _imshow, _imshow_gpu,
    _show_residules, _reg_v_plot

Three parametrised tests carry the contracts: the recruitment and control
figures, the tiled image canvas, and the statistics plots. Each row names a
helper that drives one call and returns what it observed, beside the value
the contract requires.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import re

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import plot as P


@pytest.fixture(autouse=True)
def _close_figs():
    """Fresh figure state per test, and no global rcParams leakage.

    ``_plot_recruitment`` calls ``sns.set_palette`` which mutates the global
    matplotlib rcParams, so snapshot/restore them around every test.
    """
    saved = matplotlib.rcParams.copy()
    plt.close("all")
    yield
    plt.close("all")
    matplotlib.rcParams.update(saved)


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _bind(helper, *args):
    """A fixture-free helper that runs ``helper(*args)``."""
    return lambda: helper(*args)


def _figs():
    """All live figures, in creation order."""
    return [plt.figure(n) for n in plt.get_fignums()]


# ---------------------------------------------------------------------------
# _plot_recruitment
# ---------------------------------------------------------------------------

_RECRUITMENT_EXTRA = [
    "pathogen_channel_1_cytoplasm_mean_ratio",
    "pathogen_channel_1_cytoplasm_q75_ratio",
    "pathogen_channel_1_cytoplasm_periphery_mean_ratio",
    "pathogen_channel_1_cytoplasm_outside_mean_ratio",
    "pathogen_channel_1_cytoplasm_outside_q75_ratio",
]


def _recruitment_df(channel=1, n=24, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        "condition": ["ctrl", "trt"] * (n // 2),
        "pathogen": ["wt", "mut"] * (n // 2),
    }
    for comp in ("cell", "nucleus", "cytoplasm", "pathogen"):
        data[f"{comp}_channel_{channel}_mean_intensity"] = rng.uniform(10, 100, n)
    for col in _RECRUITMENT_EXTRA + ["extra_a", "extra_b"]:
        data[col.replace('channel_1_', f'channel_{channel}_')] = rng.uniform(2, 50, n)
    return pd.DataFrame(data)


def _case_plot_recruitment_default_columns_builds_two_figures():
    """columns=None -> the five hard-coded recruitment columns on a 2x3 grid."""
    df = _recruitment_df(channel=1)

    P._plot_recruitment(df, "test", 1, figuresize=4)

    figs = _figs()
    intensity, grid = figs
    legend = intensity.axes[3].get_legend()
    return {
        # one 1x4 intensity figure + one 2xN recruitment grid
        "figures": len(figs),
        # Figure 1: cell / nucleus / cytoplasm / pathogen mean intensity.
        "intensity_panels": len(intensity.axes),
        "intensity_ylabels": [ax.get_ylabel() for ax in intensity.axes],
        "intensity_xlabels": {ax.get_xlabel() for ax in intensity.axes},
        # Only the last panel keeps a legend.
        "last_panel_has_legend": legend is not None,
        "legend_texts": {t.get_text() for t in legend.get_texts()},
        # Rotated tick labels.
        "rotations": {lbl.get_rotation()
                      for lbl in intensity.axes[0].get_xticklabels()},
        # Figure 2: 5 columns -> ceil(5/2)=3 per row -> 6 axes, last blanked.
        "grid_panels": len(grid.axes),
        "grid_ylabels": [ax.get_ylabel() for ax in grid.axes[:5]],
        "last_grid_axis_on": grid.axes[5].axison,
        # i <= 5 => ylim bottom forced to 1
        "grid_ylim_bottoms": {ax.get_ylim()[0] for ax in grid.axes[:5]},
        # Per-panel legends are removed in the grid.
        "grid_legends_removed": all(ax.get_legend() is None
                                    for ax in grid.axes[:5]),
    }


def _case_a_recruitment_plot_does_not_recolour_the_next_plot():
    """Drawing one figure must not restyle the ones the user draws after it.

    ``_plot_recruitment`` picks its own four colours, which is fine -- but it
    used to install them with ``sns.set_palette``, and that writes matplotlib's
    process-wide colour cycle. Every plot the session drew afterwards came out
    in the recruitment palette: other screens, other modules, and the palette
    the user had chosen in figure preferences, silently overridden by whichever
    figure they happened to open first.

    The bars must still be the recruitment colours -- the fix is where the
    palette is set, not what it is -- so both halves are asserted here.
    """
    before = matplotlib.rcParams["axes.prop_cycle"]

    P._plot_recruitment(_recruitment_df(channel=1), "test", 1, figuresize=4)

    # seaborn draws bars at saturation 0.75, so the expectation is computed
    # through its own desaturate rather than typed as the raw triples.
    import seaborn as sns

    intended = [(55 / 255, 155 / 255, 155 / 255),
                (155 / 255, 55 / 255, 155 / 255)]
    bars = {tuple(np.round(patch.get_facecolor()[:3], 4))
            for patch in _figs()[0].axes[0].patches}
    # The bars lost their own colours if this differs; and a recruitment plot
    # must not leave its palette on matplotlib's global colour cycle, or the
    # next figure of the session inherits it.
    return (bars == {tuple(np.round(sns.desaturate(colour, 0.75), 4))
                     for colour in intended},
            matplotlib.rcParams["axes.prop_cycle"] == before)


def _case_plot_recruitment_extra_columns_widen_grid_and_skip_ylim_after_index_5():
    """Extra user columns are prepended; the 7th panel (i=6) keeps its autoscale."""
    df = _recruitment_df(channel=0)
    user_cols = ["extra_a", "extra_b"]

    P._plot_recruitment(df, "train", 0, columns=user_cols, figuresize=4)

    # The caller's list must not be mutated (the function rebinds, not extends).
    grid = _figs()[1]
    return {
        "user_cols_unchanged": user_cols,
        # 7 columns -> ceil(7/2)=4 per row -> 8 axes, last blanked.
        "grid_panels": len(grid.axes),
        "ylabels": [ax.get_ylabel() for ax in grid.axes[:7]],
        "last_axis_on": grid.axes[7].axison,
        # i in 0..5 -> ylim bottom pinned at 1; i == 6 -> autoscaled to 0.
        "pinned_bottoms": {ax.get_ylim()[0] for ax in grid.axes[:6]},
        "seventh_bottom": grid.axes[6].get_ylim()[0],
    }


def _case_plot_recruitment_prints_the_column_list(capsys):
    df = _recruitment_df(channel=2)
    P._plot_recruitment(df, "test", 2, figuresize=4)
    out = capsys.readouterr().out
    return [col.replace('channel_1_', 'channel_2_') in out
            for col in _RECRUITMENT_EXTRA]


# ---------------------------------------------------------------------------
# _plot_controls
# ---------------------------------------------------------------------------

_COMPONENTS = ("cell", "nucleus", "pathogen", "cytoplasm")


def _controls_df(chans=(0, 1, 2, 3), conditions=("ctrl", "trt"), n=12, seed=1):
    rng = np.random.default_rng(seed)
    data = {"condition": list(conditions) * (n // len(conditions))}
    for chan in chans:
        for comp in _COMPONENTS:
            data[f"{comp}_channel_{chan}_mean_intensity"] = rng.uniform(5, 50, n)
    return pd.DataFrame(data)


def _case_plot_controls_adds_the_channel_of_interest_without_mutating(mask_chans, expected):
    """channel_of_interest is added to a copy; the list passed in is untouched."""
    n_chan = len(expected)
    df = _controls_df(chans=tuple(range(n_chan)))
    original = list(mask_chans)

    P._plot_controls(df, mask_chans, channel_of_interest=n_chan - 1, figuresize=1)

    fig = _figs()[0]
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    # subplots(len(conditions)=2, len(mask_chans)+1) -> one spare column.
    return (mask_chans == original, len(fig.axes), titled)


def _case_plot_controls_keeps_the_real_channel_numbers():
    """Channels [0, 3] are plotted as channels 0 and 3, not renumbered 0 and 1."""
    df = _controls_df(chans=(0, 3))
    P._plot_controls(df, [0], channel_of_interest=3, figuresize=1)
    fig = _figs()[0]
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    heights = [p.get_height() for p in fig.axes[1].patches]
    sub = df[df["condition"] == "ctrl"]
    return (titled[:2],
            bool(np.allclose(heights, [sub[f"{c}_channel_3_mean_intensity"].mean()
                                       for c in _COMPONENTS])))


def _case_plot_controls_bar_heights_are_the_per_condition_means():
    df = _controls_df(chans=(0, 1))

    P._plot_controls(df, [0], channel_of_interest=1, figuresize=1)

    fig = _figs()[0]
    observed = []
    for row, cond in enumerate(("ctrl", "trt")):
        sub = df[df["condition"] == cond]
        for chan in (0, 1):
            ax = fig.axes[row * 3 + chan]
            heights = [p.get_height() for p in ax.patches]
            expected = [sub[f"{c}_channel_{chan}_mean_intensity"].mean()
                        for c in _COMPONENTS]
            observed.append((bool(np.allclose(heights, expected)),
                             [t.get_text() for t in ax.get_xticklabels()],
                             ax.get_xlabel(), ax.get_ylabel()))
    return observed


def _case_plot_controls_single_condition_is_duplicated_into_two_rows():
    df = _controls_df(chans=(0,), conditions=("only",), n=8)

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    fig = _figs()[0]
    top = [p.get_height() for p in fig.axes[0].patches]
    bottom = [p.get_height() for p in fig.axes[2].patches]
    # 1 unique condition is doubled -> 2 rows; 1 channel -> 2 columns.
    return (len(fig.axes), [ax.get_title() for ax in fig.axes],
            bool(np.allclose(top, bottom)))


def _case_plot_controls_all_nan_column_is_plotted_as_zero():
    df = _controls_df(chans=(0,))
    df["cell_channel_0_mean_intensity"] = np.nan

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    ax = _figs()[0].axes[0]
    heights = [p.get_height() for p in ax.patches]
    # the NaN mean must be coerced to 0, not left as NaN
    return (heights[0], all(np.isfinite(h) and h > 0 for h in heights[1:]))


def _case_plot_controls_tolerates_a_missing_component_column():
    df = _controls_df(chans=(0,)).drop(columns=["pathogen_channel_0_mean_intensity"])

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    ax = _figs()[0].axes[0]
    return len(ax.patches)


# ---------------------------------------------------------------------------
# _imshow / _imshow_gpu
# ---------------------------------------------------------------------------

def _chw_images(n=3, c=3, h=4, w=5, seed=2):
    rng = np.random.default_rng(seed)
    return [rng.random((c, h, w)) for _ in range(n)]


def _case_imshow_tiles_images_into_a_labelled_canvas():
    imgs = _chw_images()
    labels = ["a", "b", "c"]

    fig = P._imshow(imgs, labels, nrow=2, color="red", fontsize=7)

    ax = fig.axes[0]
    canvas = np.asarray(ax.images[0].get_array())
    return {
        "is_figure": isinstance(fig, plt.Figure),
        # 3 images, 2 per row -> 2 rows x 2 cols of 4x5 tiles.
        "canvas_shape": canvas.shape,
        "tiles": [
            bool(np.allclose(canvas[0:4, 0:5], np.transpose(imgs[0], (1, 2, 0)))),
            bool(np.allclose(canvas[0:4, 5:10], np.transpose(imgs[1], (1, 2, 0)))),
            bool(np.allclose(canvas[4:8, 0:5], np.transpose(imgs[2], (1, 2, 0)))),
        ],
        # The unused 4th slot stays background (idx >= n_images).
        "spare_slot_blank": bool(np.all(canvas[4:8, 5:10] == 0)),
        # Axis off, one text per label, positioned tile-relative.
        "axis_on": ax.axison,
        "texts": [t.get_text() for t in ax.texts],
        "positions": [t.get_position() for t in ax.texts],
        "styled": all(t.get_color() == "red" and t.get_fontsize() == 7
                      for t in ax.texts),
    }


def _case_imshow_single_row_when_nrow_exceeds_image_count():
    imgs = _chw_images(n=2, h=3, w=3)

    fig = P._imshow(imgs, ["l0", "l1"], nrow=20)

    canvas = np.asarray(fig.axes[0].images[0].get_array())
    # 1 row of 20 slots; 18 empty slots
    return (canvas.shape,
            bool(np.allclose(canvas[:, 0:3], np.transpose(imgs[0], (1, 2, 0)))),
            bool(np.all(canvas[:, 6:] == 0)),
            [t.get_position()[1] for t in fig.axes[0].texts])


def _case_imshow_gpu_matches_the_numpy_implementation():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    batch = rng.random((3, 3, 4, 5)).astype(np.float32)
    labels = ["x", "y", "z"]

    fig_gpu = P._imshow_gpu(torch.from_numpy(batch), labels, nrow=2, fontsize=9)
    gpu_canvas = np.asarray(fig_gpu.axes[0].images[0].get_array())
    plt.close(fig_gpu)

    fig_cpu = P._imshow(list(batch), labels, nrow=2, fontsize=9)
    cpu_canvas = np.asarray(fig_cpu.axes[0].images[0].get_array())

    return (gpu_canvas.shape, cpu_canvas.shape,
            bool(np.allclose(gpu_canvas, cpu_canvas, atol=1e-6)),
            [t.get_text() for t in fig_gpu.axes[0].texts] == labels,
            fig_gpu.axes[0].axison)


def _case_imshow_gpu_moves_a_cuda_tensor_to_cpu_first():
    """Exercise the `if img.is_cuda` branch without a GPU by faking the flag."""
    torch = pytest.importorskip("torch")
    real = torch.rand(2, 3, 4, 4)

    class FakeCudaTensor:
        is_cuda = True

        def __init__(self, t):
            self._t = t
            self.moved = False

        def cpu(self):
            self.moved = True
            return self._t

    fake = FakeCudaTensor(real)
    fig = P._imshow_gpu(fake, ["p", "q"], nrow=2)

    canvas = np.asarray(fig.axes[0].images[0].get_array())
    # .cpu() must be called for CUDA tensors
    return (fake.moved, canvas.shape,
            bool(np.allclose(canvas[:, 0:4],
                             real[0].permute(1, 2, 0).numpy(), atol=1e-6)),
            bool(np.allclose(canvas[:, 4:8],
                             real[1].permute(1, 2, 0).numpy(), atol=1e-6)))


# ---------------------------------------------------------------------------
# _plot_histograms_and_stats
# ---------------------------------------------------------------------------

def _case_plot_histograms_and_stats_reports_exact_counts_per_condition(capsys):
    """One histogram + one stats block per condition, with the real numbers."""
    df = pd.DataFrame({
        "condition": ["ctrl"] * 5 + ["trt"] * 4,
        # ctrl: 3 over / 2 under (0.5 counts as "under"), trt: 1 over / 3 under
        "pred": [0.9, 0.8, 0.6, 0.5, 0.1, 0.7, 0.4, 0.2, 0.0],
    })

    P._plot_histograms_and_stats(df)

    out = capsys.readouterr().out
    printed = [
        "Condition: ctrl" in out, "Condition: trt" in out,
        f"Mean of pred: {df[df.condition == 'ctrl']['pred'].mean()}" in out,
        "Count of pred values over 0.5: 3" in out,
        "Count of pred values under 0.5: 2" in out,
        "Percent positive: 60.0" in out,
        "Percent negative: 40.0" in out,
        "Count of pred values over 0.5: 1" in out,
        "Percent positive: 25.0" in out,
    ]

    figs = _figs()
    per_figure = []
    for fig, cond in zip(figs, ("ctrl", "trt")):
        ax = fig.axes[0]
        mean = df[df["condition"] == cond]["pred"].mean()
        per_figure.append((
            ax.get_title(), len(ax.patches), ax.get_xlabel(), ax.get_ylabel(),
            any(np.allclose(ln.get_xdata(), mean) for ln in ax.lines),
            ax.get_legend().get_texts()[0].get_text() == f"Mean = {mean:.2f}"))
    # one histogram figure per condition
    return (printed, len(figs), per_figure)


# ---------------------------------------------------------------------------
# _show_residules
# ---------------------------------------------------------------------------

def _case_show_residules_plots_diagnostics_and_prints_shapiro(capsys):
    sm = pytest.importorskip("statsmodels.api")
    from scipy.stats import shapiro

    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 60)
    y = 2.0 * x + rng.normal(0, 1.0, 60)
    model = sm.OLS(y, sm.add_constant(x)).fit()
    resid = np.asarray(model.resid)

    P._show_residules(model)

    out = capsys.readouterr().out
    m = re.search(r"Shapiro-Wilk Test W-statistic: ([-\d.e+]+), p-value: ([-\d.e+]+)", out)
    assert m is not None, f"missing Shapiro-Wilk line in: {out!r}"
    exp_w, exp_p = shapiro(resid)
    w_matches = float(m.group(1)) == pytest.approx(exp_w, rel=1e-6)
    p_matches = float(m.group(2)) == pytest.approx(exp_p, rel=1e-6)

    axes = [ax for fig in _figs() for ax in fig.axes]
    # 1. residual histogram with 30 bins spanning the residual range
    hists = [ax for ax in axes if ax.get_title() == "Histogram of Residuals"]
    edges = [p.get_x() for p in hists[0].patches]

    # 2. residuals-vs-fitted scatter (offsets must be exactly the model values)
    fitted = np.asarray(model.fittedvalues)
    scatters = [
        coll for ax in axes for coll in ax.collections
        if np.asarray(coll.get_offsets()).shape == (len(resid), 2)
        and np.allclose(np.asarray(coll.get_offsets())[:, 0], fitted)
        and np.allclose(np.asarray(coll.get_offsets())[:, 1], resid)
    ]

    # 3. the y=0 reference line drawn under that scatter
    zero_lines = [ln for ax in axes for ln in ax.lines
                  if len(ln.get_ydata()) and np.allclose(ln.get_ydata(), 0.0)]
    return {
        "shapiro_w": w_matches,
        "shapiro_p": p_matches,
        "histograms": len(hists),
        "bins": len(hists[0].patches),
        "bins_start_at_min": min(edges) == pytest.approx(resid.min()),
        "hist_xlabel": hists[0].get_xlabel(),
        "scatters": len(scatters),
        "zero_line": bool(zero_lines),  # axhline(y=0) in the residual plot
    }


def _case_show_residules_accepts_any_object_exposing_resid_and_fittedvalues(capsys):
    """The helper is duck-typed: a light-weight stand-in must work too."""
    import types

    rng = np.random.default_rng(5)
    resid = rng.normal(0, 1, 40)
    model = types.SimpleNamespace(resid=resid, fittedvalues=np.arange(40.0))

    P._show_residules(model)

    titles = {ax.get_title() for fig in _figs() for ax in fig.axes}
    return ("Shapiro-Wilk Test W-statistic" in capsys.readouterr().out,
            "Histogram of Residuals" in titles)


# ---------------------------------------------------------------------------
# _reg_v_plot
# ---------------------------------------------------------------------------

def _case_reg_v_plot_annotates_only_significant_points():
    df = pd.DataFrame(
        {"effect": [1.5, -2.0, 0.3, -0.1], "p": [0.001, 0.02, 0.4, 0.9]},
        index=["g1", "g2", "g3", "g4"],
    )

    P._reg_v_plot(df)

    # The -log10(p) column is added in place.
    column_added = ("-log10(p)" in df.columns
                    and bool(np.allclose(df["-log10(p)"], -np.log10(df["p"]))))

    ax = _figs()[0].axes[0]
    # EVERY ROW IS SCATTERED, AND ONLY THE CALLED ONES CARRY COLOUR.
    #
    # It used to hand the scatter a scalar-mappable array of np.sign(effect)
    # under a coolwarm ramp, so every point was coloured by a fact the x axis
    # already states -- and the house rule is that colour is an ARGUMENT.
    # The restyle on 2026-08-18 gives explicit per-point colours instead:
    # a called row (p < 0.05) takes ROLES["up"] or ROLES["down"] by sign,
    # everything else is ROLES["data"] grey.
    from matplotlib.colors import to_rgb

    from spacr.figures.style import ROLES

    offsets = np.asarray(ax.collections[0].get_offsets())
    # no colormap over the sign: that colours every point, which is the
    # thing the restyle removed
    no_colormap = ax.collections[0].get_array() is None
    drawn = [tuple(round(v, 3) for v in rgba[:3])
             for rgba in ax.collections[0].get_facecolors()]
    expected = []
    for effect, p_value in zip(df["effect"], df["p"]):
        role = ("data" if p_value >= 0.05
                else ("up" if effect > 0 else "down"))
        expected.append(tuple(round(v, 3) for v in to_rgb(ROLES[role])))
    return {
        "column_added": column_added,
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        # Only p < 0.05 rows get a text label, at (effect, -log10(p)).
        "labels": [t.get_text() for t in ax.texts],
        "label_positions": [t.get_position() for t in ax.texts],
        "offsets_shape": offsets.shape,
        "x_is_effect": bool(np.allclose(offsets[:, 0], df["effect"])),
        "no_colormap": no_colormap,
        "colours_by_call": drawn == expected,
        # Significance threshold line at p = 0.05.
        "threshold_line": any(np.allclose(ln.get_ydata(), -np.log10(0.05))
                              for ln in ax.lines),
    }


def _case_reg_v_plot_without_significant_rows_adds_no_text():
    df = pd.DataFrame({"effect": [0.2, -0.4], "p": [0.5, 0.9]}, index=["a", "b"])

    P._reg_v_plot(df, grouping="row", variable="effect", plate_number=1)

    ax = _figs()[0].axes[0]
    return (len(ax.texts),
            bool(np.allclose(df["-log10(p)"],
                             [-np.log10(0.5), -np.log10(0.9)])))


def _controls_adding_rows():
    rows = []
    for mask_chans, expected in [([], [0]), ([0], [0, 1]),
                                 ([0, 1], [0, 1, 2]),
                                 ([0, 1, 2], [0, 1, 2, 3])]:
        n = len(expected)
        titles = [f"Condition: {cond} - Channel {c}"
                  for cond in ("ctrl", "trt") for c in range(n)]
        rows.append(pytest.param(
            _bind(_case_plot_controls_adds_the_channel_of_interest_without_mutating,
                  list(mask_chans), expected),
            (True, 2 * (n + 1), titles),
            id=f"controls-adds-channel-of-interest-{len(mask_chans)}-masks"))
    return rows


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_plot_recruitment_default_columns_builds_two_figures,
        {"figures": 2, "intensity_panels": 4,
         "intensity_ylabels": [f"{comp}_channel_1_mean_intensity"
                               for comp in ("cell", "nucleus", "cytoplasm",
                                            "pathogen")],
         "intensity_xlabels": {"pathogen test"},
         "last_panel_has_legend": True, "legend_texts": {"wt", "mut"},
         "rotations": {45.0}, "grid_panels": 6,
         "grid_ylabels": _RECRUITMENT_EXTRA, "last_grid_axis_on": False,
         "grid_ylim_bottoms": {1.0}, "grid_legends_removed": True},
        id="recruitment-default-columns-build-two-figures"),
    pytest.param(
        _case_a_recruitment_plot_does_not_recolour_the_next_plot,
        (True, True), id="recruitment-keeps-own-colours-not-global-cycle"),
    pytest.param(
        _case_plot_recruitment_extra_columns_widen_grid_and_skip_ylim_after_index_5,
        {"user_cols_unchanged": ["extra_a", "extra_b"], "grid_panels": 8,
         "ylabels": ["extra_a", "extra_b"] + [
             col.replace('channel_1_', 'channel_0_')
             for col in _RECRUITMENT_EXTRA],
         "last_axis_on": False, "pinned_bottoms": {1.0},
         "seventh_bottom": 0.0},
        id="recruitment-extra-columns-widen-grid"),
    pytest.param(
        _case_plot_recruitment_prints_the_column_list,
        [True] * 5, id="recruitment-prints-the-column-list"),
    *_controls_adding_rows(),
    pytest.param(
        _case_plot_controls_keeps_the_real_channel_numbers,
        (["Condition: ctrl - Channel 0", "Condition: ctrl - Channel 3"], True),
        id="controls-keep-real-channel-numbers"),
    pytest.param(
        _case_plot_controls_bar_heights_are_the_per_condition_means,
        [(True, list(_COMPONENTS), "Component", "Mean Intensity")] * 4,
        id="controls-bar-heights-are-per-condition-means"),
    pytest.param(
        _case_plot_controls_single_condition_is_duplicated_into_two_rows,
        (4, ["Condition: only - Channel 0", "",
             "Condition: only - Channel 0", ""], True),
        id="controls-single-condition-duplicated-into-two-rows"),
    pytest.param(
        _case_plot_controls_all_nan_column_is_plotted_as_zero,
        (0.0, True), id="controls-all-nan-column-plotted-as-zero"),
    pytest.param(
        _case_plot_controls_tolerates_a_missing_component_column,
        3, id="controls-missing-component-column-tolerated"),
])
def test_recruitment_and_control_figures_show_the_documented_panels(
        request, helper, expected):
    """Panels, labels, colours and bar heights of the recruitment and control
    figures follow the columns and conditions of the frame."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_imshow_tiles_images_into_a_labelled_canvas,
        {"is_figure": True, "canvas_shape": (8, 10, 3),
         "tiles": [True, True, True], "spare_slot_blank": True,
         "axis_on": False, "texts": ["a", "b", "c"],
         "positions": [(2, 15), (7, 15), (2, 19)], "styled": True},
        id="imshow-tiles-images-into-labelled-canvas"),
    pytest.param(
        _case_imshow_single_row_when_nrow_exceeds_image_count,
        ((3, 60, 3), True, True, [15, 15]),
        id="imshow-single-row-when-nrow-exceeds-count"),
    pytest.param(
        _case_imshow_gpu_matches_the_numpy_implementation,
        ((8, 10, 3), (8, 10, 3), True, True, False),
        id="imshow-gpu-matches-numpy"),
    pytest.param(
        _case_imshow_gpu_moves_a_cuda_tensor_to_cpu_first,
        (True, (4, 8, 3), True, True), id="imshow-gpu-moves-cuda-to-cpu"),
])
def test_tiled_image_canvas_places_images_and_labels(
        request, helper, expected):
    """The tiled canvas holds each image in its slot with its label, on the
    numpy and the tensor path alike."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_plot_histograms_and_stats_reports_exact_counts_per_condition,
        ([True] * 9, 2,
         [("Histogram for pred - Condition: ctrl", 30, "Pred Value", "Count",
           True, True),
          ("Histogram for pred - Condition: trt", 30, "Pred Value", "Count",
           True, True)]),
        id="histograms-report-exact-counts-per-condition"),
    pytest.param(
        _case_show_residules_plots_diagnostics_and_prints_shapiro,
        {"shapiro_w": True, "shapiro_p": True, "histograms": 1, "bins": 30,
         "bins_start_at_min": True, "hist_xlabel": "Residual Value",
         "scatters": 1, "zero_line": True},
        id="residuals-diagnostics-and-shapiro"),
    pytest.param(
        _case_show_residules_accepts_any_object_exposing_resid_and_fittedvalues,
        (True, True), id="residuals-accepts-duck-typed-model"),
    pytest.param(
        _case_reg_v_plot_annotates_only_significant_points,
        {"column_added": True, "title": "Volcano Plot",
         "xlabel": "Coefficient", "ylabel": "-log10(P-value)",
         "labels": ["g1", "g2"],
         "label_positions": [(1.5, -np.log10(0.001)),
                             (-2.0, -np.log10(0.02))],
         "offsets_shape": (4, 2), "x_is_effect": True, "no_colormap": True,
         "colours_by_call": True, "threshold_line": True},
        id="volcano-annotates-only-significant-points"),
    pytest.param(
        _case_reg_v_plot_without_significant_rows_adds_no_text,
        (0, True), id="volcano-without-significant-rows-adds-no-text"),
])
def test_statistics_plots_report_the_numbers_of_the_data(
        request, helper, expected):
    """Histogram statistics, residual diagnostics and the volcano plot carry
    the exact numbers of the frame or model handed in."""
    observed = _call(request, helper)
    assert observed == expected
