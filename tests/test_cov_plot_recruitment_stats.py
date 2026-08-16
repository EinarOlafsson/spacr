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
"""
from __future__ import annotations

import contextlib
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


def _figs():
    """All live figures, in creation order."""
    return [plt.figure(n) for n in plt.get_fignums()]


# ---------------------------------------------------------------------------
# _plot_recruitment
# ---------------------------------------------------------------------------

_RECRUITMENT_EXTRA = [
    "pathogen_cytoplasm_mean_mean",
    "pathogen_cytoplasm_q75_mean",
    "pathogen_periphery_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_q75_mean",
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
        data[col] = rng.uniform(2, 50, n)
    return pd.DataFrame(data)


def test_plot_recruitment_default_columns_builds_two_figures():
    """columns=None -> the five hard-coded recruitment columns on a 2x3 grid."""
    df = _recruitment_df(channel=1)

    P._plot_recruitment(df, "test", 1, figuresize=4)

    figs = _figs()
    assert len(figs) == 2, "one 1x4 intensity figure + one 2xN recruitment grid"

    intensity, grid = figs
    # Figure 1: cell / nucleus / cytoplasm / pathogen mean intensity.
    assert len(intensity.axes) == 4
    assert [ax.get_ylabel() for ax in intensity.axes] == [
        f"{comp}_channel_1_mean_intensity"
        for comp in ("cell", "nucleus", "cytoplasm", "pathogen")
    ]
    assert all(ax.get_xlabel() == "pathogen test" for ax in intensity.axes)
    # Only the last panel keeps a legend (the others are commented out upstream).
    assert intensity.axes[3].get_legend() is not None
    assert {t.get_text() for t in intensity.axes[3].get_legend().get_texts()} == {"wt", "mut"}
    # Rotated tick labels + font scaled from figuresize.
    assert all(lbl.get_rotation() == 45 for lbl in intensity.axes[0].get_xticklabels())

    # Figure 2: 5 columns -> ceil(5/2)=3 per row -> 6 axes, last one blanked.
    assert len(grid.axes) == 6
    assert [ax.get_ylabel() for ax in grid.axes[:5]] == _RECRUITMENT_EXTRA
    assert grid.axes[5].axison is False
    # i <= 5 => ylim bottom forced to 1
    assert all(ax.get_ylim()[0] == 1.0 for ax in grid.axes[:5])
    # Per-panel legends are removed in the grid.
    assert all(ax.get_legend() is None for ax in grid.axes[:5])


def test_a_recruitment_plot_does_not_recolour_the_next_plot():
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
    assert bars == {tuple(np.round(sns.desaturate(colour, 0.75), 4))
                    for colour in intended}, (
        "the recruitment bars lost their own colours")
    assert matplotlib.rcParams["axes.prop_cycle"] == before, (
        "drawing a recruitment plot left its palette on matplotlib's global "
        "colour cycle; the next figure of the session inherits it")


def test_plot_recruitment_extra_columns_widen_grid_and_skip_ylim_after_index_5():
    """Extra user columns are prepended; the 7th panel (i=6) keeps its autoscale."""
    df = _recruitment_df(channel=0)
    user_cols = ["extra_a", "extra_b"]

    P._plot_recruitment(df, "train", 0, columns=user_cols, figuresize=4)

    # The caller's list must not be mutated (the function rebinds, not extends).
    assert user_cols == ["extra_a", "extra_b"]

    grid = _figs()[1]
    # 7 columns -> ceil(7/2)=4 per row -> 8 axes, last blanked.
    assert len(grid.axes) == 8
    assert [ax.get_ylabel() for ax in grid.axes[:7]] == user_cols + _RECRUITMENT_EXTRA
    assert grid.axes[7].axison is False
    # i in 0..5 -> ylim bottom pinned at 1; i == 6 -> untouched (autoscaled to 0).
    assert all(ax.get_ylim()[0] == 1.0 for ax in grid.axes[:6])
    assert grid.axes[6].get_ylim()[0] == 0.0


def test_plot_recruitment_prints_the_column_list(capsys):
    df = _recruitment_df(channel=2)
    P._plot_recruitment(df, "test", 2, figuresize=4)
    out = capsys.readouterr().out
    for col in _RECRUITMENT_EXTRA:
        assert col in out


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


@pytest.mark.parametrize("mask_chans,expected", [
    ([], [0]),
    ([0], [0, 1]),
    ([0, 1], [0, 1, 2]),
    ([0, 1, 2], [0, 1, 2, 3]),
])
def test_plot_controls_normalises_mask_chans_in_place(mask_chans, expected):
    """channel_of_interest is appended, then the list is rewritten as 0..n-1."""
    n_chan = len(expected)
    df = _controls_df(chans=tuple(range(n_chan)))

    P._plot_controls(df, mask_chans, channel_of_interest=n_chan - 1, figuresize=1)

    # The caller's list is mutated by append() but the 0..n-1 rewrite is local.
    assert mask_chans == expected[:-1] + [n_chan - 1]
    fig = _figs()[0]
    # subplots(len(conditions)=2, len(mask_chans)+1) -> one spare column.
    assert len(fig.axes) == 2 * (n_chan + 1)
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titled == [
        f"Condition: {cond} - Channel {c}"
        for cond in ("ctrl", "trt") for c in range(n_chan)
    ]


def test_plot_controls_bar_heights_are_the_per_condition_means():
    df = _controls_df(chans=(0, 1))

    P._plot_controls(df, [0], channel_of_interest=1, figuresize=1)

    fig = _figs()[0]
    for row, cond in enumerate(("ctrl", "trt")):
        sub = df[df["condition"] == cond]
        for chan in (0, 1):
            ax = fig.axes[row * 3 + chan]
            heights = [p.get_height() for p in ax.patches]
            expected = [sub[f"{c}_channel_{chan}_mean_intensity"].mean()
                        for c in _COMPONENTS]
            assert np.allclose(heights, expected)
            assert [t.get_text() for t in ax.get_xticklabels()] == list(_COMPONENTS)
            assert ax.get_xlabel() == "Component"
            assert ax.get_ylabel() == "Mean Intensity"


def test_plot_controls_single_condition_is_duplicated_into_two_rows():
    df = _controls_df(chans=(0,), conditions=("only",), n=8)

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    fig = _figs()[0]
    # 1 unique condition is doubled -> 2 rows; 1 channel -> 2 columns.
    assert len(fig.axes) == 4
    assert [ax.get_title() for ax in fig.axes] == [
        "Condition: only - Channel 0", "",
        "Condition: only - Channel 0", "",
    ]
    top = [p.get_height() for p in fig.axes[0].patches]
    bottom = [p.get_height() for p in fig.axes[2].patches]
    assert np.allclose(top, bottom)


def test_plot_controls_all_nan_column_is_plotted_as_zero():
    df = _controls_df(chans=(0,))
    df["cell_channel_0_mean_intensity"] = np.nan

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    ax = _figs()[0].axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert heights[0] == 0.0, "NaN mean must be coerced to 0, not left as NaN"
    assert all(np.isfinite(h) and h > 0 for h in heights[1:])


def test_plot_controls_tolerates_a_missing_component_column():
    df = _controls_df(chans=(0,)).drop(columns=["pathogen_channel_0_mean_intensity"])

    P._plot_controls(df, [], channel_of_interest=0, figuresize=1)

    ax = _figs()[0].axes[0]
    assert len(ax.patches) == 3


# ---------------------------------------------------------------------------
# _imshow / _imshow_gpu
# ---------------------------------------------------------------------------

def _chw_images(n=3, c=3, h=4, w=5, seed=2):
    rng = np.random.default_rng(seed)
    return [rng.random((c, h, w)) for _ in range(n)]


def test_imshow_tiles_images_into_a_labelled_canvas():
    imgs = _chw_images()
    labels = ["a", "b", "c"]

    fig = P._imshow(imgs, labels, nrow=2, color="red", fontsize=7)

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    canvas = np.asarray(ax.images[0].get_array())
    # 3 images, 2 per row -> 2 rows x 2 cols of 4x5 tiles.
    assert canvas.shape == (8, 10, 3)
    assert np.allclose(canvas[0:4, 0:5], np.transpose(imgs[0], (1, 2, 0)))
    assert np.allclose(canvas[0:4, 5:10], np.transpose(imgs[1], (1, 2, 0)))
    assert np.allclose(canvas[4:8, 0:5], np.transpose(imgs[2], (1, 2, 0)))
    # The unused 4th slot stays background (idx >= n_images).
    assert np.all(canvas[4:8, 5:10] == 0)
    # Axis off, one text per label, positioned tile-relative.
    assert ax.axison is False
    assert [t.get_text() for t in ax.texts] == labels
    assert [t.get_position() for t in ax.texts] == [(2, 15), (7, 15), (2, 19)]
    assert all(t.get_color() == "red" and t.get_fontsize() == 7 for t in ax.texts)


def test_imshow_single_row_when_nrow_exceeds_image_count():
    imgs = _chw_images(n=2, h=3, w=3)

    fig = P._imshow(imgs, ["l0", "l1"], nrow=20)

    canvas = np.asarray(fig.axes[0].images[0].get_array())
    assert canvas.shape == (3, 3 * 20, 3)          # 1 row of 20 slots
    assert np.allclose(canvas[:, 0:3], np.transpose(imgs[0], (1, 2, 0)))
    assert np.all(canvas[:, 6:] == 0)              # 18 empty slots
    assert [t.get_position()[1] for t in fig.axes[0].texts] == [15, 15]


def test_imshow_gpu_matches_the_numpy_implementation():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    batch = rng.random((3, 3, 4, 5)).astype(np.float32)
    labels = ["x", "y", "z"]

    fig_gpu = P._imshow_gpu(torch.from_numpy(batch), labels, nrow=2, fontsize=9)
    gpu_canvas = np.asarray(fig_gpu.axes[0].images[0].get_array())
    plt.close(fig_gpu)

    fig_cpu = P._imshow(list(batch), labels, nrow=2, fontsize=9)
    cpu_canvas = np.asarray(fig_cpu.axes[0].images[0].get_array())

    assert gpu_canvas.shape == cpu_canvas.shape == (8, 10, 3)
    assert np.allclose(gpu_canvas, cpu_canvas, atol=1e-6)
    assert [t.get_text() for t in fig_gpu.axes[0].texts] == labels
    assert fig_gpu.axes[0].axison is False


def test_imshow_gpu_moves_a_cuda_tensor_to_cpu_first():
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

    assert fake.moved is True, ".cpu() must be called for CUDA tensors"
    canvas = np.asarray(fig.axes[0].images[0].get_array())
    assert canvas.shape == (4, 8, 3)
    assert np.allclose(canvas[:, 0:4], real[0].permute(1, 2, 0).numpy(), atol=1e-6)
    assert np.allclose(canvas[:, 4:8], real[1].permute(1, 2, 0).numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# _plot_histograms_and_stats
# ---------------------------------------------------------------------------

def test_plot_histograms_and_stats_reports_exact_counts_per_condition(capsys):
    """One histogram + one stats block per condition, with the real numbers."""
    df = pd.DataFrame({
        "condition": ["ctrl"] * 5 + ["trt"] * 4,
        # ctrl: 3 over / 2 under (0.5 counts as "under"), trt: 1 over / 3 under
        "pred": [0.9, 0.8, 0.6, 0.5, 0.1, 0.7, 0.4, 0.2, 0.0],
    })

    P._plot_histograms_and_stats(df)

    out = capsys.readouterr().out
    assert "Condition: ctrl" in out and "Condition: trt" in out
    assert f"Mean of pred: {df[df.condition == 'ctrl']['pred'].mean()}" in out
    assert "Count of pred values over 0.5: 3" in out
    assert "Count of pred values under 0.5: 2" in out
    assert "Percent positive: 60.0" in out
    assert "Percent negative: 40.0" in out
    assert "Count of pred values over 0.5: 1" in out
    assert "Percent positive: 25.0" in out

    figs = _figs()
    assert len(figs) == 2, "one histogram figure per condition"
    for fig, cond in zip(figs, ("ctrl", "trt")):
        ax = fig.axes[0]
        assert ax.get_title() == f"Histogram for pred - Condition: {cond}"
        assert len(ax.patches) == 30
        assert ax.get_xlabel() == "Pred Value" and ax.get_ylabel() == "Count"
        mean = df[df["condition"] == cond]["pred"].mean()
        assert any(np.allclose(ln.get_xdata(), mean) for ln in ax.lines)
        assert ax.get_legend().get_texts()[0].get_text() == f"Mean = {mean:.2f}"


# ---------------------------------------------------------------------------
# _show_residules
# ---------------------------------------------------------------------------

def test_show_residules_plots_diagnostics_and_prints_shapiro(capsys):
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
    assert float(m.group(1)) == pytest.approx(exp_w, rel=1e-6)
    assert float(m.group(2)) == pytest.approx(exp_p, rel=1e-6)

    axes = [ax for fig in _figs() for ax in fig.axes]
    # 1. residual histogram with 30 bins spanning the residual range
    hists = [ax for ax in axes if ax.get_title() == "Histogram of Residuals"]
    assert len(hists) == 1
    assert len(hists[0].patches) == 30
    edges = [p.get_x() for p in hists[0].patches]
    assert min(edges) == pytest.approx(resid.min())
    assert hists[0].get_xlabel() == "Residual Value"

    # 2. residuals-vs-fitted scatter (offsets must be exactly the model values)
    fitted = np.asarray(model.fittedvalues)
    scatters = [
        coll for ax in axes for coll in ax.collections
        if np.asarray(coll.get_offsets()).shape == (len(resid), 2)
        and np.allclose(np.asarray(coll.get_offsets())[:, 0], fitted)
        and np.allclose(np.asarray(coll.get_offsets())[:, 1], resid)
    ]
    assert len(scatters) == 1

    # 3. the y=0 reference line drawn under that scatter
    zero_lines = [ln for ax in axes for ln in ax.lines
                  if len(ln.get_ydata()) and np.allclose(ln.get_ydata(), 0.0)]
    assert zero_lines, "axhline(y=0) missing from the residual plot"


def test_show_residules_accepts_any_object_exposing_resid_and_fittedvalues(capsys):
    """The helper is duck-typed: a light-weight stand-in must work too."""
    import types

    rng = np.random.default_rng(5)
    resid = rng.normal(0, 1, 40)
    model = types.SimpleNamespace(resid=resid, fittedvalues=np.arange(40.0))

    P._show_residules(model)

    assert "Shapiro-Wilk Test W-statistic" in capsys.readouterr().out
    titles = {ax.get_title() for fig in _figs() for ax in fig.axes}
    assert "Histogram of Residuals" in titles


# ---------------------------------------------------------------------------
# _reg_v_plot
# ---------------------------------------------------------------------------

def test_reg_v_plot_annotates_only_significant_points():
    df = pd.DataFrame(
        {"effect": [1.5, -2.0, 0.3, -0.1], "p": [0.001, 0.02, 0.4, 0.9]},
        index=["g1", "g2", "g3", "g4"],
    )

    P._reg_v_plot(df)

    # The -log10(p) column is added in place.
    assert "-log10(p)" in df.columns
    assert np.allclose(df["-log10(p)"], -np.log10(df["p"]))

    ax = _figs()[0].axes[0]
    assert ax.get_title() == "Volcano Plot"
    assert ax.get_xlabel() == "Coefficient"
    assert ax.get_ylabel() == "-log10(P-value)"
    # Only p < 0.05 rows get a text label, placed at (effect, -log10(p)).
    assert [t.get_text() for t in ax.texts] == ["g1", "g2"]
    assert [t.get_position() for t in ax.texts] == [
        (1.5, -np.log10(0.001)), (-2.0, -np.log10(0.02)),
    ]
    # Every row is scattered, coloured by the sign of the effect.
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets.shape == (4, 2)
    assert np.allclose(offsets[:, 0], df["effect"])
    assert np.allclose(ax.collections[0].get_array(), np.sign(df["effect"]))
    # Significance threshold line at p = 0.05.
    assert any(np.allclose(ln.get_ydata(), -np.log10(0.05)) for ln in ax.lines)


def test_reg_v_plot_without_significant_rows_adds_no_text():
    df = pd.DataFrame({"effect": [0.2, -0.4], "p": [0.5, 0.9]}, index=["a", "b"])

    P._reg_v_plot(df, grouping="row", variable="effect", plate_number=1)

    ax = _figs()[0].axes[0]
    assert len(ax.texts) == 0
    assert np.allclose(df["-log10(p)"], [-np.log10(0.5), -np.log10(0.9)])
