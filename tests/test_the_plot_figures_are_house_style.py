"""``spacr/plot.py`` draws in the house style, and puts the globals back.

Instruction 136, measured 2026-08-18: this file creates 45 figures -- a third
of every figure spaCR draws -- and used the house style zero times. It is the
module a user sees most, because every other module's plots come from it.

These tests assert the DRAWING, not the source text. A panel that imported
``figure_style`` and then drew a bold red reference line in a saturated teal
cloud would pass a grep and fail a reader, so every assertion here reads the
artists that ended up on the axes: spine visibility, tick label size, the
number of distinct colours among the data marks, the weight and dash of a
reference line.

And one invariant above all the others, because it is the one that already
cost this repository a day: **after a figure is drawn, matplotlib's global
rcParams are exactly what they were before.** ``figure_style`` is a context
manager for that reason; a global ``rcParams.update`` in a long-running GUI
styles every later figure in the session, in every other module, until the
process exits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from spacr import plot as P  # noqa: E402
from spacr.figures.style import (ROLES, TYPE_SCALE, WEIGHTS,  # noqa: E402
                                 Palette, resolve_ink, theme_target)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _hex(colour):
    """A colour spec as a comparable lower-case hex string."""
    return to_hex(to_rgba(colour)).lower()


def _distinct_mark_colours(ax):
    """Every distinct colour among the data marks of ``ax``.

    Lines, patches and scatter collections together, because "how many
    colours does this panel use" is a question about what a reader sees, not
    about which matplotlib artist happened to draw it.
    """
    colours = set()
    for line in ax.lines:
        colours.add(_hex(line.get_color()))
    for patch in ax.patches:
        colours.add(_hex(patch.get_facecolor()))
    for collection in ax.collections:
        for face in np.atleast_2d(collection.get_facecolors()):
            colours.add(_hex(tuple(face)))
    return colours


def _is_reference_line(line):
    """True when ``line`` is drawn the way the house style draws a reference."""
    return (_hex(line.get_color()) == _hex(ROLES["reference"])
            and line.get_linewidth() == pytest.approx(WEIGHTS["reference"])
            and line.get_linestyle() not in ("-", "solid", "None"))


def _figures():
    return [plt.figure(n) for n in plt.get_fignums()]


@pytest.fixture(autouse=True)
def _clean_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def rcparams_guard():
    """Fails the test if the code under it left matplotlib's globals moved.

    Snapshots by ``repr`` because a few rcParams hold unhashable values
    (the colour cycle, the dash patterns) that compare badly by identity.
    """
    before = {k: repr(v) for k, v in mpl.rcParams.items()}
    yield
    after = {k: repr(v) for k, v in mpl.rcParams.items()}
    leaked = {k: (before.get(k), after.get(k))
              for k in set(before) | set(after)
              if before.get(k) != after.get(k)}
    assert leaked == {}, f"rcParams leaked out of the figure build: {leaked}"


# --------------------------------------------------------------------------- #
#  The frame and the type scale, asserted once on a representative panel
# --------------------------------------------------------------------------- #

def _pred_frame():
    return pd.DataFrame({
        "condition": ["ctrl"] * 6 + ["trt"] * 6,
        "pred": [0.9, 0.8, 0.6, 0.5, 0.1, 0.2, 0.7, 0.4, 0.2, 0.0, 0.3, 0.95],
    })


def test_the_distribution_panels_wear_the_L_frame(rcparams_guard):
    """Left and bottom spines only -- the Cell figures' framing."""
    P._plot_histograms_and_stats(_pred_frame())

    for figure in _figures():
        ax = figure.axes[0]
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_linewidth() == pytest.approx(
            WEIGHTS["spine"])


def test_the_tick_labels_are_the_house_tick_tier(rcparams_guard):
    """6.2 pt, the measured tick tier -- not matplotlib's 10."""
    P._plot_histograms_and_stats(_pred_frame())

    ax = _figures()[0].axes[0]
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    assert labels
    assert all(label.get_fontsize() == pytest.approx(TYPE_SCALE["tick"])
               for label in labels)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(TYPE_SCALE["label"])


def test_the_ink_follows_the_theme_not_a_hard_coded_black(rcparams_guard):
    """A print-palette near-black on spaCR's dark theme is invisible axes."""
    P._plot_histograms_and_stats(_pred_frame())

    ink = _hex(resolve_ink(theme_target()))
    ax = _figures()[0].axes[0]
    assert _hex(ax.spines["left"].get_edgecolor()) == ink
    assert _hex(ax.xaxis.label.get_color()) == ink


def test_no_panel_of_this_module_leaves_a_gridline(rcparams_guard):
    """No gridlines. Ever. -- the skill, and the fastest way to look like a
    spreadsheet."""
    P._plot_histograms_and_stats(_pred_frame())

    for figure in _figures():
        for ax in figure.axes:
            assert not any(line.get_visible()
                           for line in ax.get_xgridlines() + ax.get_ygridlines())


# --------------------------------------------------------------------------- #
#  _plot_histograms_and_stats
# --------------------------------------------------------------------------- #

def test_the_prediction_histogram_uses_one_fill_and_one_reference(rcparams_guard):
    """The distribution is the subject; the mean is the ruler.

    Two colours in the panel and no more: the fill and the grey reference.
    """
    P._plot_histograms_and_stats(_pred_frame())

    figures = _figures()
    assert len(figures) == 2                       # one per condition
    for figure in figures:
        ax = figure.axes[0]
        bars = {_hex(p.get_facecolor()) for p in ax.patches}
        assert bars == {_hex(ROLES["fill"])}
        assert len(ax.lines) == 1
        assert _is_reference_line(ax.lines[0])
        assert _distinct_mark_colours(ax) == {_hex(ROLES["fill"]),
                                              _hex(ROLES["reference"])}


def test_the_mean_still_reads_its_own_value(rcparams_guard):
    """A restyle may not move a number: the line sits on the mean and says so."""
    frame = _pred_frame()
    P._plot_histograms_and_stats(frame)

    for figure, condition in zip(_figures(), ("ctrl", "trt")):
        ax = figure.axes[0]
        mean = frame[frame["condition"] == condition]["pred"].mean()
        assert np.allclose(ax.lines[0].get_xdata(), mean)
        assert ax.get_legend().get_texts()[0].get_text() == f"Mean = {mean:.2f}"
        # ...and the frame around that legend is gone, which is the style.
        assert not ax.get_legend().get_frame_on()


# --------------------------------------------------------------------------- #
#  _show_residules
# --------------------------------------------------------------------------- #

def _fit():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 60)
    y = 2.0 * x + rng.normal(0, 1.0, 60)
    return sm.OLS(y, sm.add_constant(x)).fit()


def test_the_qq_plot_is_no_longer_drawn_over(rcparams_guard, capsys):
    """THE BUG THIS PASS FIXED, stated as a test.

    ``sm.qqplot`` creates a figure and leaves ITS axes current, so the
    residuals-vs-fitted scatter landed on top of the QQ panel and its title
    overwrote 'QQ Plot'. Measured on this 60-point fit: two figures came back
    instead of three, and the second held both diagnostics superimposed.
    """
    P._show_residules(_fit())
    capsys.readouterr()

    titles = [ax.get_title() for figure in _figures() for ax in figure.axes]
    assert len(_figures()) == 3
    assert sorted(titles) == ["Histogram of Residuals", "QQ Plot",
                              "Residuals vs. Fitted Values"]

    # ...and the scatter is alone on its own axes.
    residual_ax = [ax for figure in _figures() for ax in figure.axes
                   if ax.get_title() == "Residuals vs. Fitted Values"][0]
    assert len(residual_ax.collections) == 1


def test_the_residual_diagnostics_are_grey_with_grey_references(rcparams_guard,
                                                                capsys):
    """Nothing in a diagnostic is the claim, so nothing in it is coloured."""
    P._show_residules(_fit())
    capsys.readouterr()

    axes = {ax.get_title(): ax for figure in _figures() for ax in figure.axes}

    scatter = axes["Residuals vs. Fitted Values"].collections[0]
    assert {_hex(tuple(c)) for c in scatter.get_facecolors()} == {
        _hex(ROLES["data"])}
    zero = axes["Residuals vs. Fitted Values"].lines[0]
    assert _is_reference_line(zero)
    assert np.allclose(zero.get_ydata(), 0.0)

    # The 45-degree line of the QQ panel is a reference too, and qqplot draws
    # it bold red unless it is taken in hand.
    drawn = [ln for ln in axes["QQ Plot"].lines
             if ln.get_linestyle() not in ("None",)]
    assert drawn and all(_is_reference_line(ln) for ln in drawn)


def test_the_shapiro_wilk_numbers_did_not_move(rcparams_guard, capsys):
    """This is a restyle. If a statistic moved, the change is wrong."""
    from scipy.stats import shapiro

    model = _fit()
    P._show_residules(model)
    out = capsys.readouterr().out

    expected_w, expected_p = shapiro(np.asarray(model.resid))
    assert f"Shapiro-Wilk Test W-statistic: {expected_w}" in out
    assert f"p-value: {expected_p}" in out


# --------------------------------------------------------------------------- #
#  plot_histogram
# --------------------------------------------------------------------------- #

def test_plot_histogram_is_one_pale_fill_not_a_translucent_saturated_one(
        rcparams_guard, tmp_path):
    """Overplotting is handled by a pale fill, never by alpha on a strong hue.

    The old bars were ``(0, 155, 255)/255`` teal at alpha 0.6.
    """
    frame = pd.DataFrame({"recruitment": np.linspace(0.0, 4.0, 50)})
    P.plot_histogram(frame, "recruitment", dst=str(tmp_path))

    ax = _figures()[0].axes[0]
    faces = {_hex(p.get_facecolor()) for p in ax.patches}
    assert faces == {_hex(ROLES["fill"])}
    assert all(p.get_alpha() in (None, 1.0) for p in ax.patches)
    # The counts are untouched: all 50 observations are still binned.
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(50.0)
    # ...and it still went out through save_figure, in the user's format.
    assert (tmp_path / "recruitment_histogram.pdf").is_file()


# --------------------------------------------------------------------------- #
#  plot_lorenz_curves
# --------------------------------------------------------------------------- #

def _counts_csv(path, names, counts):
    pd.DataFrame({"grna_name": list(names),
                  "count": list(counts)}).to_csv(path, index=False)
    return str(path)


def test_the_lorenz_plates_are_grey_and_only_the_library_is_coloured(
        rcparams_guard, tmp_path, capsys):
    """Four plates used to be four cycle colours; the claim is the combined
    curve, so it is the only coloured mark."""
    files = [_counts_csv(tmp_path / f"p{i}.csv", [f"g{j}" for j in range(10)],
                         np.arange(1, 11) * (i + 1))
             for i in range(4)]

    P.plot_lorenz_curves(files, save=False)
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    lines = ax.get_lines()
    assert len(lines) == 5                       # four plates + combined
    assert {_hex(ln.get_color()) for ln in lines[:4]} == {_hex(ROLES["data"])}
    assert _hex(lines[4].get_color()) == _hex(ROLES["highlight"])
    assert lines[4].get_linestyle() == "--"
    # The highlight is a minority of the marks -- one in five.
    assert len(_distinct_mark_colours(ax)) == 2


def test_the_lorenz_legend_is_coloured_text_with_no_box(rcparams_guard,
                                                        tmp_path, capsys):
    """The published figures label a curve in its own colour, without a frame."""
    files = [_counts_csv(tmp_path / "p1.csv", [f"g{j}" for j in range(10)],
                         np.arange(1, 11))]

    P.plot_lorenz_curves(files, save=False)
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    assert ax.get_legend() is None
    texts = {t.get_text(): _hex(t.get_color()) for t in ax.texts}
    plate = [t for t in texts if t.startswith("plate 1 (Gini:")]
    combined = [t for t in texts if t.startswith("Combined (Gini:")]
    assert plate and combined
    assert texts[plate[0]] == _hex(ROLES["data"])
    assert texts[combined[0]] == _hex(ROLES["highlight"])
    assert all(t.get_fontsize() == pytest.approx(TYPE_SCALE["annotation"])
               for t in ax.texts)


def test_the_gini_coefficients_did_not_move(rcparams_guard, tmp_path, capsys):
    """A perfectly even library is still exactly 0.0000."""
    files = [_counts_csv(tmp_path / "p1.csv", [f"g{j}" for j in range(10)],
                         [3] * 10)]
    P.plot_lorenz_curves(files, save=False)
    assert "plate 1: Gini Coefficient = 0.0000" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  The invariant that matters most
# --------------------------------------------------------------------------- #

def test_drawing_every_distribution_figure_leaves_the_globals_alone(tmp_path,
                                                                    capsys):
    """Rule 2, asserted directly and over all four figures at once.

    Not folded into the fixture above: this is the statement that the whole
    group can be drawn in one session and the next module's figure is still
    the figure it asked for.
    """
    before = {k: repr(v) for k, v in mpl.rcParams.items()}

    P._plot_histograms_and_stats(_pred_frame())
    P._show_residules(_fit())
    P.plot_histogram(pd.DataFrame({"x": np.arange(20.0)}), "x", dst=None)
    P.plot_lorenz_curves(
        [_counts_csv(tmp_path / "p1.csv", [f"g{j}" for j in range(8)],
                     np.arange(1, 9))], save=False)
    capsys.readouterr()

    after = {k: repr(v) for k, v in mpl.rcParams.items()}
    leaked = {k: (before.get(k), after.get(k))
              for k in set(before) | set(after)
              if before.get(k) != after.get(k)}
    assert leaked == {}
