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


# --------------------------------------------------------------------------- #
#  plot_comparison_results -- four metric panels of one comparison
# --------------------------------------------------------------------------- #

def _comparison_rows(n=6):
    rng = np.random.default_rng(11)
    return [{"filename": f"f{i}",
             "jaccard_a_b": float(rng.uniform(0.6, 0.9)),
             "dice_a_b": float(rng.uniform(0.6, 0.9)),
             "boundary_f1_a_b": float(rng.uniform(0.5, 0.8)),
             "average_precision_a_b": float(rng.uniform(0.4, 0.7))}
            for i in range(n)]


def test_the_comparison_panels_are_lettered(rcparams_guard):
    """Four panels are a sheet, and a sheet is read by its letters."""
    figure = P.plot_comparison_results(_comparison_rows())

    letters = []
    for ax in figure.axes:
        letters += [t.get_text() for t in ax.texts
                    if t.get_fontsize() == pytest.approx(
                        TYPE_SCALE["panel_letter"])]
    assert letters == ["A", "B", "C", "D"]
    assert all(t.get_fontweight() == "bold"
               for ax in figure.axes for t in ax.texts)


def test_the_comparison_panels_are_grey_and_opaque(rcparams_guard):
    """No metric of the four is the claim, so no mark of them is coloured.

    The points were drawn at alpha 0.6; the style handles overplotting with
    point size and greying, never by making a colour translucent.
    """
    figure = P.plot_comparison_results(_comparison_rows())

    for ax in figure.axes:
        strip = ax.collections[0]
        assert {_hex(tuple(c)) for c in strip.get_facecolors()} == {
            _hex(Palette.GREY_DARK)}
        assert strip.get_alpha() in (None, 1.0)
        boxes = {_hex(p.get_facecolor()) for p in ax.patches}
        assert boxes <= {_hex(ROLES["data"])}


def test_the_comparison_tick_labels_rotate_and_anchor(rcparams_guard):
    """A comparison name is a pair of filenames; at 0 degrees it runs off the
    panel and at 45 degrees unanchored it drifts off its own tick."""
    figure = P.plot_comparison_results(_comparison_rows())

    for ax in figure.axes:
        for label in ax.get_xticklabels():
            assert label.get_rotation() == pytest.approx(45.0)
            assert label.get_ha() == "right"


def test_the_comparison_values_did_not_move(rcparams_guard):
    """Restyle only: each metric is still on its own panel at its own value."""
    values = {"jaccard_a_b": 0.8, "dice_a_b": 0.9,
              "boundary_f1_a_b": 0.7, "average_precision_a_b": 0.6}
    figure = P.plot_comparison_results([{"filename": "x", **values}])

    for ax, value in zip(figure.axes, values.values()):
        points = np.vstack([c.get_offsets() for c in ax.collections])
        assert points[0, 1] == pytest.approx(value)


# --------------------------------------------------------------------------- #
#  plot_permutation / plot_feature_importance
# --------------------------------------------------------------------------- #

def _importance_frame(n=12, low=0.01):
    return pd.DataFrame({
        "feature": [f"f{i}" for i in range(n)],
        "importance": np.linspace(low, 1.0, n),
        "importance_mean": np.linspace(low, 1.0, n),
        "importance_std": np.full(n, 0.02),
    })


@pytest.mark.parametrize("builder,column", [
    (lambda f: P.plot_permutation(f), "importance_mean"),
    (lambda f: P.plot_feature_importance(f), "importance"),
])
def test_an_importance_ranking_is_all_grey(rcparams_guard, builder, column):
    """The ranking IS the claim; no single bar of it is, so none is coloured.

    The two panels used to be solid teal and solid BLUE at alpha 0.6 -- and
    BLUE is the palette's highlight, the hue that means "this one".
    """
    figure = builder(_importance_frame())
    ax = figure.axes[0]

    assert {_hex(p.get_facecolor()) for p in ax.patches} == {
        _hex(ROLES["data"])}
    assert all(p.get_alpha() in (None, 1.0) for p in ax.patches)
    assert not ax.spines["top"].get_visible()


@pytest.mark.parametrize("builder,column", [
    (lambda f: P.plot_permutation(f), "importance_mean"),
    (lambda f: P.plot_feature_importance(f), "importance"),
])
def test_the_zero_rule_is_drawn_only_where_a_bar_can_cross_it(
        rcparams_guard, builder, column):
    """A negative permutation importance means shuffling the feature IMPROVED
    the model. Without a zero rule you cannot see which bars cross it -- and
    with every bar positive the rule is the axis, so it is not drawn twice."""
    positive = _importance_frame()
    assert len(builder(positive).axes[0].lines) == 0

    mixed = _importance_frame()
    mixed[column] = np.linspace(-0.4, 1.0, len(mixed))
    ax = builder(mixed).axes[0]
    assert len(ax.lines) == 1
    assert _is_reference_line(ax.lines[0])
    assert np.allclose(ax.lines[0].get_xdata(), 0.0)


def test_the_importance_bars_still_measure_what_they_measured(rcparams_guard):
    """Restyle only: the widths are the importances and the error bars the
    standard deviations."""
    frame = _importance_frame()
    ax = P.plot_permutation(frame).axes[0]

    assert [p.get_width() for p in ax.patches] == pytest.approx(
        list(frame["importance_mean"]))
    from matplotlib.container import ErrorbarContainer
    errors = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    segments = errors[0][2][0].get_segments()
    assert [s[0][0] for s in segments] == pytest.approx(
        list(frame["importance_mean"] - frame["importance_std"]))


# --------------------------------------------------------------------------- #
#  read_and_plot__vision_results
# --------------------------------------------------------------------------- #

def _vision_tree(base, models):
    for epoch, (model, scores) in models.items():
        directory = base / epoch
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"accuracy": scores}).to_csv(
            directory / f"{model}_time1700000000_test_result.csv", index=False)


def test_only_the_best_model_is_coloured(rcparams_guard, tmp_path, capsys):
    """THE SENTENCE IS "this model scored best". The rows arrive sorted
    ascending, so exactly one bar -- the last -- carries the highlight."""
    base = tmp_path / "runs"
    _vision_tree(base, {"e1": ("resnet50", [0.80, 0.90]),
                        "e2": ("vgg16", [0.70, 0.72]),
                        "e3": ("densenet", [0.60, 0.62])})

    P.read_and_plot__vision_results(str(base), y_lim=[0.5, 1.0])
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    faces = [_hex(p.get_facecolor()) for p in ax.patches]
    assert faces[:-1] == [_hex(ROLES["data"])] * (len(faces) - 1)
    assert faces[-1] == _hex(ROLES["highlight"])
    # ...and the highlighted bar is the highest-scoring model.
    assert [t.get_text() for t in ax.get_xticklabels()][-1] == "resnet50"
    assert [p.get_height() for p in ax.patches] == pytest.approx(
        [0.61, 0.71, 0.85])


def test_a_single_model_is_not_a_comparison(rcparams_guard, tmp_path, capsys):
    """One bar highlighted out of one bar is 100% of the marks coloured, which
    is a figure with no claim rather than a figure making one."""
    base = tmp_path / "runs"
    _vision_tree(base, {"e1": ("resnet50", [0.80, 0.90])})

    P.read_and_plot__vision_results(str(base), y_lim=[0.5, 1.0])
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    assert [_hex(p.get_facecolor()) for p in ax.patches] == [
        _hex(ROLES["data"])]


def test_drawing_every_score_figure_leaves_the_globals_alone(tmp_path, capsys):
    """Rule 2 again, over the second group."""
    before = {k: repr(v) for k, v in mpl.rcParams.items()}

    P.plot_comparison_results(_comparison_rows())
    P.plot_permutation(_importance_frame())
    P.plot_feature_importance(_importance_frame())
    base = tmp_path / "runs"
    _vision_tree(base, {"e1": ("a", [0.8]), "e2": ("b", [0.7])})
    P.read_and_plot__vision_results(str(base), y_lim=[0.5, 1.0])
    capsys.readouterr()

    after = {k: repr(v) for k, v in mpl.rcParams.items()}
    assert {k for k in set(before) | set(after)
            if before.get(k) != after.get(k)} == set()


# --------------------------------------------------------------------------- #
#  The volcanoes
# --------------------------------------------------------------------------- #

def _volcano_frame(n=200):
    """A genome-wide-shaped table: a few called genes among many that are not."""
    rng = np.random.default_rng(3)
    effect = rng.normal(0.0, 0.6, n)
    p = rng.uniform(0.06, 1.0, n)
    effect[:4] = [2.4, 2.0, -2.6, -2.2]
    p[:4] = [1e-6, 1e-5, 1e-6, 1e-4]
    return pd.DataFrame({"fc": effect, "p": p,
                         "name": [f"g{i}" for i in range(n)]})


def test_the_volcano_is_grey_with_a_minority_highlighted(rcparams_guard):
    """Grey is the default ink; GREEN up and RUST down are the argument.

    The hues were crimson, royalblue and lightgray -- three colours that
    appear in no other spaCR panel, so a reader who learned them here could
    carry nothing to the next figure.
    """
    figure, ax, hits = P.volcano_plot(
        _volcano_frame(), fold_change_col="fc", p_value_col="p",
        name_col="name", fold_change_threshold=1.5, p_value_threshold=1e-3,
        annotate=False, show=False)

    faces = np.atleast_2d(ax.collections[0].get_facecolors())[:, :3]
    counts = {}
    for face in faces:
        counts[to_hex(face).lower()] = counts.get(to_hex(face).lower(), 0) + 1

    assert set(counts) == {_hex(ROLES["up"]), _hex(ROLES["down"]),
                           _hex(ROLES["data"])}
    assert counts[_hex(ROLES["up"])] == 2
    assert counts[_hex(ROLES["down"])] == 2
    # ...and the highlight is a small minority of the marks, which is the rule.
    coloured = counts[_hex(ROLES["up"])] + counts[_hex(ROLES["down"])]
    assert coloured / sum(counts.values()) < 0.1


def test_the_volcano_thresholds_are_references_not_results(rcparams_guard):
    """A threshold sorts the points; it is not one of them.

    The three lines were black at 1.0 pt -- heavier than any mark on the
    panel -- plus a fourth black rule at x = 0.
    """
    figure, ax, hits = P.volcano_plot(
        _volcano_frame(), fold_change_col="fc", p_value_col="p",
        fold_change_threshold=1.5, p_value_threshold=1e-3, show=False)

    assert len(ax.lines) == 4              # two FC rules, one p rule, x = 0
    assert all(_is_reference_line(line) for line in ax.lines)


def test_an_explicit_threshold_colour_still_wins(rcparams_guard):
    """The house default is a default. A caller who asks for green gets green."""
    figure, ax, hits = P.volcano_plot(
        _volcano_frame(), fold_change_col="fc", p_value_col="p",
        fold_change_threshold=1.5, p_value_threshold=1e-3,
        threshold_line_kwargs={"color": "green", "linestyle": ":"},
        show=False)

    assert [_hex(line.get_color()) for line in ax.lines[:3]] == [
        _hex("green")] * 3
    # ...and the cosmetic x = 0 rule is not a threshold, so it does not.
    assert _is_reference_line(ax.lines[3])


def test_the_volcano_calls_did_not_move(rcparams_guard):
    """This is a restyle: the same genes are called and labelled as before."""
    frame = _volcano_frame()
    figure, ax, hits = P.volcano_plot(
        frame, fold_change_col="fc", p_value_col="p", name_col="name",
        fold_change_threshold=1.5, p_value_threshold=1e-3, show=False)

    expected = frame[(frame["fc"].abs() >= 1.5) & (frame["p"] <= 1e-3)]
    assert sorted(hits) == sorted(expected["name"])
    assert all(text.get_fontsize() == pytest.approx(TYPE_SCALE["annotation"])
               for text in ax.texts)


def test_a_volcano_with_no_thresholds_is_entirely_grey(rcparams_guard):
    """Nothing was called, so nothing is claimed, so nothing is coloured."""
    figure, ax, hits = P.volcano_plot(
        _volcano_frame(), fold_change_col="fc", p_value_col="p", show=False)

    faces = np.atleast_2d(ax.collections[0].get_facecolors())[:, :3]
    assert {to_hex(f).lower() for f in faces} == {_hex(ROLES["data"])}


def test_the_regression_volcano_stopped_colouring_every_gene(rcparams_guard):
    """``_reg_v_plot`` ran ``cmap='coolwarm'`` over ``np.sign(effect)``: every
    point coloured, by a fact the x axis already states. That is the failure
    the grey rule exists to prevent."""
    frame = pd.DataFrame(
        {"effect": [1.5, -2.0, 0.3, -0.1, 0.8, -0.6],
         "p": [0.001, 0.02, 0.4, 0.9, 0.7, 0.6]},
        index=[f"g{i}" for i in range(6)])

    P._reg_v_plot(frame)
    ax = _figures()[0].axes[0]

    faces = np.atleast_2d(ax.collections[0].get_facecolors())[:, :3]
    drawn = [to_hex(f).lower() for f in faces]
    assert drawn == [_hex(ROLES["up"]), _hex(ROLES["down"])] + \
        [_hex(ROLES["data"])] * 4
    # The p = 0.05 rule is a reference, and the canvas is a panel: 40x30
    # inches at the 300 dpi the save preference asks for is 108 megapixels.
    assert any(_is_reference_line(line) for line in ax.lines)
    assert max(_figures()[0].get_size_inches()) <= 12.0


# --------------------------------------------------------------------------- #
#  create_venn_diagram
# --------------------------------------------------------------------------- #

def _gene_csv(path, genes, coefficients):
    pd.DataFrame({"gene": list(genes),
                  "coefficient": list(coefficients)}).to_csv(path, index=False)
    return str(path)


def test_only_the_overlap_of_a_venn_is_coloured(rcparams_guard, tmp_path,
                                                capsys):
    """The sentence a Venn makes is the overlap.

    matplotlib_venn's own defaults are a red circle and a green one at alpha
    0.4 -- two arguments where the figure has one, in the one hue pair
    red-green deficiency removes.
    """
    first = _gene_csv(tmp_path / "a.csv", ["a", "b", "c"], [0.5, 0.5, 0.5])
    second = _gene_csv(tmp_path / "b.csv", ["b", "c", "d"], [0.5, 0.5, 0.5])

    result = P.create_venn_diagram(first, second, save=False)
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    faces = [_hex(p.get_facecolor()) for p in ax.patches]
    assert faces.count(_hex(ROLES["highlight"])) == 1
    assert set(faces) == {_hex(ROLES["data"]), _hex(Palette.GREY_DARK),
                          _hex(ROLES["highlight"])}
    assert all(p.get_alpha() in (None, 1.0) for p in ax.patches)
    # ...and the sets themselves are untouched.
    assert sorted(result["overlap"]) == ["b", "c"]
    assert result["unique_to_file1"] == ["a"]
    assert result["unique_to_file2"] == ["d"]


def test_a_venn_with_no_overlap_still_draws(rcparams_guard, tmp_path, capsys):
    """An empty region has no patch at all, which is a None to step over."""
    first = _gene_csv(tmp_path / "a.csv", ["a"], [0.5])
    second = _gene_csv(tmp_path / "b.csv", ["z"], [0.5])

    result = P.create_venn_diagram(first, second, save=False)
    capsys.readouterr()
    assert result["overlap"] == []


def test_drawing_every_volcano_leaves_the_globals_alone(tmp_path, capsys):
    """Rule 2, over the group that draws the figure spaCR shows most."""
    before = {k: repr(v) for k, v in mpl.rcParams.items()}

    P.volcano_plot(_volcano_frame(), fold_change_col="fc", p_value_col="p",
                   name_col="name", fold_change_threshold=1.5,
                   p_value_threshold=1e-3, show=False)
    P._reg_v_plot(pd.DataFrame({"effect": [1.0, -1.0], "p": [0.01, 0.5]},
                               index=["a", "b"]))
    P.create_venn_diagram(
        _gene_csv(tmp_path / "a.csv", ["a", "b"], [0.5, 0.5]),
        _gene_csv(tmp_path / "b.csv", ["b", "c"], [0.5, 0.5]), save=False)
    capsys.readouterr()

    after = {k: repr(v) for k, v in mpl.rcParams.items()}
    assert {k for k in set(before) | set(after)
            if before.get(k) != after.get(k)} == set()


# --------------------------------------------------------------------------- #
#  The categorical panels: recruitment, controls, jitter, proportions
# --------------------------------------------------------------------------- #

_RECRUITMENT_EXTRA = [
    "pathogen_cytoplasm_mean_mean", "pathogen_cytoplasm_q75_mean",
    "pathogen_periphery_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_mean_mean",
    "pathogen_outside_cytoplasm_q75_mean",
]


def _recruitment_frame(channel=1, n=24):
    rng = np.random.default_rng(0)
    data = {"condition": ["ctrl", "trt"] * (n // 2),
            "pathogen": ["wt", "mut"] * (n // 2)}
    for component in ("cell", "nucleus", "cytoplasm", "pathogen"):
        data[f"{component}_channel_{channel}_mean_intensity"] = rng.uniform(
            10, 100, n)
    for column in _RECRUITMENT_EXTRA:
        data[column] = rng.uniform(2, 50, n)
    return pd.DataFrame(data)


def test_the_recruitment_grid_hides_the_axes_it_did_not_fill(rcparams_guard,
                                                             capsys):
    """An empty framed box reads as a panel that failed to draw."""
    P._plot_recruitment(_recruitment_frame(), "test", 1, figuresize=4)
    capsys.readouterr()

    grid = _figures()[1]
    assert len(grid.axes) == 6
    assert grid.axes[5].axison is False
    assert all(ax.axison for ax in grid.axes[:5])


def test_the_recruitment_ticks_are_anchored_not_merely_rotated(rcparams_guard,
                                                               capsys):
    """A condition name rotated about its centre drifts off its own tick."""
    P._plot_recruitment(_recruitment_frame(), "test", 1, figuresize=4)
    capsys.readouterr()

    for label in _figures()[0].axes[0].get_xticklabels():
        assert label.get_rotation() == pytest.approx(45.0)
        assert label.get_ha() == "right"


def test_the_recruitment_palette_is_still_its_own_and_still_local(capsys):
    """The four hues stay -- the category IS the data here -- and the house
    style must not have turned the local palette into a global one."""
    before = repr(mpl.rcParams["axes.prop_cycle"])
    P._plot_recruitment(_recruitment_frame(), "test", 1, figuresize=4)
    capsys.readouterr()

    import seaborn as sns
    intended = [(55 / 255, 155 / 255, 155 / 255),
                (155 / 255, 55 / 255, 155 / 255)]
    bars = {tuple(np.round(patch.get_facecolor()[:3], 4))
            for patch in _figures()[0].axes[0].patches}
    assert bars == {tuple(np.round(sns.desaturate(colour, 0.75), 4))
                    for colour in intended}
    assert repr(mpl.rcParams["axes.prop_cycle"]) == before


def _controls_frame(chans=(0, 1), conditions=("ctrl", "trt"), n=12):
    rng = np.random.default_rng(1)
    data = {"condition": list(conditions) * (n // len(conditions))}
    for chan in chans:
        for component in ("cell", "nucleus", "pathogen", "cytoplasm"):
            data[f"{component}_channel_{chan}_mean_intensity"] = rng.uniform(
                5, 50, n)
    return pd.DataFrame(data)


def test_the_control_components_are_not_coloured_twice(rcparams_guard):
    """The four components are already the x axis of every panel; giving each
    one a hue as well argues nothing."""
    P._plot_controls(_controls_frame(), [0], channel_of_interest=1,
                     figuresize=1)

    figure = _figures()[0]
    faces = {_hex(patch.get_facecolor())
             for ax in figure.axes for patch in ax.patches}
    assert faces == {_hex(ROLES["data"])}


def test_the_control_means_did_not_move(rcparams_guard):
    """Restyle only: each bar is still its condition's mean intensity."""
    frame = _controls_frame(chans=(0, 1))
    P._plot_controls(frame, [0], channel_of_interest=1, figuresize=1)

    ax = _figures()[0].axes[0]
    expected = [frame[frame["condition"] == "ctrl"]
                [f"{component}_channel_0_mean_intensity"].mean()
                for component in ("cell", "nucleus", "pathogen", "cytoplasm")]
    assert [p.get_height() for p in ax.patches] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
#  plot_proportion_stacked_bars
# --------------------------------------------------------------------------- #

def _proportion_frame(n=120):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "group": rng.choice(["a", "b"], n),
        "bin": rng.choice(["b1", "b2", "b3"], n),
        "prc": rng.choice([f"p1_r1_c{i}" for i in range(4)], n),
    })


def test_the_ordered_bins_get_a_single_hue_ramp(rcparams_guard, capsys):
    """Volume bins are ORDERED, so their encoding is one hue light-to-dark.

    viridis was the literal every internal call site was written with, not a
    choice; it is treated as unset here exactly as plot_plates treats it.
    """
    from matplotlib import colormaps

    results, pairwise, figure = P.plot_proportion_stacked_bars(
        {"verbose": False}, _proportion_frame(), "group", "bin",
        level="object")
    capsys.readouterr()

    ax = figure.axes[0]
    faces = [_hex(p.get_facecolor()) for p in ax.patches]
    blues = colormaps[Palette.SEQUENTIAL]
    assert set(faces) <= {_hex(blues(v)) for v in np.linspace(0, 1, 256)}
    # ...and a caller who names another map still gets it.
    _r, _p, other = P.plot_proportion_stacked_bars(
        {"verbose": False}, _proportion_frame(), "group", "bin",
        level="object", cmap="autumn")
    capsys.readouterr()
    autumn = colormaps["autumn"]
    assert {_hex(p.get_facecolor()) for p in other.axes[0].patches} <= {
        _hex(autumn(v)) for v in np.linspace(0, 1, 256)}


def test_the_proportions_and_the_chi_squared_did_not_move(rcparams_guard,
                                                          capsys):
    """A restyle that silently changes a number is the worst outcome here."""
    from scipy.stats import chi2_contingency

    frame = _proportion_frame()
    results, pairwise, figure = P.plot_proportion_stacked_bars(
        {"verbose": False}, frame, "group", "bin", level="object")
    capsys.readouterr()

    counts = frame.groupby(["group", "bin"], observed=True).size().unstack(
        fill_value=0)
    chi2, p, _dof, _expected = chi2_contingency(counts)
    assert results["chi_squared_stat"][0] == pytest.approx(chi2)
    assert results["p_value"][0] == pytest.approx(p)
    assert figure.axes[0].get_ylim() == (0.0, 1.0)


# --------------------------------------------------------------------------- #
#  jitterplot_by_annotation
# --------------------------------------------------------------------------- #

def test_the_jitter_classes_are_grey_and_carry_a_mean_bar(rcparams_guard,
                                                          monkeypatch, capsys):
    """The annotation classes ARE the x axis. A viridis ramp over them said
    nothing the axis had not said, and implied an order that is not there.

    The mean bar is ADDED, and this test says so: a dot strip without its
    mean is a cloud, and GREY_DARK is the palette's own role for a mean bar.
    """
    import spacr.io as sio

    frame = pd.DataFrame({
        "annotation": ["pos"] * 6 + ["neg"] * 6,
        "recruitment": np.concatenate([np.linspace(1.0, 2.0, 6),
                                       np.linspace(3.0, 4.0, 6)]),
        "plateID": ["p1"] * 12,
        "rowID": ["r1"] * 12,
        "columnID": [f"c{i % 3}" for i in range(12)],
        "prcfo": [f"o{i}" for i in range(12)],
    })
    monkeypatch.setattr(sio, "_read_and_merge_data",
                        lambda *a, **k: (frame.copy(), []))
    monkeypatch.setattr(sio, "_read_db",
                        lambda *a, **k: [frame[["prcfo"]].copy()])

    out = P.jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    capsys.readouterr()

    ax = _figures()[0].axes[0]
    strips = list(ax.collections)
    assert len(strips) == 2                      # one collection per class
    for strip in strips:
        assert {_hex(tuple(c)) for c in strip.get_facecolors()} == {
            _hex(ROLES["data"])}

    drawn = sorted(round(float(line.get_ydata()[0]), 6) for line in ax.lines)
    expected = sorted(
        round(float(out.loc[out["annotation"] == group,
                            "recruitment"].mean()), 6)
        for group in ("pos", "neg"))
    assert drawn == expected
    assert {_hex(line.get_color()) for line in ax.lines} == {
        _hex(Palette.GREY_DARK)}
    for label in ax.get_xticklabels():
        assert label.get_ha() == "right"


def test_drawing_every_categorical_figure_leaves_the_globals_alone(capsys):
    """Rule 2, over the group that draws the most axes at once."""
    before = {k: repr(v) for k, v in mpl.rcParams.items()}

    P._plot_recruitment(_recruitment_frame(), "test", 1, figuresize=4)
    P._plot_controls(_controls_frame(), [0], channel_of_interest=1,
                     figuresize=1)
    P.plot_proportion_stacked_bars({"verbose": False}, _proportion_frame(),
                                   "group", "bin", level="object")
    capsys.readouterr()

    after = {k: repr(v) for k, v in mpl.rcParams.items()}
    assert {k for k in set(before) | set(after)
            if before.get(k) != after.get(k)} == set()
