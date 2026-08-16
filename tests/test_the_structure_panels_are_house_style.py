"""The four structure QC panels, in the house style.

The cluster: ``coefficient_forest``, ``vif``, ``condition_number`` and
``predictor_correlation`` -- ranked bars, a forest and a matrix. What they
were before:

* the forest drew every coefficient in the same teal whether its interval
  cleared zero or not, and drew the zero line itself in the loudest colour in
  the module;
* the VIF panel was a **traffic light** -- green under 5, orange under 10, red
  above -- which colours every bar and so states nothing, and which spends
  GREEN on "healthy" in a report whose forest one panel over uses GREEN for
  "called, upregulated";
* the conditioning panel wrote its headline over its own bars and printed a
  condition number of 1.4e16 as nineteen digits;
* the correlation matrix carried a sentence title, 90-degree ticks and a
  caption that named the worst-correlated pair by their shared 14-character
  prefix.

These tests assert the artists on the axes, not the imports: a panel that
imports the style module and then draws in red would pass an import test.

The statistics are asserted too. This is a restyle -- if any number moved,
the change is wrong.
"""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sm = pytest.importorskip("statsmodels.api")

from spacr import regression_qc as rq  # noqa: E402
from spacr.figures.style import (INK_PRINT, ROLES, TYPE_SCALE,  # noqa: E402
                                 WEIGHTS, Palette, resolve_ink)

#: The cluster, in report order.
STRUCTURE = ("vif", "condition_number", "predictor_correlation",
             "coefficient_forest")

#: The five module-level hexes these panels used to draw with. None is in the
#: published palette; none may survive in this cluster.
OLD_PALETTE = ("#1f6f8b", "#d1495b", "#8d99ae", "#e07a3f", "#2a9d8f")

#: The real screen. Read-only.
REAL_DESIGN = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/test/"
               "results/plate1_dv/ols/list/regression_data.csv")


# --------------------------------------------------------------------------- #
#  Fixtures: two designs, because one of them exercises no warning at all
# --------------------------------------------------------------------------- #

def _axes(size=(5.6, 4.4)):
    figure = Figure(figsize=size, dpi=140)
    return figure, figure.subplots()


#: How many predictors the clean design carries, and how many of them do
#: anything. Two real effects among twelve, which is the shape of a screen and
#: the shape the grey-plus-highlight rule needs: a highlight that is a
#: minority of the marks.
CLEAN_PREDICTORS = tuple("abcdefghijkl")
CLEAN_EFFECTS = ("a", "b")


@pytest.fixture(scope="module")
def clean():
    """A well-conditioned design: no VIF above 10, nothing aliased."""
    rng = np.random.default_rng(11)
    n = 240
    frame = pd.DataFrame({"Intercept": np.ones(n)})
    for name in CLEAN_PREDICTORS:
        frame[name] = rng.normal(size=n)
    y = 1.0 + 4.0 * frame["a"] - 3.0 * frame["b"] + rng.normal(size=n) * 0.4
    return rq.build_context(sm.OLS(y, frame).fit(), frame, y,
                            regression_type="ols")


@pytest.fixture(scope="module")
def collinear():
    """The dummy-variable trap: exactly aliased columns, rank deficiency.

    Every warning branch in the cluster fires on this one -- infinite VIFs,
    a condition number of 1e16, an off-diagonal |r| of exactly 1.
    """
    rng = np.random.default_rng(12)
    n = 200
    a = rng.normal(size=n)
    frame = pd.DataFrame({
        "Intercept": np.ones(n),
        "left": (np.arange(n) % 2).astype(float),
        "right": 1.0 - (np.arange(n) % 2).astype(float),
        "cells": a,
        "cells_thousands": a / 1000.0,
        "other": rng.normal(size=n)})
    y = a + rng.normal(size=n) * 0.3
    return rq.build_context(sm.OLS(y, frame).fit(), frame, y,
                            regression_type="ols")


def _hexes(ax):
    """Every colour that ended up on the axes, as lower-case hex."""
    out = set()
    for patch in ax.patches:
        out.add(to_hex(patch.get_facecolor()))
    for line in ax.lines:
        out.add(to_hex(line.get_color()))
    for text in ax.texts:
        out.add(to_hex(text.get_color()))
    for collection in ax.collections:
        for array in (collection.get_facecolor(), collection.get_edgecolor()):
            for rgba in np.atleast_2d(array):
                if len(rgba) >= 3:
                    out.add(to_hex(rgba))
    return {value.lower() for value in out}


# --------------------------------------------------------------------------- #
#  Everything is grey except what the sentence is about
# --------------------------------------------------------------------------- #

def test_the_forest_greys_every_term_whose_interval_crosses_zero(clean):
    """Grey unless the interval excludes zero, and then the sign picks the
    colour -- the rule spacr.figures.panels.effect_rank already uses, so a
    reader who learned green/rust from the volcano sheet reads this for free.
    """
    figure, ax = _axes()
    rq.draw_panel("coefficient_forest", clean, ax)

    intervals = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert intervals, "the forest drew no intervals at all"
    colours = [to_hex(rgba).lower()
               for rgba in np.atleast_2d(intervals[0].get_color())]
    assert set(colours) <= {ROLES["data"].lower(), ROLES["up"].lower(),
                            ROLES["down"].lower()}, set(colours)
    grey = sum(1 for c in colours if c == ROLES["data"].lower())
    assert grey > len(colours) / 2, (
        f"only {grey} of {len(colours)} terms are grey; a forest where half "
        f"the terms are coloured has no claim")
    # ...and the claim is actually made: this design has two huge effects.
    assert grey < len(colours), "nothing was called at all"


def test_the_forest_colours_nothing_when_the_fit_offers_no_interval(clean):
    """A penalised fit has no covariance matrix, so nothing has been called,
    so nothing may be coloured. `crosses_zero` is all-False in that branch,
    and a naive port of the interval rule would paint every single term."""

    class Penalised:
        """A sklearn-shaped estimator: coefficients, no inference."""

        coef_ = np.asarray(clean.model.params, dtype=float)

        def predict(self, X):
            return np.asarray(X, dtype=float) @ self.coef_

    ctx = rq.build_context(Penalised(), clean.X, clean.y,
                           regression_type="lasso")
    figure, ax = _axes()
    stats = rq.draw_panel("coefficient_forest", ctx, ax)

    assert stats["has_intervals"] is False
    marks = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert marks, "the forest drew no coefficients"
    colours = {to_hex(rgba).lower()
               for rgba in np.atleast_2d(marks[0].get_facecolor())}
    assert colours == {ROLES["data"].lower()}, colours


def test_the_vif_traffic_light_is_gone(clean, collinear):
    """Grey and RUST only.

    GREEN is `up` in the shared vocabulary. Spending it on "healthy VIF" in a
    report that also carries a coefficient forest using `up`/`down` for
    exactly that makes "no collinearity problem" read as "called".
    """
    for ctx in (clean, collinear):
        figure, ax = _axes()
        rq.draw_panel("vif", ctx, ax)
        bars = {to_hex(p.get_facecolor()).lower() for p in ax.patches}
        assert Palette.GREEN.lower() not in bars, bars
        assert bars <= {ROLES["data"].lower(), ROLES["down"].lower()}, bars

    # ...and it still separates: nothing is coloured on a clean design, and
    # the aliased predictors are on the collinear one.
    figure, ax = _axes()
    rq.draw_panel("vif", clean, ax)
    assert {to_hex(p.get_facecolor()).lower()
            for p in ax.patches} == {ROLES["data"].lower()}

    figure, ax = _axes()
    stats = rq.draw_panel("vif", collinear, ax)
    assert stats["n_aliased"] >= 2
    assert ROLES["down"].lower() in {to_hex(p.get_facecolor()).lower()
                                     for p in ax.patches}


def test_the_conditioning_bars_are_grey_and_the_verdict_carries_the_colour(
        clean, collinear):
    """A singular-value spectrum has no minority to highlight -- its shape is
    the claim -- so the bars stay grey. The verdict is the only warning on the
    panel, so it is the only thing entitled to a colour, and that colour is
    never GREEN."""
    figure, ax = _axes()
    rq.draw_panel("condition_number", clean, ax)
    assert {to_hex(p.get_facecolor()).lower()
            for p in ax.patches} == {ROLES["data"].lower()}
    assert Palette.GREEN.lower() not in _hexes(ax), (
        "a healthy condition number is still drawn in the 'called up' colour")

    figure, ax = _axes()
    stats = rq.draw_panel("condition_number", collinear, ax)
    assert stats["condition_number"] >= 30
    warned = [t for t in ax.texts
              if to_hex(t.get_color()).lower() == ROLES["down"].lower()]
    assert warned, "a severely collinear design said nothing in the warning ink"
    assert any("scaled condition number" in t.get_text() for t in warned)


def test_a_condition_number_of_1e16_is_printed_in_a_form_that_fits(collinear):
    """`f"{1.3583e16:,.1f}"` is `13,583,837,847,143,152.0` -- nineteen
    characters straight across the axes, and nobody reads to the end."""
    figure, ax = _axes()
    stats = rq.draw_panel("condition_number", collinear, ax)
    assert stats["condition_number"] > 1e10, "the fixture is not singular"
    headline = [t.get_text() for t in ax.texts
                if "scaled condition number" in t.get_text()]
    assert headline, "the panel no longer states its own headline number"
    assert len(headline[0]) < 45, headline[0]
    assert "," not in headline[0].split("=")[-1], headline[0]


# --------------------------------------------------------------------------- #
#  Reference lines are thin, dashed and grey -- never a result
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,expected", [("vif", (5.0, 10.0)),
                                           ("coefficient_forest", (0.0,))])
def test_the_thresholds_are_thin_grey_dashes(clean, name, expected):
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    rules = {round(float(line.get_xdata()[0]), 6): line for line in ax.lines
             if len(set(line.get_xdata())) == 1}
    for value in expected:
        assert value in rules, f"{name} lost its rule at x = {value}"
        line = rules[value]
        assert to_hex(line.get_color()).lower() == ROLES["reference"].lower()
        assert line.get_linewidth() <= WEIGHTS["reference"] + 1e-9, (
            f"{name}'s rule at {value} is {line.get_linewidth()}pt; a "
            f"reference is not a result")
        assert line.get_linestyle() not in ("-", "solid")


def test_the_vif_thresholds_are_drawn_over_the_bars_not_under_them(collinear):
    """style.reference_line parks a rule at zorder 0, which is right for a
    scatter and wrong for bars: an aliased design fills the panel to the
    right of both guides and would bury them."""
    figure, ax = _axes()
    rq.draw_panel("vif", collinear, ax)
    bars = max(p.get_zorder() for p in ax.patches)
    rules = [line for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert rules
    assert all(line.get_zorder() > bars for line in rules)


# --------------------------------------------------------------------------- #
#  No boxes, no sentence titles, no framed legends
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", STRUCTURE)
def test_no_note_is_drawn_inside_a_box(clean, name):
    """`_note` wrapped every statistics block in a white rounded rectangle.
    The style has no other boxes for it to match, and white is invisible on a
    transparent ground."""
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    boxed = [t.get_text()[:40] for t in ax.texts if t.get_bbox_patch()]
    assert not boxed, boxed


@pytest.mark.parametrize("name", STRUCTURE)
def test_no_panel_carries_a_sentence_title(clean, name):
    """"No panel titles as sentences. The axis labels carry the content. If a
    panel needs a descriptor it is 2-4 words above the axes." """
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    title = ax.get_title()
    assert title, f"{name} lost its descriptor entirely"
    assert "\n" not in title, title
    assert len(title.split()) <= 4, title
    assert title == title.lower(), title
    assert "(" not in title and "=" not in title, title


@pytest.mark.parametrize("name", STRUCTURE)
def test_no_panel_draws_a_framed_legend(clean, name):
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    assert ax.get_legend() is None


@pytest.mark.parametrize("name", STRUCTURE)
def test_none_of_the_old_hardcoded_hexes_survive(clean, name):
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    left = _hexes(ax) & {h.lower() for h in OLD_PALETTE}
    assert not left, f"{name} still draws with {sorted(left)}"


# --------------------------------------------------------------------------- #
#  The chrome, which the context manager cannot reach on its own
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", STRUCTURE)
def test_the_chrome_takes_the_reports_ink_and_the_house_type(clean, name):
    """The axes exists before the panel is called, so rcParams inside the
    `with` block cannot colour its spines, ticks or axis labels. Whatever the
    panel does about that, the result has to be the house ink."""
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    ink = resolve_ink(rq._REPORT_TARGET).lower()

    for side in ("left", "bottom"):
        spine = ax.spines[side]
        assert spine.get_visible()
        assert to_hex(spine.get_edgecolor()).lower() == ink
        assert spine.get_linewidth() == pytest.approx(WEIGHTS["spine"])
    for side in ("top", "right"):
        assert not ax.spines[side].get_visible(), (
            f"{name} is box-framed; this figure is L-framed")

    assert to_hex(ax.xaxis.label.get_color()).lower() == ink
    assert ax.xaxis.label.get_fontsize() == pytest.approx(TYPE_SCALE["label"])
    assert to_hex(ax.title.get_color()).lower() == ink
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    assert labels, f"{name} has no tick labels at all"
    assert max(t.get_fontsize() for t in labels) <= TYPE_SCALE["tick"] + 1e-9


def test_the_report_ink_contrasts_with_the_page_the_report_writes():
    """The decision this module had to make, pinned.

    Every panel here is written to `<results>/regression_qc/` as a file and
    read on a page. The report driver builds its `Figure` before any style
    context is entered, so the figure keeps matplotlib's white facecolor and
    `_save` writes that white into the PDF. `theme_target()` answers a
    different question -- what the GUI theme is doing -- and returns 'screen'
    for every user who has not explicitly chosen a white figure background,
    which resolves to INK_SCREEN (#E8EDEE). #E8EDEE on a white page is
    invisible.
    """
    assert rq._REPORT_TARGET == "print"
    ink = resolve_ink(rq._REPORT_TARGET)
    assert ink == INK_PRINT

    page = np.asarray(Figure().get_facecolor()[:3])
    assert page.mean() > 0.9, "the report driver no longer writes a white page"
    assert np.asarray(matplotlib.colors.to_rgb(ink)).mean() < 0.4, (
        f"{ink} on a white page is unreadable")


@pytest.mark.parametrize("name", STRUCTURE)
def test_each_panel_opens_the_style_as_a_context_manager(name):
    """Not `rcParams.update`. spaCR draws from a long-lived GUI, and a
    process-wide style change restyles every figure drawn afterwards until
    the process exits. Parsed rather than grepped, so a comment about the
    rule cannot pass for the rule."""
    _title, _group, fn = rq._PANEL_BY_NAME[name]
    tree = ast.parse(inspect.getsource(fn).lstrip())
    opened = [item for node in ast.walk(tree)
              if isinstance(node, ast.With)
              for item in node.items
              if isinstance(item.context_expr, ast.Call)
              and getattr(item.context_expr.func, "id", "") == "figure_style"]
    assert opened, f"{name} never enters figure_style()"

    assert not [node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and getattr(node.func.value, "attr", "") == "rcParams"]


@pytest.mark.parametrize("name", STRUCTURE)
def test_drawing_a_structure_panel_leaves_no_rcparams_behind(clean, name):
    before = dict(plt.rcParams)
    figure, ax = _axes()
    rq.draw_panel(name, clean, ax)
    changed = {k for k in before if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, sorted(changed)


# --------------------------------------------------------------------------- #
#  What must NOT change
# --------------------------------------------------------------------------- #

def test_the_correlation_matrix_keeps_its_diverging_map(collinear):
    """A refusal, pinned so a later sweep cannot undo it.

    Pearson r on [-1, 1] is a genuinely signed quantity, which is the one case
    the skill permits a diverging colormap. Mapping it to the categorical
    palette or to Palette.SEQUENTIAL would put r = -0.9 and r = +0.9 at the
    same lightness and destroy the panel's only encoding. "Everything grey
    except the claim" does not apply either: the whole matrix is the claim.
    """
    figure, ax = _axes()
    rq.draw_panel("predictor_correlation", collinear, ax)
    images = ax.get_images()
    assert len(images) == 1
    assert images[0].get_cmap().name == "RdBu_r"
    assert images[0].get_clim() == (-1, 1)


def test_the_correlation_caption_names_the_pair_it_found(collinear):
    """18 characters truncated every spaCR predictor inside its own bracket:
    "between fraction:grna[2130 and fraction:grna[2276" names no pair."""
    figure, ax = _axes()
    stats = rq.draw_panel("predictor_correlation", collinear, ax)
    caption = " ".join(t.get_text() for t in ax.texts)
    for name in stats["max_pair"]:
        assert name in caption, (name, caption)


@pytest.mark.parametrize("degrees", [45])
def test_the_correlation_ticks_rotate_45_not_90(collinear, degrees):
    """"Long categorical tick labels rotate 45 degrees, right-aligned." """
    figure, ax = _axes()
    rq.draw_panel("predictor_correlation", collinear, ax)
    rotations = {t.get_rotation() for t in ax.get_xticklabels()}
    assert rotations == {float(degrees)}, rotations


def test_the_statistics_are_untouched(clean, collinear):
    """A restyle. Every number a panel returns is what it returned before."""
    out = {}
    for ctx, tag in ((clean, "clean"), (collinear, "collinear")):
        for name in STRUCTURE:
            figure, ax = _axes()
            out[(tag, name)] = rq.draw_panel(name, ctx, ax)

    width = len(CLEAN_PREDICTORS) + 1          # + the intercept

    forest = out[("clean", "coefficient_forest")]
    assert forest["n_total"] == width and forest["n_shown"] == width
    assert forest["has_intervals"] is True
    assert forest["largest_term"] == "a"
    assert forest["largest_coefficient"] == pytest.approx(
        float(clean.model.params["a"]))

    vif = out[("clean", "vif")]
    assert vif["n_above_10"] == 0 and vif["n_aliased"] == 0
    assert vif["n_constant"] == 1              # the intercept
    assert vif["max_vif"] < 1.5

    cond = out[("clean", "condition_number")]
    assert cond["condition_number"] < 10
    assert cond["verdict"] == "no collinearity problem"
    assert cond["rank"] == cond["n_singular_values"] == width

    corr = out[("clean", "predictor_correlation")]
    assert corr["n_predictors"] == len(CLEAN_PREDICTORS)   # intercept dropped
    assert corr["truncated"] is False

    bad = out[("collinear", "condition_number")]
    assert bad["condition_number"] > 1e10
    assert bad["rank"] < bad["n_singular_values"]
    assert "severe collinearity" in bad["verdict"]
    assert out[("collinear", "vif")]["n_aliased"] >= 2
    assert out[("collinear", "predictor_correlation")][
        "max_abs_offdiagonal"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", STRUCTURE)
def test_a_panel_that_cannot_be_computed_still_says_why(name):
    """A skipped panel must name what was missing, not raise something else."""
    frame = pd.DataFrame({"Intercept": np.ones(6)})
    y = pd.Series(np.arange(6, dtype=float))
    ctx = rq.build_context(sm.OLS(y, frame).fit(), frame, y,
                           regression_type="ols")
    figure, ax = _axes()
    try:
        rq.draw_panel(name, ctx, ax)
    except rq.PanelUnavailable as exc:
        assert str(exc), f"{name} skipped without a reason"


# --------------------------------------------------------------------------- #
#  The real screen
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_DESIGN),
                    reason="the tsg101 screen is not on this machine")
def test_the_real_screen_draws_every_structure_panel_in_the_house_style():
    """1,945 wells, 824 gRNA columns, the design spaCR actually fits.

    The clean fixture cannot stand in for it: this design is well conditioned,
    so every VIF bar and every singular value comes out grey, and the forest
    calls 5 of its 25 strongest terms. That is the house rule working -- the
    highlight is a minority and a healthy design makes no claim at all.
    """
    frame = pd.read_csv(REAL_DESIGN).dropna(subset=["pred", "fraction", "grna"])
    dummies = pd.get_dummies(frame["grna"].astype(str), dtype=float)
    X = dummies.mul(frame["fraction"].to_numpy(), axis=0)
    X.columns = [f"fraction:grna[{c}]" for c in X.columns]
    X.insert(0, "Intercept", 1.0)
    y = frame["pred"].astype(float)

    fit = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
    intervals = np.asarray(fit.conf_int())

    class Named:
        """The fit with its column names back on, and an interval each."""

        def __init__(self):
            self.params = pd.Series(np.asarray(fit.params), index=X.columns)

        def conf_int(self, *args, **kwargs):
            return intervals

        def __getattr__(self, item):
            return getattr(fit, item)

    ctx = rq.build_context(Named(), X, y, regression_type="ols")
    assert ctx.n == 1945 and ctx.p == 824

    allowed = {ROLES["data"].lower(), ROLES["up"].lower(),
               ROLES["down"].lower(), ROLES["reference"].lower(),
               ROLES["fill"].lower(), resolve_ink(rq._REPORT_TARGET).lower()}
    for name in STRUCTURE:
        figure, ax = _axes()
        stats = rq.draw_panel(name, ctx, ax)
        assert stats
        if name == "predictor_correlation":
            continue           # the diverging map is the encoding, see above
        stray = _hexes(ax) - allowed
        assert not stray, f"{name} drew {sorted(stray)} on the real screen"

    figure, ax = _axes()
    rq.draw_panel("coefficient_forest", ctx, ax)
    intervals_drawn = [c for c in ax.collections
                       if isinstance(c, LineCollection)][0]
    colours = [to_hex(rgba).lower()
               for rgba in np.atleast_2d(intervals_drawn.get_color())]
    grey = sum(1 for c in colours if c == ROLES["data"].lower())
    assert grey == 20 and len(colours) == 25, (grey, len(colours))
