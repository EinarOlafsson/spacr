"""The QC report's three distribution panels, drawn in the house style.

The cluster: ``residual_distribution``, ``response_distribution`` and
``p_value_histogram`` — every panel in :mod:`spacr.regression_qc` whose mark
is a histogram bar.

Before this file existed all three drew a saturated teal bar at 55-75% alpha,
a crimson reference curve heavier than the data, a framed matplotlib legend,
a stats block inside a white rounded box, and a two-line sentence title
carrying the n. The visual system in ``.claude/skills/apicomplexan-figures``
says the opposite of nearly all of that, so each rule it states is one test
here.

THE ONE THAT MATTERS MOST is the leak test. ``figure_style`` is a
``plt.rc_context``; if a panel ever applies the style by writing rcParams
instead, drawing one QC report restyles every figure the GUI draws afterwards
for the rest of the session.

The second-most important is :func:`test_the_statistics_are_untouched`. This
was a restyle, not a re-analysis, and the numbers each panel returns are the
part a reviewer quotes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spacr import regression_qc as rq  # noqa: E402
from spacr.figures.style import (INK_PRINT, ROLES, TYPE_SCALE, WEIGHTS,
                                 resolve_ink)  # noqa: E402

#: Every panel this file owns.
CLUSTER = ("residual_distribution", "response_distribution",
           "p_value_histogram")


def _axes():
    """A bare Axes on a pyplot-free Figure — the shape the report driver uses.

    Deliberately built OUTSIDE any style context, because that is what
    ``regression_qc_report`` does: it makes the ``Figure`` and then hands the
    axes to the panel. A panel that only styles artists it creates itself
    leaves this axes with black library-default ticks.
    """
    figure = Figure(figsize=(5.6, 4.4))
    return figure, figure.subplots()


def _context(n=240, seed=11, n_coefficients=600):
    """An OLS fit plus a coefficient table, the two inputs this cluster needs."""
    rng = np.random.default_rng([seed, 2])
    design = pd.DataFrame({"Intercept": np.ones(n),
                           "x1": rng.normal(size=n),
                           "x2": rng.normal(size=n)})
    y = 1.0 + 2.0 * design["x1"] - 0.5 * design["x2"] + rng.normal(size=n) * .4
    coefficients = pd.DataFrame({
        "feature": [f"grna[g{i}]" for i in range(n_coefficients)],
        "coefficient": rng.normal(size=n_coefficients),
        "p_value": rng.uniform(size=n_coefficients)})
    return rq.build_context(sm.OLS(y, design).fit(), design, y,
                            coef_df=coefficients, regression_type="ols")


def _draw(name, ctx=None):
    """Draw one panel of the cluster and hand back the axes and its stats."""
    figure, ax = _axes()
    stats = rq.draw_panel(name, ctx if ctx is not None else _context(), ax)
    return figure, ax, stats


def _bars(ax):
    """The histogram rectangles, which is every patch these panels draw."""
    from matplotlib.patches import Rectangle
    return [p for p in ax.patches if isinstance(p, Rectangle)]


# --------------------------------------------------------------------------- #
#  The style is scoped
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", CLUSTER)
def test_drawing_a_distribution_panel_does_not_leak_into_the_process(name):
    """spaCR draws from a long-lived GUI. A global rcParams update here is
    how every later figure in the session inherits this panel's font."""
    before = dict(plt.rcParams)
    _draw(name)
    changed = {k for k in before if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, f"{name} left these rcParams changed: {sorted(changed)}"


# --------------------------------------------------------------------------- #
#  Everything is grey except the claim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", CLUSTER)
def test_the_bars_are_the_house_histogram_fill(name):
    """CORAL, the skill's stated density/histogram fill — not a saturated
    teal, and not a category colour that means something else elsewhere."""
    _figure, ax, _stats = _draw(name)
    bars = _bars(ax)
    assert bars, f"{name} drew no histogram bars"
    fills = {to_hex(bar.get_facecolor()) for bar in bars}
    assert fills == {ROLES["fill"].lower()}, f"{name} bar fills: {fills}"


@pytest.mark.parametrize("name", CLUSTER)
def test_the_bars_are_opaque_and_have_no_edge(name):
    """"Density/histogram fills: solid but pale, not a translucent saturated
    colour." A white bar edge also draws white gaps on a transparent ground."""
    _figure, ax, _stats = _draw(name)
    for bar in _bars(ax):
        assert to_rgba(bar.get_facecolor())[3] == 1.0, f"{name} bar is translucent"
        assert bar.get_alpha() in (None, 1.0), f"{name} bar carries alpha"
        assert to_rgba(bar.get_edgecolor())[3] == 0.0, (
            f"{name} bar has a visible edge: {to_hex(bar.get_edgecolor())}")


def test_the_kde_is_the_claim_and_the_normal_fit_is_only_a_reference():
    """The panel asks whether the residuals are normal. The empirical density
    is the answer, so it carries the colour; the normal curve it is compared
    against is a reference and references are thin, dashed and grey."""
    _figure, ax, _stats = _draw("residual_distribution")
    curves = {to_hex(line.get_color()): line for line in ax.lines}

    kde = curves.get(ROLES["highlight"].lower())
    assert kde is not None, f"no highlighted KDE; drew {sorted(curves)}"
    assert kde.get_linestyle() == "-"

    normal = curves.get(ROLES["reference"].lower())
    assert normal is not None, f"the normal fit is not grey; drew {sorted(curves)}"
    assert normal.get_linestyle() != "-", "a reference curve must be dashed"
    assert normal.get_linewidth() <= WEIGHTS["reference"], (
        f"the reference is heavier than the data: {normal.get_linewidth()}")


def test_the_uniform_expectation_is_a_grey_dashed_reference():
    """It was crimson at lw=1.2 — louder than the bars it was a guide to."""
    _figure, ax, stats = _draw("p_value_histogram")
    lines = [line for line in ax.lines
             if to_hex(line.get_color()) == ROLES["reference"].lower()]
    assert lines, "the uniform expectation is not drawn as a reference"
    assert all(line.get_linewidth() <= WEIGHTS["reference"] for line in lines)
    assert all(line.get_linestyle() != "-" for line in lines)
    assert stats["n"] == 600


def test_a_broken_p_value_shape_is_the_only_thing_that_gets_a_warning_colour():
    """A verdict nobody can see is a verdict nobody acts on — but a verdict
    that is red when the fit is FINE trains the reader to ignore red."""
    rng = np.random.default_rng([43, 2])
    def context(p_values):
        base = _context()
        base.coef_df["p_value"] = p_values
        return base

    conservative = np.concatenate([rng.uniform(.9, 1., 300),
                                   rng.uniform(size=300)])
    _f1, bad_ax, bad = _draw("p_value_histogram", context(conservative))
    _f2, ok_ax, ok = _draw("p_value_histogram", context(rng.uniform(size=600)))

    assert bad["verdict"] == "excess-large" and ok["verdict"] == "uniform"
    bad_colours = {to_hex(t.get_color()) for t in bad_ax.texts
                   if "conservative" in t.get_text()}
    assert bad_colours == {ROLES["down"].lower()}, bad_colours
    ok_colours = {to_hex(t.get_color()) for t in ok_ax.texts
                  if "flat" in t.get_text()}
    assert ROLES["down"].lower() not in ok_colours, (
        f"a healthy p-value shape was drawn as a warning: {ok_colours}")


# --------------------------------------------------------------------------- #
#  No boxes, no framed legends, no sentence titles
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", CLUSTER)
def test_no_annotation_is_drawn_inside_a_box(name):
    """The published figures never draw one, and a white rounded box is
    invisible text the moment the ground is dark or transparent."""
    _figure, ax, _stats = _draw(name)
    boxed = [t.get_text()[:40] for t in ax.texts if t.get_bbox_patch() is not None]
    assert not boxed, f"{name} boxed these annotations: {boxed}"


@pytest.mark.parametrize("name", CLUSTER)
def test_there_is_no_framed_legend(name):
    """Legends are coloured text. A framed legend costs a corner of the axes
    and adds the only box in the figure."""
    _figure, ax, _stats = _draw(name)
    assert ax.get_legend() is None, f"{name} drew a matplotlib legend"


def test_the_residual_legend_is_coloured_text_matching_its_curves():
    """The legend entry has to be the same hue as the thing it names, or it
    is a caption rather than a legend."""
    _figure, ax, _stats = _draw("residual_distribution")
    entries = {t.get_text(): to_hex(t.get_color()) for t in ax.texts}
    assert entries.get("KDE") == ROLES["highlight"].lower(), entries
    assert entries.get("normal fit") == ROLES["reference"].lower(), entries


@pytest.mark.parametrize("name", CLUSTER)
def test_the_title_is_a_descriptor_and_not_a_sentence(name):
    """"No panel titles as sentences. If a panel needs a descriptor it is 2-4
    words above the axes." The n moved to the annotation, where it belongs."""
    _figure, ax, _stats = _draw(name)
    title = ax.get_title()
    assert "\n" not in title, f"{name} title is a two-line sentence: {title!r}"
    assert 2 <= len(title.split()) <= 4, f"{name} title: {title!r}"
    assert title == title.lower(), f"{name} title is not lower-case: {title!r}"
    assert "n = " not in title, f"{name} still carries its n in the title"


@pytest.mark.parametrize("name,unit", [("residual_distribution", "wells"),
                                       ("response_distribution", "wells"),
                                       ("p_value_histogram", "coefficients")])
def test_the_n_is_still_stated_on_the_panel_face(name, unit):
    """Moving it out of the title must not lose it: a distribution without
    its n cannot be read at all."""
    _figure, ax, stats = _draw(name)
    drawn = " ".join(t.get_text() for t in ax.texts)
    expected = stats.get("n_points", stats.get("n"))
    assert f"n = {expected:,} {unit}" in drawn, drawn


@pytest.mark.parametrize("name", CLUSTER)
def test_axis_labels_are_lower_case_and_there_are_no_gridlines(name):
    _figure, ax, _stats = _draw(name)
    for label in (ax.get_xlabel(), ax.get_ylabel()):
        assert label, f"{name} left an axis unlabelled"
        assert label == label.lower(), f"{name} label is not lower-case: {label!r}"
    drawn = [line for line in ax.get_xgridlines() + ax.get_ygridlines()
             if line.get_visible()]
    assert not drawn, f"{name} drew {len(drawn)} gridlines"


# --------------------------------------------------------------------------- #
#  The ink follows the theme, on an axes that already existed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", CLUSTER)
def test_the_chrome_takes_the_reports_ink_not_the_library_default(name):
    """The report driver builds the Figure before the panel is called, so the
    spines and tick labels exist with matplotlib's black already baked in.
    Opening a style context does not retro-colour them; the panel has to."""
    ink = resolve_ink(rq._REPORT_TARGET).lower()
    assert ink == INK_PRINT.lower(), (
        "these panels are written to a PDF whose page is white, so print ink "
        "is the only one that is visible in the delivered file")
    _figure, ax, _stats = _draw(name)

    assert to_hex(ax.xaxis.label.get_color()) == ink
    assert to_hex(ax.yaxis.label.get_color()) == ink
    assert to_hex(ax.title.get_color()) == ink
    ticks = ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels()
    assert ticks and {to_hex(t.get_color()) for t in ticks} == {ink}
    for side in ("left", "bottom"):
        assert to_hex(ax.spines[side].get_edgecolor()) == ink
        assert ax.spines[side].get_linewidth() == WEIGHTS["spine"]


def test_the_ink_is_visible_on_the_page_the_report_actually_writes(tmp_path):
    """The ink has to be chosen for the FILE, not for the GUI theme.

    ``regression_qc_report`` builds its ``Figure`` before any style context is
    entered, so the figure keeps matplotlib's white facecolor and ``_save``
    writes that white into the PDF. Resolving the ink from ``theme_target()``
    returns ``'screen'`` for every user who has not explicitly set a white
    figure background, and #E8EDEE on a white page is an invisible panel.
    This asserts the page really is white and the ink really is dark on it.
    """
    from matplotlib.colors import to_rgb
    import matplotlib.image as mpimg

    ctx_source = _context()
    manifest = rq.regression_qc_report(
        ctx_source.model, ctx_source.X, ctx_source.y, dst=str(tmp_path),
        coef_df=ctx_source.coef_df, regression_type="ols",
        panels=CLUSTER, combined=False, fmt="png")
    assert len(manifest["written"]) == len(CLUSTER), manifest["skipped"]

    ink_luminance = sum(to_rgb(resolve_ink(rq._REPORT_TARGET))) / 3.0
    for path in manifest["written"]:
        page = mpimg.imread(path)
        corner = page[1, 1]
        assert corner[:3].min() > 0.95, (
            f"{path} is not written on a white page: {corner}")
        assert ink_luminance < 0.5, (
            f"the ink is lighter than the page it is written on: "
            f"{resolve_ink(rq._REPORT_TARGET)}")
        assert page[..., :3].min() < 0.4, (
            f"{path} has nothing dark on it — the panel drew invisibly")


@pytest.mark.parametrize("name", CLUSTER)
def test_the_frame_is_l_shaped_and_the_type_is_the_house_scale(name):
    _figure, ax, _stats = _draw(name)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.xaxis.label.get_fontsize() == TYPE_SCALE["label"]
    assert ax.yaxis.label.get_fontsize() == TYPE_SCALE["label"]
    assert ax.title.get_fontsize() == TYPE_SCALE["label"]


# --------------------------------------------------------------------------- #
#  A restyle is not a re-analysis
# --------------------------------------------------------------------------- #

def test_the_statistics_are_untouched():
    """Every number these three panels return, pinned. The colours moved; the
    analysis did not, and this is what proves it."""
    ctx = _context()
    _f1, _a1, residual = _draw("residual_distribution", ctx)
    _f2, _a2, response = _draw("response_distribution", ctx)
    _f3, _a3, pvalues = _draw("p_value_histogram", ctx)

    assert residual["n_points"] == 240 and residual["n_bins"] == 15
    assert residual["skew"] == pytest.approx(-0.045054, abs=1e-5)
    assert residual["excess_kurtosis"] == pytest.approx(-0.074942, abs=1e-5)
    assert residual["normality_p"] == pytest.approx(0.957426, abs=1e-5)

    assert response["n"] == 240 and response["family"].startswith("Gaussian")
    assert response["mean"] == pytest.approx(0.970238, abs=1e-5)
    assert response["sd"] == pytest.approx(2.158248, abs=1e-5)
    assert response["min"] == pytest.approx(-4.712281, abs=1e-5)
    assert response["max"] == pytest.approx(7.261750, abs=1e-5)

    assert pvalues["verdict"] == "uniform" and pvalues["n"] == 600
    assert pvalues["source"] == "coefficient table"
    assert pvalues["frac_below_0.05"] == pytest.approx(29 / 600, abs=1e-9)
    assert pvalues["first_bin_ratio"] == pytest.approx(0.966667, abs=1e-5)
    assert pvalues["last_bin_ratio"] == pytest.approx(0.966667, abs=1e-5)


@pytest.mark.parametrize("name", CLUSTER)
def test_a_panel_that_cannot_be_computed_still_says_why(name):
    """The style must not have swallowed the degradation contract."""
    ctx = _context(n=240)
    ctx.resid = np.array([np.nan, np.nan])
    ctx.y = np.array([np.nan, np.nan])
    ctx.coef_df = None
    ctx.model = object()
    with pytest.raises(rq.PanelUnavailable) as raised:
        _draw(name, ctx)
    assert str(raised.value)
