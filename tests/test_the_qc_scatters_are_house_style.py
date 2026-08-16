"""The eight scatter-with-reference-line QC panels, in the house style.

The cluster: ``residuals_vs_fitted``, ``scale_location``, ``qq_residuals``,
``observed_vs_predicted``, ``cooks_distance``, ``influence``, ``dffits`` and
``cell_count_vs_effect``. Every one of them was a saturated teal cloud with a
bold red threshold line, a sentence title carrying the n, and a white rounded
box around its statistics -- the exact inversion of the rule the house style
is built on: **everything is grey except what the sentence is about**, and a
reference line is thin, dashed and grey because a reference is not a result.

These tests assert the style, not the drawing library. They are deliberately
about the artists that end up on the axes, because that is the only thing a
reader sees; asserting that the code imports the style module would pass on a
panel that imported it and then drew in red anyway.

The statistics are asserted too. This is a restyle: if any number moved, the
change is wrong.
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
from spacr.figures.style import (ROLES, TYPE_SCALE, WEIGHTS,  # noqa: E402
                                 resolve_ink)

#: The cluster, in report order.
SCATTERS = ("residuals_vs_fitted", "scale_location", "qq_residuals",
            "observed_vs_predicted", "cooks_distance", "influence", "dffits")

#: ...plus the one that needs a per-well cell count to draw at all.
ALL_SCATTERS = SCATTERS + ("cell_count_vs_effect",)

#: The panels whose sentence is "these specific wells are the problem", so a
#: highlight must exist and must be a minority of the marks.
HIGHLIGHTERS = ("cooks_distance", "dffits", "influence", "cell_count_vs_effect")

#: The five module-level hexes the panels used to draw with. None of them is
#: in the published palette and none of them may survive in this cluster.
OLD_PALETTE = ("#1f6f8b", "#d1495b", "#8d99ae", "#e07a3f", "#2a9d8f")

REAL_TABLE = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/test/"
              "results/plate1_dv/ols/list/regression_data.csv")


# --------------------------------------------------------------------------- #
#  Fixtures: a fit with real structure in it, so the highlights have something
#  to highlight.
# --------------------------------------------------------------------------- #

def _axes():
    figure = Figure(figsize=(5.6, 4.4))
    return figure, figure.subplots()


def _context(n=240, seed=7):
    """An OLS fit with planted influential wells and a per-well cell count."""
    rng = np.random.default_rng([seed, 1])
    X = pd.DataFrame({
        "Intercept": np.ones(n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    counts = np.exp(rng.uniform(np.log(50), np.log(2000), n))
    y = 0.4 + 1.3 * X["x1"] - 0.6 * X["x2"] + rng.normal(scale=0.5, size=n)
    # Four wells that the influence panels have to find, and that are also the
    # four smallest wells on the plate -- so the cell-count panel has a real
    # "the tails are the small wells" set to colour.
    y.iloc[:4] += 9.0
    counts[:4] = [20.0, 21.0, 22.0, 23.0]
    meta = pd.DataFrame({
        "prc": [f"plate1_r{i // 24 + 1}_c{i % 24 + 1}" for i in range(n)],
        "cell_count": counts,
    })
    model = sm.OLS(y.to_numpy(dtype=float), X).fit()
    return rq.build_context(model, X, y.to_numpy(dtype=float), metadata=meta,
                            regression_type="ols")


@pytest.fixture(scope="module")
def ctx():
    return _context()


def _draw(ctx, name):
    figure, ax = _axes()
    stats = rq.draw_panel(name, ctx, ax)
    return figure, ax, stats


# --------------------------------------------------------------------------- #
#  Reading the artists back off the axes
# --------------------------------------------------------------------------- #

def _mark_colours(ax):
    """``{hex: number of marks}`` over every scatter and every stem."""
    tally = {}
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            faces = coll.get_facecolor()
            count = len(coll.get_offsets())
        elif isinstance(coll, LineCollection):
            faces = coll.get_colors()
            count = len(coll.get_segments())
        else:
            continue
        if not len(faces) or not count:
            continue
        if len(faces) == 1:
            tally[to_hex(faces[0])] = tally.get(to_hex(faces[0]), 0) + count
        else:
            for face in faces:
                key = to_hex(face)
                tally[key] = tally.get(key, 0) + 1
    return tally


def _guides(ax):
    """The axis-parallel guide lines: an axhline or an axvline, nothing else."""
    found = []
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        if xs.size == 2 and (len(set(xs)) == 1 or len(set(ys)) == 1):
            found.append(line)
    return found


def _panel_source(name):
    _, _, fn = rq._PANEL_BY_NAME[name]
    return inspect.getsource(fn)


# --------------------------------------------------------------------------- #
#  The rule that matters most: grey by default, colour only for the claim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_the_data_marks_are_grey(ctx, name):
    """Teal points are a decision the panel has not earned. The data is the
    default ink; if a mark is coloured, the panel is claiming something about
    it."""
    figure, ax, _ = _draw(ctx, name)
    try:
        tally = _mark_colours(ax)
        assert tally, f"{name} drew no marks at all"
        grey = to_hex(ROLES["data"])
        assert grey in tally, (
            f"{name} draws no grey data marks; it drew {tally}")
        assert tally[grey] / sum(tally.values()) > 0.5, (
            f"{name}: the highlight is not a minority of the marks: {tally}")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", HIGHLIGHTERS)
def test_the_flagged_wells_are_the_only_coloured_ones(ctx, name):
    """These four panels exist to name wells. Before this change the flagged
    set was stated only in a text box while every mark on the panel wore the
    same colour -- the claim was in the prose and not in the picture."""
    figure, ax, _ = _draw(ctx, name)
    try:
        tally = _mark_colours(ax)
        coloured = {k: v for k, v in tally.items()
                    if k != to_hex(ROLES["data"])}
        assert coloured, f"{name} highlights nothing; its whole point is which wells"
        assert set(coloured) <= {to_hex(ROLES["down"])}, (
            f"{name} highlights in a colour that is not the 'down' rust: "
            f"{coloured}")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_no_panel_still_carries_the_old_hardcoded_palette(name):
    """Parsed from the source, not from the rendered figure: a colour passed
    through a variable would still be a violation, and a panel that has been
    converted must have no way back to the old hexes."""
    tree = ast.parse(inspect.getsource(rq))
    body = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == rq._PANEL_BY_NAME[name][2].__name__)
    names = {n.id for n in ast.walk(body) if isinstance(n, ast.Name)}
    banned = {"_POINT", "_ACCENT", "_GUIDE", "_TREND", "_OK"} & names
    assert not banned, f"{name} still draws with {sorted(banned)}"
    literals = {n.value for n in ast.walk(body)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not {lit.lower() for lit in literals} & set(OLD_PALETTE), (
        f"{name} inlines an old palette hex")


# --------------------------------------------------------------------------- #
#  A reference is not a result
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_reference_lines_are_thin_dashed_and_grey(ctx, name):
    """Thresholds were drawn in the loudest colour in the file, at up to
    twice the weight of the data. The skill's wording: reference lines are
    grey, thin, dashed or dotted -- never bold."""
    figure, ax, _ = _draw(ctx, name)
    try:
        guides = _guides(ax)
        for line in guides:
            assert to_hex(line.get_color()) == to_hex(ROLES["reference"]), (
                f"{name}: a guide at "
                f"{line.get_xdata()[0]},{line.get_ydata()[0]} is "
                f"{to_hex(line.get_color())}, not the reference grey")
            assert line.get_linewidth() <= WEIGHTS["reference"] + 1e-9, (
                f"{name}: a guide is {line.get_linewidth()} wide; the "
                f"reference weight is {WEIGHTS['reference']}")
            assert line.get_linestyle() != "-", (
                f"{name}: a guide is drawn solid")
    finally:
        plt.close(figure)


def test_the_qq_reference_line_stopped_shouting(ctx):
    """The Q-Q quartile line was #d1495b, solid, at lw 1.4 -- the boldest
    artist on a panel whose only claim is that the points follow it."""
    figure, ax, _ = _draw(ctx, "qq_residuals")
    try:
        diagonals = [line for line in ax.lines if line not in _guides(ax)]
        assert diagonals, "the quartile line is gone"
        for line in diagonals:
            assert to_hex(line.get_color()) == to_hex(ROLES["reference"])
            assert line.get_linewidth() <= WEIGHTS["reference"] + 1e-9
            assert line.get_linestyle() != "-"
    finally:
        plt.close(figure)


def test_the_identity_diagonal_is_a_reference_and_not_a_series(ctx):
    figure, ax, _ = _draw(ctx, "observed_vs_predicted")
    try:
        diagonals = [line for line in ax.lines if line not in _guides(ax)]
        assert len(diagonals) == 1
        assert to_hex(diagonals[0].get_color()) == to_hex(ROLES["reference"])
        assert diagonals[0].get_linewidth() <= WEIGHTS["reference"] + 1e-9
    finally:
        plt.close(figure)


def test_the_smoother_is_the_claim_and_wears_the_highlight(ctx):
    """Residuals-vs-fitted and scale-location both exist to answer one
    question, and the LOWESS curve is the answer. It is the one artist on
    those panels entitled to a colour."""
    for name in ("residuals_vs_fitted", "scale_location"):
        figure, ax, _ = _draw(ctx, name)
        try:
            curves = [line for line in ax.lines
                      if len(line.get_xdata()) > 2]
            assert len(curves) == 1, f"{name} has {len(curves)} smoothers"
            assert to_hex(curves[0].get_color()) == to_hex(ROLES["highlight"])
        finally:
            plt.close(figure)


# --------------------------------------------------------------------------- #
#  Titles, notes, legends, type
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_no_panel_carries_a_sentence_title(ctx, name):
    """"No panel titles as sentences. The axis labels carry the content. If a
    panel needs a descriptor it is 2-4 words above the axes." Every one of
    these carried a Title-Case sentence with the n bolted on in brackets."""
    figure, ax, _ = _draw(ctx, name)
    try:
        title = ax.get_title()
        assert "\n" not in title, f"{name} has a two-line title: {title!r}"
        assert "(n =" not in title, (
            f"{name} still puts the n in the title: {title!r}")
        assert title == title.lower(), (
            f"{name} title is not lower case: {title!r}")
        assert 1 <= len(title.split()) <= 4, (
            f"{name} descriptor is {len(title.split())} words: {title!r}")
        assert ax.title.get_fontsize() == TYPE_SCALE["label"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_the_n_did_not_disappear_with_the_title(ctx, name):
    """Moving the n off the title must not lose it: a diagnostic panel that
    does not say how many wells it drew cannot be read on its own."""
    figure, ax, _ = _draw(ctx, name)
    try:
        text = " ".join(t.get_text() for t in ax.texts)
        assert "n = " in text and "wells" in text, (
            f"{name} no longer states its n anywhere: {text!r}")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_no_note_is_drawn_in_a_box(ctx, name):
    """A white rounded box is invisible text on a transparent ground and the
    published figures never draw one -- "no frame, no box"."""
    figure, ax, _ = _draw(ctx, name)
    try:
        boxed = [t.get_text()[:40] for t in ax.texts
                 if t.get_bbox_patch() is not None]
        assert not boxed, f"{name} boxes its notes: {boxed}"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_no_panel_draws_a_matplotlib_legend(ctx, name):
    """Legends are coloured text with no frame and no sample marker."""
    figure, ax, _ = _draw(ctx, name)
    try:
        assert ax.get_legend() is None, f"{name} drew a matplotlib legend"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_the_type_sizes_come_from_the_type_scale(ctx, name):
    """7 pt labels, 6.2 pt ticks, 6 pt annotations -- measured off the
    published figures and pinned in the style module, not chosen per panel."""
    figure, ax, _ = _draw(ctx, name)
    try:
        assert ax.xaxis.label.get_fontsize() == TYPE_SCALE["label"]
        assert ax.yaxis.label.get_fontsize() == TYPE_SCALE["label"]
        ticks = ax.get_xticklabels()
        assert ticks and ticks[0].get_fontsize() == TYPE_SCALE["tick"]
        for text in ax.texts:
            assert text.get_fontsize() <= TYPE_SCALE["annotation"] + 1e-9, (
                f"{name}: an annotation is {text.get_fontsize()} pt")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_the_frame_is_an_l_at_the_house_weight(ctx, name):
    figure, ax, _ = _draw(ctx, name)
    try:
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        # `_REPORT_TARGET`, not `theme_target()`: this report writes files,
        # and the driver builds the Figure outside any style context, so the
        # page it saves is matplotlib's white. Screen ink on a white page is
        # invisible.
        ink = to_hex(resolve_ink(rq._REPORT_TARGET))
        for side in ("left", "bottom"):
            spine = ax.spines[side]
            assert spine.get_visible()
            assert spine.get_linewidth() == pytest.approx(WEIGHTS["spine"])
            assert to_hex(spine.get_edgecolor()) == ink
        assert to_hex(ax.xaxis.label.get_color()) == ink
        assert to_hex(ax.title.get_color()) == ink
    finally:
        plt.close(figure)


def test_the_scatter_panels_resolve_the_same_ink_as_the_rest_of_the_report(ctx):
    """One page, one ink. The combined QC page redraws every panel onto one
    figure, so a cluster that resolved its ink differently from its
    neighbours would ship a page with black text in one column and near-white
    text in the next."""
    figure, ax, _ = _draw(ctx, "residuals_vs_fitted")
    other, other_ax = _axes()
    try:
        rq.draw_panel("residual_distribution", ctx, other_ax)
        assert (to_hex(ax.spines["left"].get_edgecolor())
                == to_hex(other_ax.spines["left"].get_edgecolor()))
    finally:
        plt.close(figure)
        plt.close(other)


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_the_axis_labels_are_lower_case(ctx, name):
    figure, ax, _ = _draw(ctx, name)
    try:
        for label in (ax.get_xlabel(), ax.get_ylabel()):
            assert label, f"{name} lost an axis label"
            letters = [c for c in label.split("$")[0] if c.isalpha()]
            assert not any(c.isupper() for c in letters), (
                f"{name} axis label is not lower case: {label!r}")
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The style is scoped
# --------------------------------------------------------------------------- #

def test_drawing_every_scatter_panel_leaves_no_rcparams_behind(ctx):
    """spaCR draws from a long-lived GUI. A style applied globally restyles
    every later figure in the session, in every other module, until the
    process exits."""
    before = {k: str(v) for k, v in plt.rcParams.items()}
    for name in ALL_SCATTERS:
        figure, ax, _ = _draw(ctx, name)
        plt.close(figure)
    changed = sorted(k for k, v in before.items()
                     if str(plt.rcParams[k]) != v)
    assert not changed, f"these rcParams were left changed: {changed}"


@pytest.mark.parametrize("name", ALL_SCATTERS)
def test_each_panel_opens_the_style_as_a_context_manager(name):
    """Grepping for `figure_style` would pass on a module that imported it;
    this checks the panel body is inside a `with`."""
    tree = ast.parse(_panel_source(name).lstrip())
    fn = tree.body[0]
    withs = [node for node in ast.walk(fn) if isinstance(node, ast.With)]
    opened = [item for node in withs for item in node.items
              if isinstance(item.context_expr, ast.Call)
              and getattr(item.context_expr.func, "id", "") == "figure_style"]
    assert opened, f"{name} does not draw inside `with figure_style(...)`"


# --------------------------------------------------------------------------- #
#  A restyle moves no numbers
# --------------------------------------------------------------------------- #

def test_the_statistics_are_untouched(ctx):
    """The panels' returned numbers are checked against their definitions
    again here, because the one thing a restyle must not do is re-analyse."""
    from scipy import stats as sps

    _, _, ovp = _draw(ctx, "observed_vs_predicted")
    rss = float(np.sum((ctx.y - ctx.fitted) ** 2))
    tss = float(np.sum((ctx.y - ctx.y.mean()) ** 2))
    assert ovp["r2"] == pytest.approx(1 - rss / tss, rel=1e-12)
    assert ovp["r2"] == pytest.approx(ctx.model.rsquared, rel=1e-9)

    _, _, cooks = _draw(ctx, "cooks_distance")
    reference = ctx.model.get_influence().cooks_distance[0]
    assert cooks["threshold"] == pytest.approx(4.0 / ctx.n)
    assert cooks["max_cooks"] == pytest.approx(np.nanmax(reference), rel=1e-8)
    assert cooks["n_above"] == int(np.sum(reference > 4.0 / ctx.n))

    _, _, dff = _draw(ctx, "dffits")
    assert dff["threshold"] == pytest.approx(2 * np.sqrt(ctx.p / ctx.n))
    assert dff["max_abs_dffits"] == pytest.approx(
        np.nanmax(np.abs(ctx.model.get_influence().dffits[0])), rel=1e-8)

    _, _, sl = _draw(ctx, "scale_location")
    root = np.sqrt(np.abs(ctx.std_resid))
    rho, _p = sps.spearmanr(ctx.fitted, root)
    assert sl["spearman_rho"] == pytest.approx(float(rho), rel=1e-12)

    _, _, qq = _draw(ctx, "qq_residuals")
    sample = np.sort(ctx.std_resid[np.isfinite(ctx.std_resid)])
    q1_t, q3_t = sps.norm.ppf([0.25, 0.75])
    q1_s, q3_s = np.quantile(sample, [0.25, 0.75])
    assert qq["slope"] == pytest.approx((q3_s - q1_s) / (q3_t - q1_t),
                                        rel=1e-12)


def test_the_flagged_wells_are_the_same_wells_the_stats_name(ctx):
    """The highlight has to be drawn from the set the panel returns, not from
    a second, parallel rule that could drift from it."""
    figure, ax, stats = _draw(ctx, "cooks_distance")
    try:
        rust = to_hex(ROLES["down"])
        stems = [c for c in ax.collections if isinstance(c, LineCollection)]
        drawn = sum(len(c.get_segments()) for c in stems
                    if to_hex(c.get_colors()[0]) == rust)
        assert drawn == stats["n_above"], (
            f"{drawn} stems highlighted, {stats['n_above']} wells flagged")
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The real screen
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_TABLE),
                    reason="the tsg101 screen table is not on this machine")
def test_the_real_screen_draws_every_scatter_panel_in_the_house_style():
    """610 real wells from the tsg101 screen, refit small enough that
    leverage is defined. Synthetic residuals are well behaved by
    construction; this table is not, which is the point."""
    long = pd.read_csv(REAL_TABLE)
    top = long["grna"].value_counts().head(20).index.tolist()
    wide = (long[long["grna"].isin(top)]
            .pivot_table(index="prc", columns="grna", values="fraction",
                         aggfunc="sum")
            .reindex(sorted(long["prc"].unique())).fillna(0.0))
    wells = long.groupby("prc").agg(log_pred=("log_pred", "first"),
                                    cell_count=("cell_count", "first"))
    wells = wells.reindex(wide.index)
    X = wide.copy()
    X.insert(0, "Intercept", 1.0)
    X.columns = [str(c) for c in X.columns]
    y = wells["log_pred"].to_numpy(dtype=float)
    meta = pd.DataFrame({"prc": list(wide.index),
                         "cell_count": wells["cell_count"].to_numpy(float)})
    context = rq.build_context(sm.OLS(y, X).fit(), X, y, metadata=meta,
                               regression_type="ols")
    assert context.n == 610

    for name in ALL_SCATTERS:
        figure, ax, stats = _draw(context, name)
        try:
            tally = _mark_colours(ax)
            grey = to_hex(ROLES["data"])
            assert tally.get(grey, 0) / sum(tally.values()) > 0.5, (
                f"{name} on the real screen: {tally}")
            assert "\n" not in ax.get_title()
            assert not [t for t in ax.texts if t.get_bbox_patch() is not None]
            for line in _guides(ax):
                assert to_hex(line.get_color()) == to_hex(ROLES["reference"])
        finally:
            plt.close(figure)
