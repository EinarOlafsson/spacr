"""One house style, one panel catalog, one publication-ready sheet.

Asked for on 2026-08-16: "the figures themselves need to be remade from
scratch. i want the same data (and more) shown ... i want you to start from
scratch regarding how the figures are made the color used and the libraries
used", "the all figures section should look like a publication ready figure",
and "i generally like figures in the style of Sebastian Lourido (papers he is
last author on)".

The visual system is `.claude/skills/apicomplexan-figures`, derived by direct
inspection of published Lourido-lab figures rather than from design taste.
This file pins the rules that are easy to break by accident.

THE ONE THAT WOULD COST THE MOST: the skill's own helper applies its style by
calling `matplotlib.rcParams.update` at module scope. spaCR draws from a
long-lived GUI, so a process-wide style change means drawing one figure
restyles every figure drawn afterwards, in every other module, until the
process exits. Five spaCR functions were doing exactly that and were fixed
the same day this module was written. Everything here is a context manager,
and the first test is the one that proves it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures import (REGISTRY, SHEET_ORDER, build_panel, build_sheet,
                           figure_style, rc)
from spacr.figures.panels import label_series
from spacr.figures.style import INK_PRINT, INK_SCREEN, ROLES, TRANSPARENT


def _results(n=400, seed=0, with_q=True):
    """A coefficient table shaped like the real one."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, .35, n)
    effect[:12] += rng.choice([-3.5, 3.5], 12)
    p = rng.uniform(size=n)
    p[:12] = rng.uniform(1e-12, 1e-4, 12)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i // 4}_{i % 4}]" for i in range(n)],
        "coefficient": effect,
        "p_value": p,
        "grna": [f"{i // 4}_{i % 4}" for i in range(n)],
        "gene": [None] * n,
        "condition": rng.choice(["nc", "pc", "control", "other"], n,
                                p=[.03, .03, .06, .88]),
    })
    if with_q:
        frame["q_value"] = np.minimum(frame["p_value"] * 4, 1.0)
    intercept = pd.DataFrame([{
        "feature": "Intercept", "coefficient": .19, "p_value": 3e-46,
        "grna": None, "gene": None, "condition": "other",
        **({"q_value": np.nan} if with_q else {})}])
    return pd.concat([intercept, frame], ignore_index=True)


# --------------------------------------------------------------------------- #
#  The style is scoped. This is the important one.
# --------------------------------------------------------------------------- #

def test_the_style_does_not_leak_into_the_process():
    """Draw one figure in the house style and every LATER figure must be
    unaffected. A global rcParams update is how a UMAP on the dark theme
    leaves the next screen's figures with white-on-white text."""
    before = dict(plt.rcParams)

    with figure_style("print"):
        figure = plt.figure()
        figure.add_subplot(111).plot([0, 1], [0, 1])
        plt.close(figure)

    changed = {k for k in before
               if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, f"these rcParams were left changed: {sorted(changed)}"


def test_building_a_whole_sheet_does_not_leak_either():
    before = dict(plt.rcParams)
    sheet = build_sheet(_results(120))
    plt.close(sheet.figure)

    changed = {k for k in before
               if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, sorted(changed)


def test_there_is_no_rcparams_update_call_anywhere_in_the_style():
    """Named so that copying the skill's `use()` in verbatim fails here.

    Parsed, not grepped: the module DOCUMENTS why it must not call
    rcParams.update, so a text search matches its own explanation and passes
    whatever the code does.
    """
    import ast
    import inspect

    from spacr.figures import style

    tree = ast.parse(inspect.getsource(style))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "update"
             and isinstance(node.func.value, ast.Attribute)
             and node.func.value.attr == "rcParams"]
    assert not calls, (
        "the style writes rcParams globally; it must yield a context instead")


# --------------------------------------------------------------------------- #
#  The rules the skill states
# --------------------------------------------------------------------------- #

def test_there_are_no_gridlines_ever():
    assert rc()["axes.grid"] is False


def test_the_ground_is_transparent_by_default():
    """"not black not white just transparent" -- instruction 118."""
    params = rc()
    assert params["figure.facecolor"] == TRANSPARENT
    assert params["axes.facecolor"] == TRANSPARENT
    assert params["savefig.transparent"] is True


def test_the_ink_follows_where_the_figure_is_going():
    """The published palette is near-black on white. On spaCR's dark theme
    that is invisible axes, so screen and print resolve differently -- while
    the HUES stay fixed, because a strain colour that moves between panels
    is worse than no colour at all."""
    assert rc("print")["text.color"] == INK_PRINT
    assert rc("screen")["text.color"] == INK_SCREEN
    assert ROLES["up"] == "#2E7D4F" and ROLES["down"] == "#C4441C"


def test_the_frame_is_l_or_box_and_nothing_else():
    assert rc(frame="L")["axes.spines.top"] is False
    assert rc(frame="box")["axes.spines.top"] is True


# --------------------------------------------------------------------------- #
#  Everything is grey except the claim
# --------------------------------------------------------------------------- #

def test_the_volcano_greys_what_it_did_not_call():
    """"If half the points are coloured, the figure has no claim."""
    figure, panel = build_panel("volcano", _results(400))
    try:
        assert panel.drawn
        collections = figure.axes[0].collections
        counts = {tuple(np.round(c.get_facecolor()[0][:3], 3)): len(c.get_offsets())
                  for c in collections if len(c.get_offsets())}
        total = sum(counts.values())
        from matplotlib.colors import to_rgb
        grey = tuple(np.round(to_rgb(ROLES["data"]), 3))
        assert counts.get(grey, 0) / total > 0.5, (
            f"the highlight is not a minority of the marks: {counts}")
    finally:
        plt.close(figure)


def test_the_volcano_leaves_the_nuisance_terms_off():
    """The intercept is not a hypothesis, and plotting it makes the y-axis
    3.6x taller than the data."""
    figure, panel = build_panel("volcano", _results(200))
    try:
        drawn = sum(len(c.get_offsets()) for c in figure.axes[0].collections)
        assert drawn == 200, f"drew {drawn} of 200 tested coefficients"
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  Labels that are actually names
# --------------------------------------------------------------------------- #

def test_a_row_is_never_labelled_nan():
    """`gene` is empty on the per-guide rows and `grna` on the per-gene rows,
    so either column alone labels half the volcano 'nan' -- which is what the
    first pass drew."""
    names = label_series(_results(40))
    assert not names.str.lower().eq("nan").any(), names.head().tolist()


def test_the_design_boilerplate_is_stripped():
    frame = pd.DataFrame({"feature": ["fraction:grna[233460_1]",
                                      "gene_fraction:gene[233460]"]})
    assert label_series(frame).tolist() == ["233460_1", "233460"]


# --------------------------------------------------------------------------- #
#  The sheet
# --------------------------------------------------------------------------- #

def test_the_sheet_draws_every_panel_the_table_supports():
    sheet = build_sheet(_results(300))
    try:
        assert [p.key for p in sheet.panels] == list(SHEET_ORDER)
        assert not sheet.skipped
    finally:
        plt.close(sheet.figure)


def test_reading_order_is_the_argument():
    """Result first, then what it rests on, then whether the model was
    entitled to say it. A reader who stops after B has the result; one who
    reads to G knows whether to believe it."""
    assert SHEET_ORDER[0] == "volcano"
    assert SHEET_ORDER.index("controls") < SHEET_ORDER.index("qq")
    assert SHEET_ORDER[-1] == "qq"


def test_panels_are_lettered_bold_and_upper_case():
    sheet = build_sheet(_results(150))
    try:
        letters = [t.get_text() for ax in sheet.figure.axes
                   for t in ax.texts if len(t.get_text()) == 1]
        assert letters[:3] == ["A", "B", "C"], letters[:5]
        bold = [t for ax in sheet.figure.axes for t in ax.texts
                if t.get_text() == "A"][0]
        assert bold.get_fontweight() == "bold"
    finally:
        plt.close(sheet.figure)


def test_a_missing_panel_is_named_not_drawn_as_an_empty_frame():
    """A blank box in a figure sheet reads as a panel that failed, which is
    worse than a gap and much worse than a sentence saying why."""
    frame = _results(120).drop(columns=["condition"])
    sheet = build_sheet(frame)
    try:
        assert "controls" not in [p.key for p in sheet.panels]
        assert "controls" in [p.key for p in sheet.skipped]
        assert "Not shown" in sheet.legend()
        assert "condition" in sheet.legend()
    finally:
        plt.close(sheet.figure)


def test_the_legend_is_generated_from_the_panels():
    """A legend maintained by hand beside the code that draws the figure is
    a legend that describes last month's figure."""
    sheet = build_sheet(_results(200))
    try:
        legend = sheet.legend()
        assert legend.startswith("(A)")
        assert "(B)" in legend and "(G)" in legend
        assert "tested coefficients" in legend
    finally:
        plt.close(sheet.figure)


# --------------------------------------------------------------------------- #
#  Every backend's table has to display
# --------------------------------------------------------------------------- #

def test_a_backend_with_no_p_value_still_gets_a_sheet():
    """lasso and elasticnet report a selection frequency and no p-value at
    all. The panels that need one say so; the rest still draw."""
    frame = _results(150).drop(columns=["p_value", "q_value"])
    frame["selection_frequency"] = np.linspace(0, 1, len(frame))

    sheet = build_sheet(frame)
    try:
        drawn = {p.key for p in sheet.panels}
        assert "effect_rank" in drawn and "effect_distribution" in drawn
        assert "p_histogram" not in drawn
        reasons = {p.key: p.reason for p in sheet.skipped}
        assert "no p-value" in reasons["p_histogram"]
    finally:
        plt.close(sheet.figure)


@pytest.mark.parametrize("statistic", ["z value", "t value", "z_value"])
def test_the_statistic_column_is_found_whatever_it_is_called(statistic):
    """OLS reports t, GLM and Poisson report z. Both tables must work."""
    from spacr.figures import statistic_column

    frame = _results(60)
    frame[statistic] = 1.0
    assert statistic_column(frame) == statistic


def test_a_table_with_no_q_value_falls_back_to_p():
    frame = _results(150, with_q=False)
    figure, panel = build_panel("volcano", frame)
    try:
        assert panel.drawn
        assert "p" in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  Nothing raises on the edges
# --------------------------------------------------------------------------- #

def test_an_empty_table_does_not_raise():
    sheet = build_sheet(pd.DataFrame(columns=["feature", "coefficient",
                                              "p_value"]))
    try:
        assert sheet.panels == [] or all(p.drawn for p in sheet.panels)
    finally:
        plt.close(sheet.figure)


@pytest.mark.parametrize("key", list(SHEET_ORDER))
def test_every_registered_panel_survives_a_one_row_table(key):
    """A screen with almost nothing in it is a real case -- a run that
    filtered hard, or a sweep trial on a tiny design. Every panel must come
    back with an answer rather than an exception, and a panel that cannot
    draw must SAY so rather than returning a blank one."""
    frame = _results(200).head(2)
    figure = plt.figure()
    try:
        panel = REGISTRY[key](figure.add_subplot(111), frame)
        assert panel.key, "a panel came back without identifying itself"
        assert panel.drawn or panel.reason, (
            f"{key} refused to draw and gave no reason")
    finally:
        plt.close(figure)
