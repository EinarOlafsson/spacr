"""The plate/row/column panels and the volcano signpost, in the house style.

The four panels in this file are the "screen-level structure" corner of
:mod:`spacr.regression_qc`: residuals grouped by plate, by plate row and by
plate column, plus the card that names where the volcano was written instead
of drawing a second one.

WHAT THESE TESTS ARE FOR. The house style in :mod:`spacr.figures.style` is not
decoration, and the way it fails is silent: a panel that colours every mark
still draws, still saves and still reads as a figure -- it just no longer says
anything, because a highlight that is on everything is on nothing. So each
test below plants a situation whose ANSWER IS KNOWN and asserts that the ink
follows the answer:

* a plate row with a planted edge artefact must be the only coloured thing;
* a plate with no positional effect at all must come out entirely grey,
  because "colour is an argument" and there is no argument to make;
* the same numbers must come back either way, since this is a restyle.

Plus the two failure modes that cost a day each in this repository: a style
applied by writing ``rcParams`` globally (it restyles every later figure in
the GUI session), and a light-page ink assumption on a dark ground (the panel
draws, saves, and is invisible).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import statsmodels.api as sm                               # noqa: E402
from matplotlib.colors import to_hex                       # noqa: E402
from matplotlib.figure import Figure                       # noqa: E402

from spacr import regression_qc as rq                      # noqa: E402
from spacr import schema                                   # noqa: E402
from spacr.figures.style import (INK_PRINT, INK_SCREEN, ROLES,  # noqa: E402
                                 TYPE_SCALE)

#: The palette this module used before the house style, kept here so a
#: half-finished conversion cannot pass: teal points, red accents, grey-blue
#: guides, orange trend, green "healthy".
_OLD_PALETTE = {"#1f6f8b", "#d1495b", "#8d99ae", "#e07a3f", "#2a9d8f"}


def _hex(colour):
    return to_hex(colour).lower()


def _role(name):
    return ROLES[name].lower()


# --------------------------------------------------------------------------- #
#  Fixtures: a plate whose answer is known
# --------------------------------------------------------------------------- #

def _metadata(n_rows=8, n_cols=12, plates=("plate1",)):
    records = [{schema.PLATE_KEY: plate,
                schema.ROW_KEY: f"r{r}",
                schema.COLUMN_KEY: f"c{c}",
                schema.PRC_KEY: f"{plate}_r{r}_c{c}"}
               for plate in plates
               for r in range(1, n_rows + 1)
               for c in range(1, n_cols + 1)]
    return pd.DataFrame.from_records(records)


def _fit(edge_delta=0.0, seed=7, n_rows=8, n_cols=12, plates=("plate1",)):
    """An OLS fit whose outer ROWS are shifted by ``edge_delta``.

    The row term is deliberately left out of the design, so the shift lands in
    the residuals -- which is the artefact the row panel exists to reveal.
    """
    meta = _metadata(n_rows=n_rows, n_cols=n_cols, plates=plates)
    n = len(meta)
    rng = np.random.default_rng([seed, 2])
    X = pd.DataFrame({"Intercept": np.ones(n), "x1": rng.normal(size=n)},
                     index=meta.index)
    row = meta[schema.ROW_KEY].str.extract(r"(\d+)")[0].astype(int)
    outer = (row == 1) | (row == n_rows)
    y = 1.0 + 2.0 * X["x1"] + rng.normal(size=n) * 0.3 + outer * edge_delta
    model = sm.OLS(y, X).fit()
    return rq.build_context(model, X, y, metadata=meta, regression_type="ols")


def _draw(panel, ctx, ground="white"):
    """Draw one panel on a figure whose ground is decided BEFORE the draw."""
    figure = Figure(figsize=(5.6, 4.4), dpi=110, facecolor=ground)
    ax = figure.subplots()
    ax.set_facecolor(ground)
    return ax, rq.draw_panel(panel, ctx, ax)


def _mark_colours(ax):
    """``{colour: number of scatter points}`` for the jittered residuals."""
    counts = {}
    for collection in ax.collections:
        points = len(collection.get_offsets())
        if not points:
            continue
        faces = collection.get_facecolor()
        if not len(faces):
            continue
        counts[_hex(faces[0])] = counts.get(_hex(faces[0]), 0) + points
    return counts


# --------------------------------------------------------------------------- #
#  The style is scoped, and the ink follows the page
# --------------------------------------------------------------------------- #

def test_the_positional_panels_do_not_leak_the_style_into_the_process():
    """spaCR draws from a long-lived GUI. An rcParams update made while
    drawing a QC panel would restyle every figure the session draws
    afterwards, in every other module, until the process exits."""
    ctx = _fit()
    before = dict(plt.rcParams)
    for panel in ("row_effects", "column_effects", "volcano_reference"):
        _draw(panel, ctx)
    changed = {key for key in before if str(before[key]) != str(plt.rcParams[key])}
    assert not changed, f"these rcParams were left changed: {sorted(changed)}"


@pytest.mark.parametrize("panel", ["row_effects", "volcano_reference"])
@pytest.mark.parametrize("ground,ink", [("white", INK_PRINT),
                                        ("#2b2b2b", INK_SCREEN)])
def test_the_ink_follows_the_ground_the_panel_is_drawn_on(panel, ground, ink):
    """The QC suite is written to files, and the caller owns the ground.

    Near-black text on the dark ground is an invisible panel; near-white text
    on the white page matplotlib gives a bare ``Figure`` is the same panel
    invisible the other way. The volcano card is the sharpest case, because
    its entire content is text.
    """
    ctx = _fit()
    ax, _ = _draw(panel, ctx, ground=ground)
    drawn = [_hex(text.get_color()) for text in ax.texts
             if text.get_text().strip()]
    assert drawn, "the panel drew no text at all"
    assert _hex(ink) in drawn, (
        f"on a {ground} ground the panel's ink is {set(drawn)}, "
        f"and it needs {_hex(ink)}")


# --------------------------------------------------------------------------- #
#  Everything is grey except what the sentence is about
# --------------------------------------------------------------------------- #

def test_a_plate_with_no_positional_effect_is_drawn_entirely_grey():
    """Colour is an argument. With Kruskal-Wallis unable to reject there is
    no argument to make, so nothing may be highlighted -- a panel that always
    colours its most extreme group teaches a reader to see an effect in
    noise."""
    ctx = _fit(edge_delta=0.0)
    ax, stats = _draw("row_effects", ctx)

    assert stats["kruskal_p"] > 0.05, (
        "fixture is wrong: this plate is supposed to have no row effect")
    assert stats["highlighted_groups"] == []
    assert set(_mark_colours(ax)) == {_role("data")}, _mark_colours(ax)


def test_a_planted_edge_artefact_is_stated_in_colour_not_in_a_faint_wash():
    """The outer rows were 1.5 units dim, and the panel has to say so.

    It used to say it with an ``axvspan`` at 8% alpha -- invisible at report
    size -- and an ASCII arrow in a boxed note. Now the wells themselves carry
    RUST, which is the one thing a reader cannot miss.
    """
    ctx = _fit(edge_delta=1.5)
    ax, stats = _draw("row_effects", ctx)

    assert stats["edge_minus_interior_median"] > 1.0, "fixture lost its edge"
    assert stats["highlighted_groups"] == ["r1", "r8"], stats["highlighted_groups"]

    counts = _mark_colours(ax)
    assert _role("down") in counts, counts
    assert counts[_role("data")] > sum(
        n for colour, n in counts.items() if colour != _role("data")), (
        f"the highlight is not a minority of the marks: {counts}")
    assert not ax.patches, (
        "the edge groups are stated by colouring their wells; the faint "
        "axvspan wash must be gone")


def test_the_named_group_is_the_only_coloured_one_when_the_test_rejects():
    """A real gradient down the columns: exactly one group carries BLUE, and
    it is the one the annotation names."""
    meta = _metadata(n_rows=8, n_cols=12)
    n = len(meta)
    rng = np.random.default_rng([11, 2])
    column = meta[schema.COLUMN_KEY].str.extract(r"(\d+)")[0].astype(int)
    X = pd.DataFrame({"Intercept": np.ones(n), "x1": rng.normal(size=n)},
                     index=meta.index)
    y = 1.0 + X["x1"] + 0.25 * column + rng.normal(size=n) * 0.2
    ctx = rq.build_context(sm.OLS(y, X).fit(), X, y, metadata=meta)

    ax, stats = _draw("column_effects", ctx)
    assert stats["kruskal_p"] < 1e-6
    assert stats["highlighted_groups"] == [stats["worst_group"]]
    assert _role("highlight") in _mark_colours(ax)
    named = [text.get_text() for text in ax.texts
             if _hex(text.get_color()) == _role("highlight")]
    assert named and stats["worst_group"] in named[0], named


def test_none_of_the_old_module_palette_survives_on_these_panels():
    """Teal points and a red median bar: the loudest thing on the old panel
    was the box's median line, which is chrome, not the claim."""
    ctx = _fit(edge_delta=1.5)
    for panel in ("plate_effects", "row_effects", "column_effects",
                  "volcano_reference"):
        ctx_for = _fit(edge_delta=1.5, plates=("plate1", "plate2"))
        ax, _ = _draw(panel, ctx_for if panel == "plate_effects" else ctx)
        used = ({_hex(line.get_color()) for line in ax.lines}
                | set(_mark_colours(ax))
                | {_hex(text.get_color()) for text in ax.texts})
        assert not used & _OLD_PALETTE, f"{panel} still draws {used & _OLD_PALETTE}"


# --------------------------------------------------------------------------- #
#  Type, chrome and the things the skill forbids outright
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("panel,expected",
                         [("plate_effects", "residuals by plate"),
                          ("row_effects", "residuals by row"),
                          ("column_effects", "residuals by column")])
def test_the_panels_carry_a_descriptor_and_not_a_sentence_title(panel, expected):
    """"No panel titles as sentences. If a panel needs a descriptor it is
    2-4 words above the axes." The old title was a sentence plus a
    parenthesised n on a second line."""
    ctx = _fit(plates=("plate1", "plate2"))
    ax, _ = _draw(panel, ctx)

    title = ax.get_title()
    assert title == expected
    assert title == title.lower() and "\n" not in title
    assert 2 <= len(title.split()) <= 4
    assert ax.title.get_fontsize() == TYPE_SCALE["label"]


def test_the_statistics_block_is_not_drawn_in_a_box():
    """The published figures have no framed annotations anywhere, and a white
    rounded box is also a light-page assumption sitting on a page whose
    colour the panel does not own."""
    ctx = _fit()
    ax, _ = _draw("row_effects", ctx)
    notes = [text for text in ax.texts if "Kruskal-Wallis" in text.get_text()]
    assert len(notes) == 1, [t.get_text() for t in ax.texts]
    assert notes[0].get_bbox_patch() is None
    assert notes[0].get_fontsize() == TYPE_SCALE["annotation"]


def test_the_zero_line_is_a_thin_dashed_grey_reference():
    """"Reference lines, thresholds and limits of detection are grey, thin,
    dashed or dotted -- never bold." A reference is not a result."""
    ctx = _fit()
    ax, _ = _draw("row_effects", ctx)
    zero = [line for line in ax.lines
            if list(line.get_ydata()) == [0.0, 0.0] and line.get_linestyle() != "-"]
    assert len(zero) == 1, "expected exactly one dashed line at y = 0"
    assert _hex(zero[0].get_color()) == _role("reference")
    assert zero[0].get_linewidth() <= 0.7


def test_the_ticks_and_spines_are_on_the_house_scale():
    """The axes is built by the caller before the style context exists, so a
    panel that only wraps its own drawing leaves 8 pt labels and 0.8 pt black
    spines behind. It has to re-ink what it inherited."""
    ctx = _fit()
    ax, _ = _draw("row_effects", ctx)

    assert [name for name, spine in ax.spines.items() if spine.get_visible()] \
        == ["left", "bottom"]
    for name in ("left", "bottom"):
        assert _hex(ax.spines[name].get_edgecolor()) == _hex(INK_PRINT)
        assert ax.spines[name].get_linewidth() < 0.7
    labels = ax.get_xticklabels()
    assert labels and labels[0].get_fontsize() == TYPE_SCALE["tick"]
    assert _hex(labels[0].get_color()) == _hex(INK_PRINT)
    assert ax.xaxis.label.get_fontsize() == TYPE_SCALE["label"]
    assert not any(line.get_visible() for line in ax.get_xgridlines())


def test_short_tick_labels_stay_flat_and_long_ones_rotate():
    """45 degrees is for labels that would not fit flat. Rotating `r1`
    through `r8` costs a third of the panel's height and buys nothing;
    `plate1` genuinely does not fit."""
    ctx = _fit(plates=("plate1", "plate2"))
    rows, _ = _draw("row_effects", ctx)
    plates, _ = _draw("plate_effects", ctx)

    assert rows.get_xticklabels()[0].get_rotation() == 0
    assert plates.get_xticklabels()[0].get_rotation() == 45


def test_the_signpost_does_not_shout_louder_than_a_real_panel():
    """A card that says "the figure is elsewhere" was set in 10 pt bold, which
    made it the loudest element on a combined page whose real panels label
    their axes at 7 pt."""
    ctx = _fit()
    ax, stats = _draw("volcano_reference", ctx)

    assert stats["state"] == "unlocated"
    sizes = [text.get_fontsize() for text in ax.texts]
    assert sizes and max(sizes) <= TYPE_SCALE["label"], sizes
    heading = [t for t in ax.texts if t.get_text() == "volcano plot"]
    assert len(heading) == 1, [t.get_text() for t in ax.texts]


def test_the_signpost_keeps_the_run_s_own_path_inside_the_panel():
    """The real screen writes to a 79-character directory, which ran off both
    sides of the card. Wrapping it is the whole fix -- the path itself is the
    only reason this panel exists."""
    ctx = _fit()
    # The real one, from the tsg101 screen this panel was verified against.
    ctx.volcano_path = ("/mnt/firecuda2/Claude/toxoplasma_projects/"
                        "tsg101_screen/test/results/plate1_dv/ols/list/"
                        "volcano_plot.pdf")
    ax, stats = _draw("volcano_reference", ctx)

    assert stats["state"] == "referenced"
    body = [t.get_text() for t in ax.texts if "volcano_plot.pdf" in t.get_text()]
    assert body, [t.get_text() for t in ax.texts]
    assert max(len(line) for line in body[0].splitlines()) <= 56, body[0]


# --------------------------------------------------------------------------- #
#  A restyle moves no numbers
# --------------------------------------------------------------------------- #

def test_the_restyle_moved_no_numbers():
    """Every statistic this panel reported before the house style, reported
    unchanged after it -- the colour decision reads them, it does not feed
    them."""
    ctx = _fit(edge_delta=1.5)
    _, stats = _draw("row_effects", ctx)

    assert stats["n_groups"] == 8
    assert stats["groups"] == [f"r{i}" for i in range(1, 9)]
    assert stats["worst_group"] in ("r1", "r8")
    assert stats["kruskal_p"] < 1e-6
    assert stats["edge_minus_interior_median"] > 1.0
    assert len(stats["medians"]) == 8
    # The one new key: which groups were given colour, so the decision the
    # figure makes is also in the text report.
    assert set(stats) == {"n_groups", "groups", "medians", "kruskal_p",
                          "worst_group", "worst_median",
                          "edge_minus_interior_median", "highlighted_groups"}
