"""Instruction 149: the volcano's y axis is continuous again.

The report was "gra14 and 225160 allways have the same adjusted p value, why
is there a sealing... with BH FDR adjustment it looks like the p values have
been binned and are not continuous".

THE OBSERVATION IS CORRECT AND IT IS NOT A BUG, which is why this file starts
with the maths rather than with the widget. Benjamini-Hochberg's adjusted P is
a cumulative minimum taken from the largest P downwards, so the moment a later
rank produces a smaller value every earlier rank is pulled down onto it and a
whole block collapses. Measured on the maintainer's runs: 823 q values, 31
distinct, 19 tied levels covering 811 coefficients.

The answer is not to move the points. It is to put the RAW P on the axis --
continuous, and the evidence per test -- and let the correction decide the
COLOUR and the LINE, which is the discrete thing it actually is. The line is
exact: every correction here is monotone within a family, so the set it calls
is a lower set and one horizontal line divides the plot exactly as the FDR
does.

Instructions 151 and 152 are here too, because they are the same three
controls: the restyle menu's colours and widths.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.multiple_testing import (LOCAL_FDR_MIN_TESTS, METHODS,
                                    adjust_p_values, critical_p_value,
                                    local_fdr)


ALPHA = 0.05


def _p_values(n: int = 823, hits: int = 23, seed: int = 0) -> np.ndarray:
    """A screen: mostly null, with a handful of real effects."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(0, 1, n - hits),
                           10.0 ** (-rng.uniform(4, 8, hits))])


def _frame(p=None, *, level: str = "grna", method: str = "fdr_bh",
           seed: int = 0) -> pd.DataFrame:
    """A coefficient table shaped like the one `perform_regression` writes."""
    p = _p_values(seed=seed) if p is None else np.asarray(p, dtype=float)
    n = p.size
    rng = np.random.default_rng(seed + 1)
    if level == "grna":
        feature = [f"fraction:grna[{100000 + i}_{i % 4 + 1}]" for i in range(n)]
    else:
        feature = [f"gene_fraction:gene[{100000 + i}]" for i in range(n)]
    q, _ = adjust_p_values(p, method=method, alpha=ALPHA)
    return pd.DataFrame({"feature": feature,
                         "coefficient": rng.normal(0, 1, n),
                         "p_value": p, "q_value": q,
                         "multiple_testing_method": method})


# --------------------------------------------------------------------------- #
#  The maths, before the widget
# --------------------------------------------------------------------------- #

def test_benjamini_hochberg_really_does_tie_and_it_is_the_procedure_working():
    """The diagnosis, checked rather than asserted. If BH stopped tying, the
    whole design this file tests would be answering a question nobody has."""
    p = _p_values()
    q, _ = adjust_p_values(p, "fdr_bh", ALPHA)

    assert len(np.unique(q)) < len(np.unique(p)) / 4, (
        "BH is meant to collapse blocks of coefficients onto one q")


def test_the_critical_line_is_the_exact_bh_identity_not_an_approximation():
    """q_(i) <= alpha if and only if p_(i) <= alpha * i / n, at the largest
    such i. Computed from the correction's own rejection call so the line
    cannot disagree with the colours beside it -- and here checked against
    the textbook formula, which is the thing it has to equal."""
    p = _p_values()
    n = p.size
    ranked = np.sort(p)
    satisfied = ranked <= ALPHA * np.arange(1, n + 1) / n
    k = int(np.nonzero(satisfied)[0].max()) + 1

    assert critical_p_value(p, "fdr_bh", ALPHA) == ranked[k - 1]


def test_the_line_is_not_alpha_and_that_is_the_mistake_it_replaces():
    """Drawing the threshold at -log10(0.05) is the UNCORRECTED cut. It calls
    far too much of the screen, which is the whole reason the corrected
    threshold has to be computed rather than assumed."""
    p = _p_values()
    critical = critical_p_value(p, "fdr_bh", ALPHA)

    assert critical < ALPHA / 100
    # Measured on this screen: 56 called at the uncorrected 0.05 against 24
    # at the corrected threshold. Drawing the line at alpha would present
    # more than twice as many points as survivors of a correction they never
    # went through.
    assert int(np.sum(p <= ALPHA)) > 2 * int(np.sum(p <= critical))


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_correction_calls_a_lower_set_so_one_line_can_divide_it(method):
    """The premise of the whole design: there is a rank k such that every
    test with p <= p_(k) is called and every test above it is not. If any of
    the thirteen were not monotone in the raw P, a single horizontal line
    would be a lie for that one."""
    p = _p_values()
    _, rejected = adjust_p_values(p, method=method, alpha=ALPHA)
    if not rejected.any():
        pytest.skip(f"{method} calls nothing on this screen")

    threshold = critical_p_value(p, method, ALPHA)

    assert threshold == p[rejected].max()
    assert not np.any((p <= threshold) & ~rejected), (
        f"{method}'s rejection region is not a lower set")


def test_nothing_called_has_no_critical_value_and_that_is_a_finding():
    """"When NOTHING is called there is no k and no line: say so, do not draw
    one." An empty result is a result."""
    rng = np.random.default_rng(4)

    assert critical_p_value(rng.uniform(0.2, 1.0, 200), "fdr_bh", ALPHA) is None


# --------------------------------------------------------------------------- #
#  Section B: a genuinely continuous FDR quantity
# --------------------------------------------------------------------------- #

def test_the_local_fdr_is_distinct_where_benjamini_hochberg_is_tied():
    """The point of offering it. BH's q is a tail area and steps; the local
    FDR is a density ratio and is a strictly monotone function of the raw P,
    so two different P values never land on the same height."""
    p = _p_values()
    q, _ = adjust_p_values(p, "fdr_bh", ALPHA)
    lfdr = local_fdr(p)

    assert len(np.unique(q)) < p.size / 4
    assert len(np.unique(lfdr)) == len(np.unique(p)) == p.size


def test_the_local_fdr_is_monotone_in_the_p_value():
    """An FDR that fell as the evidence weakened would be a ranking that
    disagrees with itself."""
    p = _p_values()
    order = np.argsort(p)

    assert np.all(np.diff(local_fdr(p)[order]) >= -1e-12)


def test_a_family_too_small_to_read_a_density_off_says_one_rather_than_a_shape():
    """Fitting a two-parameter mixture to a dozen numbers and drawing the
    result is showing the user the fit, not the screen."""
    small = np.linspace(0.001, 0.9, LOCAL_FDR_MIN_TESTS - 1)

    assert np.all(local_fdr(small) == 1.0)
    assert len(np.unique(local_fdr(_p_values(n=LOCAL_FDR_MIN_TESTS * 4)))) > 1


def test_the_local_fdr_keeps_nans_and_does_not_count_them():
    """A guide that could not be tested is not a test."""
    values = local_fdr(np.array([0.001, np.nan, 0.5]))

    assert np.isnan(values[1])
    assert np.isfinite(values[[0, 2]]).all()


def test_a_p_value_of_exactly_zero_is_a_real_result_not_a_crash():
    """It is an underflow, and log(0) would take the whole fit with it."""
    values = local_fdr(np.concatenate([[0.0], _p_values(n=100, hits=5)]))

    assert np.isfinite(values).all()
    assert values[0] == values.min()


# --------------------------------------------------------------------------- #
#  Section E: the family the recorrection is applied to
# --------------------------------------------------------------------------- #

def test_family_labels_says_the_same_thing_as_the_two_functions_it_replaces():
    """It is a vectorised spelling of `tested_family` plus `guide_of`, not a
    second rule. A second copy of that parse is how a dot in the volcano and
    a row in the hit list come to name different guides."""
    from spacr.hits import family_labels, guide_of, tested_family

    terms = ["Intercept", "fraction:grna[233460_1]", "gene_fraction:gene[233460]",
             "rowID[T.2]", "columnID[T.c3]", "C(condition)[T.pc]",
             "no brackets at all", "x[T.233460_12]", "y[]", "plateID[T.p1]",
             None, float("nan"), 5]
    expected = ["" if not tested else
                ("grna" if guide_of(term) is not None else "gene")
                for term, tested in zip(terms, tested_family(terms))]

    assert family_labels(terms).tolist() == expected


def test_pooling_two_families_would_change_every_number():
    """Which is why the plot corrects within a level. The correction applies
    within a family, a run at level='both' fits twice, and pooling changes n
    and therefore every q on the screen."""
    guides, genes = _p_values(seed=1), _p_values(n=200, hits=6, seed=2)

    apart, _ = adjust_p_values(guides, "fdr_bh", ALPHA)
    pooled, _ = adjust_p_values(np.concatenate([guides, genes]), "fdr_bh", ALPHA)

    assert not np.allclose(apart, pooled[:guides.size])


# --------------------------------------------------------------------------- #
#  The widget: what a user sees
# --------------------------------------------------------------------------- #

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")


@pytest.fixture()
def volcano(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame())
    return plot


def _entries(plot) -> list:
    from spacr.qt.widgets.fast_plots import menu_entries

    return [action.text() for action in menu_entries(plot.build_style_menu())]


def _action(plot, fragment: str):
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(plot.build_style_menu()):
        if fragment in action.text():
            return action
    raise AssertionError(f"no entry containing {fragment!r}: {_entries(plot)}")


def _line_labels(plot) -> list:
    return [item.label.textItem.toPlainText() for item in plot.line_items()
            if getattr(item, "label", None) is not None]


def _drawn_y(plot) -> np.ndarray:
    for item in plot.plot.plotItem.items:
        if hasattr(item, "data") and hasattr(item, "setBrush"):
            return np.asarray(item.data["y"], dtype=float)
    raise AssertionError("nothing is drawn as points")


def test_the_default_height_is_the_raw_p_and_it_is_continuous(volcano):
    """The whole instruction in one assertion: the axis the user gets without
    asking is the one with no steps in it."""
    frame = _frame()
    drawn = _drawn_y(volcano)

    assert volcano.p_axis() == "raw"
    assert volcano.plot.getAxis("left").labelText == "-log10(p)"
    assert np.allclose(np.sort(drawn),
                       np.sort(-np.log10(frame["p_value"].to_numpy())))
    assert len(np.unique(drawn)) == len(drawn)


def test_the_caption_says_which_number_is_the_height_and_which_the_colour(
        volcano):
    """A volcano whose height is the raw P and whose colour is the FDR, with
    nothing saying so, is a figure a reader misreads in the direction of
    over-confidence. It is the reason this default is safe to ship."""
    caption = volcano.caption()

    assert "raw p" in caption
    assert "colour" in caption and "line" in caption
    assert "Benjamini-Hochberg" in caption
    assert caption in volcano._status.text()


def test_the_threshold_line_is_the_corrected_cut_and_not_alpha(volcano):
    """"THE LINE IS NOT alpha. Drawing the line at -log10(0.05) is the
    mistake this replaces"."""
    critical = critical_p_value(_frame()["p_value"], "fdr_bh", ALPHA)
    horizontal = [item for item in volcano.line_items()
                  if getattr(item, "angle", 90) == 0]

    assert len(horizontal) == 1
    assert horizontal[0].value() == pytest.approx(-np.log10(critical))
    assert horizontal[0].value() != pytest.approx(-np.log10(ALPHA))
    assert f"p<={critical:.3g}" in _line_labels(volcano)[0]


def test_a_screen_that_calls_nothing_draws_no_line_and_says_so(qtbot):
    """"WHEN NOTHING IS CALLED there is no k and no line: say so, do not draw
    one." An empty result is a finding."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    rng = np.random.default_rng(9)
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(rng.uniform(0.2, 1.0, 400)))

    assert [item for item in plot.line_items()
            if getattr(item, "angle", 90) == 0] == []
    assert "Nothing is called" in plot.caption()


def test_the_colour_carries_the_call_and_the_legend_names_it(volcano):
    """Continuous height, binary colour, and the colour doing what the house
    style says colour is for: carrying the claim."""
    called = int(np.sum(volcano._called))

    assert called
    assert f"called ({called})" in volcano._legend_colours
    assert volcano._legend_box.isEnabled()


def test_the_adjusted_axis_is_still_offered_and_still_shows_the_steps(volcano):
    """"Honest, stepped, and kept -- because that is what BH is, and a user
    comparing against a published figure drawn that way needs to be able to
    reproduce it. It is not the default"."""
    volcano.set_p_axis("adjusted")
    drawn = _drawn_y(volcano)

    assert volcano.p_axis() == "adjusted"
    assert "adjusted p" in volcano.plot.getAxis("left").labelText
    assert len(np.unique(drawn)) < len(drawn) / 4, (
        "the adjusted axis has to keep its ties -- that is what BH is")


def test_two_tied_q_values_land_on_exactly_the_same_height(volcano):
    """The one thing this plot will not do is jitter the y axis to separate
    them: it moves a point away from its own value, and this instruction
    exists because a plot was showing something the data did not say."""
    volcano.set_p_axis("adjusted")
    q = volcano._q_values
    drawn = _drawn_y(volcano)
    ties = np.flatnonzero(q == np.sort(q)[len(q) // 2])

    assert ties.size > 1, "the fixture must carry a tie"
    assert len(np.unique(drawn[ties])) == 1


def test_the_local_fdr_axis_separates_what_the_adjusted_axis_ties(volcano):
    volcano.set_p_axis("adjusted")
    stepped = len(np.unique(_drawn_y(volcano)))

    volcano.set_p_axis("lfdr")

    assert "local FDR" in volcano.plot.getAxis("left").labelText
    assert len(np.unique(_drawn_y(volcano))) > 4 * stepped
    assert "continuous by construction" in volcano.caption()


def test_the_local_fdr_is_not_computed_until_it_is_asked_for(qtbot):
    """Measured at 25 ms of a 40 ms redraw on the real screen's 1,215
    coefficients -- more than drawing the plot -- and the default axis does
    not use it."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame())

    assert plot._lfdr_values is None
    assert plot.local_fdr_values() is not None
    assert plot._lfdr_values is not None


def test_the_three_axes_are_on_the_plots_own_menu(volcano):
    entries = _entries(volcano)

    assert any("raw p" in text for text in entries), entries
    assert any("adjusted p" in text for text in entries), entries
    assert any("local FDR" in text for text in entries), entries


def test_a_family_too_small_for_a_density_greys_the_local_fdr_entry(qtbot):
    """Instruction 106's rule: greyed AND saying why, rather than absent or
    present-but-inert."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(_p_values(n=8, hits=1, seed=5)))

    action = _action(plot, "local FDR")

    assert not action.isEnabled()
    assert "too small" in action.text()


# --------------------------------------------------------------------------- #
#  Section E: the correction, chosen on the graph
# --------------------------------------------------------------------------- #

def test_every_correction_spacr_knows_is_on_the_menu(volcano):
    from spacr.multiple_testing import method_label

    entries = " || ".join(_entries(volcano))

    for key in METHODS:
        assert method_label(key) in entries, key


def test_choosing_a_correction_recomputes_the_line_on_the_spot(volcano):
    before = volcano.families()["grna"][0]

    volcano.set_correction("bonferroni")

    assert volcano.correction() == "bonferroni"
    assert volcano.families()["grna"][0] == critical_p_value(
        _frame()["p_value"], "bonferroni", ALPHA)
    assert volcano.families()["grna"][0] <= before, (
        "Bonferroni is never more permissive than Benjamini-Hochberg")


def test_the_plot_says_which_correction_it_draws_and_whether_it_is_the_runs(
        volcano):
    """"It MUST NOT SILENTLY DISAGREE WITH THE EXPORTED TABLE... The plot
    SAYS which correction it is drawing and whether that is the run's"."""
    assert "the run's" in " ".join(_entries(volcano))
    assert "the run used" not in volcano.caption().lower()

    volcano.set_correction("bonferroni")

    assert "The run used fdr_bh" in volcano.caption()
    assert "VIEW" in volcano.caption()


def test_the_recorrection_is_within_a_level_and_not_across_two(qtbot):
    """"IT MUST RECORRECT THE RIGHT FAMILY... Recomputing over whatever
    happens to be on screen would pool two families and quietly change every
    number"."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    guides = _frame(seed=1)
    genes = _frame(_p_values(n=200, hits=6, seed=2), level="gene", seed=2)
    both = pd.concat([guides, genes], ignore_index=True)

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(both)

    assert set(plot.families()) == {"grna", "gene"}
    assert plot.families()["grna"][2] == len(guides)
    assert plot.families()["gene"][2] == len(genes)
    apart, _ = adjust_p_values(guides["p_value"], "fdr_bh", ALPHA)
    assert np.allclose(plot._q_values[:len(guides)], apart)
    assert "within each level separately" in plot.caption()


def test_two_families_with_different_cuts_get_a_line_each(qtbot):
    """One horizontal line divides the plot exactly as the FDR does -- for
    ONE family. Two families with different critical values need two, each
    named, or the line is exact for one of them and wrong for the other."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    guides = _frame(seed=1)
    genes = _frame(_p_values(n=300, hits=40, seed=3), level="gene", seed=3)
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(pd.concat([guides, genes], ignore_index=True))

    labels = [text for text in _line_labels(plot) if "p<=" in text]

    assert len(labels) == 2, labels
    assert any("grna" in text for text in labels)
    assert any("gene" in text for text in labels)


def test_a_plot_that_cannot_reproduce_the_tables_q_values_says_so(qtbot):
    """The check that catches a family mismatch. It fired for real while this
    was being written: a q column corrected over 823 rows against a plot
    correcting the 822 that are hypotheses."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    frame = _frame()
    frame.loc[0, "q_value"] = 0.5           # a table that is not this family's
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(frame)

    assert "WARNING" in plot.caption()
    assert "disagree" in plot.caption()


def test_the_recorrected_table_can_be_written_out(volcano, tmp_path,
                                                  monkeypatch):
    """"Offer to write the re-corrected table, or say plainly that it is a
    view only. Do not leave it ambiguous"."""
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "recorrected.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    volcano.set_correction("bonferroni")

    _action(volcano, "Write this correction as a table").trigger()

    written = pd.read_csv(target)
    assert set(written["multiple_testing_method"]) == {"bonferroni"}
    expected, _ = adjust_p_values(_frame()["p_value"], "bonferroni", ALPHA)
    assert np.allclose(written["q_value"], expected)
    assert written["called_at_0.05"].sum() == int(np.sum(volcano._called))


def test_clicking_a_dot_says_the_q_that_decided_its_colour(volcano):
    """On a raw-P axis the colour is an assertion the reader cannot check
    unless the number behind it is one click away."""
    detail = volcano._detail(int(np.argmin(_frame()["p_value"].to_numpy())))

    assert "q=" in detail and "fdr_bh" in detail and "called" in detail


# --------------------------------------------------------------------------- #
#  Section D: the y-axis split
# --------------------------------------------------------------------------- #

def _split_frame() -> pd.DataFrame:
    """A screen with a genuine empty stretch: nothing between 1e-4 and 1e-20."""
    rng = np.random.default_rng(3)
    return _frame(np.concatenate([rng.uniform(1e-4, 1, 295),
                                  10.0 ** (-rng.uniform(20, 30, 5))]))


def test_the_y_axis_can_be_split_from_the_plots_own_menu(qtbot):
    """"the option to insert an axis split on the y axis", asked for by
    name, and it belongs on the plot's menu rather than in a setting."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())

    assert "Split the y axis…" in _entries(plot)
    assert plot.set_y_split(5.0, 19.0) == ""
    assert plot.y_split() == (5.0, 19.0)
    assert "Y axis split (5-19): remove" in _entries(plot)


def test_the_split_axis_still_prints_the_datas_own_numbers(qtbot):
    """What makes it a broken axis rather than a lie: the ruler is piecewise
    linear and every number printed beside it is the number the mark has."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())
    tall = _drawn_y(plot).max()
    plot.set_y_split(5.0, 19.0)

    compressed = _drawn_y(plot).max()

    assert compressed < tall - 13, "the empty stretch was not taken out"
    assert plot._to_data(compressed, "y") == pytest.approx(tall)
    axis = plot.plot.getAxis("left")
    for spacing, values in axis.tickValues(0.0, compressed, 500):
        for value in values:
            printed = axis.tickStrings([value], 1, spacing)[0]
            assert float(printed) == pytest.approx(
                plot._to_data(value, "y"), abs=0.6), printed


def test_the_split_is_refused_when_the_band_holds_points(qtbot):
    """A mark inside the hidden band has no number left on the ruler to sit
    at, and drawing it in the break would put a point somewhere its value is
    not -- which is the jitter this instruction forbids, by another route."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())

    reason = plot.set_y_split(1.0, 2.0)

    assert "points sit inside" in reason
    assert plot.y_split() is None
    assert reason in plot._style_note


def test_the_split_says_what_it_fixes_and_what_it_does_not(qtbot):
    """"Build it, and do not sell it as the answer to the ties." It fixes
    dynamic range; BH's steps are the procedure working."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())
    plot.set_y_split(5.0, 19.0)

    assert "does not make a stepped adjusted P continuous" in plot._style_note
    assert "split, 5-19 not drawn" in plot.plot.getAxis("left").labelText


def test_a_split_can_be_taken_back_off(qtbot):
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())
    before = _drawn_y(plot).copy()
    plot.set_y_split(5.0, 19.0)

    _action(plot, "Y axis split").trigger()

    assert plot.y_split() is None
    assert np.allclose(_drawn_y(plot), before)
    assert "split" not in plot.plot.getAxis("left").labelText


def test_a_typed_y_limit_survives_a_split(qtbot):
    """The window the user was looking at is meaningless once the ruler
    changes, so the axis re-fits -- but a limit they TYPED is a number they
    chose, and releasing it would throw it away."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_split_frame())
    plot.set_axis_limits(y=(0.0, 30.0))

    plot.set_y_split(5.0, 19.0)

    assert plot.axis_limits()[1] == pytest.approx((0.0, 30.0))


# --------------------------------------------------------------------------- #
#  Instruction 151 and 152: the colour controls
# --------------------------------------------------------------------------- #

def test_this_module_names_no_colour_dialog_of_its_own():
    """Every colour picker here goes through the SHARED helper, which is the
    one place the non-native flag is passed. There were seven call sites in
    this file's neighbourhood, not the six instruction 151 counted --
    `_ask_style_value`, the figure style's own colour fields, was missed --
    which is exactly why the rule is "go through the helper" rather than
    "pass the flag"."""
    source = Path(__file__).resolve().parents[1] / "spacr" / "qt" / \
        "widgets" / "fast_plots.py"
    text = source.read_text()

    assert not re.findall(r"QColorDialog\.getColor\(", text)
    assert "from .colour_picker import pick_colour" in text


def test_the_font_control_reaches_every_piece_of_text(volcano):
    """"Font colour -- EVERY piece of text: title, axis labels, TICK LABELS,
    legend, annotations"."""
    volcano._legend_box.setChecked(True)
    volcano.set_font_colour("#ff00ff")

    for edge in ("bottom", "left"):
        axis = volcano.plot.getAxis(edge)
        assert axis.textPen().color().name() == "#ff00ff"
    assert "#ff00ff" in volcano.plot.plotItem.titleLabel.text.lower() or \
        volcano.plot.plotItem.titleLabel.opts["color"] == "#ff00ff"
    captions = [item.label for item in volcano.line_items()
                if getattr(item, "label", None) is not None]
    assert captions and all(label.color.name() == "#ff00ff"
                            for label in captions)


def test_the_line_control_reaches_the_spines_and_the_tick_marks(volcano):
    """"Line colour -- EVERY line: ... AND THE AXIS SPINES AND TICK MARKS."
    They took `_foreground` at construction and no control changed them
    afterwards, which is exactly what the first report said."""
    volcano.set_line_colour("#00ff00")

    for axis in volcano.axis_items():
        assert axis.pen().color().name() == "#00ff00"
        assert axis.tickPen().color().name() == "#00ff00"
    for item in volcano.line_items():
        assert volcano._pen_of(item).color().name() == "#00ff00"


def test_the_two_controls_do_not_reach_into_each_other(volcano):
    """Tick marks are lines and tick labels are text. That split is the one
    place the two controls meet, so it is the one worth asserting."""
    volcano.set_line_colour("#00ff00")
    volcano.set_font_colour("#ff00ff")

    axis = volcano.plot.getAxis("left")
    assert axis.pen().color().name() == "#00ff00"
    assert axis.tickPen().color().name() == "#00ff00"
    assert axis.textPen().color().name() == "#ff00ff"


def test_the_selection_ring_and_the_scatters_stay_out_of_the_line_control(
        volcano):
    """The ring is a cursor, not a mark; the points have their own control."""
    volcano.highlight_key(_frame()["feature"].iloc[0])
    volcano.set_line_colour("#00ff00")

    assert volcano._highlight is not None
    assert volcano._highlight not in volcano.line_items()
    assert all(item not in volcano.line_items()
               for item in volcano._scatter_items())


def test_a_dash_pattern_survives_the_extended_reach(volcano):
    """Extending the control to the axes must not lose the pen copying that
    lets a threshold line stay dashed."""
    from PySide6.QtCore import Qt

    before = [volcano._pen_of(item).style() for item in volcano.line_items()]
    assert Qt.DashLine in before

    volcano.set_line_colour("#00ff00")

    assert [volcano._pen_of(item).style()
            for item in volcano.line_items()] == before


def test_follow_the_theme_appears_only_once_there_is_something_to_undo(
        volcano):
    assert "Follow the theme (colours)" not in _entries(volcano)

    volcano.set_font_colour("#ff00ff")

    assert "Follow the theme (colours)" in _entries(volcano)


def test_a_line_added_after_the_control_was_used_still_obeys_it(volcano):
    """A redraw puts new threshold lines on the plot. Without this they
    arrive in the default red beside the ones the user recoloured."""
    volcano.set_line_colour("#00ff00")

    volcano.redraw()

    assert volcano.line_items()
    for item in volcano.line_items():
        assert volcano._pen_of(item).color().name() == "#00ff00"


# --------------------------------------------------------------------------- #
#  Who owns the axis
# --------------------------------------------------------------------------- #

def test_a_host_may_seed_the_axis_and_a_person_may_overrule_it(qtbot):
    """`RegressionResultsPanel` switches raw/adjusted by handing over a
    different column, so that keeps working -- until someone picks an axis
    off the plot's own menu. After that the host redraws on every level,
    baseline and compartment change, and any one of them silently putting
    the axis back would be the user watching their own choice undo itself."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    frame = _frame()
    plot = VolcanoPlot()
    qtbot.addWidget(plot)

    plot.set_results(frame, p_column="q_value")
    assert plot.p_axis() == "adjusted"
    plot.set_results(frame, p_column="p_value")
    assert plot.p_axis() == "raw"

    plot.set_p_axis("lfdr")
    plot.set_results(frame, p_column="q_value")

    assert plot.p_axis() == "lfdr"


def test_an_empty_redraw_leaves_no_numbers_from_the_last_one(qtbot):
    """The worst kind of wrong number is one that used to be right."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame())
    assert plot.families()

    plot.set_results(pd.DataFrame({"feature": ["Intercept"],
                                   "coefficient": [1.0], "p_value": [0.5]}))

    assert plot.families() == {}
    assert plot.caption() == ""
    assert plot._detail(0) == ""


def test_a_permutation_p_is_told_it_is_quantised_before_bh_ever_runs(qtbot):
    """Section C. The 4%-distinct case is the permutation run and its raw P
    is quantised to 1/(n+1) BEFORE the correction sees it -- 1,000
    permutations cannot express a p below 1e-3. Saying so is the difference
    between a user raising `guide_permutations` and a user concluding the
    plot is broken."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    rng = np.random.default_rng(2)
    permutations = 1000
    counts = rng.binomial(permutations, rng.uniform(0, 0.5, 823))
    p_values = (1 + counts) / (permutations + 1)
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.set_results(_frame(p_values))

    assert "RAW p is itself quantised" in plot.caption()
    assert "1/(permutations + 1)" in plot.caption()
    # The OBSERVED smallest, not the theoretical floor: a run whose best
    # guide beat the null only 3 times in 1,000 has not reached 1/1001, and
    # claiming it had would be the plot inventing a number.
    assert f"{p_values.min():.3g}" in plot.caption()
    assert f"{len(np.unique(p_values)):,} distinct" in plot.caption()


def test_a_continuous_p_is_not_accused_of_being_quantised(volcano):
    assert "quantised" not in volcano.caption()
