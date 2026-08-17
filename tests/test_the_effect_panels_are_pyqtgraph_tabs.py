"""The two panels the sheet has always drawn and the screen had no twin of.

Instruction 129 B, which measured the gap against the two registries rather
than guessing at it:

    "i would like you to generate all plots with the pyqtgraph and have each
     represented as a tab under results."

    THE GAP IS TWO: `effect_rank` (every gene ranked by effect, as a dot with
    its interval) and `effect_distribution`.

WHY EITHER IS WORTH A TAB, measured on the TSG101 screen rather than argued:
its strongest effect is ``fraction:grna[000000_22]`` at +4.371 with q = 3e-05,
and its THIRD strongest is ``fraction:grna[252190_3]`` at -4.220 with
q = 0.063. Two effects the same size, one called and one not. A volcano ranks
by significance and puts them nowhere near each other; only a list ranked by
the effect itself, with the interval drawn through each dot, shows that the
difference between them is precision and not size.

THE SORT IS THE TRAP AND IT IS THE WHOLE OF THIS FILE. instruction 119 B:

    "A regression table sorted by effect and a scatter drawn in input order
     are the same points in two orders, and joining them by index highlights
     the wrong guide -- silently, and in exactly the direction a user would
     not question, because SOMETHING lights up."

The ranking is sorted by |effect|, so its drawn dot n is the nth LARGEST
coefficient and almost never row n of anything. Every test below therefore
first asserts that the position join and the key join give DIFFERENT answers;
if a change ever makes the fixture stop distinguishing them, that assertion
fails and says so rather than letting the rest of the file pass for free.

The distribution is the other kind of mark: a bar stands for many rows, so it
narrows the table to what it holds instead of guessing which of them was
meant -- the same rule :class:`PValueHistogram` already follows, and now the
same code, because two copies of a half-open binning rule drift.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

#: The real screen, when this is the machine that has it. Every number this
#: file claims about the TSG101 fit was measured there first.
REAL_RESULTS = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/"
                "claude/results/ols/results.csv")


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

def _frame(seed: int = 5, *, standard_error: bool = True,
           corrected: bool = True) -> pd.DataFrame:
    """A screen whose effect order is nothing like its row order.

    Deliberately shaped so a positional join is wrong in every direction:

      * the coefficients are random, so ranking by |effect| reorders the frame
        completely;
      * the significance is INDEPENDENT of the effect, so the ranking's colour
        and its order disagree -- which is the finding the panel exists for;
      * two coefficients are blank, so they are on no dot at all and their
        rows must not be claimed by the dots drawn beside them;
      * an Intercept sits at row 0, in the table and off both panels.
    """
    rng = np.random.default_rng(seed)
    n = 60
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i // 3}_{i % 3}]" for i in range(n)],
        "coefficient": rng.normal(scale=1.5, size=n),
        "p_value": rng.uniform(size=n),
        "gene": [f"{i // 3}" for i in range(n)],
        "condition": ["nc" if i % 17 == 0 else "other" for i in range(n)],
    })
    if standard_error:
        frame["std_err"] = np.abs(rng.normal(0.4, 0.12, n)) + 0.05
    if corrected:
        frame["q_value"] = rng.uniform(size=n) ** 3
    # The nuisance term the fit writes: in the table, on neither effect panel,
    # and carrying a q_value of NaN because the correction never covered it.
    intercept = {"feature": "Intercept", "coefficient": 0.19,
                 "p_value": 3.1e-46, "gene": None, "condition": "other"}
    if standard_error:
        intercept["std_err"] = 0.02
    if corrected:
        intercept["q_value"] = np.nan
    frame = pd.concat([pd.DataFrame([intercept]), frame], ignore_index=True)
    # Two coefficients that did not come out. They are drawn nowhere.
    frame.loc[frame.index[[4, 21]], "coefficient"] = np.nan
    return frame


@pytest.fixture()
def results():
    return _frame()


@pytest.fixture()
def panel(qtbot, results):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    assert widget.set_frame(results, source="results.csv")
    # Sort the table as well, so neither end of the link is in frame order.
    widget.table.table.sortItems(1)
    return widget


def points_of(plot):
    """Every drawn dot on ``plot``, carrying the frame row it came from."""
    found = []
    for item in plot.plot.listDataItems():
        if item is plot._highlight or not hasattr(item, "points"):
            continue
        found.extend(item.points())
    return found


def rows_of(plot):
    return [int(point.data()) for point in points_of(plot)]


def intervals_of(plot):
    """``{rank: (low, high)}`` for every interval bar actually drawn.

    The bars are three ``PlotCurveItem``s with ``connect="pairs"`` -- one per
    ink -- rather than one item per coefficient, so they are read back out of
    the arrays rather than counted as artists.
    """
    import pyqtgraph as pg

    found = {}
    for item in plot.plot.plotItem.items:
        if not isinstance(item, pg.PlotCurveItem):
            continue
        x, y = item.getData()
        for start in range(0, len(x), 2):
            found[int(round(float(y[start])))] = (float(x[start]),
                                                  float(x[start + 1]))
    return found


def selected_feature(panel):
    items = panel.table.table.selectedItems()
    if not items:
        return None
    column = list(panel._frame.columns).index("feature")
    return panel.table.table.item(items[0].row(), column).text()


def tab_names(panel):
    return [panel.tabs.tabText(i) for i in range(panel.tabs.count())]


def title_of(plot) -> str:
    return str(plot.plot.plotItem.titleLabel.text)


# --------------------------------------------------------------------------- #
#  They are tabs, and an unfillable one says why rather than being absent
# --------------------------------------------------------------------------- #

class TestTheyAreTabsOfTheirOwn:

    def test_both_panels_have_a_tab_named_for_the_graph(self, panel):
        """"ONE TAB PER GRAPH, named for the graph" -- 129 B."""
        names = tab_names(panel)

        assert "Effect rank" in names, names
        assert "Effect distribution" in names, names

    def test_they_sit_with_the_volcano_they_are_the_other_half_of(self, panel):
        """Reading order is the argument: the result, then how big it is and
        how sure, then whether the model was entitled to say it."""
        names = tab_names(panel)

        assert names.index("Effect rank") == names.index("Volcano") + 1
        assert names.index("Effect distribution") == names.index(
            "Effect rank") + 1

    def test_a_tab_with_no_run_yet_says_what_it_is_waiting_for(self, qtbot):
        """An empty plot behind a tab nobody has opened is indistinguishable
        from a broken one, which is the failure 129 B names for an ABSENT tab
        and which a present empty one commits just as quietly."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)

        for plot in (widget.effect_rank, widget.effect_distribution):
            said = plot._status.text()
            assert "No coefficient table yet" in said, said
            assert "Load results" in said, said

    def test_a_table_with_no_effect_column_says_so_in_both_tabs(self, qtbot):
        """A specific, checkable fact about the file rather than a shrug: a
        frame with no fitted effect is not a regression result, and no amount
        of re-running will make it one."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        assert widget.set_frame(pd.DataFrame({
            "feature": ["fraction:grna[1_1]", "fraction:grna[1_2]"],
            "p_value": [0.01, 0.4], "condition": ["nc", "other"]}))

        for plot in (widget.effect_rank, widget.effect_distribution):
            said = plot._status.text()
            assert "No fitted-effect column" in said, said
            assert "coefficient, coef, effect and estimate" in said, said
        assert not points_of(widget.effect_rank)

    def test_that_table_opens_at_all_rather_than_taking_the_panel_down(
            self, qtbot):
        """Found while writing the test above, and it is not about these tabs.

        The effect-size cut asked for `_effect_column(frame)`, which answers
        "coefficient" for a table that has none of the four spellings -- a
        DEFAULT, not a finding -- and then indexed the frame with it.
        `frame.loc[mask, "coefficient"]` raises KeyError, and the raise came
        out through `refresh_views` into `set_frame`, so such a table opened
        NOTHING: no volcano, no coefficient table, no message. A panel whose
        whole doctrine is "every way this fails says so" was failing in the
        one way that says nothing at all.

        It needs a `condition` column to reach, which is why an ordinary
        coefficient table never tripped it.
        """
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        odd = pd.DataFrame({
            "feature": ["fraction:grna[1_1]", "fraction:grna[1_2]"],
            "beta": [1.0, -2.0],          # not one of the four spellings
            "p_value": [0.01, 0.4],
            "condition": ["nc", "other"]})

        assert widget.set_frame(odd), widget.status_text()

        assert widget.table.table.rowCount() == 2
        assert widget._current_threshold() is None
        assert "no effect-size cut" in widget._threshold_sentence().lower()


# --------------------------------------------------------------------------- #
#  The ranking: sorted by effect, joined on the key
# --------------------------------------------------------------------------- #

class TestTheRankingIsJoinedOnTheKey:

    def test_a_position_join_would_pick_a_different_coefficient(self, panel):
        """Proves the rest of this class is testing something. If the drawn
        rows ever equal 0, 1, 2 ... the fixture has stopped distinguishing the
        two joins and every test below passes for free."""
        rows = rows_of(panel.effect_rank)

        assert rows, "the ranking drew nothing"
        assert rows[:20] != list(range(20)), (
            "the fixture no longer reorders the ranking -- change the effects")

    def test_the_nth_dot_is_the_nth_largest_effect(self, panel, results):
        """The sort, stated as the rows it produces."""
        tested = results["feature"] != "Intercept"
        family = results.loc[tested].reset_index(drop=True)
        magnitude = family["coefficient"].abs().to_numpy()
        expected = [row for row in np.argsort(-magnitude, kind="stable")
                    if np.isfinite(magnitude[row])]

        assert rows_of(panel.effect_rank) == expected

    def test_a_coefficient_that_did_not_come_out_is_on_no_dot(self, panel,
                                                              results):
        """Its row must not be claimed by the dot drawn next to it."""
        family = results.loc[results["feature"] != "Intercept"].reset_index(
            drop=True)
        blank = [int(row) for row in
                 family.index[family["coefficient"].isna()]]

        assert blank, "the fixture has no blank coefficient to lose"
        assert not set(blank) & set(rows_of(panel.effect_rank))

    def test_clicking_a_dot_selects_that_coefficients_row(self, panel):
        point = points_of(panel.effect_rank)[7]
        wanted = panel.effect_rank.key_for_row(int(point.data()))
        assert wanted is not None, "the ranking has no identifiers at all"

        panel.effect_rank._on_points_clicked(None, [point])

        assert selected_feature(panel) == wanted

    def test_clicking_selects_the_key_not_the_drawing_position(self, panel):
        """Dot 7 is the 8th largest effect and not row 7, so a plot reporting
        its own drawing index would name a real coefficient nobody clicked."""
        point = points_of(panel.effect_rank)[7]
        by_key = panel.effect_rank.key_for_row(int(point.data()))
        by_position = panel.effect_rank.key_for_row(7)
        assert by_key != by_position, "fixture no longer distinguishes them"

        panel.effect_rank._on_points_clicked(None, [point])

        assert selected_feature(panel) == by_key
        assert selected_feature(panel) != by_position

    def test_the_ring_lands_on_that_dot_and_not_near_it(self, panel):
        """Present is not enough. A marker in the wrong place is the bug this
        whole file exists to catch, and "an item exists" does not catch it."""
        point = points_of(panel.effect_rank)[12]
        key = panel.effect_rank.key_for_row(int(point.data()))

        assert panel.effect_rank.highlight_key(key)

        drawn = panel.effect_rank._highlight.getData()
        assert float(drawn[0][0]) == pytest.approx(point.pos().x(), abs=1e-9)
        assert float(drawn[1][0]) == pytest.approx(point.pos().y(), abs=1e-9)

    def test_selecting_a_row_rings_it_on_the_ranking_too(self, panel):
        """The other direction: "is my hit actually a big effect" is what this
        panel is for, and it needs the link both ways."""
        table = panel.table.table
        table.selectRow(6)
        column = list(panel._frame.columns).index("feature")
        shown = table.item(6, column).text()

        assert panel.effect_rank._selected_key == shown
        assert panel.effect_rank._highlight is not None

    def test_a_click_reports_the_effect_and_its_interval(self, panel,
                                                          results):
        """"press every dot and get its information" -- for a ranked effect
        that is the number and the range around it, which is the entire
        content of the panel."""
        point = points_of(panel.effect_rank)[2]
        row = int(point.data())
        key = panel.effect_rank.key_for_row(row)
        source = results.loc[results["feature"] == key].iloc[0]

        panel.effect_rank._on_points_clicked(None, [point])

        said = panel.effect_rank._status.text()
        assert key in said
        assert f"effect = {float(source['coefficient']):.3g}" in said, said
        low = float(source["coefficient"]) - 1.96 * float(source["std_err"])
        assert f"[{low:.3g}," in said, said

    def test_the_nuisance_term_is_not_ranked_and_the_panel_says_so(
            self, panel):
        """It is a covariate: its q_value is NaN, so it could never be called,
        and a permanently grey row halfway down a list of hypotheses is a
        different experiment from the one the q-values describe."""
        assert "Intercept" not in panel.effect_rank._keys

        said = panel.effect_rank._status.text()
        assert "1 nuisance term not ranked" in said, said
        assert "covariates" in said, said


# --------------------------------------------------------------------------- #
#  The interval, which is the reason it is dots and not bars
# --------------------------------------------------------------------------- #

class TestTheIntervalIsDrawnAndIsTheRightWidth:

    def test_each_dot_carries_a_1_96_standard_error_interval(self, panel,
                                                              results):
        """A bar chart of coefficients hides the uncertainty that decides
        whether to believe any of them. Drawn is not enough -- drawn at the
        right width is the claim."""
        bars = intervals_of(panel.effect_rank)
        rows = rows_of(panel.effect_rank)
        assert len(bars) == len(rows), (len(bars), len(rows))

        family = results.loc[results["feature"] != "Intercept"].reset_index(
            drop=True)
        for rank in (0, 5, 19, len(rows) - 1):
            row = rows[rank]
            centre = float(family["coefficient"].iloc[row])
            half = 1.96 * float(family["std_err"].iloc[row])
            low, high = bars[rank]
            assert low == pytest.approx(centre - half, abs=1e-9)
            assert high == pytest.approx(centre + half, abs=1e-9)

    def test_a_table_with_no_standard_error_says_the_dots_have_none(
            self, qtbot):
        """The penalised backends report no standard error and never will.
        Silence there would leave a reader taking point estimates for exact
        ones -- which is the whole failure a bar chart commits."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        assert widget.set_frame(_frame(standard_error=False))

        assert intervals_of(widget.effect_rank) == {}
        said = widget.effect_rank._status.text()
        assert "no standard error" in said, said
        assert "point estimates" in said, said

    def test_the_intervals_are_reachable_by_the_line_restyle(self, panel):
        """"line color and width" must reach every line on the plot. A control
        that recoloured the zero line and left 60 interval bars behind would
        be worse than one that was greyed out."""
        plot = panel.effect_rank
        assert plot.line_reason() == ""
        assert len(plot.line_items()) >= 2, "the intervals are not lines here"

        touched = plot.set_line_style(colour="#123456", width=3.0)

        assert touched == len(plot.line_items())
        for item in plot.line_items():
            pen = plot._pen_of(item)
            assert pen.color().name() == "#123456"
            assert pen.widthF() == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
#  The colouring, which is the house rule and the saved panel's rule
# --------------------------------------------------------------------------- #

class TestTheColouringSaysWhatTheSentenceIs:

    def _coloured(self, plot):
        from spacr.qt.widgets.fast_plots import EffectRankPlot

        wanted = {EffectRankPlot.UP_INK.lower(),
                  EffectRankPlot.DOWN_INK.lower()}
        return sum(point.brush().color().name().lower() in wanted
                   for point in points_of(plot))

    def test_only_the_called_coefficients_are_coloured(self, panel, results):
        """Everything grey except what the sentence is about."""
        family = results.loc[results["feature"] != "Intercept"]
        called = int(((family["q_value"] <= 0.05)
                      & family["coefficient"].notna()).sum())
        assert 0 < called < len(family), "the fixture has no contrast to show"

        assert self._coloured(panel.effect_rank) == called
        assert f"{called} called at q_value ≤ 0.05" in \
            panel.effect_rank._status.text()

    def test_a_table_with_no_corrected_p_colours_nothing_and_says_why(
            self, qtbot):
        """`spacr.figures.panels.effect_rank` colours on a q and nothing else.
        Calling hits off an uncorrected p across a thousand tests is the
        multiple-testing error a screen panel exists to make visible, and the
        tab must not disagree with the figure the same run writes to disk."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        assert widget.set_frame(_frame(corrected=False))

        assert self._coloured(widget.effect_rank) == 0
        said = widget.effect_rank._status.text()
        assert "no corrected p-value" in said, said
        assert "uncorrected p" in said, said

    def test_a_penalised_fit_is_not_coloured_by_its_meaningless_p(self,
                                                                  qtbot):
        """`spacr.ml` writes an OLS-style p-value into a lasso results.csv,
        computed as though there were no penalty -- which is why
        `spacr.hits.NO_P_VALUE_TYPES` exists. A ranking that went looking for
        a significance column would colour its dots by a number nobody
        tested, and it would look entirely correct doing it."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        loaded = _frame()
        assert "q_value" in loaded.columns, "the trap needs a column to fall for"
        assert widget.set_frame(loaded, source="/screen/results/x/lasso/list")
        assert widget.ranking()[0] == "selection-frequency"

        assert self._coloured(widget.effect_rank) == 0
        assert "no corrected p-value" in widget.effect_rank._status.text()
        # The effects themselves are real, so the ranking is still drawn: a
        # penalised fit has coefficients, it simply has no hypothesis test.
        assert len(points_of(widget.effect_rank)) > 0


# --------------------------------------------------------------------------- #
#  The distribution: a bar is not a point
# --------------------------------------------------------------------------- #

class TestTheDistribution:

    def test_a_bar_hands_back_exactly_the_rows_it_drew(self, panel, results):
        """The count a bar draws and the rows it reports must be one number.
        Any drift between them is a bar that lies about its own height."""
        plot = panel.effect_distribution
        family = results.loc[results["feature"] != "Intercept"].reset_index(
            drop=True)
        values = family["coefficient"].to_numpy()

        assert int(np.nansum(plot._counts)) == int(np.isfinite(values).sum())
        for index, rows in enumerate(plot._bin_rows):
            assert len(rows) == int(plot._counts[index])
            inside = values[rows]
            assert ((inside >= plot._edges[index])
                    & (inside <= plot._edges[index + 1])).all()

    def test_a_bar_of_many_does_not_pretend_to_select_one(self, panel):
        """Picking the first, the strongest or the nearest would be a guess
        dressed up as an answer -- the same mistake as joining on position."""
        plot = panel.effect_distribution
        index = next(i for i, rows in enumerate(plot._bin_rows)
                     if len(rows) > 1)
        singles, sets = [], []
        plot.key_selected.connect(singles.append)
        plot.keys_selected.connect(sets.append)

        keys = plot.select_bin(index)

        assert len(keys) > 1
        assert singles == [], "a bar of many claimed to select one of them"
        assert sets == [keys]
        said = plot._status.text()
        assert "not one point" in said, said
        assert said.count("effect ") >= 1, "the bar did not name its quantity"

    def test_a_bar_holding_one_row_selects_it_like_any_other_point(self,
                                                                   panel):
        plot = panel.effect_distribution
        index = next(i for i, rows in enumerate(plot._bin_rows)
                     if len(rows) == 1)
        wanted = plot.key_for_row(int(plot._bin_rows[index][0]))

        plot.select_bin(index)

        assert selected_feature(panel) == wanted

    def test_clicking_a_bar_narrows_the_table_to_it(self, panel):
        """"Show me the eleven" is the question a bar CAN answer exactly."""
        plot = panel.effect_distribution
        index = next(i for i, rows in enumerate(plot._bin_rows)
                     if len(rows) > 2)

        keys = plot.select_bin(index)

        shown = sum(not panel.table.table.isRowHidden(row)
                    for row in range(panel.table.table.rowCount()))
        assert shown == len(keys)
        assert "narrowed to" in panel.table._count.text()

    def test_a_selected_row_marks_the_bar_it_falls_in(self, panel, results):
        """Not a ring floating over the bars: this plot never drew that row as
        a mark of its own, and the bar is where the coefficient actually is."""
        family = results.loc[results["feature"] != "Intercept"].reset_index(
            drop=True)
        row = int(family.index[family["coefficient"].notna()][9])
        key = str(family["feature"].iloc[row])

        assert panel.effect_distribution.highlight_key(key)

        outlined = panel.effect_distribution._highlight.opts
        low, high = float(outlined["x0"][0]), float(outlined["x1"][0])
        assert low <= float(family["coefficient"].iloc[row]) <= high

    def test_the_nuisance_term_is_outside_the_family(self, panel):
        """σ is measured over the coefficients the q-values describe. A
        covariate inside it would be measuring the spread of a mixture."""
        assert "Intercept" not in panel.effect_distribution._keys
        said = panel.effect_distribution._status.text()
        assert "1 nuisance term not counted" in said, said

    def test_sigma_is_the_number_the_saved_panel_prints(self, panel, results):
        """The screen and the disk must not describe one screen two ways.

        Read off the matplotlib panel's own annotation rather than recomputed
        here: `spacr.figures.panels.effect_distribution` is what a run writes
        beside the results, and a tab quoting a different σ would leave a
        reader with two numbers and no way to tell which is the fit's.
        """
        from matplotlib.figure import Figure

        from spacr.figures.panels import effect_distribution

        figure = Figure()
        try:
            effect_distribution(figure.add_subplot(111), results)
            saved = "\n".join(text.get_text()
                              for text in figure.axes[0].texts)
        finally:
            figure.clf()
        sigma = [line for line in saved.splitlines() if "σ (MAD)" in line]
        assert sigma, saved

        assert sigma[0].strip() in panel.effect_distribution._status.text()

    def test_a_screen_with_no_spread_says_so_instead_of_drawing_sigma_zero(
            self, qtbot):
        """±0σ is three lines on top of each other pretending to be a cut."""
        from spacr.qt.widgets.fast_plots import EffectDistribution

        plot = EffectDistribution()
        qtbot.addWidget(plot)

        assert plot.set_effects([0.4] * 12) == 12

        said = plot._status.text()
        assert "no spread" in said, said
        assert "σ" in said


# --------------------------------------------------------------------------- #
#  The filter reaches both, and a redraw does not lose the reader's place
# --------------------------------------------------------------------------- #

class TestTheFilterAndTheRedraw:

    def test_the_gene_guide_filter_reaches_both_tabs(self, qtbot):
        """A filter that reaches nine tabs of eleven is worse than none: two
        then disagree with the rest on screen at the same time, with nothing
        saying which is which."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        rows = []
        for gene in range(20):
            rows.append({"feature": f"gene_fraction:gene[{gene}]",
                         "coefficient": 0.1 * gene, "p_value": 0.5,
                         "q_value": 0.5, "std_err": 0.1})
            for guide in range(3):
                rows.append({"feature": f"fraction:grna[{gene}_{guide}]",
                             "coefficient": -0.2 * gene - guide,
                             "p_value": 0.01, "q_value": 0.01,
                             "std_err": 0.1})
        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        assert widget.set_frame(pd.DataFrame(rows))
        whole = len(points_of(widget.effect_rank))

        widget.set_level("gene")

        assert len(points_of(widget.effect_rank)) == 20
        assert len(points_of(widget.effect_rank)) < whole
        assert int(np.nansum(widget.effect_distribution._counts)) == 20

    def test_the_family_is_written_into_both_titles(self, panel):
        """A tab label is above the tab bar and a status line is overwritten
        by whatever was last clicked. The title is neither."""
        panel.set_level("grna")

        assert title_of(panel.effect_rank) == "Effect rank — guides only"
        assert title_of(panel.effect_distribution) == \
            "Effect distribution — guides only"
        assert "Effect rank (guides)" in tab_names(panel)
        assert "Effect distribution (guides)" in tab_names(panel)

    def test_the_selection_survives_a_redraw_of_both(self, panel):
        """A redraw clears the scene, so the marker has to be put back or the
        user loses their place every time they touch a control."""
        panel.table.table.selectRow(5)
        chosen = panel.effect_rank._selected_key
        assert chosen
        assert panel.effect_distribution._highlight is not None

        panel.refresh_views()

        assert panel.effect_rank._selected_key == chosen
        assert panel.effect_rank._highlight is not None
        assert panel.effect_distribution._highlight is not None

    def test_a_new_run_drops_the_ring_on_both(self, panel, results):
        """Each plot re-marks its key at the end of its own draw. The other
        edge of that: a plot whose selection is not cleared on load cheerfully
        re-rings the NEW run at the OLD key."""
        panel.table.table.selectRow(5)
        assert panel.effect_rank._highlight is not None
        assert panel.effect_distribution._highlight is not None

        panel.set_frame(results.iloc[:30].copy(), source="other.csv")

        assert panel.effect_rank._highlight is None
        assert panel.effect_distribution._highlight is None

    def test_a_new_run_does_not_inherit_the_old_keys(self, qtbot):
        """A key list left over from the previous table names rows that do not
        exist in this one, and would join two experiments together."""
        from spacr.qt.widgets.fast_plots import EffectRankPlot

        plot = EffectRankPlot()
        qtbot.addWidget(plot)
        plot.set_results(pd.DataFrame({
            "feature": ["a", "b"], "coefficient": [1.0, 2.0]}))
        assert plot.highlight_key("a")

        plot.set_results(pd.DataFrame({
            "feature": ["c", "d"], "coefficient": [1.0, 2.0]}))

        assert plot.highlight_key("a") is False
        assert plot.key_for_row(0) == "c"

    def test_a_plot_with_no_keys_still_draws_and_claims_nothing(self, qtbot):
        """Keys are optional. A caller with no unique column gets a readable
        ranking that simply does not invite a click it cannot honour."""
        from spacr.qt.widgets.fast_plots import EffectRankPlot

        plot = EffectRankPlot()
        qtbot.addWidget(plot)

        assert plot.set_results(pd.DataFrame(
            {"coefficient": [1.0, -3.0, 2.0]})) == 3

        assert plot.key_for_row(0) is None
        assert "Click a dot" not in plot._status.text()


# --------------------------------------------------------------------------- #
#  pyqtgraph is optional, and these two constructors are new chains through it
# --------------------------------------------------------------------------- #

def test_both_panels_build_and_speak_with_pyqtgraph_absent():
    """Each of these constructors opens a NEW chain through the stand-in.

    ``EffectRankPlot`` runs ``self.plot.getViewBox().invertY(True)`` and
    ``BinnedPlot`` runs ``self.plot.scene().sigMouseClicked.connect(...)``,
    both before any data arrives. `_Absorbs` exists precisely because a chain
    has to survive WHOLE rather than one link at a time, and a machine with
    PySide6 and no pyqtgraph is a real install -- it took down every module in
    the application once already.

    A SUBPROCESS, because an import cannot be undone: pyqtgraph is installed
    here and is in ``sys.modules`` before any test runs, so patching it out
    in-process would test a half-loaded state rather than a machine that never
    had it.
    """
    script = textwrap.dedent("""
        import builtins, os, sys
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        _real = builtins.__import__
        def _blocked(name, *a, **k):
            if name == 'pyqtgraph' or name.startswith('pyqtgraph.'):
                raise ImportError('blocked')
            return _real(name, *a, **k)
        builtins.__import__ = _blocked
        for _m in [m for m in sys.modules if m.startswith('pyqtgraph')]:
            del sys.modules[_m]
        from PySide6.QtWidgets import QApplication, QLabel
        _app = QApplication([])
        from spacr.qt.widgets import fast_plots
        assert fast_plots.HAVE_PYQTGRAPH is False, 'the block did not take'
        import pandas as pd
        rank = fast_plots.EffectRankPlot()
        dist = fast_plots.EffectDistribution()
        print('AVAILABLE', rank.plots_available, dist.plots_available)
        frame = pd.DataFrame({'feature': ['fraction:grna[1_1]'],
                              'coefficient': [1.0]})
        print('RANK', rank.set_results(frame))
        print('DIST', dist.set_effects(frame['coefficient']))
        said = ' '.join(w.text() for w in rank.findChildren(QLabel))
        print('SAYS', 'pyqtgraph' in said, 'spacr[qt]' in said)
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]

    assert "AVAILABLE False False" in out.stdout, out.stdout
    # They still ANSWER -- a plot that cannot draw still counts what it was
    # given, so a caller reading the return value is not lied to either.
    assert "RANK 1" in out.stdout, out.stdout
    assert "DIST 1" in out.stdout, out.stdout
    # And the empty box says what is missing and how to fix it.
    assert "SAYS True True" in out.stdout, out.stdout


# --------------------------------------------------------------------------- #
#  The ranking is only worth a tab because it disagrees with the volcano
# --------------------------------------------------------------------------- #

def test_the_ranking_and_the_volcano_do_not_agree_about_the_top(panel,
                                                                results):
    """If ranking by effect gave the same order as ranking by significance,
    this tab would be the volcano with a different axis and would not be worth
    a tab. It does not: the two answer different questions, which is exactly
    why 129 B asked for both."""
    by_effect = panel.effect_rank.key_for_row(rows_of(panel.effect_rank)[0])
    family = results.loc[results["feature"] != "Intercept"]
    by_significance = str(family.loc[family["q_value"].idxmin(), "feature"])

    assert by_effect != by_significance, (
        "the fixture no longer separates size from significance")


# --------------------------------------------------------------------------- #
#  Against the screen itself
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_RESULTS),
                    reason="the TSG101 screen is not on this machine")
class TestAgainstTheRealScreen:
    """Synthetic frames prove the join; the screen proves the shape.

    1,213 coefficients, one of them the Intercept. 54 called at q ≤ 0.05, no
    standard-error column at all, and σ (MAD) = 0.229.
    """

    @pytest.fixture()
    def screen(self):
        return pd.read_csv(REAL_RESULTS)

    @pytest.fixture()
    def panel(self, qtbot, screen):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        widget.set_frame(screen, source=REAL_RESULTS)
        return widget

    def test_it_ranks_the_family_the_q_values_describe(self, panel, screen):
        from spacr.hits import tested_family

        family = int(tested_family(screen["feature"]).sum())
        assert family == 1212

        assert len(points_of(panel.effect_rank)) == family
        assert "Intercept" not in panel.effect_rank._keys

    def test_the_intercept_is_dropped_for_its_family_not_for_its_size(
            self, screen):
        """The obvious reason is the wrong one here, and writing it down is
        how a later 'simplification' does not reinstate it. The volcano drops
        the intercept because it OWNS the p-axis. By EFFECT it is 0.190
        against a tested maximum of 4.371, i.e. rank 547 of 1,213 -- it
        stretches nothing. It comes out because its q_value is NaN and a
        permanently uncallable row is not a hypothesis."""
        from spacr.hits import tested_family

        tested = tested_family(screen["feature"])
        effects = screen["coefficient"].to_numpy()
        intercept = int(np.nonzero(~tested)[0][0])
        order = list(np.argsort(-np.abs(effects), kind="stable"))

        assert order.index(intercept) + 1 == 547
        assert float(abs(effects[intercept])) == pytest.approx(0.190, abs=1e-3)
        assert float(np.abs(effects[tested]).max()) == pytest.approx(4.371,
                                                                     abs=1e-3)
        assert bool(screen["q_value"].isna().iloc[intercept])

    def test_the_strongest_effect_is_not_the_strongest_hit(self, panel,
                                                            screen):
        """The whole argument for the tab, on the real fit: the third largest
        effect in the screen (-4.220) is not called (q = 0.063) while the
        largest (+4.371) is (q = 3e-05). A volcano puts those two nowhere near
        each other and never says they are the same size."""
        rows = rows_of(panel.effect_rank)
        family = screen.loc[screen["feature"] != "Intercept"].reset_index(
            drop=True)

        biggest = family.iloc[rows[0]]
        third = family.iloc[rows[2]]
        assert abs(float(third["coefficient"])) == pytest.approx(4.220,
                                                                 abs=1e-3)
        assert float(biggest["q_value"]) <= 0.05
        assert float(third["q_value"]) > 0.05

    def test_it_says_this_screen_has_no_standard_errors(self, panel, screen):
        """A real and useful finding rather than an empty picture: spaCR's own
        results writer emits no std_err, so every dot here is a point estimate
        and the panel says so instead of implying exactness."""
        assert not [c for c in screen.columns if "err" in c.lower()]

        assert intervals_of(panel.effect_rank) == {}
        assert "no standard error" in panel.effect_rank._status.text()

    def test_the_called_count_is_the_screens_own(self, panel, screen):
        called = int((screen["q_value"] <= 0.05).sum())
        assert called == 54

        assert f"{called} called at q_value ≤ 0.05" in \
            panel.effect_rank._status.text()

    def test_the_distribution_reports_the_screens_sigma(self, panel, screen):
        from spacr.hits import tested_family

        values = screen.loc[tested_family(screen["feature"]),
                            "coefficient"].to_numpy()
        sigma = float(np.median(np.abs(values - np.median(values))) * 1.4826)
        assert sigma == pytest.approx(0.229228, abs=1e-6)

        assert f"σ (MAD) = {sigma:.3g}" in \
            panel.effect_distribution._status.text()
