"""Every point that is a gene or a guide can be pressed, on every plot.

Instruction 124 section F, quoting the maintainer:

    "id like to be able to presson the datapoints of all graphs where data is
     represented as genes and grnas, e.g. like the Q-Q plots"

Only the volcano hit-tested. A Q-Q point IS a coefficient; so is a dot in the
control panel and a gene in the guide-agreement plot, and a histogram bar is a
hundred of them.

THE TRAP THIS FILE EXISTS FOR IS THE SAME ONE
test_row_to_point_is_joined_on_the_key.py DOCUMENTS, ONE STEP WORSE.

There, the table was sorted and the plot was not. Here the PLOT itself is
reordered before it is drawn:

    Q-Q          sorted by p-value, so drawn point n is the nth smallest p
    controls     split into groups, so the negatives are all drawn first
    agreement    one point per gene, from a frame that has one row per guide

so a point's position on the plot is not its row in the table, is not its
position in the array the plot was handed, and in the agreement plot is not
even a row of the same frame. A positional join lights up a real, wrong guide
in every one of them -- silently, and in the direction nobody questions,
because something did light up.

Every test below therefore first asserts that the position join and the key
join give DIFFERENT answers. If a future change makes the fixture stop
distinguishing them, that assertion fails and says so, rather than letting the
rest of the file pass for the wrong reason.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

#: The real screen, when this is the machine that has it. Every number the
#: fixtures below claim about shape was measured here first.
REAL_RESULTS = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/"
                "claude/results/ols/results.csv")


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def results():
    """A screen shaped like the real one, in an order that breaks positions.

    Four guides per gene plus a gene-level term for each, an Intercept, some
    controls, and -- deliberately -- p-values that do NOT ascend with the row
    number, so sorting the Q-Q genuinely reorders the frame.
    """
    rng = np.random.default_rng(11)
    genes = [f"{200000 + i * 10}" for i in range(40)]
    rows = []
    for index, gene in enumerate(genes):
        guides = 1 if index % 7 == 0 else (2 + index % 3)
        for guide in range(guides):
            rows.append({
                "feature": f"fraction:grna[{gene}_{guide}]",
                "coefficient": float(rng.normal()),
                "p_value": float(rng.uniform()),
                "gene": gene,
                "grna": f"{gene}_{guide}",
                "condition": "other",
            })
        rows.append({
            "feature": f"gene_fraction:gene[{gene}]",
            "coefficient": float(rng.normal()),
            "p_value": float(rng.uniform()),
            "gene": gene,
            "grna": None,
            "condition": "other",
        })
    # Controls, in the middle of the table rather than at either end: the
    # control panel draws them first whatever their rows are, which is the
    # reordering that breaks a positional join there.
    for index in range(12):
        rows.insert(30 + index, {
            "feature": f"fraction:grna[000000_{index}]",
            "coefficient": float(rng.normal(-2 if index % 2 else 2, 0.3)),
            "p_value": float(rng.uniform(0, 0.01)),
            "gene": "000000",
            "grna": f"000000_{index}",
            "condition": "nc" if index % 2 else "pc",
        })
    # The control block gets a gene-level term of its own, as it does on the
    # real screen -- and it is therefore one "gene" carrying twelve guides
    # while every other gene carries one to four, which is what stretches the
    # agreement plot's x-axis.
    rows.append({
        "feature": "gene_fraction:gene[000000]", "coefficient": -0.06,
        "p_value": 0.84, "gene": "000000", "grna": None, "condition": "other"})
    frame = pd.DataFrame(rows)
    # The nuisance term: in the table, off the volcano, and -- because it is
    # the smallest p in the screen -- the FIRST point on the Q-Q.
    intercept = pd.DataFrame([{
        "feature": "Intercept", "coefficient": 0.19, "p_value": 3.1e-46,
        "gene": None, "grna": None, "condition": "other"}])
    frame = pd.concat([intercept, frame], ignore_index=True)
    # A coefficient with no usable p-value, which is drawn on no plot at all
    # and must not have its row claimed by the point next to it.
    frame.loc[frame.index[4], "p_value"] = np.nan
    return frame


@pytest.fixture()
def panel(qtbot, results):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    widget.set_frame(results, source="results.csv")
    # Sort the table too, so neither end of the link is in frame order.
    widget.table.table.sortItems(1)
    return widget


def points_of(plot):
    """Every drawn point on ``plot``, with the frame row each carries."""
    found = []
    for item in plot.plot.listDataItems():
        if item is plot._highlight or not hasattr(item, "points"):
            continue
        found.extend(item.points())
    return found


def rows_of(plot):
    return [int(point.data()) for point in points_of(plot)]


def selected_feature(panel):
    items = panel.table.table.selectedItems()
    if not items:
        return None
    column = list(panel._frame.columns).index("feature")
    return panel.table.table.item(items[0].row(), column).text()


# --------------------------------------------------------------------------- #
#  The Q-Q, which is the one the maintainer named
# --------------------------------------------------------------------------- #

class TestTheQQPlot:

    def test_a_position_join_would_pick_a_different_coefficient(self, panel):
        """Proves the rest of this class is testing something.

        The Q-Q is sorted by p, so drawn point n carries the row of the nth
        smallest p-value. If that ever equals n the fixture has stopped
        distinguishing the two joins and every test below passes for free.
        """
        rows = rows_of(panel.qq)
        assert rows, "the Q-Q drew nothing"
        assert rows[:20] != list(range(20)), (
            "the fixture no longer reorders the Q-Q -- change the p-values")

    def test_the_nth_drawn_point_is_the_nth_smallest_p_value(self, panel,
                                                             results):
        """The sort, stated as the rows it produces."""
        rows = rows_of(panel.qq)
        usable = results["p_value"].notna() & (results["p_value"] > 0)
        expected = list(results.loc[usable, "p_value"]
                        .sort_values(kind="stable").index)
        assert rows == expected

    def test_a_coefficient_with_no_p_value_is_on_no_point(self, panel,
                                                          results):
        """Its row must not be claimed by the point drawn next to it."""
        missing = int(results.index[results["p_value"].isna()][0])
        assert missing not in rows_of(panel.qq)

    def test_clicking_a_point_selects_that_guides_row(self, panel):
        """The feature the instruction asked for."""
        point = points_of(panel.qq)[17]
        row = int(point.data())
        wanted = panel.qq.key_for_row(row)
        # Named, not merely equal: a Q-Q that knows no keys answers None to
        # both sides of the assertion below and passes it having done nothing.
        assert wanted is not None, "the Q-Q has no identifiers at all"

        panel.qq._on_points_clicked(None, [point])

        assert selected_feature(panel) == wanted

    def test_clicking_selects_the_key_not_the_drawing_position(self, panel):
        """The same click, against what a positional join would have done.

        Point 17 on the Q-Q is not row 17 of the frame, so a plot that
        reported its own drawing index would name a real coefficient that the
        user did not click.
        """
        point = points_of(panel.qq)[17]
        by_key = panel.qq.key_for_row(int(point.data()))
        by_position = panel.qq.key_for_row(17)
        assert by_key != by_position, "fixture no longer distinguishes them"

        panel.qq._on_points_clicked(None, [point])

        assert selected_feature(panel) == by_key
        assert selected_feature(panel) != by_position

    def test_the_ring_lands_on_that_point_not_near_it(self, panel, results):
        """Present is not enough; a marker in the wrong place is the bug."""
        rows = rows_of(panel.qq)
        row = rows[25]
        key = panel.qq.key_for_row(row)

        assert panel.qq.highlight_key(key)

        n = len(rows)
        expected_x = -np.log10((25 + 1 - 0.5) / n)
        expected_y = -np.log10(float(results["p_value"].iloc[row]))
        drawn = panel.qq._highlight.getData()
        assert float(drawn[0][0]) == pytest.approx(expected_x, abs=1e-9)
        assert float(drawn[1][0]) == pytest.approx(expected_y, abs=1e-9)

    def test_selecting_a_row_rings_it_on_the_qq_as_well(self, panel):
        """The other direction: "is my hit the one lifting off the diagonal"
        is the question the Q-Q exists for, and it needs the link both ways."""
        table = panel.table.table
        table.selectRow(6)
        column = list(panel._frame.columns).index("feature")
        shown = table.item(6, column).text()

        assert panel.qq._selected_key == shown
        assert panel.qq._highlight is not None

    def test_the_inflation_figure_survives_a_click(self, panel):
        """The status line carries the number the panel exists for. Replacing
        it with the name of whatever was clicked trades the panel's content
        for a string the user can already read in the table."""
        point = points_of(panel.qq)[3]
        panel.qq._on_points_clicked(None, [point])

        text = panel.qq._status.text()
        assert "Inflation" in text
        assert panel.qq.key_for_row(int(point.data())) in text

    def test_a_plot_with_no_keys_still_draws_and_claims_nothing(self, qtbot,
                                                               results):
        """Keys are optional. A caller with no unique column gets a readable
        Q-Q that simply does not invite a click it cannot honour."""
        from spacr.qt.widgets.fast_plots import QQPlot

        plot = QQPlot()
        qtbot.addWidget(plot)
        assert plot.set_p_values(results["p_value"]) > 0
        assert plot.key_for_row(0) is None
        assert "Click a point" not in plot._status.text()


# --------------------------------------------------------------------------- #
#  The control panel, reordered by grouping rather than by sorting
# --------------------------------------------------------------------------- #

class TestTheControlPanel:

    def test_a_dots_own_index_is_not_a_row_of_anybody_elses_table(self, panel,
                                                                 results):
        """The panel lays its groups out in one flat sequence of its own, so a
        dot's index there is not the frame row -- which is precisely why the
        index is private and the key is the contract. Reading the index as a
        row selects a real, wrong guide."""
        point = points_of(panel.controls)[2]
        index = int(point.data())
        by_key = panel.controls.key_for_row(index)
        by_position = str(results["feature"].iloc[index])
        assert by_key is not None, "the control panel has no identifiers at all"
        assert by_key != by_position, (
            "the fixture no longer distinguishes the two joins")

    def test_the_negatives_are_drawn_first_whatever_their_rows(self, panel,
                                                              results):
        """The grouping IS the reordering, and it is not a sort: the negative
        controls sit in the middle of the table and at the left of the plot."""
        negatives = results.loc[results["condition"] == "nc", "feature"]
        assert min(negatives.index) > 0, "the fixture put the controls at the top"

        keys = [panel.controls.key_for_row(row)
                for row in rows_of(panel.controls)]
        assert keys[:len(negatives)] == list(negatives.astype(str))

    def test_clicking_a_control_dot_selects_that_coefficient(self, panel):
        point = points_of(panel.controls)[2]
        wanted = panel.controls.key_for_row(int(point.data()))
        assert wanted is not None

        panel.controls._on_points_clicked(None, [point])

        assert selected_feature(panel) == wanted

    def test_the_dot_reports_its_group_and_its_effect(self, panel, results):
        """"press every dot and get its information" -- for a control that is
        which class it is in, which is the whole content of the panel."""
        point = points_of(panel.controls)[0]
        key = panel.controls.key_for_row(int(point.data()))
        row = int(results.index[results["feature"] == key][0])

        panel.controls._on_points_clicked(None, [point])

        text = panel.controls._status.text()
        assert "negative" in text
        assert f"{float(results['coefficient'].iloc[row]):.3g}" in text

    def test_selecting_a_row_rings_its_control_dot(self, panel, results):
        row = int(results.index[results["condition"] == "pc"][0])
        key = str(results["feature"].iloc[row])

        assert panel.controls.highlight_key(key)

        x, y = panel.controls._row_xy[panel.controls._key_rows[key]]
        assert y == pytest.approx(float(results["coefficient"].iloc[row]))
        drawn = panel.controls._highlight.getData()
        assert float(drawn[0][0]) == pytest.approx(x)
        assert float(drawn[1][0]) == pytest.approx(y)

    def test_a_group_with_no_keys_reports_no_key_rather_than_a_blank_one(
            self, qtbot):
        """Unidentified rows must not all answer to the same empty string --
        that is one bogus identifier several unrelated rows share."""
        from spacr.qt.widgets.fast_plots import ControlSeparation

        plot = ControlSeparation()
        qtbot.addWidget(plot)
        plot.set_groups({"negative": np.array([1.0, 2.0]),
                         "positive": np.array([3.0, 4.0])},
                        keys={"negative": ["a", "b"]})
        assert plot.key_for_row(0) == "a"
        assert plot.key_for_row(2) is None
        assert plot.highlight_key("") is False


# --------------------------------------------------------------------------- #
#  The histogram, whose marks stand for many rows each
# --------------------------------------------------------------------------- #

class TestThePValueHistogram:

    def test_a_bar_hands_back_exactly_the_rows_it_drew(self, panel, results):
        """The count a bar draws and the rows it reports must be one number.
        Any drift between them is a bar that lies about its own height."""
        counts = panel.p_values._counts
        for index in (0, 1, 20, 49):
            rows = panel.p_values._bin_rows[index]
            assert len(rows) == int(counts[index])
            edges = panel.p_values._edges
            values = results["p_value"].to_numpy()[rows]
            assert ((values >= edges[index]) & (values <= edges[index + 1])).all()

    def test_a_bar_of_many_does_not_pretend_to_select_one(self, panel):
        """Picking the first, the strongest or the nearest would be a guess
        dressed up as an answer -- the same mistake as joining on position."""
        index = next(i for i, rows in enumerate(panel.p_values._bin_rows)
                     if len(rows) > 1)
        singles, sets = [], []
        panel.p_values.key_selected.connect(singles.append)
        panel.p_values.keys_selected.connect(sets.append)

        keys = panel.p_values.select_bin(index)

        assert len(keys) > 1
        assert singles == [], "a bar of many claimed to select one of them"
        assert sets == [keys]
        assert "not one point" in panel.p_values._status.text()

    def test_a_bar_holding_one_row_selects_it_like_any_other_point(self,
                                                                  panel):
        index = next(i for i, rows in enumerate(panel.p_values._bin_rows)
                     if len(rows) == 1)
        wanted = panel.p_values.key_for_row(int(panel.p_values._bin_rows[index][0]))

        panel.p_values.select_bin(index)

        assert selected_feature(panel) == wanted

    def test_clicking_a_bar_narrows_the_table_to_it(self, panel):
        """"Show me the hundred" is the question a bar CAN answer exactly."""
        index = next(i for i, rows in enumerate(panel.p_values._bin_rows)
                     if len(rows) > 2)
        keys = panel.p_values.select_bin(index)

        shown = [panel.table.table.item(row, 0)
                 for row in range(panel.table.table.rowCount())
                 if not panel.table.table.isRowHidden(row)]
        assert len(shown) == len(keys)
        assert "narrowed to" in panel.table._count.text()

    def test_typing_in_the_filter_box_clears_that_narrowing(self, panel):
        """Otherwise the two filters AND together and the user types a gene
        they can see in the plot, gets nothing, and cannot find out why."""
        index = next(i for i, rows in enumerate(panel.p_values._bin_rows)
                     if len(rows) > 2)
        panel.p_values.select_bin(index)
        assert panel.table._key_restriction is not None

        panel.table._filter.setText("grna")

        assert panel.table._key_restriction is None
        assert "narrowed to" not in panel.table._count.text()

    def test_a_click_lands_in_the_bar_under_the_cursor(self, panel):
        """The bin is worked out from the x coordinate, which is the only
        definition of "which bar" that stays right when the axis is zoomed."""
        edges = panel.p_values._edges
        middle = float((edges[7] + edges[8]) / 2)
        assert panel.p_values.bin_at(middle) == 7
        assert panel.p_values.bin_at(float(edges[7])) == 7
        assert panel.p_values.bin_at(-0.5) is None
        assert panel.p_values.bin_at(1.5) is None
        # 1.0 exactly belongs to the last bar, as np.histogram counted it.
        assert panel.p_values.bin_at(1.0) == len(panel.p_values._counts) - 1

    def test_a_real_mouse_press_reaches_the_bar_under_it(self, panel, qtbot):
        """Driven through Qt rather than by calling the handler.

        A bar has no ``sigClicked`` to connect to, so this path is scene
        plumbing -- mapping a viewport pixel back to a data coordinate -- and
        calling ``select_bin`` directly would test everything except the part
        that is easy to get wrong.
        """
        from PySide6.QtCore import QEvent, QPointF, Qt as QtCore_Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication

        plot = panel.p_values
        plot.resize(700, 500)
        plot.show()
        qtbot.waitExposed(plot)
        got = []
        plot.keys_selected.connect(got.append)

        edges = plot._edges
        middle = float((edges[0] + edges[1]) / 2)
        height = float(plot._counts[0]) / 2 or 1.0
        viewbox = plot.plot.plotItem.vb
        where = plot.plot.mapFromScene(
            viewbox.mapViewToScene(QPointF(middle, height)))
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            QApplication.sendEvent(plot.plot.viewport(), QMouseEvent(
                kind, where, QtCore_Qt.LeftButton, QtCore_Qt.LeftButton,
                QtCore_Qt.NoModifier))

        assert got, "a real click on the first bar reached nothing"
        assert set(got[0]) == set(plot.keys_in_bin(0))

    def test_a_right_press_is_the_style_menu_and_selects_nothing(self, panel,
                                                                 qtbot):
        """Right-click already means "restyle this plot" on every plot here.
        Making it select as well would fire a selection behind the menu."""
        from PySide6.QtCore import QEvent, QPointF, Qt as QtCore_Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication

        plot = panel.p_values
        plot.resize(700, 500)
        plot.show()
        qtbot.waitExposed(plot)
        got = []
        plot.keys_selected.connect(got.append)

        viewbox = plot.plot.plotItem.vb
        where = plot.plot.mapFromScene(
            viewbox.mapViewToScene(QPointF(float(plot._edges[1]) / 2, 1.0)))
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            QApplication.sendEvent(plot.plot.viewport(), QMouseEvent(
                kind, where, QtCore_Qt.RightButton, QtCore_Qt.RightButton,
                QtCore_Qt.NoModifier))

        assert got == []

    def test_a_selected_row_marks_the_bar_it_falls_in(self, panel, results):
        """Not a ring floating over the bars: this plot never drew that row as
        a mark of its own, and the bar is where the coefficient actually is."""
        row = 12
        key = str(results["feature"].iloc[row])
        assert panel.p_values.highlight_key(key)

        edges = panel.p_values._edges
        outlined = panel.p_values._highlight.opts
        low = float(outlined["x0"][0])
        p = float(results["p_value"].iloc[row])
        assert low <= p <= low + float(edges[1] - edges[0])


# --------------------------------------------------------------------------- #
#  The guide-agreement plot, whose points are genes rather than guides
# --------------------------------------------------------------------------- #

class TestTheGuideAgreementPlot:

    def test_it_draws_one_point_per_gene(self, panel, results):
        from spacr.guide_concordance import guide_support

        expected = len(guide_support(results))
        assert len(points_of(panel.agreement)) == expected

    def test_a_gene_joins_on_its_gene_level_term_not_its_bare_id(self, panel):
        """The support table is indexed by ``244480``; every other view here
        joins on ``gene_fraction:gene[244480]``. A bare id would be a second
        key space that nothing else can resolve."""
        keys = [key for key in panel.agreement._keys if key]
        assert keys, "the agreement plot has no keys"
        assert all(key.startswith("gene_fraction:gene[") for key in keys)

    def test_clicking_a_gene_selects_its_row_in_the_coefficient_table(self,
                                                                     panel):
        point = points_of(panel.agreement)[9]
        wanted = panel.agreement.key_for_row(int(point.data()))

        panel.agreement._on_points_clicked(None, [point])

        assert selected_feature(panel) == wanted

    def test_a_position_join_would_pick_a_different_gene(self, panel,
                                                        results):
        """The agreement plot is one row per GENE from a frame with one row
        per guide, so its row numbers are not the table's at all."""
        point = points_of(panel.agreement)[9]
        by_key = panel.agreement.key_for_row(int(point.data()))
        by_frame_position = str(results["feature"].iloc[int(point.data())])
        assert by_key != by_frame_position

    def test_the_ring_lands_on_the_jittered_dot_the_user_sees(self, panel):
        """The x is jittered, so a marker placed on the un-jittered lattice
        point would sit beside the dot that was clicked."""
        point = points_of(panel.agreement)[4]
        row = int(point.data())
        key = panel.agreement.key_for_row(row)
        assert panel.agreement.highlight_key(key)

        drawn = panel.agreement._highlight.getData()
        assert float(drawn[0][0]) == pytest.approx(point.pos().x(), abs=1e-9)
        assert float(drawn[1][0]) == pytest.approx(point.pos().y(), abs=1e-9)
        assert float(drawn[0][0]) != pytest.approx(round(float(drawn[0][0])))

    def test_only_the_single_guide_genes_are_coloured(self, panel, results):
        """The house rule: everything grey except what the sentence is about,
        and the sentence is "these ones rest on a single guide"."""
        from spacr.qt.widgets.fast_plots import GuideAgreementPlot
        from spacr.guide_concordance import guide_support

        support = guide_support(results)
        alone = int(support["single_guide"].sum())
        assert 0 < alone < len(support), "fixture has no contrast to show"

        coloured = 0
        for point in points_of(panel.agreement):
            name = point.brush().color().name()
            coloured += int(name.lower() == GuideAgreementPlot.SINGLE.lower())
        assert coloured == alone

    def test_it_says_which_genes_rest_on_one_guide(self, panel, results):
        from spacr.guide_concordance import guide_support

        alone = int(guide_support(results)["single_guide"].sum())
        assert f"{alone} rest on a single guide" in panel.agreement._status.text()

    def test_one_over_represented_gene_does_not_own_the_axis(self, panel,
                                                             results):
        """The control block parses as a single gene carrying all its guides,
        and on autorange that one point stretches the x-axis several times
        wider than the data -- the same failure the intercept caused on the
        volcano. The opening view covers the library; the outlier is still
        drawn, still clickable, and said out loud."""
        from spacr.guide_concordance import guide_support

        counts = guide_support(results)["n_guides"]
        assert counts.max() > 3 * counts.median(), (
            "the fixture has no over-represented gene to bound")

        low, high = panel.agreement.plot.viewRange()[0]
        assert high < counts.max(), "the opening view is stretched by one gene"
        assert high >= counts[counts < counts.max()].max()
        assert "beyond the opening view" in panel.agreement._status.text()

    def test_the_gene_beyond_the_view_is_still_drawn_and_still_clickable(
            self, panel, results):
        """Left out of the OPENING VIEW, not left out. Dropping it would lose
        a real gene, which is not the same argument as the volcano's."""
        from spacr.guide_concordance import guide_support

        support = guide_support(results)
        biggest = str(support["n_guides"].idxmax())
        term = f"gene_fraction:gene[{biggest}]"

        assert term in panel.agreement._keys
        assert panel.agreement.highlight_key(term)

    def test_a_gene_with_no_gene_level_term_has_no_key_rather_than_a_guess(
            self, qtbot, results):
        """A fit that never fitted a gene-level term leaves those genes with
        nothing to join on. Naming them by their bare id would build a key
        nothing else can resolve; saying None is the truthful answer."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        guides_only = results[~results["feature"].astype(str)
                              .str.startswith("gene_fraction:gene")]
        widget = RegressionResultsPanel()
        qtbot.addWidget(widget)
        widget.set_frame(guides_only.reset_index(drop=True), source="x.csv")

        assert len(points_of(widget.agreement)) > 0, "the genes still plot"
        assert not any(widget.agreement._keys)

    def test_an_empty_support_table_says_so_rather_than_drawing_nothing(
            self, qtbot):
        from spacr.qt.widgets.fast_plots import GuideAgreementPlot

        plot = GuideAgreementPlot()
        qtbot.addWidget(plot)
        assert plot.set_support(None) == 0
        assert "guide support is unknown" in plot._status.text()


# --------------------------------------------------------------------------- #
#  What a redraw and a new run must not do
# --------------------------------------------------------------------------- #

class TestTheLinkSurvivesTheThingsThatBreakIt:

    def test_a_redraw_does_not_leave_a_marker_at_the_old_coordinates(
            self, qtbot):
        """``plot.clear()`` takes the artists and leaves the dictionary that
        pointed at them, so a redrawn plot would ring where a point USED to
        be -- a marker in the wrong place, which is the whole failure mode."""
        from spacr.qt.widgets.fast_plots import QQPlot

        plot = QQPlot()
        qtbot.addWidget(plot)
        keys = [f"g{i}" for i in range(6)]
        plot.set_p_values([0.5, 0.4, 0.3, 0.2, 0.1, 0.05], keys=keys)
        plot.highlight_key("g0")
        first = plot._highlight.getData()[1][0]

        plot.set_p_values([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], keys=keys)

        assert plot.highlight_key("g0")
        assert plot._highlight.getData()[1][0] != pytest.approx(first)

    def test_a_new_run_does_not_inherit_the_old_keys(self, qtbot):
        """A key list left over from the previous table names rows that do not
        exist in this one, and would join the two experiments together."""
        from spacr.qt.widgets.fast_plots import QQPlot

        plot = QQPlot()
        qtbot.addWidget(plot)
        plot.set_p_values([0.1, 0.2, 0.3], keys=["a", "b", "c"])
        assert plot.highlight_key("a")

        plot.set_p_values([0.1, 0.2, 0.3])

        assert plot.highlight_key("a") is False
        assert plot.key_for_row(0) is None

    def test_a_missing_key_is_none_not_the_string_nan(self, qtbot):
        """A frame carries its blanks as float NaN, and str() turns every one
        of them into the same four characters."""
        from spacr.qt.widgets.fast_plots import QQPlot

        plot = QQPlot()
        qtbot.addWidget(plot)
        plot.set_p_values([0.1, 0.2, 0.3],
                          keys=pd.Series(["a", np.nan, "c"]))
        assert plot.key_for_row(1) is None
        assert plot.highlight_key("nan") is False

    def test_a_reloaded_run_drops_the_ring_on_EVERY_plot(self, panel,
                                                          results):
        """Each plot re-marks its selected key at the end of its own draw, so
        that a restyle does not lose the user's place. The other edge of that:
        a plot whose selection is not cleared on load cheerfully re-rings the
        NEW run at the OLD key -- a mark on a point that means something else
        now. Only the volcano was being cleared."""
        panel.table.table.selectRow(5)
        marked = [name for name in ("volcano", "qq", "controls", "agreement",
                                    "p_values")
                  if getattr(panel, name)._highlight is not None]
        assert len(marked) > 1, "nothing was marked, so nothing is being tested"

        panel.set_frame(results.iloc[:120].copy(), source="other.csv")

        assert panel._selected_key is None
        still = [name for name in ("volcano", "qq", "controls", "agreement",
                                   "p_values")
                 if getattr(panel, name)._highlight is not None]
        assert still == [], f"{still} kept a ring from the previous run"

    def test_a_reloaded_run_drops_a_set_chosen_on_the_old_one(self, panel,
                                                             results):
        """A set of keys chosen off the last table names nothing in this one,
        and leaving it on would hide every row."""
        index = next(i for i, rows in enumerate(panel.p_values._bin_rows)
                     if len(rows) > 2)
        panel.p_values.select_bin(index)
        assert panel.table._key_restriction is not None

        panel.set_frame(results.iloc[:80].copy(), source="other.csv")

        assert panel.table._key_restriction is None
        shown = sum(not panel.table.table.isRowHidden(row)
                    for row in range(panel.table.table.rowCount()))
        assert shown == 80


# --------------------------------------------------------------------------- #
#  Against the screen itself
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists(REAL_RESULTS),
                    reason="the TSG101 screen is not on this machine")
class TestAgainstTheRealScreen:
    """Synthetic frames prove the join; the screen proves the shape.

    1,213 coefficients, 1,213 distinct features, 389 genes of which 102 rest
    on a single guide, and one Intercept whose p of 3.15e-46 makes it the
    first point on the Q-Q.
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

    def test_the_first_qq_point_is_the_intercept_and_it_says_so(self, panel):
        """The strongest p in the screen belongs to a nuisance term that the
        volcano refuses to plot. Clicking it must name it and admit the
        volcano has no point for it -- not ring the nearest thing it has."""
        first = points_of(panel.qq)[0]     # smallest p, top right of the plot
        assert panel.qq.key_for_row(int(first.data())) == "Intercept"

        panel.qq._on_points_clicked(None, [first])

        assert "not on this plot" in panel.volcano._status.text()

    def test_every_qq_point_names_the_guide_its_p_value_came_from(self,
                                                                 panel,
                                                                 screen):
        """The join, checked on all 1,213 of them rather than on a sample."""
        rows = rows_of(panel.qq)
        drawn = [item for item in panel.qq.plot.listDataItems()
                 if hasattr(item, "points")][0]
        observed = np.asarray([point.pos().y() for point in drawn.points()])
        expected = -np.log10(screen["p_value"].to_numpy()[rows])
        assert np.allclose(observed, expected, atol=1e-9)

        keys = [panel.qq.key_for_row(row) for row in rows]
        assert keys == list(screen["feature"].to_numpy()[rows])

    def test_the_hardest_hit_in_the_screen_round_trips(self, panel, screen):
        """244480: gene p 2.9e-13, one surviving guide, top of the list. It is
        the gene the guide-support panel was built to catch, so it is the one
        worth being able to click on every plot that draws it."""
        term = "gene_fraction:gene[244480]"
        assert term in set(screen["feature"])

        assert panel.table.select_key(term)

        assert panel.qq._selected_key == term
        assert panel.qq._highlight is not None
        assert panel.agreement._highlight is not None
        assert panel.volcano._highlight is not None

    def test_the_controls_are_clickable_and_are_where_the_table_says(
            self, panel, screen):
        """24 control guides, 3 positive controls, 1,186 screen guides."""
        rows = rows_of(panel.controls)
        assert len(rows) == int(screen["condition"].isin(
            ["nc", "pc", "control", "other"]).sum())
        # The first three dots are the positive controls, which sit at rows
        # 277, 278 and 951 of the table -- scattered through it, and nowhere
        # near the front. The panel's own ordering is not the table's.
        positives = screen.loc[screen["condition"] == "pc", "feature"]
        assert list(positives.index) != list(range(len(positives)))
        for offset, feature in enumerate(positives.astype(str)):
            key = panel.controls.key_for_row(rows[offset])
            assert key == feature
            _, y = panel.controls._row_xy[rows[offset]]
            row = int(screen.index[screen["feature"] == feature][0])
            assert y == pytest.approx(float(screen["coefficient"].iloc[row]))

    def test_the_agreement_plot_finds_the_102_single_guide_genes(self, panel,
                                                                 screen):
        from spacr.guide_concordance import guide_support

        support = guide_support(screen)
        assert len(support) == 389
        alone = int(support["single_guide"].sum())
        assert alone == 102
        assert f"{alone} rest on a single guide" in \
            panel.agreement._status.text()

    def test_the_first_histogram_bar_hands_back_its_whole_bin(self, panel,
                                                              screen):
        keys = panel.p_values.select_bin(0)
        edges = panel.p_values._edges
        expected = screen.loc[screen["p_value"] <= edges[1], "feature"]
        assert len(keys) == len(expected)
        assert set(keys) == set(expected.astype(str))
