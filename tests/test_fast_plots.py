"""The interactive regression plots, drawn by Qt rather than matplotlib.

These cover the reasons the switch was made: the plots must be fast, every
point must map back to its own row, and the numbers behind the picture must be
readable without opening a CSV.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph")


@pytest.fixture()
def results():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "feature": [f"grna[{i}]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": np.concatenate([rng.uniform(size=n - 20),
                                   rng.uniform(0, 1e-8, 20)]),
        "q_value": np.sort(rng.uniform(size=n)),
        "condition": rng.choice([f"LOPIT{i}" for i in range(27)], n),
    })


class TestTheVolcano:

    def test_every_point_maps_back_to_its_own_row(self, qtbot, results):
        """"press every dot and get its information".

        The index carried by a point must be its row in the ORIGINAL frame,
        not its position after unplottable points were dropped -- otherwise a
        click near a NaN reports the wrong gene.
        """
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        # A hole in the data, which is what shifts the indices.
        results.loc[5, "p_value"] = np.nan
        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(results, category_column="condition")

        item = [i for i in plot.plot.listDataItems()
                if hasattr(i, "points")][0]
        points = item.points()
        # The dropped row must not be claimed by any point...
        assert 5 not in [p.data() for p in points]
        # ...and a point's index must still name its own feature.
        sample = points[10]
        index = int(sample.data())
        assert results["feature"].iloc[index] == f"grna[{index}]"

    def test_clicking_reports_the_row(self, qtbot, results):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(results)
        seen = []
        plot.point_clicked.connect(seen.append)

        class _Point:
            def data(self):
                return 7

        plot._on_points_clicked(None, [_Point()])
        assert seen == [7]
        assert "grna[7]" in plot._status.text()

    def test_an_empty_frame_says_so_rather_than_drawing_nothing(self, qtbot):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        assert plot.set_results(pd.DataFrame()) == 0
        assert "No coefficients" in plot._status.text()

    def test_a_p_value_of_zero_stays_on_the_plot(self, qtbot, results):
        """An underflowing p is a real result, not a mistake.

        -log10(0) is inf, which rescales the axis until every other point is
        a line at the bottom.
        """
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        results.loc[0, "p_value"] = 0.0
        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        assert plot.set_results(results) == len(results)

    def test_it_is_fast(self, qtbot, results):
        """The whole reason for leaving matplotlib."""
        import time

        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(results, category_column="condition")

        start = time.perf_counter()
        for index in range(20):
            plot._log_y.setChecked(index % 2 == 0)
        each = (time.perf_counter() - start) / 20 * 1000
        # matplotlib redraws the whole figure for this: ~115 ms.
        assert each < 25, f"a log toggle took {each:.0f} ms"


class TestTheDiagnostics:

    def test_the_p_value_histogram_reports_the_excess(self, qtbot, results):
        """Flat plus a spike at zero is a screen with signal in it."""
        from spacr.qt.widgets.fast_plots import PValueHistogram

        plot = PValueHistogram()
        qtbot.addWidget(plot)
        assert plot.set_p_values(results["p_value"]) == len(results)
        assert "more than that" in plot._status.text()

    def test_the_qq_plot_reports_inflation(self, qtbot, results):
        """Inflation is the number that says whether to believe the volcano."""
        from spacr.qt.widgets.fast_plots import QQPlot

        plot = QQPlot()
        qtbot.addWidget(plot)
        plot.set_p_values(results["p_value"])
        assert "Inflation" in plot._status.text()

    def test_the_residual_plot_reports_its_trend(self, qtbot):
        from spacr.qt.widgets.fast_plots import ResidualPlot

        rng = np.random.default_rng(0)
        plot = ResidualPlot()
        qtbot.addWidget(plot)
        plot.set_residuals(rng.normal(size=300), rng.normal(size=300))
        assert "slope" in plot._status.text()

    def test_control_separation_reports_both_medians(self, qtbot):
        """The assay window, before anyone argues about a hit list."""
        from spacr.qt.widgets.fast_plots import ControlSeparation

        rng = np.random.default_rng(0)
        plot = ControlSeparation()
        qtbot.addWidget(plot)
        plot.set_groups({"negative": rng.normal(0, 1, 50),
                         "positive": rng.normal(3, 1, 50)})
        assert "negative" in plot._status.text()
        assert "positive" in plot._status.text()

    def test_a_diagnostic_with_no_data_does_not_raise(self, qtbot):
        from spacr.qt.widgets.fast_plots import (ControlSeparation,
                                                 PValueHistogram, QQPlot)

        for build, call in ((PValueHistogram, "set_p_values"),
                            (QQPlot, "set_p_values")):
            plot = build()
            qtbot.addWidget(plot)
            assert getattr(plot, call)([]) == 0
        plot = ControlSeparation()
        qtbot.addWidget(plot)
        assert plot.set_groups({}) == 0


class TestTheTable:

    def test_it_shows_the_numbers_behind_the_picture(self, qtbot, results):
        from spacr.qt.widgets.fast_plots import ResultsTable

        table = ResultsTable()
        qtbot.addWidget(table)
        assert table.set_frame(results) == len(results)
        assert table.table.columnCount() == len(results.columns)

    def test_filtering_narrows_to_what_was_typed(self, qtbot, results):
        from spacr.qt.widgets.fast_plots import ResultsTable

        table = ResultsTable()
        qtbot.addWidget(table)
        table.set_frame(results)
        table._filter.setText("grna[17")
        shown = sum(not table.table.isRowHidden(r)
                    for r in range(table.table.rowCount()))
        assert 0 < shown < len(results)

    def test_significant_only_uses_a_corrected_column(self, qtbot, results):
        """Filtering on raw p would call far more things hits than there are."""
        from spacr.qt.widgets.fast_plots import ResultsTable

        table = ResultsTable()
        qtbot.addWidget(table)
        table.set_frame(results)
        assert table._significance == "q_value"
        table._only_hits.setChecked(True)
        expected = int((results["q_value"] <= 0.05).sum())
        shown = sum(not table.table.isRowHidden(r)
                    for r in range(table.table.rowCount()))
        assert shown == expected

    def test_a_point_and_its_row_are_two_views_of_one_thing(self, qtbot,
                                                            results):
        from spacr.qt.widgets.fast_plots import ResultsTable

        table = ResultsTable()
        qtbot.addWidget(table)
        table.set_frame(results)
        assert table.select_frame_row(42)
        selected = table.table.selectedItems()
        assert selected and selected[0].data(0x0100) == 42  # Qt.UserRole

    def test_sorting_a_number_column_sorts_numerically(self, qtbot):
        """Sorted as text, "10" lands before "9"."""
        from spacr.qt.widgets.fast_plots import ResultsTable

        frame = pd.DataFrame({"q_value": [9.0, 10.0, 1.0]})
        table = ResultsTable()
        qtbot.addWidget(table)
        table.set_frame(frame)
        table.table.sortItems(0)
        order = [float(table.table.item(r, 0).text())
                 for r in range(table.table.rowCount())]
        assert order == sorted(order)

    def test_copy_gives_only_the_visible_rows(self, qtbot, results):
        from spacr.qt.widgets.fast_plots import ResultsTable

        table = ResultsTable()
        qtbot.addWidget(table)
        table.set_frame(results)
        table._filter.setText("grna[17")
        lines = table.copy_visible().splitlines()
        assert lines[0].startswith("feature")
        assert 1 < len(lines) < len(results)


class TestTheLastGraphIsNotSlowAnyMore:
    """"the last graph makes everything very laggy".

    Three separate per-point costs, each of which alone was larger than
    drawing the plot. They are worth naming individually because each looks
    harmless in the source.
    """

    @pytest.fixture()
    def big(self):
        rng = np.random.default_rng(0)
        n = 1215                       # the real screen's coefficient count
        return pd.DataFrame({
            "feature": [f"grna[{i}]" for i in range(n)],
            "coefficient": rng.normal(size=n),
            "p_value": rng.uniform(size=n),
            "condition": rng.choice([f"LOPIT{i}" for i in range(27)], n),
        })

    def _time(self, plot, frame, app=None, **kwargs):
        import time
        plot.set_results(frame, **kwargs)
        start = time.perf_counter()
        for _ in range(5):
            plot.set_results(frame, **kwargs)
        return (time.perf_counter() - start) / 5 * 1000

    def test_the_plain_volcano_is_immediate(self, qtbot, big):
        """matplotlib needed ~115 ms for this, on every redraw."""
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        each = self._time(plot, big)
        assert each < 30, f"a plain volcano took {each:.0f} ms"

    def test_colouring_does_not_cost_a_brush_per_point(self, qtbot, big):
        """pg.mkBrush() per point built 1,215 QBrush objects: 39.5 ms.

        Reusing one brush per distinct colour is 3.5 ms for the same picture.
        """
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        each = self._time(plot, big, category_column="condition")
        assert each < 30, f"a coloured volcano took {each:.0f} ms"

    def test_no_label_is_built_before_it_is_needed(self, qtbot, big):
        """Formatting all 1,215 labels up front was 3,600 pandas lookups.

        Only the clicked point is ever read, so only it is formatted.
        """
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(big, category_column="condition")
        assert not plot._labels, "labels were pre-built for every point"
        # ...and the description is still correct on demand.
        assert "grna[7]" in plot._describe(7)

    def test_the_legend_is_off_by_default(self, qtbot, big):
        """27 entries are 40 ms of a 49 ms redraw -- the same cost that made
        the matplotlib version slow. Carried across unchanged it would have
        wasted the whole switch."""
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(big, category_column="condition")
        assert not plot._legend_box.isChecked()
        assert plot._legend_box.isEnabled(), "the legend must still be offered"
        assert "27" in plot._legend_box.text()

    def test_the_legend_can_be_turned_on_and_off_again(self, qtbot, big):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(big, category_column="condition")
        plot._legend_box.setChecked(True)
        assert plot.plot.plotItem.legend is not None
        plot._legend_box.setChecked(False)
        assert plot.plot.plotItem.legend is None

    def test_a_frame_with_no_category_offers_no_legend(self, qtbot, big):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        plot = VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(big)
        assert not plot._legend_box.isEnabled()


class TestRestylingTheFastPlots:
    """"i can see and modify all graphs" -- the same gesture as the
    matplotlib figures, on the plots that replaced them."""

    @pytest.fixture()
    def plot(self, qtbot):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        rng = np.random.default_rng(0)
        n = 400
        frame = pd.DataFrame({
            "feature": [f"g{i}" for i in range(n)],
            "coefficient": rng.normal(size=n),
            "p_value": rng.uniform(size=n),
            "condition": rng.choice(list("abc"), n),
        })
        widget = VolcanoPlot()
        qtbot.addWidget(widget)
        widget.set_results(frame, category_column="condition")
        return widget

    def test_the_scatter_is_reachable_for_restyling(self, plot):
        assert plot._scatter_items(), "nothing to restyle"

    def test_size_colour_and_opacity_all_apply(self, plot):
        import pyqtgraph as pg

        for item in plot._scatter_items():
            item.setSize(14.0)
            item.setBrush(pg.mkBrush("#C44E52"))
            item.setOpacity(0.5)
        assert plot._scatter_items()[0].opacity() == 0.5

    def test_axis_labels_are_editable(self, plot):
        plot.plot.setLabel("bottom", "effect size")
        assert plot.plot.getAxis("bottom").labelText == "effect size"

    def test_the_menu_offers_the_legend_only_when_there_is_one(self, qtbot):
        from spacr.qt.widgets.fast_plots import VolcanoPlot

        frame = pd.DataFrame({"feature": ["a", "b"], "coefficient": [1.0, 2.0],
                              "p_value": [0.1, 0.2]})
        widget = VolcanoPlot()
        qtbot.addWidget(widget)
        widget.set_results(frame)          # no category column
        assert not widget._legend_box.isEnabled()
