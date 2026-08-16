"""The panel a finished regression opens into.

Covers the thing that was missing: the volcano and the numbers behind it are
two views of one table, and both are fast.
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
    n = 1215
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(size=n),
        "q_value": np.sort(rng.uniform(size=n)),
        "condition": rng.choice(["nc", "pc", "control", "other"], n,
                                p=[0.05, 0.05, 0.1, 0.8]),
    })


class TestItOpensIntoTheResults:

    def test_every_view_is_populated(self, qtbot, results):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        assert panel.set_frame(results, source="results.csv")

        assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == \
            ["Volcano", "p-values", "Q-Q", "Controls", "Guide support"]
        assert panel.table.table.rowCount() == len(results)
        assert "Inflation" in panel.qq._status.text()
        assert "negative" in panel.controls._status.text()
        assert "positive" in panel.controls._status.text()

    def test_a_dot_and_its_row_are_linked_both_ways(self, qtbot, results):
        """The point of putting them beside each other.

        Joined on the KEY. The link used to carry a position, which is only
        meaningful to the frame it came from -- see the sorting test below.
        """
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        panel.set_frame(results)

        wanted = results["feature"].iloc[42]
        panel.volcano.key_selected.emit(wanted)
        selected = panel.table.table.selectedItems()
        assert selected, "clicking a point selected no row"
        row = selected[0].row()
        assert panel.table.table.item(row, 0).text() == wanted

        other = results["feature"].iloc[7]
        panel.table.key_selected.emit(other)
        assert panel.volcano._selected_key == other
        assert other in panel.volcano._status.text()

    def test_it_is_fast_enough_to_recolour_interactively(self, qtbot, results):
        """The lag that started all of this was on exactly this redraw."""
        import time

        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        panel.set_frame(results)

        start = time.perf_counter()
        for _ in range(5):
            panel._redraw_volcano()
        each = (time.perf_counter() - start) / 5 * 1000
        assert each < 30, f"the volcano took {each:.0f} ms (matplotlib: 115)"

    def test_only_plausible_categories_are_offered_for_colour(self, qtbot,
                                                              results):
        """A column with one value per row is not a category."""
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        panel.set_frame(results)
        offered = [panel._colour_by.itemData(i)
                   for i in range(panel._colour_by.count())]
        assert "condition" in offered
        assert "feature" not in offered, \
            "a unique-per-row column was offered as a category"

    def test_an_empty_table_says_so(self, qtbot):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        assert not panel.set_frame(pd.DataFrame())
        assert "empty" in panel._source.text().lower()


class TestFindingTheResultsOnDisk:

    def test_a_folder_a_parent_or_the_file_itself(self, tmp_path, results):
        """The three things a user has to hand when reopening a run."""
        from spacr.qt.widgets.regression_results import find_results_table

        run = tmp_path / "results" / "plate1_dv" / "ols" / "list"
        run.mkdir(parents=True)
        target = run / "results.csv"
        results.to_csv(target, index=False)

        assert find_results_table(str(target)) == str(target)
        assert find_results_table(str(run)) == str(target)
        assert find_results_table(str(tmp_path)) == str(target)

    def test_nothing_there_is_not_an_error(self, tmp_path):
        from spacr.qt.widgets.regression_results import find_results_table

        assert find_results_table(str(tmp_path)) is None
        assert find_results_table(None) is None
        assert find_results_table("/does/not/exist") is None

    def test_loading_from_disk_populates_the_panel(self, qtbot, tmp_path,
                                                   results):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        run = tmp_path / "results" / "x" / "ols" / "list"
        run.mkdir(parents=True)
        results.to_csv(run / "results.csv", index=False)

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        assert panel.load(str(tmp_path))
        assert panel.table.table.rowCount() == len(results)

    def test_a_bad_path_reports_instead_of_raising(self, qtbot, tmp_path):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        assert not panel.load(str(tmp_path / "nowhere"))
        assert "No results table" in panel._source.text()


class TestTheGuideSupportTab:
    """The one thing a volcano structurally cannot show."""

    def _frame(self):
        return pd.DataFrame({
            "feature": ["fraction:grna[244480_3]", "gene_fraction:gene[244480]",
                        "fraction:grna[225160_1]", "fraction:grna[225160_2]",
                        "fraction:grna[225160_3]", "gene_fraction:gene[225160]"],
            "coefficient": [2.0, 2.0, 0.4, 0.6, 0.5, 0.5],
            "p_value": [1.6e-12, 1.6e-12, 0.51, 0.14, 0.27, 4.6e-08],
        })

    def test_it_names_the_single_guide_gene(self, qtbot):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        panel.set_frame(self._frame())

        assert "Guide support" in [panel.tabs.tabText(i)
                                   for i in range(panel.tabs.count())]
        verdicts = {}
        headers = [panel.support.table.horizontalHeaderItem(c).text()
                   for c in range(panel.support.table.columnCount())]
        gene_col, verdict_col = headers.index("gene"), headers.index("verdict")
        for row in range(panel.support.table.rowCount()):
            verdicts[panel.support.table.item(row, gene_col).text()] = \
                panel.support.table.item(row, verdict_col).text()

        assert "single guide" in verdicts["244480"]
        assert "agreement is the evidence" in verdicts["225160"]

    def test_a_table_with_no_guide_terms_empties_the_tab(self, qtbot):
        from spacr.qt.widgets.regression_results import RegressionResultsPanel

        panel = RegressionResultsPanel()
        qtbot.addWidget(panel)
        panel.set_frame(pd.DataFrame({
            "feature": ["Intercept"], "coefficient": [1.0], "p_value": [0.01]}))
        assert panel.support.table.rowCount() == 0
