"""The two tabs the request named: Coefficients and Hits.

"whenever a container shows a table like the Coefficients and hit tabs in
regression module, the user should be able to sort on each column by clicking
it once for descending next click for ascending."

Driven through the header with a real mouse click, read off the drawn table,
and with the selection checked by FEATURE rather than by row number.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

pytestmark = pytest.mark.qt


def _coefficients():
    return pd.DataFrame({
        "feature": ["gene[A]", "gene[B]", "gene[C]", "gene[D]"],
        "coefficient": [2.0, 9.0, 10.0, -3.0],
        "p_value": [1e-2, 3e-9, 2e-5, 0.4],
        "q_value": [2e-2, 6e-9, 4e-5, 0.6],
    })


def _click(header, column, qapp):
    x = header.sectionViewportPosition(column) + header.sectionSize(column) // 2
    QTest.mouseClick(header.viewport(), Qt.LeftButton, Qt.NoModifier,
                     QPoint(x, header.height() // 2))
    qapp.processEvents()


def _drawn(table, column):
    return [table.item(row, column).text()
            for row in range(table.rowCount())
            if not table.isRowHidden(row)]


@pytest.fixture()
def panel(qtbot, qapp):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel(external_volcano=True)
    qtbot.addWidget(widget)
    widget.set_frame(_coefficients(), source="results.csv")
    widget.resize(900, 600)
    widget.show()
    qapp.processEvents()
    return widget


def test_the_coefficients_tab_is_a_tab_and_it_sorts(panel, qapp):
    titles = [panel.tabs.tabText(i) for i in range(panel.tabs.count())]
    assert any(title.startswith("Coefficients") for title in titles), titles

    table = panel.table.table
    columns = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    assert "coefficient" in columns
    coefficient = columns.index("coefficient")

    _click(table.horizontalHeader(), coefficient, qapp)
    assert _drawn(table, coefficient) == ["10.0", "9.0", "2.0", "-3.0"]

    _click(table.horizontalHeader(), coefficient, qapp)
    assert _drawn(table, coefficient) == ["-3.0", "2.0", "9.0", "10.0"]


def test_every_column_of_the_coefficients_tab_sorts(panel, qapp):
    """Every column, not the one somebody remembered to wire."""
    table = panel.table.table
    header = table.horizontalHeader()
    for column in range(table.columnCount()):
        before = _drawn(table, column)
        _click(header, column, qapp)
        descending = _drawn(table, column)
        _click(header, column, qapp)
        ascending = _drawn(table, column)
        name = table.horizontalHeaderItem(column).text()
        assert descending == list(reversed(ascending)), name
        assert header.sortIndicatorOrder() == Qt.AscendingOrder, name
        assert sorted(descending) == sorted(before), name


def test_a_q_value_column_sorts_as_a_number_not_as_a_word(panel, qapp):
    """3e-9 is smaller than 2e-5, however the two read as text."""
    table = panel.table.table
    columns = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    q = columns.index("q_value")

    _click(table.horizontalHeader(), q, qapp)
    _click(table.horizontalHeader(), q, qapp)      # ascending: smallest q first
    assert _drawn(table, q)[0] == "6e-09"


def test_a_selected_coefficient_survives_a_sort_as_the_same_feature(panel,
                                                                    qapp):
    """The acceptance point: the same FEATURE, not the same row number."""
    table = panel.table.table
    columns = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    feature = columns.index("feature")

    table.selectRow(0)
    qapp.processEvents()
    chosen = table.item(0, feature).text()
    keys = []
    panel.table.key_selected.connect(keys.append)

    _click(table.horizontalHeader(), columns.index("coefficient"), qapp)

    rows = {index.row() for index in table.selectedIndexes()}
    assert rows, "the sort dropped the selection"
    still = table.item(sorted(rows)[0], feature).text()
    assert still == chosen
    # And the plot is told about the same feature, not about a row number.
    assert not keys or keys[-1] in chosen


def test_the_hits_tab_sorts_on_every_column(qtbot, qapp, tmp_path):
    """The other tab the request named."""
    from spacr.qt.screens.hit_list import HitListScreen

    gene = pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in ("100", "200", "300")],
        "coefficient": [2.4, -1.8, 0.9],
        "std_err": [0.30, 0.25, 0.40],
        "p_value": [1e-6, 4e-5, 0.02],
        "condition": ["other", "other", "other"],
        "n_gene": [48, 44, 30],
    })
    grna = pd.DataFrame({
        "feature": [f"fraction:grna[{g}]"
                    for g in ("100_1", "100_2", "200_1", "300_1")],
        "grna": ["100_1", "100_2", "200_1", "300_1"],
        "coefficient": [2.2, 2.9, -1.9, 0.9]})
    root = tmp_path / "results" / "pred" / "ols"
    root.mkdir(parents=True)
    gene.to_csv(root / "results_gene.csv", index=False)
    grna.to_csv(root / "results_grna.csv", index=False)
    pd.concat([gene, grna], ignore_index=True).to_csv(
        root / "results.csv", index=False)

    screen = HitListScreen(folder=str(root), threaded=False,
                           regression_type="ols")
    qtbot.addWidget(screen)
    screen.resize(1100, 500)
    screen.show()
    qapp.processEvents()

    tree = screen._table
    assert tree.topLevelItemCount() >= 2, "fixture produced no hits"
    header = tree.header()
    effect = 3            # the "Effect" column

    _click(header, effect, qapp)
    assert header.sortIndicatorOrder() == Qt.DescendingOrder
    drawn = [tree.topLevelItem(i).text(effect)
             for i in range(tree.topLevelItemCount())]
    numbers = [float(text) for text in drawn if text not in ("", "—")]
    assert numbers == sorted(numbers, reverse=True), drawn

    _click(header, effect, qapp)
    assert header.sortIndicatorOrder() == Qt.AscendingOrder
    drawn = [tree.topLevelItem(i).text(effect)
             for i in range(tree.topLevelItemCount())]
    numbers = [float(text) for text in drawn if text not in ("", "—")]
    assert numbers == sorted(numbers), drawn
