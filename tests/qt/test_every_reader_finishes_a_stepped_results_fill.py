"""Every reader of a Regression results table finishes a stepped fill first.

A large table fills a few thousand cells per event-loop turn. Searching,
sorting, selecting a row or key, restricting to keys and copying all read
the rows, so each completes the fill before it looks; once the fill is done
a read starts nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets.fast_plots import ResultsTable  # noqa: E402
from tests.qt.test_a_large_results_table_fills_in_steps import _frame  # noqa: E402

pytestmark = pytest.mark.qt


def _filling(qtbot):
    table = ResultsTable()
    table.fill_in_steps = True
    qtbot.addWidget(table)
    assert table.set_frame(_frame()) == 200
    assert table._filling is not None, "the fill ran in one go"
    return table


@pytest.mark.parametrize("read", [
    lambda t: t.table.findItems("gene_fraction:gene[G199]", Qt.MatchExactly),
    lambda t: t.table.sortItems(1, Qt.DescendingOrder),
    lambda t: t.select_frame_row(199),
    lambda t: t.select_key("gene_fraction:gene[G199]"),
    lambda t: t.show_keys(["gene_fraction:gene[G5]"]),
    lambda t: t.copy_visible(),
], ids=["find", "sort", "select-row", "select-key", "show-keys", "copy"])
def test_a_read_during_the_fill_sees_every_row(qtbot, read):
    table = _filling(qtbot)
    read(table)
    assert table._filling is None
    assert table.table.rowCount() == 200
    texts = {table.table.item(r, 0).text() for r in range(200)}
    assert "gene_fraction:gene[G199]" in texts


def test_a_search_finds_the_last_row_mid_fill(qtbot):
    table = _filling(qtbot)
    found = table.table.findItems("gene_fraction:gene[G199]", Qt.MatchExactly)
    assert [item.text() for item in found] == ["gene_fraction:gene[G199]"]


def test_a_read_after_the_fill_starts_no_work(qtbot):
    table = _filling(qtbot)
    qtbot.waitUntil(lambda: table._filling is None, timeout=10000)
    table.table._finish_reading = None
    assert table.table.item(0, 0).text() == "gene_fraction:gene[G0]"
    assert table._finish_the_fill() is False


def test_a_frame_set_while_a_step_runs_replaces_the_old_fill(qtbot):
    table = _filling(qtbot)
    newer = _frame()
    newer["feature"] = [f"newer[{i}]" for i in range(200)]

    def interrupted():
        table.set_frame(newer)
        yield

    table._filling = interrupted()
    table._fill_a_step()
    assert table._filling is not None, "the newer fill was dropped"
    qtbot.waitUntil(lambda: table._filling is None, timeout=10000)
    assert table.table.rowCount() == 200
    assert {table.table.item(r, 0).text() for r in range(200)} == {f"newer[{i}]" for i in range(200)}


def test_a_grouped_plot_falls_back_to_its_default_mark_when_the_choice_cannot_be_read(monkeypatch):
    from spacr import graph_types
    from spacr.qt.widgets.fast_plots import GroupedPlot

    def unreadable(_shape):
        raise OSError("the graph-type preferences cannot be read")

    monkeypatch.setattr(graph_types, "chosen_for", unreadable)
    assert GroupedPlot.starting_mark((3, 4)) == (GroupedPlot.DEFAULT_MARK, "")
