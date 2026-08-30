"""Sorting must not break what a row stands for.

Three screens looked a selected row up by its POSITION in a list. That is
correct exactly until the table can be sorted, which it now can everywhere:
the third row stops being the third job, the third model and the third fit
the moment a header is clicked. Each one is asserted here by sorting the
table and reading the selection back as the THING, not as the row number.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt  # noqa: E402

pytestmark = pytest.mark.qt


def test_the_batch_queue_finds_the_selected_job_after_a_sort(qtbot, qapp,
                                                             tmp_path):
    from spacr import batch as bt
    from spacr.qt.screens.batch import BatchScreen

    screen = BatchScreen()
    qtbot.addWidget(screen)
    for label in ("zeta", "alpha", "mu"):
        screen.queue().add(bt.Job(module="mask",
                                  settings={"src": str(tmp_path / label)},
                                  label=label), validate=False)
    screen._refresh_table()
    qapp.processEvents()

    table = screen._table
    assert table.rowCount() == 3
    label_column = 3                      # "Label"
    screen._table.selectRow(0)
    chosen = table.item(0, label_column).text()
    assert screen.selected_job().label == chosen

    # Sort by label; the same row number now holds a different job.
    table.sortItems(label_column, Qt.AscendingOrder)
    qapp.processEvents()
    row = table.currentRow()
    assert table.item(row, label_column).text() == chosen
    assert screen.selected_job() is not None
    assert screen.selected_job().label == chosen


def test_the_model_zoo_returns_the_model_the_row_shows(qtbot, qapp):
    from spacr.qt.screens import model_zoo as zoo_screen
    from spacr import model_zoo as zoo

    screen = zoo_screen.ModelZooScreen()
    qtbot.addWidget(screen)
    entries = [
        zoo.ModelEntry(key="z", name="zeta", path="/tmp/zeta.pth",
                       kind="cellpose", source="local", size_bytes=3),
        zoo.ModelEntry(key="a", name="alpha", path="/tmp/alpha.pth",
                       kind="cellpose", source="local", size_bytes=2),
        zoo.ModelEntry(key="m", name="mu", path="/tmp/mu.pth",
                       kind="cellpose", source="local", size_bytes=1),
    ]
    screen.set_entries(entries)
    qapp.processEvents()

    table = screen._table
    table.sortItems(0, Qt.AscendingOrder)      # by name: alpha, mu, zeta
    qapp.processEvents()
    screen.select(0)
    qapp.processEvents()

    shown = table.item(0, 0).text()
    chosen = screen.selected_entries()
    assert chosen and chosen[0].name == shown, (
        f"row 0 shows {shown} but the screen returned "
        f"{chosen[0].name if chosen else None}")


def test_a_size_column_sorts_on_bytes_not_on_the_unit_printed(qtbot, qapp):
    """"900 KB" reads as 900 and would sit above "12 MB"."""
    from spacr.qt.screens import model_zoo as zoo_screen
    from spacr import model_zoo as zoo

    screen = zoo_screen.ModelZooScreen()
    qtbot.addWidget(screen)
    screen.set_entries([
        zoo.ModelEntry(key="s", name="small", path="/tmp/a",
                       kind="cellpose", source="local",
                       size_bytes=900 * 1024),
        zoo.ModelEntry(key="b", name="big", path="/tmp/b",
                       kind="cellpose", source="local",
                       size_bytes=12 * 1024 * 1024),
    ])
    qapp.processEvents()

    table = screen._table
    table.sortItems(4, Qt.DescendingOrder)
    qapp.processEvents()
    assert table.item(0, 0).text() == "big", [
        table.item(r, 4).text() for r in range(table.rowCount())]


def test_a_sorted_dose_response_row_draws_its_own_curve(qtbot, qapp):
    """The clicked row's curve, not the curve whose fit sits at that index."""
    import numpy as np
    import pandas as pd

    pytest.importorskip("matplotlib")
    from spacr.qt.screens.dose_response import DoseResponseScreen
    from spacr.qt.widgets.dose_response import four_parameter_logistic

    doses = 27.0 / 3.0 ** np.arange(10)
    parts = []
    for gene, ec50, seed in (("zeta", 1.0, 1), ("alpha", 0.05, 2),
                             ("mu", 0.3, 3)):
        rng = np.random.default_rng(seed)
        dose = np.repeat(doses, 3)
        clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(ec50),
                                        -1.0)
        parts.append(pd.DataFrame({
            "gene": gene, "conc_uM": dose,
            "signal": clean + rng.normal(0.0, 1.0, dose.size)}))
    screen = DoseResponseScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(pd.concat(parts, ignore_index=True), label="synthetic")
    screen.concentration_picker.setCurrentText("conc_uM")
    screen.response_picker.setCurrentText("signal")
    screen.group_picker.setCurrentText("gene")
    screen.fit()
    qapp.processEvents()

    table = screen.table
    assert table.rowCount() == 3
    # The engine hands its fits over grouped alphabetically, so sorting on
    # the group column proves nothing. EC50 puts them in a different order.
    ec50 = 4
    table.sortItems(ec50, Qt.DescendingOrder)
    qapp.processEvents()
    order = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert order != sorted(order), (
        f"the sort left the rows alphabetical, so this proves nothing: "
        f"{order}")

    table.clearSelection()
    table.selectRow(0)
    qapp.processEvents()

    shown = table.item(0, 0).text()
    assert shown in screen.report.toPlainText(), (
        f"row 0 shows {shown} but the report describes something else:\n"
        + screen.report.toPlainText()[:300])


def test_a_row_carrying_no_index_falls_back_to_its_own_position(qtbot, qapp):
    """The identity stamp is what survives a sort, and it can be absent.

    Every row is stamped with the entry's index under ``Qt.UserRole``, which
    is what makes a sorted table still name the right model. A row rebuilt by
    something that did not stamp it -- a filter that repopulated the table, a
    plugin adding a row -- has no stamp, and falling back to the ROW NUMBER is
    right for exactly as long as nothing has been sorted.

    Asserted with the table unsorted, because that is the only state in which
    the fallback is correct, and it is better than returning nothing at all.
    """
    from spacr.qt.screens import model_zoo as zoo_screen
    from spacr import model_zoo as zoo

    screen = zoo_screen.ModelZooScreen()
    qtbot.addWidget(screen)
    screen.set_entries([
        zoo.ModelEntry(key="a", name="alpha", path="/tmp/a.pth",
                       kind="cellpose", source="local", size_bytes=1),
        zoo.ModelEntry(key="b", name="beta", path="/tmp/b.pth",
                       kind="cellpose", source="local", size_bytes=2),
    ])
    qapp.processEvents()

    # Strip the stamp from row 1, the way a repopulate that forgot it would.
    screen._table.item(1, 0).setData(Qt.UserRole, None)
    screen.select(1)
    qapp.processEvents()

    chosen = screen.selected_entries()

    assert [entry.name for entry in chosen] == ["beta"]


def test_a_row_whose_index_points_past_the_entries_is_dropped(qtbot, qapp):
    """A stale stamp names a model that is no longer in the list.

    Returning it would raise IndexError inside whatever the user clicked --
    delete, benchmark, set as default -- so the row is simply not offered.
    """
    from spacr.qt.screens import model_zoo as zoo_screen
    from spacr import model_zoo as zoo

    screen = zoo_screen.ModelZooScreen()
    qtbot.addWidget(screen)
    screen.set_entries([
        zoo.ModelEntry(key="a", name="alpha", path="/tmp/a.pth",
                       kind="cellpose", source="local", size_bytes=1),
    ])
    qapp.processEvents()

    screen._table.item(0, 0).setData(Qt.UserRole, 99)
    screen.select(0)
    qapp.processEvents()

    assert screen.selected_entries() == []
