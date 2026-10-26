"""Screens redraws keep their items (stale-wrapper sweep, group screens_a).

Agreement, Classifier Evaluation and Control Charts used to delete every
Python-made item in a redraw a click can trigger; with accessibility on that
segfaulted the tutorial host. Each redraw now rewrites the items it has.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QImage
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    interface = QAccessible.queryAccessibleInterface(view)
    if interface is None:
        return
    for index in range(min(interface.childCount(), 80)):
        child = interface.child(index)
        if child is not None:
            child.text(QAccessible.Name)


def _list_items(listing):
    return [listing.item(r) for r in range(listing.count())]


def _cells(table):
    return [table.item(r, c) for r in range(table.rowCount())
            for c in range(table.columnCount()) if table.item(r, c)]


def _build_db(tmp_path, name, annotators, rows):
    src = tmp_path / name
    meas = src / "measurements"
    pngs = src / "cell_png"
    meas.mkdir(parents=True)
    pngs.mkdir(parents=True)
    crops = []
    for i in range(len(rows)):
        crop = pngs / f"plate1_A01_1_{i}.png"
        image = QImage(8, 8, QImage.Format_RGB32)
        image.fill(0xFF3366AA)
        assert image.save(str(crop), "PNG")
        crops.append(str(crop))
    con = sqlite3.connect(meas / "measurements.db")
    try:
        meta_sql = ", ".join(f'"{c}" TEXT' for c in _META)
        con.execute(f"CREATE TABLE png_list ({meta_sql})")
        for col in annotators:
            con.execute(f'ALTER TABLE png_list ADD COLUMN "{col}" INTEGER')
        marks = ", ".join("?" * (len(_META) + len(annotators)))
        con.executemany(
            f"INSERT INTO png_list VALUES ({marks})",
            [(crops[i], os.path.basename(crops[i]), "plate1", "r1", "c1",
              "f1", f"plate1_A01_1_o{i}", f"o{i}", *labels)
             for i, labels in enumerate(rows)])
        con.commit()
    finally:
        con.close()
    return str(src)


def test_agreement_redraws(qtbot, qt_theme_applied, accessible, tmp_path):
    from spacr.qt.screens.agreement import AgreementScreen
    two = _build_db(tmp_path, "two", ["alice", "bob"],
                    [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1), (2, 2)])
    three = _build_db(tmp_path, "three", ["a", "b", "c"],
                      [(1, 1, 1), (1, 2, 2), (2, 2, 2), (1, 2, 1), (1, 1, 2)])
    one = _build_db(tmp_path, "one", ["only"], [(1,), (2,)])
    screen = AgreementScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for path, names in ((two, ["alice", "bob"]), (three, ["a", "b", "c"]),
                        (one, ["only"]), (two, ["alice", "bob"]),
                        (three, ["a", "b", "c"])):
        screen.set_database(path)
        if len(names) > 1:
            screen.compute()
        for view in (screen._columns_list, screen._kappa_table,
                     screen._confusion_table, screen._review_table):
            _walk(view)
        QApplication.processEvents()
        assert screen.available_columns() == names
        assert all(i.flags() & Qt.ItemIsUserCheckable
                   for i in _list_items(screen._columns_list))
        seen.extend(_list_items(screen._columns_list))
        for table in (screen._kappa_table, screen._confusion_table,
                      screen._review_table):
            seen.extend(_cells(table))
        assert all(shiboken6.isValid(i) for i in seen)
        if len(names) > 1:
            assert screen._kappa_table.rowCount() >= 1
            assert screen._confusion_table.rowCount() >= 2
            assert screen._confusion_table.horizontalHeaderItem(0) is not None
            assert all(not (c.flags() & Qt.ItemIsEditable)
                       for c in _cells(screen._confusion_table))
        else:
            assert screen._kappa_table.rowCount() == 0
    screen.set_database("")
    assert screen._columns_list.count() == 0
    assert screen._kappa_table.rowCount() == 0
    assert all(shiboken6.isValid(i) for i in seen)


@pytest.fixture
def evaluation_root(tmp_path):
    from spacr.classifier_evaluation import (
        LeakageReport, evaluate_predictions, write_evaluation_bundle)
    paths = ["plate1_A01_1_1.png", "plate1_A02_1_2.png",
             "plate2_B01_1_3.png", "plate2_B02_1_4.png"]
    evaluation = evaluate_predictions(
        [0, 1, 0, 1],
        np.asarray([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]),
        paths, classes=["negative", "positive"], fold_ids=[1, 1, 2, 2],
        calibration_method="none", calibration_bins=4)
    report = LeakageReport(
        group_by="well", train_samples=2, validation_samples=2,
        overlap_counts={"exact": 0, "augmentation_family": 0, "object": 0,
                        "field": 0, "well": 0, "plate": 1},
        examples={"exact": [], "augmentation_family": [], "object": [],
                  "field": [], "well": [], "plate": ["plate1"]},
        split_name="outer_1")
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation,
        leakage_reports=[report]).parents[1]


def test_classifier_evaluation_redraws(qtbot, qt_theme_applied, accessible,
                                       evaluation_root):
    from spacr.qt.screens.classifier_evaluation import (
        ClassifierEvaluationScreen)
    screen = ClassifierEvaluationScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._source.setText(str(evaluation_root))
    seen = []
    for _ in range(3):
        screen.scan()
        for cell in ((0, 1), (1, 2), (0, 2), (1, 1)):
            screen._on_confusion_cell(*cell)
            for view in (screen._high_list, screen._low_list,
                         screen._confusion, screen._per_plate,
                         screen._calibration):
                _walk(view)
            QApplication.processEvents()
            for listing in (screen._high_list, screen._low_list):
                assert listing.count() >= 1
                seen.extend(_list_items(listing))
            for table in (screen._confusion, screen._per_plate,
                          screen._calibration):
                seen.extend(_cells(table))
            assert all(shiboken6.isValid(i) for i in seen)
        screen._threshold.setValue(0.35)
        screen._threshold.setValue(0.75)
        assert screen._confusion.rowCount() == 2
        assert screen._confusion.columnCount() == 3
        assert screen._confusion.horizontalHeaderItem(1) is not None
        assert screen._confusion.item(0, 0).text()
        assert not (screen._confusion.item(0, 0).flags() & Qt.ItemIsEditable)
        screen._clear_bundle()
        assert screen._confusion.rowCount() == 0
        assert screen._high_list.count() == 0
        assert all(shiboken6.isValid(i) for i in seen)


def _campaign(plates=12):
    rows = []
    rng = np.random.default_rng(3)
    for index in range(plates):
        drift = 0.0 if index < 8 else 3.0 * (index - 7)
        for _ in range(3):
            for kind, base, sd in (("neg", 100.0 + drift, 2),
                                   ("pos", 20.0, 2), ("sample", 60.0, 9)):
                rows.append({"plateID": f"P{index + 1:02d}",
                             "run_date": f"2026-01-{index + 1:02d}",
                             "well_type": kind,
                             "signal": base + rng.normal(0, sd)})
    return pd.DataFrame(rows)


def test_control_chart_redraws(qtbot, qt_theme_applied, accessible):
    from spacr.qt.screens.control_chart import ControlChartScreen
    screen = ControlChartScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    base = _campaign(12)
    frames = [base, base.assign(
        well_type=base["well_type"].replace({"sample": "other"}))]
    for step in range(4):
        frame = frames[step % 2]
        screen.set_frame(frame)
        for lvl in ("neg", "pos"):
            for i in range(screen._levels.count()):
                item = screen._levels.item(i)
                item.setSelected(item.text() == lvl)
            for view in (screen._levels, screen.violations, screen.hit_table,
                         screen.anomaly_table, screen.sar_table):
                _walk(view)
            QApplication.processEvents()
            seen.extend(_list_items(screen._levels))
            for table in (screen.violations, screen.hit_table,
                          screen.anomaly_table, screen.sar_table):
                seen.extend(_cells(table))
            assert all(shiboken6.isValid(i) for i in seen)
        names = [screen._levels.item(i).text()
                 for i in range(screen._levels.count())]
        assert names == sorted(set(frame["well_type"]))
        if screen.result is not None:
            assert screen.violations.rowCount() == len(
                screen.result.violations)
    screen._show_refusal("nothing")
    assert screen.violations.rowCount() == 0
    screen._show_hit_refusal("nothing")
    assert screen.hit_table.rowCount() == 0
    assert all(shiboken6.isValid(i) for i in seen)


@pytest.mark.parametrize("headers", [False, True], ids=["rows", "clear_like"])
def test_retiring_a_sorted_table_announces_no_layout_to_accessibility(
        qtbot, accessible, headers):
    """No ``layoutChanged`` while a sorted table's cells are being retired.

    A native macOS run crashed in ``QAccessibleCache::deleteInterface`` under
    ``QAbstractItemModel::layoutChanged`` under ``setSortingEnabled``, from
    ``_retire_table_cells`` as Agreement's ``set_database`` cleared its
    results. Switching sorting back on sorted rows that held nothing any
    more, and the announcement reached an accessibility cache that still
    had interfaces for what had just been taken out. With accessibility
    active, the cells' interfaces are made first, as a screen reader would.
    """
    from PySide6.QtWidgets import QTableWidget

    from spacr.qt.widgets.sortable_table import _retire_table_cells, table_item

    table = QTableWidget(4, 3)
    qtbot.addWidget(table)
    table.setHorizontalHeaderLabels(["a", "b", "c"])
    for row in range(4):
        for column in range(3):
            table.setItem(row, column, table_item(f"{3 - row}{column}"))
    table.setSortingEnabled(True)
    table.sortByColumn(0, Qt.AscendingOrder)
    table.show()
    qtbot.waitExposed(table)
    _walk(table)
    announced = []
    table.model().layoutChanged.connect(lambda *a: announced.append("layout"))
    table.model().layoutAboutToBeChanged.connect(
        lambda *a: announced.append("about"))
    _retire_table_cells(table, headers=headers)
    QApplication.processEvents()
    _walk(table)
    assert announced == [], "a sort of retired cells was announced"
    assert table.isSortingEnabled(), "sorting was not put back"
    assert not _cells(table)
    table.setRowCount(2)
    for row in range(2):
        for column in range(3):
            table.setItem(row, column, table_item(f"{row}{column}"))
    table.sortByColumn(0, Qt.DescendingOrder)
    assert table.item(0, 0).text() == "10", "sorting no longer works"
