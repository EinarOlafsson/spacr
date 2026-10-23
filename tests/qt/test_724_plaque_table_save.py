"""Plaque table gestures save the corresponding unrounded visible data."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMenu, QMessageBox

from spacr.qt.widgets import plaque_preview as ppv


_REAL_DIALOG_EXEC = QDialog.exec


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    Image.fromarray(np.full((400, 400, 3), 30, np.uint8)).save(tmp_path / "figure.png")
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    widget.apply_settings({"plaque_mode": "figure", "confirm_annotations": False})
    widget.load_source_async(str(tmp_path))

    def detect(image, weights, **kwargs):
        return [SimpleNamespace(x0=x, y0=100, x1=x + 80, y1=180, confidence=.9)
                for x in (100, 220)]

    def segment(crop):
        labels = np.zeros(crop.shape[:2], np.int32)
        labels[10:20, 10:20] = 7
        labels[30:37, 30:37] = 42
        return labels

    assert widget.run_preview(detect=detect, read_text=lambda path: [])
    assert widget.find_plaques_in_all_wells(segment=segment)
    for index, annotation in enumerate(widget._annotations):
        annotation.condition = f"condition-{index}"
        annotation.pixels_per_um = 1.23456789012345 + index
    widget._refresh_calibration_scales()
    widget._fill_table()
    widget._fill_plaque_table()
    return widget


def _choose_save(path, *, accepted=True):
    confirmations = []

    def choose(dialog, failures):
        assert isinstance(dialog, QFileDialog)
        assert dialog.acceptMode() == QFileDialog.AcceptSave
        assert not dialog.testOption(QFileDialog.DontConfirmOverwrite)
        chosen = next(label for label in dialog.nameFilters()
                      if f"*.{path.suffix[1:]}" in label)
        dialog.selectNameFilter(chosen)
        dialog.filterSelected.emit(chosen)
        assert dialog.defaultSuffix() == path.suffix[1:]
        assert dialog.selectedFiles()[0].endswith(path.suffix)
        dialog.selectFile(str(path))
        if not accepted:
            dialog.reject()
            return
        if path.exists():
            def confirm():
                box = QApplication.activeModalWidget()
                try:
                    assert isinstance(box, QMessageBox)
                    yes = box.button(QMessageBox.Yes)
                    assert yes is not None
                    confirmations.append(str(path))
                    yes.click()
                except Exception as error:
                    failures.append(error)
                    if box is not None:
                        box.reject()
                    dialog.reject()
            QTimer.singleShot(0, confirm)
        dialog.accept()

    choose.confirmations = confirmations
    return choose


def _save_gesture(monkeypatch, table, choose, *, expect_error=False):
    menus, failures, errors = [], [], []
    monkeypatch.setattr(QFileDialog, "exec", _REAL_DIALOG_EXEC)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda parent, title, text: errors.append(text))

    def choose_dialog():
        dialog = QApplication.activeModalWidget()
        try:
            choose(dialog, failures)
        except Exception as error:
            failures.append(error)
            if dialog is not None:
                dialog.reject()

    def choose_menu():
        menu = QApplication.activePopupWidget()
        try:
            assert isinstance(menu, QMenu)
            menus.append([action.text() for action in menu.actions()])
            action = menu.actions()[0]
            QTimer.singleShot(0, choose_dialog)
            QTest.mouseClick(menu, Qt.LeftButton,
                             pos=menu.actionGeometry(action).center())
        except Exception as error:
            failures.append(error)
        finally:
            if menu is not None:
                menu.close()

    assert table.contextMenuPolicy() == Qt.CustomContextMenu
    QTimer.singleShot(0, choose_menu)
    table.customContextMenuRequested.emit(QPoint(4, 4))
    assert not failures, failures
    assert len(menus) == 1 and menus[0][0] == ppv.tr("Save table…")
    if not expect_error:
        assert errors == []
    return errors


@pytest.mark.parametrize("which", ("_table", "_plaque_table"))
@pytest.mark.parametrize("suffix", ("csv", "tsv"))
def test_the_clicked_table_saves_visible_sorted_typed_values(
        panel, monkeypatch, tmp_path, which, suffix):
    table = getattr(panel, which)
    table.sortItems(0, Qt.DescendingOrder)
    table.setRowHidden(1, True)
    table.horizontalHeader().moveSection(4, 0)
    table.selectRow(0)
    if which == "_table":
        table.item(0, ppv.CONDITION_COLUMN).setText("edited well")
        table.item(0, ppv.OK_COLUMN).setCheckState(Qt.Checked)
    table.setColumnHidden(1, True)
    selected = [(item.row(), item.column()) for item in table.selectedItems()]
    visible = [row for row in range(table.rowCount()) if not table.isRowHidden(row)]
    expected_ids = [int(table.item(row, 0).text()) for row in visible]
    condition_column = ppv.CONDITION_COLUMN if which == "_table" else 2
    expected_conditions = [table.item(row, condition_column).text() for row in visible]
    expected_plaque_ids = ([int(table.item(row, 3).text()) for row in visible]
                          if which == "_plaque_table" else [])
    header = table.horizontalHeader()
    expected_columns = [table.horizontalHeaderItem(header.logicalIndex(position)).text()
                        for position in range(table.columnCount())
                        if not table.isColumnHidden(header.logicalIndex(position))]
    destination = tmp_path / f"{which}.{suffix}"
    _save_gesture(monkeypatch, table, _choose_save(destination))
    actual = pd.read_csv(destination, sep="\t" if suffix == "tsv" else ",",
                         float_precision="round_trip")
    assert list(actual.columns) == expected_columns
    assert actual.columns[0] == ppv.tr("Condition" if which == "_table" else "Area (px)")
    assert ppv.tr("Panel") not in actual.columns
    assert len(actual) == len(visible) == table.rowCount() - 1
    condition = ppv.tr("Condition")
    assert actual[condition].tolist() == expected_conditions
    if which == "_table":
        assert actual[ppv.tr("#")].tolist() == expected_ids
        assert actual[ppv.tr("Mean area")].tolist() == [74.5]
        assert actual[condition].tolist() == ["edited well"]
        assert actual[ppv.tr("OK")].tolist() == [True]
        assert actual[ppv.tr("Pixels per µm")].tolist() == [
            1.23456789012345 + actual[ppv.tr("#")].iloc[0] - 1]
    else:
        assert actual[ppv.tr("Well")].tolist() == expected_ids
        assert actual[ppv.tr("Plaque")].tolist() == expected_plaque_ids
        assert actual[ppv.tr("Area (px)")].tolist() == [
            {7: 100, 42: 49}[identifier] for identifier in expected_plaque_ids]
        assert set(actual[ppv.tr("Plaque")]) <= {7, 42}
        area = ppv.tr("Area (mm²)")
        assert any(value != round(value, 2) for value in actual[area])
        independently_scaled = actual[ppv.tr("Area (px)")] / (
            (1.23456789012345 + actual[ppv.tr("Well")] - 1) * 1000) ** 2
        assert actual[area].tolist() == pytest.approx(independently_scaled.tolist())
    assert [(item.row(), item.column()) for item in table.selectedItems()] == selected


@pytest.mark.parametrize("which", ("_table", "_plaque_table"))
def test_cancelling_a_table_save_leaves_the_existing_file_unchanged(
        panel, monkeypatch, tmp_path, which):
    destination = tmp_path / "existing.csv"
    destination.write_bytes(b"previous file\n")
    _save_gesture(monkeypatch, getattr(panel, which),
                  _choose_save(destination, accepted=False))
    assert destination.read_bytes() == b"previous file\n"
    assert not list(tmp_path.glob(".*.pending*"))


def test_a_partial_write_failure_preserves_the_existing_table(
        panel, monkeypatch, tmp_path):
    from spacr import tabular

    destination = tmp_path / "existing.csv"
    destination.write_bytes(b"previous file\n")
    def fail(frame, path, **kwargs):
        from pathlib import Path
        Path(path).write_bytes(b"incomplete")
        raise OSError("disk unavailable")

    monkeypatch.setattr(tabular, "write_table", fail)
    choose = _choose_save(destination)
    errors = _save_gesture(monkeypatch, panel._plaque_table, choose, expect_error=True)
    assert choose.confirmations == [str(destination)]
    assert destination.read_bytes() == b"previous file\n"
    assert len(errors) == 1 and "disk unavailable" in errors[0]
    assert not list(tmp_path.glob(".*.pending*"))


@pytest.mark.parametrize("which", ("_table", "_plaque_table"))
def test_cleared_results_save_only_current_headers_without_stale_rows(
        panel, monkeypatch, tmp_path, which):
    panel._clear_figure()
    table = getattr(panel, which)
    destination = tmp_path / "empty.csv"
    _save_gesture(monkeypatch, table, _choose_save(destination))
    actual = pd.read_csv(destination)
    assert actual.empty
    assert list(actual.columns) == list(panel._table_export_frame(table).columns)
