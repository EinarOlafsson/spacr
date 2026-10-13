"""Record Plate Viewer's public example through its real controls."""
from pathlib import Path
import math
import time

from capture_database import _digest, _readonly
from capture_plate_retention import verify_export, verify_wells


def record_plate_example(app, window, screen, stage, captures, capture,
                         settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit
    from spacr.qt.screens.plate_view import PlateViewScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    database = Path(stage) / 'example_data/plate1/measurements/measurements.db'
    original = _digest(database)
    with _readonly(database) as connection:
        rows = connection.execute('SELECT plateID,rowID,columnID,COUNT(*),AVG(cell_area) '
                                  'FROM cell GROUP BY plateID,rowID,columnID').fetchall()
    independent = {(str(p), str(r), str(c)): dict(n=n, mean=mean)
                   for p, r, c, n, mean in rows}
    assert len(rows) == 4 and sum(row[3] for row in rows) == 2341
    minimum = math.ceil((min(row[3] for row in rows) + max(row[3] for row in rows)) / 2)
    deadline = time.monotonic() + timeout
    proof = dict(accepted=False, example_database_sha256=original,
                 input_rows=2341, input_wells=4, states={}, published=False,
                 input_method='Actual Load test data button with a private cached copy of the public Annotate example')

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise RuntimeError('Unavailable Plate Viewer control: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle():
        settle(.3)
        while panel.is_busy() or panel.active_jobs() or panel._recompute_timer.isActive():
            if time.monotonic() > deadline:
                raise TimeoutError('Plate Viewer did not finish')
            settle(.1)
        if panel.last_error:
            raise RuntimeError(panel.last_error)

    def fill(widget, text):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(text))
        QTest.keyClick(widget, Qt.Key_Return)
        settle(.2)

    def select(widget, text):
        index = widget.findText(text)
        assert index >= 0
        click(widget)
        QTest.keyClick(widget.view(), Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(widget.view(), Qt.Key_Down)
        QTest.keyClick(widget.view(), Qt.Key_Return)
        idle()
        assert widget.currentText() == text

    capture('01_graph_builder_host')
    folds = [w for w in screen.findChildren(FoldButton) if w.app_key == 'plate_view' and w.isVisible()]
    assert len(folds) == 1
    click(folds[0]); settle(.5)
    panels = [w for w in window.findChildren(PlateViewScreen) if w.isVisible()]
    assert len(panels) == 1
    panel = panels[0]
    capture('02_empty_viewer')
    click(panel._test_data_button)
    idle()
    assert panel.current_table() == 'cell' and panel.current_value_column() == 'cell_area'
    select(panel._grouping_combo, 'mean')
    click(panel._btn_render); idle()
    records = panel._layout_df.to_dict('records')
    proof['states']['mean'] = verify_wells(records, independent, 'mean', 0)
    capture('03_example_mean')
    click(panel._scale_combo); settle(.2); capture('04_colour_scale', desktop=True)
    QTest.keyClick(panel._scale_combo.view(), Qt.Key_Escape); settle(.2)
    row = records[0]
    QTest.mouseClick(panel._grid, Qt.LeftButton,
                     pos=panel._grid.cell_rect(int(row['row_index']), int(row['column_index'])).center().toPoint())
    settle(.3)
    proof['well_detail'] = panel.well_info_text()
    assert str(row['well']) in proof['well_detail']
    capture('05_well_detail')
    select(panel._grouping_combo, 'count')
    proof['states']['count'] = verify_wells(panel._layout_df.to_dict('records'), independent, 'count', 0)
    capture('06_count')
    fill(panel._min_count_box, minimum); idle()
    records = panel._layout_df.to_dict('records')
    proof['minimum_count'] = minimum
    proof['states']['filtered'] = verify_wells(records, independent, 'count', minimum)
    assert 0 < len(records) < 4
    capture('07_filtered')
    exported = Path(stage) / 'example_well_counts.csv'
    errors, accepted = [], []

    def save():
        dialog = app.activeModalWidget()
        try:
            assert isinstance(dialog, QFileDialog)
            dialog.accepted.connect(lambda: accepted.append(True))
            field = dialog.findChild(QLineEdit, 'fileNameEdit')
            click(field)
            QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(field, str(exported))
            capture('08_save_csv')
            click(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save))
        except Exception as error:
            errors.append(str(error))
            if isinstance(dialog, QFileDialog):
                dialog.reject()

    QTimer.singleShot(500, save)
    click(panel._btn_export); idle()
    if errors or not accepted:
        raise RuntimeError('CSV export did not complete: ' + repr(errors))
    proof['export'] = verify_export(exported, records)
    capture('09_exported')
    fill(panel._min_count_box, 0); idle()
    proof['states']['restored'] = verify_wells(panel._layout_df.to_dict('records'), independent, 'count', 0)
    proof['database_unchanged'] = _digest(database) == original
    assert proof['database_unchanged']
    proof['accepted'] = True
    proof['active_jobs_at_exit'] = panel.active_jobs()
    write_json(Path(captures) / 'scientific_acceptance.json', proof)
    panel.close(); settle(.3)
