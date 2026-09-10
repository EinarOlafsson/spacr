"""Inspect a real saved evaluation bundle through Classify's actual fold."""
import json
from pathlib import Path
import time

from build_evaluation_example import sha


def record_evaluation(app, window, stage, captures, capture, settle, write_json, timeout, source):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    stage, source = Path(stage).resolve(), Path(source).resolve()
    if not source.is_relative_to(stage / 'evaluation_runs'):
        raise ValueError('Only the private prepared tutorial bundle may be inspected')
    prepared = json.loads((source / 'preparation.json').read_text())
    if not prepared['input_prepared'] or prepared['database_identity_audit']['passed']:
        raise ValueError('The known-overlap bundle was not prepared as documented')
    before = {str(p): sha(p) for p in source.rglob('*') if p.is_file()}
    proof = dict(lesson='39_classifier_evaluation', accepted=False,
        source=str(source), prepared_manifest=prepared['manifest'],
        actual_gui_loaded=False, app_source_modified=False, published=False,
        original_inputs_preserved=False, snapshots={})
    write_json(captures / 'scientific_acceptance.json', proof)
    deadline = time.monotonic() + timeout

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('The native evaluation inspector exceeded its time limit')
        settle(.1)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual evaluation control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    options = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey') == 'classify_merged' or w.property('navKey') == 'classify_merged')]
    click(max(options, key=lambda w: w.width() * w.height()))
    host = window._screens['classify_merged']
    capture('01_classify_host')
    folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'classifier_evaluation']
    if len(folds) != 1:
        raise ValueError('No unique actual Classifier Evaluation fold')
    click(folds[0])
    candidates = [w for w in window.findChildren(ClassifierEvaluationScreen) if w.isVisible()]
    if len(candidates) != 1:
        raise ValueError('The actual evaluation page did not open')
    panel = candidates[0]
    capture('02_actual_evaluation_page')
    errors, accepted = [], []
    timer, watchdog = QTimer(window), QTimer(window)
    timer.setSingleShot(True)
    watchdog.setSingleShot(True)

    def choose():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise ValueError('The real evaluation folder dialog did not open')
            dialog.accepted.connect(lambda: accepted.append(True))
            dialog.resize(1400, 950)
            field = dialog.findChild(QLineEdit, 'fileNameEdit')
            click(field)
            QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(field, str(source))
            capture('03_actual_bundle_folder')
            box = dialog.findChild(QDialogButtonBox)
            buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
            if len(buttons) != 1:
                raise ValueError('No unique real folder acceptance button')
            click(buttons[0])
        except Exception as error:
            errors.append(str(error))
            if dialog is not None:
                dialog.reject()

    def abort_dialog():
        errors.append('The real folder dialog timed out')
        if app.activeModalWidget() is not None:
            app.activeModalWidget().reject()

    timer.timeout.connect(choose)
    watchdog.timeout.connect(abort_dialog)
    timer.start(400)
    watchdog.start(20000)
    loaded = []
    panel.evaluation_loaded.connect(loaded.append)
    try:
        click(panel._browse)
    finally:
        timer.stop()
        watchdog.stop()
    if errors or not accepted:
        raise ValueError('; '.join(errors) or 'Folder selection was not accepted')
    while panel._busy or not loaded or panel._jobs:
        tick()
        if panel.last_error:
            raise ValueError(panel.last_error)
    proof['actual_gui_loaded'] = True
    proof['loaded_signals'] = loaded
    proof['status_text'] = panel._status.text()

    def table(widget):
        return dict(columns=[widget.horizontalHeaderItem(c).text() for c in range(widget.columnCount())],
            rows=[[widget.item(r, c).text() if widget.item(r, c) else ''
                for c in range(widget.columnCount())] for r in range(widget.rowCount())])

    def snapshot(name):
        capture(name)
        proof['snapshots'][name] = dict(tab=panel._tabs.tabText(panel._tabs.currentIndex()),
            summary=panel._overview.toPlainText(), leakage=panel._leakage.toPlainText(),
            confusion=table(panel._confusion), per_plate=table(panel._per_plate),
            calibration=table(panel._calibration), predictions=table(panel._predictions),
            prediction_filter=panel._prediction_filter.text(), threshold=panel._threshold.value(),
            cell_summary=panel._cell_summary.text(), cell_breakdown=panel._cell_breakdown.text(),
            high=[panel._high_list.item(i).text() for i in range(panel._high_list.count())],
            low=[panel._low_list.item(i).text() for i in range(panel._low_list.count())],
            high_open_enabled=panel._high_open.isEnabled(), low_open_enabled=panel._low_open.isEnabled())
        write_json(captures / 'scientific_acceptance.json', proof)

    def tab(index):
        bar = panel._tabs.tabBar()
        QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(index).center())
        settle(.3)

    snapshot('04_actual_summary')
    tab(5)
    snapshot('05_actual_leakage_requires_review')
    tab(1)
    snapshot('06_actual_confusion_counts')
    row = next(r for r in range(panel._confusion.rowCount())
               if panel._confusion.item(r, 0).text() == 'infected_1')
    col = next(c for c in range(panel._confusion.columnCount())
               if panel._confusion.horizontalHeaderItem(c).text() == 'infected_2')
    item = panel._confusion.item(row, col)
    panel._confusion.scrollToItem(item)
    QTest.mouseClick(panel._confusion.viewport(), Qt.LeftButton,
                    pos=panel._confusion.visualItemRect(item).center())
    settle(.3)
    snapshot('07_actual_error_cell')
    click(panel._threshold)
    QTest.keyClick(panel._threshold, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(panel._threshold, '0.95')
    QTest.keyClick(panel._threshold, Qt.Key_Tab)
    settle(.3)
    if panel._threshold.value() != .95:
        raise ValueError('The actual confidence threshold differs')
    snapshot('08_actual_confidence_threshold')
    tab(2)
    snapshot('09_actual_per_plate')
    tab(3)
    snapshot('10_actual_calibration_table')
    tab(4)
    snapshot('11_actual_all_predictions')
    click(panel._prediction_filter)
    QTest.keyClicks(panel._prediction_filter, 'r5_c2')
    settle(.3)
    snapshot('12_actual_prediction_filter')
    QTest.keyClick(panel._prediction_filter, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClick(panel._prediction_filter, Qt.Key_Backspace)
    settle(.3)
    snapshot('13_actual_filter_cleared')
    tab(5)
    snapshot('14_actual_final_leakage')
    proof['bundle_unchanged'] = all(sha(p) == value for p, value in before.items())
    proof['original_inputs_preserved'] = all(sha(p) == value for p, value in prepared['original_inputs'].items())
    if not proof['bundle_unchanged'] or not proof['original_inputs_preserved']:
        raise ValueError('Inspecting the results changed an input')
    proof['next_gate'] = 'Independent comparisons of actual tables, filters, confidence splits and saved inputs'
    write_json(captures / 'scientific_acceptance.json', proof)
