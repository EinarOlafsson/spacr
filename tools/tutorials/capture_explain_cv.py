"""Record the actual surrogate workbench with real, previously saved predictions."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import pandas as pd

from build_evaluation_example import sha


def record_explain(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.qt.screens.model_explanation import ModelExplanationScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    stage = Path(stage)
    source = stage / 'annotate_fresh/example_data/plate1'
    database = source / 'measurements/measurements.db'
    predictions = source / 'datasets/training_1/model/resnet18/rgb/epochs_1/resnet18_time_260909_test_acc.csv'
    before = {str(p): sha(p) for p in (database, predictions)}
    parent = stage / 'explain_cv_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='REAL-recorded-predictions-', dir=parent))
    copies = [work / 'measurements.db', work / 'recorded_predictions.csv']
    for src, dst in zip((database, predictions), copies):
        shutil.copy2(src, dst)
        if sha(dst) != before[str(src)]:
            raise ValueError('Private surrogate input differs from the recorded source')
    output = work / 'surrogate'; output.mkdir()
    proof = dict(lesson='70_explain_cv', accepted=False, private_folder=str(work),
        original_inputs=before, original_inputs_preserved=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        recorded_predictions=int(len(pd.read_csv(predictions))),
        cv_model_retrained=False, independent_classifier_validation=False,
        synthetic_predictions=False, application_modified=False, published=False,
        snapshots={})
    write_json(captures / 'scientific_acceptance.json', proof)
    deadline = time.monotonic() + timeout

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('The bounded native surrogate run timed out')
        settle(.15)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The real surrogate control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    def fill(widget, text):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(text)); QTest.keyClick(widget, Qt.Key_Tab); settle(.2)

    def picker(row, path, name):
        errors, accepted = [], []
        def choose():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The actual surrogate file/folder picker did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1550, 1000)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path); capture(name)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1: raise ValueError('Actual picker acceptance button is ambiguous')
                click(buttons[0])
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None: dialog.reject()
        timer = QTimer(window); timer.setSingleShot(True)
        timer.timeout.connect(lambda: app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(300, choose); timer.start(15000); click(row.button); timer.stop()
        if errors or not accepted or row.text() != str(path):
            raise ValueError('; '.join(errors) or 'The real surrogate input was not selected')
        window.raise_(); window.activateWindow(); settle(.3)

    def choice(combo, text):
        index = combo.findText(text)
        if index < 0: raise ValueError('Requested real input column/choice unavailable: ' + text)
        click(combo); QTest.keyClick(combo, Qt.Key_Home)
        for _ in range(index): QTest.keyClick(combo, Qt.Key_Down)
        QTest.keyClick(combo, Qt.Key_Return); settle(.3)
        if combo.currentText() != text: raise ValueError('Native choice did not take effect')

    def table(widget):
        return dict(columns=[widget.horizontalHeaderItem(c).text() for c in range(widget.columnCount())],
            rows=[[widget.item(r,c).text() if widget.item(r,c) else None
                   for c in range(widget.columnCount())] for r in range(widget.rowCount())])

    panel = None
    try:
        buttons = [b for b in window.findChildren(QAbstractButton) if b.isVisible() and
            (b.property('moduleAppKey') == 'classify_merged' or b.property('navKey') == 'classify_merged')]
        click(max(buttons, key=lambda b:b.width()*b.height())); host = window._screens['classify_merged']
        capture('01_classify_host')
        folds = [b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key == 'explain_cv']
        if len(folds) != 1: raise ValueError('Actual Explain CV fold is not unique')
        click(folds[0])
        screens = [s for s in window.findChildren(ModelExplanationScreen) if s.isVisible()]
        if len(screens) != 1: raise ValueError('The real surrogate screen did not open')
        panel = screens[0].explain; capture('02_actual_surrogate_form')
        picker(panel.database, copies[0], '03_matching_measurements_picker')
        picker(panel.predictions, copies[1], '04_recorded_predictions_picker')
        # Browsing sets the path but does not emit editingFinished. Retype
        # that same real path and Tab out to trigger the genuine header read.
        proof['columns_after_browse'] = dict(
            path=[panel.path_column.itemText(i) for i in range(panel.path_column.count())],
            prediction=[panel.prediction_column.itemText(i) for i in range(panel.prediction_column.count())])
        fill(panel.predictions.edit, copies[1])
        QTest.keyClick(panel.predictions.edit, Qt.Key_Return); settle(.4)
        proof['columns_after_retyping'] = dict(
            path=[panel.path_column.itemText(i) for i in range(panel.path_column.count())],
            prediction=[panel.prediction_column.itemText(i) for i in range(panel.prediction_column.count())],
            entered=panel.predictions.text(), status=panel.status.text(),
            path_exists=Path(panel.predictions.text()).is_file())
        print(json.dumps(proof['columns_after_retyping']), flush=True)
        choice(panel.path_column, 'filename'); choice(panel.prediction_column, 'predicted_label')
        choice(panel.backend, 'Random Forest'); choice(panel.split, 'well')
        picker(panel.output, output, '05_private_output_folder')
        capture('06_ready_existing_predictions_not_retraining')
        started = time.monotonic(); click(panel.run_button); capture('07_actual_fitting')
        while not panel.run_button.isEnabled(): tick()
        proof.update(elapsed_seconds=time.monotonic()-started, status=panel.status.text())
        if panel.result is None:
            capture('08_actual_run_refusal')
            proof['hold'] = panel.status.text()
            return
        result = panel.result
        for index in range(panel.results.count()):
            QTest.mouseClick(panel.results.tabBar(), Qt.LeftButton,
                pos=panel.results.tabBar().tabRect(index).center()); settle(.5)
            name = f'{8+index:02d}_result_tab_{index}'
            capture(name)
            proof['snapshots'][name] = dict(tab=panel.results.tabText(index),
                summary=panel.summary.toPlainText(), status=panel.status.text())
        proof['tables'] = {name: table(getattr(panel,name)) for name in
            ('importance','metrics','confusion','shap','correlations','held_out','distributions')}
        proof.update(computational_run_completed=True, fidelity=float(result.fidelity),
            baseline=float(result.baseline), faithful=bool(result.is_faithful),
            n_objects=int(result.n_objects), features=len(result.feature_columns),
            warnings=result.warnings, split_report=result.split_report,
            open_objects_enabled=panel.open_objects_button.isEnabled(),
            next_gate='Independent source-identity, numeric table and held-out split verification')
    finally:
        proof['original_inputs_preserved'] = all(sha(p)==s for p,s in before.items())
        proof['private_inputs_preserved'] = all(sha(dst)==before[str(src)]
            for src,dst in zip((database,predictions),copies))
        write_json(captures / 'scientific_acceptance.json', proof)
