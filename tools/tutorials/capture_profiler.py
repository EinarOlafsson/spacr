"""Record native Prediction Profiler using an explicitly synthetic fitted OLS example."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

from profiler_example import digest, prepare, verify_curve


def record_profiler(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QDialogButtonBox, QLineEdit, QSplitter
    from spacr.qt.screens.profiler import ProfilerScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    parent = Path(stage)/'profiler_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='SYNTHETIC-OLS-', dir=parent))/'example'
    manifest = prepare(work)
    coefficients = work/'SYNTHETIC_OLS_coefficients.csv'
    before = {p.name: digest(p) for p in work.iterdir()}
    proof = dict(lesson='53_prediction_profiler', accepted=False, synthetic=True,
        biological_claim=False, application_modified=False, published=False,
        fitted_in_profiler=False, private_example=str(work), manifest=manifest, snapshots={})
    deadline = time.monotonic()+timeout
    panel = None

    def tick():
        if time.monotonic() >= deadline: raise TimeoutError('Native profiler capture timed out')
        settle(.1)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual profiler control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle():
        while panel.is_busy() or panel.active_jobs(): tick()
        if panel.last_error: raise ValueError(panel.last_error)

    def snapshot(name, variable, held, points):
        idle()
        curve = panel.curve().to_dict()
        checked = verify_curve(curve, manifest['coefficients'], variable=variable, held=held, points=points)
        if panel._canvas.curve() is not panel.curve() or len(panel._canvas.points()) != points:
            raise ValueError('Displayed canvas does not carry the checked curve')
        if panel._link.currentText() != 'identity' or panel.design().shape != (2,3):
            raise ValueError('The GUI must use its actual default-range identity-link file route')
        if 'No design matrix was supplied' not in panel._status.text():
            raise ValueError('The native default-range caveat is absent')
        proof['snapshots'][name] = dict(curve=curve, verification=checked,
            status=panel._status.text(), held=panel.held_values(),
            canvas_size=[panel._canvas.width(), panel._canvas.height()],
            canvas_points=panel._canvas.points())
        capture(name)

    def slider(name, key):
        control = panel._sliders[name]; click(control); QTest.keyClick(control, key); settle(.2)

    def select(variable):
        items = [panel._inputs.topLevelItem(i) for i in range(panel._inputs.topLevelItemCount())]
        item = next(i for i in items if i.data(0, Qt.UserRole) == variable)
        QTest.mouseClick(panel._inputs.viewport(), Qt.LeftButton,
                        pos=panel._inputs.visualItemRect(item).center()); settle(.2)
        if panel.variable() != variable: raise ValueError('The actual ranked-input selection did not change')

    try:
        buttons = [b for b in window.findChildren(QAbstractButton) if b.isVisible() and
                   (b.property('moduleAppKey') == 'regression' or b.property('navKey') == 'regression')]
        click(max(buttons, key=lambda b:b.width()*b.height())); host = window._screens['regression']
        capture('01_regression_host')
        folds = [b for b in host.findChildren(FoldButton) if b.isVisible() and b.app_key == 'profiler']
        if len(folds) != 1: raise ValueError('The real Regression -> Profiler fold is not unique')
        click(folds[0])
        panels = [p for p in window.findChildren(ProfilerScreen) if p.isVisible()]
        if len(panels) != 1: raise ValueError('The actual profiler did not open')
        panel = panels[0]
        dialog = panel.window()
        if dialog is not window:
            dialog.resize(3300,1650); dialog.move(500,180)
        settle(.3)
        capture('02_native_empty_profiler')
        errors, accepted = [], []

        def choose():
            picker = app.activeModalWidget()
            try:
                if not isinstance(picker, QFileDialog): raise ValueError('Actual coefficient picker missing')
                picker.resize(1900,1050); picker.move(900,400)
                picker.accepted.connect(lambda:accepted.append(True))
                edit = picker.findChild(QLineEdit,'fileNameEdit'); click(edit)
                QTest.keyClick(edit,Qt.Key_A,Qt.ControlModifier); QTest.keyClicks(edit,str(coefficients))
                capture('03_explicit_synthetic_coefficients_picker')
                box = picker.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1: raise ValueError('Actual picker accept button ambiguous')
                click(buttons[0])
            except Exception as exc:
                errors.append(str(exc))
                if picker is not None: picker.reject()
        timer = QTimer(window); timer.setSingleShot(True)
        timer.timeout.connect(lambda:app.activeModalWidget().reject() if app.activeModalWidget() else None)
        QTimer.singleShot(350,choose); timer.start(15000); click(panel._browse_button); timer.stop()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'Coefficient picker timed out')
        idle()
        if panel._path_edit.text() != str(coefficients): raise ValueError('The imported file differs')
        loaded = {str(k):float(v) for k,v in panel.model().params.items()}
        if any(abs(loaded[k]-v)>1e-12 for k,v in manifest['coefficients'].items()):
            raise ValueError('Imported coefficients differ from the saved fit')
        proof['loaded_coefficients'] = loaded
        splitter = panel.findChild(QSplitter)
        handle = splitter.handle(1)
        start = handle.rect().center(); finish = QPoint(start.x()+400,start.y())
        QTest.mousePress(handle,Qt.LeftButton,pos=start)
        QTest.mouseMove(handle,finish,delay=80); QTest.mouseRelease(handle,Qt.LeftButton,pos=finish); settle(.2)
        header = panel._inputs.header()
        for column in (0,1):
            start = header.sectionViewportPosition(column)
            edge = start+header.sectionSize(column)-1
            point = QPoint(edge,header.height()//2); target = QPoint(start+260,point.y())
            QTest.mousePress(header.viewport(),Qt.LeftButton,pos=point)
            QTest.mouseMove(header.viewport(),target,delay=80)
            QTest.mouseRelease(header.viewport(),Qt.LeftButton,pos=target); settle(.2)
        proof['ranking'] = [r.to_dict() for r in panel.ranked_inputs()]
        if [r.variable for r in panel.ranked_inputs()] != ['input_a','input_b']:
            raise ValueError('The native sensitivity order differs from the fitted slopes')
        snapshot('04_loaded_identity_model', 'input_a', {'input_b':.5}, 61)
        click(panel._link); capture('05_link_options_keep_identity')
        QTest.keyClick(panel._link,Qt.Key_Escape); settle(.2)
        slider('input_b',Qt.Key_End)
        snapshot('06_hold_b_at_one', 'input_a', {'input_b':1.}, 61)
        slider('input_b',Qt.Key_Home)
        snapshot('07_hold_b_at_zero', 'input_a', {'input_b':0.}, 61)
        click(panel._reset_button)
        snapshot('08_reset_midpoints', 'input_a', {'input_b':.5}, 61)
        select('input_b')
        snapshot('09_sweep_second_input', 'input_b', {'input_a':.5}, 61)
        click(panel._points); QTest.keyClick(panel._points,Qt.Key_A,Qt.ControlModifier)
        QTest.keyClicks(panel._points,'11'); QTest.keyClick(panel._points,Qt.Key_Tab); settle(.2)
        snapshot('10_eleven_grid_points', 'input_b', {'input_a':.5}, 11)
        slider('input_a',Qt.Key_End)
        snapshot('11_hold_a_at_one', 'input_b', {'input_a':1.}, 11)
        click(panel._reset_button)
        snapshot('12_restore_midpoint', 'input_b', {'input_a':.5}, 11)
        proof.update(accepted=True, scope='Synthetic fitted OLS import and native control semantics only',
                     checked_curve_values=sum(v['verification']['points'] for v in proof['snapshots'].values()))
    finally:
        if panel is not None:
            while panel.is_busy() or panel.active_jobs(): settle(.05)
        proof['input_bytes_preserved'] = before == {p.name:digest(p) for p in work.iterdir()}
        if not proof['input_bytes_preserved']: proof['accepted'] = False
        write_json(Path(captures)/'scientific_acceptance.json',proof)
