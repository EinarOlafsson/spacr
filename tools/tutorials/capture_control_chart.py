"""Record QC's real Control Charts fold, with a disclosed synthetic CSV."""
from pathlib import Path
import tempfile
import time

from control_chart_evidence import FIXTURE, digest, expected_points, verify_points, verify_export


def record_control_chart(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QPushButton
    from shiboken6 import isValid
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.control_chart import ControlChartScreen

    deadline = time.monotonic() + timeout
    output = Path(captures) / 'scientific_acceptance.json'
    proof = {'lesson': '51_control_charts', 'accepted': False, 'published': False,
             'scope': 'Software workflow using a disclosed synthetic campaign only',
             'synthetic_inputs': True, 'biological_validation': False,
             'acquired_test_data': False, 'model_run': False, 'ai_request_sent': False,
             'download_control_available': False, 'app_key': 'control_chart',
             'host_app_key': 'qc_dashboard', 'states': {}}
    before = digest(FIXTURE)
    proof['source'] = {'path': str(FIXTURE), 'sha256': before, 'rows': 270,
                       'plates': 30, 'wells_per_control_per_plate': 3,
                       'generator': 'control_chart_evidence.generate; NumPy default_rng seed 11',
                       'origin': 'Adapted from tests/qt/test_control_chart_screen.py:campaign'}
    panel = None
    jobs = []

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded Control Charts capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise RuntimeError('The requested Control Charts control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.15)

    def idle():
        settle(.2)
        while panel.is_busy() or panel.active_jobs():
            tick()
            settle(.1)
        settle(.4)
        tick()

    def fill(widget, text):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(text))

    def choose(widget, index):
        if index < 0:
            raise ValueError('Missing requested Control Charts dropdown option')
        if widget.currentIndex() != index:
            click(widget)
            view = widget.view()
            if not view.isVisible():
                raise ValueError('The actual dropdown did not open')
            QTest.keyClick(view, Qt.Key_Home)
            for _ in range(index):
                QTest.keyClick(view, Qt.Key_Down)
            QTest.keyClick(view, Qt.Key_Return)
            idle()
        if widget.currentIndex() != index:
            raise ValueError('The dropdown selected a different option')

    def select_level(name):
        matches = [panel._levels.item(i) for i in range(panel._levels.count())
                   if panel._levels.item(i).text() == name]
        if len(matches) != 1:
            raise ValueError('The synthetic control is missing from the actual list')
        # This is a multi-selection list: a plain click toggles a level.
        # Explicitly untick the previous control, then tick the wanted one.
        for item in list(panel._levels.selectedItems()):
            if item.text() != name:
                QTest.mouseClick(panel._levels.viewport(), Qt.LeftButton,
                                 pos=panel._levels.visualItemRect(item).center())
                idle()
        if not matches[0].isSelected():
            QTest.mouseClick(panel._levels.viewport(), Qt.LeftButton,
                             pos=panel._levels.visualItemRect(matches[0]).center())
        idle()
        if panel._selected_levels() != (name,):
            raise ValueError('The actual list did not select exactly one control')

    def picker(button, path, name, save=False):
        accepted, errors = [], []
        opener, watchdog = QTimer(window), QTimer(window)
        opener.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The actual button did not open a file dialog')
                dialog.accepted.connect(lambda: accepted.append(True))
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(name)
                standard = QDialogButtonBox.Save if save else QDialogButtonBox.Open
                click(dialog.findChild(QDialogButtonBox).button(standard))
            except Exception as error:
                errors.append(str(error))
                if isinstance(dialog, QFileDialog) and isValid(dialog):
                    dialog.reject()

        def stalled():
            errors.append('Actual file picker timed out')
            dialog = app.activeModalWidget()
            if isinstance(dialog, QFileDialog):
                dialog.reject()

        opener.timeout.connect(handle)
        watchdog.timeout.connect(stalled)
        opener.start(400)
        watchdog.start(15000)
        try:
            click(button)
        finally:
            for timer in (opener, watchdog):
                timer.stop()
                timer.deleteLater()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'Actual file picker was not accepted')
        idle()

    def inspect(name, level='neg', baseline=20, limits_only=True):
        result = panel.result
        if result is None or result.estimator != 'subgroup_s':
            raise ValueError('No real X-bar/S result is displayed')
        expected = expected_points(FIXTURE, level, baseline)
        data = verify_points(result.points_frame().to_dict('records'), expected,
                             limits_only=limits_only)
        if result.control_levels != (level,) or len(result.baseline) != baseline:
            raise ValueError('The result does not match the selected control/baseline')
        axes = panel.canvas.figure.axes
        if len(axes) != 1 or not any(list(line.get_ydata()) == list(result.values)
                                    for line in axes[0].lines):
            raise ValueError('The actual plotted values do not match the checked result')
        if panel.report.toPlainText() != result.report():
            raise ValueError('The actual text report is stale')
        if panel.violations.rowCount() != len(result.violations):
            raise ValueError('The actual rule table has the wrong number of rows')
        data.update(centre=result.centre, sigma_within=result.sigma_within,
                    sigma_of_plotted_mean=result.sigma,
                    report=panel.report.toPlainText(), rules=list(result.rules),
                    flagged_plates=[p for p, f in zip(result.plates, result.flagged) if f],
                    points=result.points_frame().to_dict('records'),
                    visible_rule_rows=panel.violations.rowCount())
        proof['states'][name] = data
        capture(name)
        return expected

    def show_choices(widget, name):
        """Expose full labels using the real popup, not a widened fake field."""
        before = widget.currentIndex()
        click(widget)
        if not widget.view().isVisible():
            raise ValueError('The real choices popup did not open')
        proof.setdefault('dropdowns', {})[name] = [widget.itemText(i)
                                                  for i in range(widget.count())]
        # Qt combo popups are separate QWidget windows, not QDialogs/QMenus;
        # a parent-window grab omits them. Capture this private desktop.
        capture(name, desktop=True)
        QTest.keyClick(widget.view(), Qt.Key_Escape)
        idle()
        if widget.currentIndex() != before:
            raise ValueError('Showing choices unexpectedly changed the selection')

    try:
        folds = [w for w in screen.findChildren(FoldButton)
                 if w.app_key == 'control_chart' and w.isVisible()]
        if len(folds) != 1:
            raise ValueError('QC has no unique visible Control Charts fold')
        capture('01_qc_host')
        click(folds[0])
        panels = [w for w in window.findChildren(ControlChartScreen) if w.isVisible()]
        if len(panels) != 1:
            raise ValueError('The actual Control Charts fold did not open')
        panel = panels[0]
        if not panel._jobs._threaded:
            raise ValueError('The real screen is not using threaded chart jobs')
        panel._jobs.job_finished.connect(lambda ok: jobs.append(bool(ok)))
        buttons = {w.text(): w for w in panel.findChildren(QPushButton) if w.isVisible()}
        if 'Load test data' in buttons:
            raise ValueError('A new downloadable example exists; inspect it before using synthetic data')
        capture('02_control_charts')
        picker(buttons['Load table…'], FIXTURE, '03_load_synthetic_csv')
        if len(panel._frame) != 270 or panel._path != str(FIXTURE):
            raise ValueError('The real CSV reader loaded a different campaign')
        proof['initial_unfiltered_report'] = panel.report.toPlainText()
        capture('04_check_column_guesses')
        for widget, text in ((panel._plate, 'plateID'), (panel._order, 'run_date'),
                             (panel._value, 'signal'), (panel._control_column, 'well_type')):
            choose(widget, widget.findText(text))
        select_level('neg')
        inspect('05_negative_control_default_rules', limits_only=False)
        show_choices(panel._estimator, '05a_sigma_choices')
        show_choices(panel._rules, '05b_rule_choices')
        choose(panel._rules, 2)
        inspect('06_limits_only')
        select_level('pos')
        inspect('07_positive_control', level='pos')
        select_level('neg')
        inspect('08_negative_restored')
        fill(panel._baseline, '12')
        QTest.keyClick(panel._baseline, Qt.Key_Return)
        idle()
        inspect('09_shorter_baseline', baseline=12)
        fill(panel._baseline, '20')
        QTest.keyClick(panel._baseline, Qt.Key_Return)
        idle()
        expected = inspect('10_baseline_restored')
        parent = Path(stage) / 'control_chart_runs'
        parent.mkdir(exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix='synthetic-', dir=parent))
        exported = private / 'synthetic_negative_control_points.csv'
        picker(buttons['Export points…'], exported, '11_actual_export_picker', save=True)
        proof['export'] = verify_export(exported, expected)
        capture('12_exported')
        proof['accepted'] = True
    except Exception as error:
        proof['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        proof['source_unchanged'] = digest(FIXTURE) == before
        proof['jobs_finished'] = jobs
        if panel is not None:
            until = time.monotonic() + 20
            while (panel.is_busy() or panel.active_jobs()) and time.monotonic() < until:
                settle(.1)
            proof['workers'] = {'busy': panel.is_busy(), 'active_jobs': panel.active_jobs()}
            if panel.is_busy() or panel.active_jobs():
                proof['accepted'] = False
        if not proof['source_unchanged']:
            proof['accepted'] = False
        write_json(output, proof)
