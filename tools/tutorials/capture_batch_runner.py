"""Exercise the current Batch Runner with real subprocesses and real pickers."""
from __future__ import annotations

import json
from pathlib import Path
import time

from batch_data import digest, prepare, verify_outputs


def record_batch(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit
    from spacr import batch as bt

    deadline = time.monotonic() + timeout
    inputs = prepare(stage)
    evidence = Path(captures) / 'scientific_acceptance.json'
    proof = dict(lesson='37_batch', accepted=False, published=False, inputs=inputs,
                 app_source_modified=False, injected_runner=False,
                 model_run=False, ai_request_sent=False)
    write_json(evidence, proof)
    screen = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Batch Runner recording timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Requested Batch control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.08)

    def wait_for(predicate):
        while not predicate():
            tick()
            settle(.03)

    def type_text(widget, text):
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClicks(widget, text)
        if widget.text() != text:
            raise ValueError('The actual editor did not receive the intended value')

    def picker(button, path, frame, save=False):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('The genuine file picker did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                type_text(dialog.findChild(QLineEdit, 'fileNameEdit'), str(path))
                capture(frame)
                key = QDialogButtonBox.Save if save else QDialogButtonBox.Open
                click(dialog.findChild(QDialogButtonBox).button(key))
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        def stalled():
            errors.append('The actual file picker timed out')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        timer.timeout.connect(handle)
        watchdog.timeout.connect(stalled)
        timer.start(300)
        watchdog.start(15000)
        try:
            click(button)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'The file picker was not accepted')

    def choose(combo, value):
        index = combo.findData(value)
        if index < 0:
            raise ValueError(f'Missing native dropdown choice {value}')
        click(combo)
        view = combo.view()
        QTest.keyClick(view, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(view, Qt.Key_Down)
        QTest.keyClick(view, Qt.Key_Return)
        settle(.2)
        if combo.currentData() != value:
            raise ValueError('Native dropdown selected the wrong value')

    def select(job_id):
        row = next(i for i in range(screen._table.rowCount())
                   if screen._table.item(i, 1).text() == job_id)
        cell = screen._table.item(row, 1)
        screen._table.scrollToItem(cell)
        QTest.mouseClick(screen._table.viewport(), Qt.LeftButton,
                         pos=screen._table.visualItemRect(cell).center())
        settle(.2)
        if screen.selected_job() is None or screen.selected_job().id != job_id:
            raise ValueError('Wrong actual table selection')

    def record(frame):
        proof.setdefault('states', {})[frame] = {
            'order': screen.queue().ids,
            'jobs': [{'id': j.id, 'status': j.status, 'depends_on': j.depends_on,
                      'overrides': j.override_args, 'log': j.log_path,
                      'started': j.started, 'finished': j.finished,
                      'run_status': j.run_status} for j in screen.queue()],
            'busy': screen.is_busy(), 'run_enabled': screen._btn_run.isEnabled(),
            'status_text': screen.status_text(), 'problems': screen.problems_text()}
        capture(frame)

    try:
        action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = action.menu()
        choice = [a for a in menu.actions() if a.text().replace('&', '') == 'Batch runner']
        if len(choice) != 1:
            raise ValueError('No unique Help -> Batch runner route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(action).center())
        settle(.2)
        capture('01_help_batch')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choice[0]).center())
        wait_for(lambda: window._screens.get('batch') is not None)
        screen = window._screens['batch']
        settle(.6)
        if not screen._threaded or screen._runner is not None or len(screen.queue()):
            raise ValueError('The real threaded, empty Batch Runner is required')
        proof['normal_threaded_runner'] = True
        header = screen._table.horizontalHeader()
        for col, width in enumerate((65, 270, 190, 550, 270, 220, 220)):
            header.resizeSection(col, width)
        record('02_empty_batch')
        for job in inputs['jobs']:
            number = job['number']
            choose(screen._module_combo, 'convert')
            picker(screen._btn_pick, job['settings_file'], f'03_choose_settings_{number}')
            type_text(screen._label_edit, f'field {number}: four real channels')
            type_text(screen._depends_edit, 'convert-1' if number == 2 else '')
            type_text(screen._overrides_edit, 'plate_naming=name' if number == 2 else '')
            capture(f'04_configure_job_{number}')
            click(screen._btn_add)
            if screen.queue().ids != [f'convert-{i}' for i in range(1, number + 1)] or screen.has_errors():
                raise ValueError('The actual job did not validate: ' + screen.problems_text())
            resolved = bt.resolve_job_settings(screen.queue().jobs[-1])
            expected = {**job['settings'], 'plate_naming': 'name' if number == 2 else 'index'}
            if any(resolved.get(k) != v for k, v in expected.items()):
                raise ValueError('Saved settings or the native override changed')
            record(f'05_added_job_{number}')
        select('convert-2')
        click(screen._btn_up)
        if not screen.has_errors() or screen._btn_run.isEnabled():
            raise ValueError('A later prerequisite failed to block the run')
        record('06_invalid_dependency_order')
        # The stock problem pane is height-limited. Use its real scrollbar
        # rather than changing application minimums or hiding the error text.
        bar = screen._problems_view.verticalScrollBar()
        before_scroll = bar.value()
        QTest.keyClick(bar, Qt.Key_End)
        settle(.2)
        if bar.maximum() <= 0 or bar.value() != bar.maximum():
            raise ValueError('The actual error scrollbar did not reach its last line')
        proof['error_scrollbar'] = {'before': before_scroll, 'after': bar.value(),
                                    'maximum': bar.maximum()}
        record('06b_dependency_error_scrolled')
        click(screen._btn_down)
        if screen.has_errors() or screen.queue().ids != ['convert-1', 'convert-2', 'convert-3']:
            raise ValueError('Restoring valid dependency order did not clear the error')
        select('convert-3')
        click(screen._btn_up)
        if screen.queue().ids != ['convert-1', 'convert-3', 'convert-2']:
            raise ValueError('Native move did not preserve the intended order')
        click(screen._btn_validate)
        if screen.has_errors() or screen.problems_text():
            raise ValueError('The restored real queue is invalid')
        record('07_valid_reordered_queue')
        choose(screen._on_error_combo, 'stop')
        record('08_stop_policy')
        choose(screen._on_error_combo, 'continue')
        record('09_continue_policy')
        queue_path = Path(inputs['root']) / 'real_conversion_queue.json'
        picker(screen._btn_save, queue_path, '10_save_queue_picker', save=True)
        before = queue_path.read_bytes()
        select('convert-3')
        click(screen._btn_remove)
        if len(screen.queue()) != 2:
            raise ValueError('Remove did not remove the selected private pending row')
        picker(screen._btn_load, queue_path, '11_load_queue_picker')
        if queue_path.read_bytes() != before or screen.queue().ids != ['convert-1', 'convert-3', 'convert-2']:
            raise ValueError('Save/load did not restore the exact queued plan')
        record('12_restored_queue')
        proof['saved_queue'] = str(queue_path)
        proof['saved_plan_sha256'] = digest(queue_path)
        click(screen._btn_run)
        wait_for(lambda: any(j.status == bt.STATUS_RUNNING for j in screen.queue()))
        record('13_real_subprocess_running')
        click(screen._btn_stop)
        wait_for(lambda: not screen.is_busy() and screen.active_jobs() == 0)
        if [j.status for j in screen.queue()] != [bt.STATUS_SUCCESS, bt.STATUS_NOT_RUN, bt.STATUS_NOT_RUN]:
            raise ValueError('Controlled Stop did not settle only the active real job: ' + screen.status_text())
        record('14_stopped_between_jobs')
        first_times = (screen.queue().jobs[0].started, screen.queue().jobs[0].finished)
        click(screen._btn_run)
        wait_for(lambda: not screen.is_busy() and screen.active_jobs() == 0)
        if screen.result() is None or not screen.result().ok or any(j.status != bt.STATUS_SUCCESS for j in screen.queue()):
            raise ValueError('The real subprocess queue did not complete: ' + screen.status_text())
        if (screen.queue().jobs[0].started, screen.queue().jobs[0].finished) != first_times:
            raise ValueError('Continuing repeated the first successful job')
        record('15_completed_real_queue')
        proof['outputs'] = [verify_outputs(job) for job in inputs['jobs']]
        proof['saved_final'] = json.loads(queue_path.read_text())
        saved = bt.load_queue(queue_path)
        if saved.ids != screen.queue().ids or any(j.status != bt.STATUS_SUCCESS for j in saved):
            raise ValueError('The persisted final queue differs from the successful GUI')
        for job in screen.queue():
            select(job.id)
            if not screen.log_text().strip() or 'exit code 0' not in Path(job.log_path).read_text():
                raise ValueError('A successful row lacks the real subprocess log')
            record('16_log_' + job.id)
        proof['first_job_not_repeated'] = True
        proof['accepted'] = True
    except Exception as error:
        proof['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        proof['original_unchanged'] = all(digest(row['original']) == row['sha256']
            and digest(row['source']) == row['sha256'] for job in inputs['jobs'] for row in job['records'])
        if screen is not None:
            proof['worker_running'] = screen.is_busy() or screen.active_jobs() != 0
            if proof['worker_running']:
                proof['accepted'] = False
        if not proof['original_unchanged']:
            proof['accepted'] = False
        write_json(evidence, proof)
