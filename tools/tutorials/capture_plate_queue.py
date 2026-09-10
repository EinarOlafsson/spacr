"""Exercise Plate Queue with real isolated Recruitment reruns, not a stub.

Two copies of the SAME four-field dataset demonstrate sequential execution,
not two independent experiments. Image previews are not rerun here. The normal
pipeline writes its numeric exports. It creates plots, but this Queue runner
does not route them to the module's figure bridge or save them; no such GUI
plot/export is claimed. Agg matches the ordinary app entry point's backend.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_frozen_rows(database, tables):
    """Pin every original field value and identity, independent of SQLite headers."""
    from recruitment_data import IDENTITY, _row_hash, _rows_hash
    if set(tables) != {'cell', 'nucleus', 'pathogen', 'cytoplasm'}:
        raise ValueError('All four frozen measurement tables are required')
    summaries = {}
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        for name, expected in tables.items():
            if name not in ('cell', 'nucleus', 'pathogen', 'cytoplasm'):
                raise ValueError('Unexpected frozen measurement table')
            cursor = connection.execute('SELECT * FROM "' + name + '"')
            columns = [x[0] for x in cursor.description]
            hashes = {}
            for row in cursor:
                values = dict(zip(columns, row))
                key = tuple(values[k] for k in IDENTITY)
                if key in hashes:
                    raise ValueError('Duplicate source object identity')
                hashes[key] = _row_hash(row)
            if len(hashes) != expected['row_count'] or _rows_hash(hashes) != expected['rows_sha256']:
                raise ValueError('The frozen source rows changed')
            summaries[name] = {'row_count': len(hashes), 'rows_sha256': _rows_hash(hashes)}
    return summaries


def prepare_replays(stage):
    """Copy the accepted small measurement subset; never copy generated results."""
    original_capture = Path(stage) / 'captures/recruitment_release'
    accepted = json.loads((original_capture / 'scientific_acceptance.json').read_text())
    manifest = json.loads((original_capture / 'input_manifest.json').read_text())
    settings = json.loads((original_capture / 'batch_settings.json').read_text())
    if not accepted['accepted']:
        raise ValueError('The prerequisite real Recruitment example is not accepted')
    source = Path(manifest['destination']) / 'measurements/measurements.db'
    if source.stat().st_size > 64 * 1024**2:
        raise ValueError('The accepted small measurement subset grew')
    wal = Path(str(source) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('The source database has uncheckpointed changes')
    frozen = verify_frozen_rows(source, manifest['tables'])
    expected = digest(source)
    parent = Path(stage) / 'plate_queue_runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='real-replays-', dir=parent))
    projects = []
    for number in (1, 2):
        project = root / f'replay_{number}'
        (project / 'measurements').mkdir(parents=True)
        (project / 'settings').mkdir()
        target = project / 'measurements/measurements.db'
        shutil.copyfile(source, target)
        if digest(target) != expected or target.stat().st_ino == source.stat().st_ino:
            raise ValueError('The replay is not a byte-identical independent copy')
        chosen = {**settings, 'src': str(project)}
        path = project / 'settings/recruitment_settings.csv'
        with path.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('Key', 'Value'))
            writer.writerows((key, str(value)) for key, value in chosen.items())
        projects.append({'project': str(project), 'database': str(target),
                         'settings': chosen, 'settings_csv': str(path)})
    return {'source': str(source), 'sha256': expected, 'root': str(root),
            'pre_analysis_database_sha256': manifest['subset_database_sha256'],
            'frozen_measurement_rows': frozen,
            'projects': projects, 'original_unchanged': digest(source) == expected,
            'independent_experiments': False, 'same_data_replayed_twice': True}


def record_queue(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt.plate_queue import PlateQueue
    from recruitment_evidence import inspect_results

    deadline = time.monotonic() + timeout
    proof = {'lesson': '30_plate_queue', 'accepted': False, 'published': False,
             'pipeline_replaced': False, 'app_source_modified': False,
             'independent_experiments': False, 'same_real_data_replayed_twice': True,
             'ai_request_sent': False, 'model_run': False,
             'scope': 'Actual sequential jobs on two isolated copies of one accepted measurement subset'}
    evidence = Path(captures) / 'scientific_acceptance.json'
    inputs = prepare_replays(stage)
    proof['inputs'] = inputs
    write_json(evidence, proof)
    queue = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded Plate Queue capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('The requested queue control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.08)

    def wait_for(predicate):
        while not predicate():
            tick()
            settle(.03)

    def state():
        return [(item.id, item.app_key, item.settings['src'], item.status.value)
                for item in queue.queue().items()]

    def record(name):
        expected = state()
        # Wait for the real queued signal to update the displayed table.
        def visible_state():
            return [tuple(queue._table.item(row, col).text() for col in (0, 1, 2, 3))
                    for row in range(queue._table.rowCount())]
        wait_for(lambda: [tuple(row) for row in visible_state()] == expected)
        proof.setdefault('states', {})[name] = expected
        capture(name)

    def open_queue():
        action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Plate queue']
        if len(choices) != 1:
            raise ValueError('No unique current Help -> Plate queue route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(action).center())
        settle(.2)
        capture('01_help_queue')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('queue') is not None)
        settle(.7)
        return window._screens['queue']

    def drop(project):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(project))])
        point = QPoint(50, 100)
        enter = QDragEnterEvent(point, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        app.sendEvent(queue, enter)
        if not enter.isAccepted():
            raise ValueError('The actual queue drop target refused the drag')
        event = QDropEvent(QPointF(point), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        app.sendEvent(queue, event)
        if not event.isAccepted():
            raise ValueError('The actual queue drop event was refused')

    try:
        queue = open_queue()
        private_state = Path(stage) / 'queue_state' / Path(captures).name
        actual = Path.home() / '.spacr/queue.json'
        # Compare the backing directory inode across the bind mount before
        # allowing any operation that could change persistent user data.
        if private_state.stat().st_ino != actual.parent.stat().st_ino or len(queue.queue()):
            raise ValueError('The queue is not a fresh, bind-mounted private state directory')
        proof['private_state'] = str(private_state)
        header = queue._table.horizontalHeader()
        proof['native_column_widths_before'] = [header.sectionSize(i) for i in range(6)]
        # Resize existing, user-adjustable header sections only. No table
        # values, application minimum sizes or size policies are changed.
        for column, width in enumerate((220, 260, 2000, 200, 180)):
            header.resizeSection(column, width)
        proof['native_column_widths_after'] = [header.sectionSize(i) for i in range(6)]
        proof['visible_buttons'] = [w.text() for w in queue.findChildren(QAbstractButton)
                                    if w.isVisible()]
        record('02_empty_queue')
        for number, project in enumerate(inputs['projects'], 1):
            drop(project['project'])
            wait_for(lambda: len(queue.queue()) == number)
            item = queue.queue().items()[-1]
            if item.app_key != 'recruitment' or item.settings != project['settings']:
                raise ValueError('The dropped snapshot changed its module or settings')
            # Pin legibility of identifiers, module and complete status words.
            from PySide6.QtGui import QFontMetrics
            metrics = QFontMetrics(queue._table.font())
            for col, text in ((0, item.id), (1, item.app_key), (3, 'success')):
                if metrics.horizontalAdvance(text) + 20 > header.sectionSize(col):
                    raise ValueError('The actual queue identity or status column is too narrow')
            record(f'03_added_replay_{number}')
        expected = state()
        if [(i.id, i.app_key, i.settings['src'], i.status.value)
                for i in PlateQueue(path=actual)] != expected:
            raise ValueError('The actual persisted queue did not reload both items')
        proof['persistence_before_run'] = True
        click(queue._btn_run)
        wait_for(lambda: any(i.status.value == 'running' for i in queue.queue()))
        record('04_first_job_running')
        click(queue._btn_stop)
        record('05_stop_requested')
        wait_for(lambda: queue._runner is None or not queue._runner.isRunning())
        settle(.4)
        if [i.status.value for i in queue.queue()] != ['success', 'queued']:
            raise ValueError('Stop did not finish the current item and preserve the next queued item')
        record('06_stopped_between_jobs')
        first = queue.queue().items()[0]
        first_times = (first.start_ts, first.end_ts)
        click(queue._btn_run)
        wait_for(lambda: queue._runner is None or not queue._runner.isRunning())
        settle(.4)
        if [i.status.value for i in queue.queue()] != ['success', 'success']:
            raise ValueError('Both real queue jobs did not finish successfully')
        if (first.start_ts, first.end_ts) != first_times:
            raise ValueError('Resuming repeated the already completed job')
        record('07_both_jobs_finished')
        proof['outputs'] = []
        for project in inputs['projects']:
            checked = inspect_results(Path(project['project']), project['settings'])
            if not checked['accepted']:
                raise ValueError('A success badge is not supported by independent numeric results')
            if digest(project['database']) != inputs['sha256']:
                raise ValueError('The measured source database was modified')
            proof['outputs'].append(checked)
        import matplotlib
        proof['matplotlib_backend'] = matplotlib.get_backend()
        proof['queue_figures_saved_or_shown'] = False
        proof['persisted_final'] = json.loads(actual.read_text())
        proof['first_job_not_repeated'] = True
        # These are disposable replay queue records only, not source data.
        # Preserve their JSON evidence before exercising Clear finished.
        click(queue._btn_clear)
        if len(queue.queue()) or len(PlateQueue(path=actual)):
            raise ValueError('Clear finished did not remove the private completed records')
        record('08_finished_records_cleared')
        if any(not Path(item['project']).exists() for item in inputs['projects']):
            raise ValueError('Clear finished removed a replay project')
        proof['accepted'] = True
    except Exception as error:
        proof['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        proof['original_unchanged'] = digest(inputs['source']) == inputs['sha256']
        if not proof['original_unchanged']:
            proof['accepted'] = False
        if queue is not None:
            proof['worker_running'] = bool(queue._runner and queue._runner.isRunning())
            if proof['worker_running']:
                proof['accepted'] = False
        write_json(evidence, proof)
