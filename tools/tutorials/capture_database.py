"""Refresh Database visuals without changing the retained eight sentences.

Only a new, byte-identical private database copy enters the actual GUI. The
independent checks below use stdlib SQLite, bounded pages and streamed CSV rows;
they do not import the application's query/export implementation or pandas.
No edit mode, backend replacement, direct result injection or training is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time


SOURCE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/'
              'tutorials/measurements/measurements.db')
SOURCE_SHA256 = '9028eabff2bab4d7ef5447e1870f7e0aa3879fa641497ae150ad31b8d2f97d5d'
ENGLISH_SHA256 = 'ecc2920e5657b8cedad33eb4bf090450238fb86c69281c8787e4c954fee98c6f'
IDENTITY = ('plateID', 'rowID', 'columnID', 'fieldID', 'object_label')
NARRATIONS = (
    'Database Browser inspects large spaCR databases in a bounded, read-only view.',
    'Choose a measurements database or its run folder and open it.',
    'Select a table after reviewing its purpose and read-only status.',
    'The preview loads a bounded page while reporting the complete row and column counts separately.',
    'Search columns to make wide feature tables readable without changing the underlying rows.',
    'Build a structured filter. Column names and operators are validated, and values are bound safely.',
    'The exact filtered count describes the full query even though only one page is drawn.',
    'Export the complete current filter and visible-column selection. The source database remains unchanged.',
)
SCENE_FRAMES = ('01_overview', '02_open', '03_tables', '04_cell_table',
                '05_column_search', '08_filter', '09_filtered', '10_export')


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _quiescent(path):
    for suffix in ('-wal', '-journal'):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError('A nonempty WAL/journal requires a quiescent, checkpointed source')


def file_bundle(path):
    """Fingerprint the main file AND existing sidecars without changing them."""
    path = Path(path).resolve(strict=True)
    _quiescent(path)
    result = {}
    for suffix in ('', '-wal', '-shm', '-journal'):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            before = _signature(candidate)
            digest = _digest(candidate)
            if _signature(candidate) != before:
                raise ValueError('Source changed while its fingerprint was read')
            result[str(candidate)] = {'sha256': digest, 'signature': list(before)}
    _quiescent(path)
    return result


def require_unchanged_source(source, before):
    if file_bundle(source) != before:
        raise ValueError('Source database or its sidecars changed')


def prepare_database_copy(source, destination, *, expected_sha256=None):
    """Copy a quiescent database into one nonexistent, private filename.

    Nonempty WAL/journal files are refused, never silently discarded. An
    immutable read-only inspection is safe only after this quiescence check.
    Failure can leave a partial PRIVATE copy for diagnosis; it is never accepted.
    """
    source = Path(source).resolve(strict=True)
    requested = Path(destination).absolute()
    if any(os.path.lexists(str(requested) + suffix)
           for suffix in ('', '-wal', '-shm', '-journal')):
        raise FileExistsError(f'Destination already exists: {requested}')
    destination = requested.resolve()
    if source.parent == destination.parent or source.parent in destination.parents:
        raise ValueError('The private copy must be outside the source directory')
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    before = file_bundle(source)
    original_hash = before[str(source)]['sha256']
    if expected_sha256 is not None and original_hash != expected_sha256:
        raise ValueError('The original database no longer matches the audited source hash')
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    if _digest(destination) != original_hash:
        raise ValueError('The private database is not a byte-identical copy')
    require_unchanged_source(source, before)
    return {'source': str(source), 'database': str(destination),
            'source_bundle': before, 'database_sha256': original_hash,
            'copy_is_byte_identical': True, 'source_opened_in_gui': False}


@contextmanager
def _readonly(path):
    path = Path(path).resolve(strict=True)
    _quiescent(path)
    # Never create/update the original's shared-memory sidecar during checks.
    con = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        con.execute('PRAGMA query_only=ON')
        con.execute('PRAGMA cache_size=-2048')
        con.execute('PRAGMA temp_store=FILE')
        yield con
    finally:
        con.close()


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _schema(con, table):
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    if table not in tables:
        raise ValueError('Requested table is not in the database schema')
    kind = con.execute('SELECT type FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
    if kind != 'table':
        raise ValueError('This bounded identity proof requires a real table')
    columns = [r[1] for r in con.execute(f'PRAGMA table_info({_quote(table)})')]
    if not set(IDENTITY + ('cell_area',)).issubset(columns):
        raise ValueError('The cell table is missing complete object identity or area columns')
    alias = next((name for name in ('_rowid_', 'rowid', 'oid')
                  if name not in {c.casefold() for c in columns}), None)
    if alias is None:
        raise ValueError('No unshadowed SQLite row identity is available')
    try:
        con.execute(f'SELECT {_quote(alias)} FROM {_quote(table)} LIMIT 1').fetchone()
        # Quoted unknown names can be string literals in SQLite; a WITHOUT
        # ROWID table must not pass that permissive behaviour.
        con.execute(f'SELECT {alias} FROM {_quote(table)} LIMIT 1').fetchone()
    except sqlite3.Error as error:
        raise ValueError('An ordinary rowid table is required') from error
    return tables, columns, alias


def _query(con, *, table='cell', threshold=None, columns=None, sort=None, limit=None):
    _, schema, alias = _schema(con, table)
    selected = schema if columns is None else list(columns)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(schema):
        raise ValueError('Export/preview columns must be unique actual schema columns')
    sql = f'SELECT {", ".join(map(_quote, selected))} FROM {_quote(table)}'
    params = []
    if threshold is not None:
        if type(threshold) not in (int, float) or not math.isfinite(threshold):
            raise ValueError('The numeric area threshold must be finite')
        sql += ' WHERE "cell_area" >= ?'
        params.append(threshold)
    order = _quote(alias) + ' ASC'
    if sort is not None:
        if (not isinstance(sort, tuple) or len(sort) != 2
                or sort[0] not in schema or type(sort[1]) is not bool):
            raise ValueError('Invalid whole-table sort request')
        order = f'{_quote(sort[0])} {"DESC" if sort[1] else "ASC"}, ' + order
    sql += ' ORDER BY ' + order
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Independent preview reads are limited to 1–1000 rows')
        sql += ' LIMIT ?'
        params.append(limit)
    return selected, con.execute(sql, params)


def inspect_database(path, *, table='cell', threshold=10000, max_rows=200000):
    """Read schema, exact counts and a single duplicate/null identity probe."""
    with _readonly(path) as con:
        tables, columns, alias = _schema(con, table)
        total = con.execute(f'SELECT COUNT(*) FROM {_quote(table)}').fetchone()[0]
        if not 0 < total <= max_rows:
            raise ValueError('The tutorial table exceeds its bounded row budget or is empty')
        _query(con, table=table, threshold=threshold, limit=1)
        filtered = con.execute(f'SELECT COUNT(*) FROM {_quote(table)} WHERE "cell_area" >= ?',
                               (threshold,)).fetchone()[0]
        keys = ','.join(map(_quote, IDENTITY))
        duplicate = con.execute(f'SELECT {keys} FROM {_quote(table)} GROUP BY {keys} '
                                'HAVING COUNT(*)>1 LIMIT 1').fetchone()
        missing = con.execute(f'SELECT 1 FROM {_quote(table)} WHERE ' + ' OR '.join(
            f'{_quote(key)} IS NULL' for key in IDENTITY) + ' LIMIT 1').fetchone()
        if duplicate is not None or missing is not None:
            raise ValueError('Full object identities are missing or duplicated')
        if not 0 < filtered < total:
            raise ValueError('The demonstrated numeric filter must retain a nonempty strict subset')
        return {'tables': tables, 'table': table, 'columns': columns, 'rows': total,
                'filtered_rows': filtered, 'filter_column': 'cell_area',
                'filter_operator': '>=', 'filter_value': threshold,
                'rowid_alias': alias, 'identity_columns': list(IDENTITY),
                'full_object_identities_unique': True}


def verify_preview(database, columns, rows, total_count, *, threshold=None, sort=None,
                   table='cell', max_loaded=1000):
    """Compare the bounded GUI page, including complete identities and values."""
    if not rows or not 0 < len(rows) <= max_loaded <= 1000:
        raise ValueError('Preview is empty or exceeds the bounded page budget')
    with _readonly(database) as con:
        _, schema, _ = _schema(con, table)
        if list(columns) != schema:
            raise ValueError('Preview schema changed or concealed underlying columns')
        _, cursor = _query(con, table=table, threshold=threshold, sort=sort, limit=len(rows))
        expected = cursor.fetchall()  # <=1000 rows, never the full large table.
        if [tuple(row) for row in rows] != expected:
            raise ValueError('Preview identities, values or whole-table order differ from SQLite')
        sql = f'SELECT COUNT(*) FROM {_quote(table)}'
        params = ()
        if threshold is not None:
            sql += ' WHERE "cell_area" >= ?'
            params = (threshold,)
        count = con.execute(sql, params).fetchone()[0]
        if type(total_count) is not int or total_count != count:
            raise ValueError('The displayed complete count differs from SQLite')
    return {'loaded_rows': len(rows), 'exact_total_rows': count,
            'columns': len(schema), 'identities_and_values_exact': True,
            'sort': list(sort) if sort is not None else None}


def verify_export(database, exported, columns, expected_count, *, threshold=10000,
                  table='cell', max_rows=200000):
    """Stream all exported rows and full keys against an independent query."""
    if not set(IDENTITY).issubset(columns):
        raise ValueError('This export proof requires every full identity column')
    if type(expected_count) is not int or not 0 < expected_count <= max_rows:
        raise ValueError('Invalid bounded export row count')
    count, identities = 0, hashlib.sha256()
    with _readonly(database) as con, Path(exported).open(encoding='utf-8', newline='') as stream:
        selected, cursor = _query(con, table=table, threshold=threshold, columns=columns)
        reader = csv.reader(stream)
        if next(reader, None) != selected:
            raise ValueError('CSV header differs from the actual visible-column selection')
        indices = [selected.index(key) for key in IDENTITY]
        for row in reader:
            count += 1
            if count > max_rows:
                raise ValueError('CSV exceeds the bounded export row budget')
            expected = cursor.fetchone()
            if expected is None or row != ['' if value is None else str(value) for value in expected]:
                raise ValueError('CSV rows, object identities or measurement values differ from SQLite')
            identities.update(json.dumps([expected[i] for i in indices],
                                         ensure_ascii=False, separators=(',', ':')).encode())
            identities.update(b'\n')
        if cursor.fetchone() is not None or count != expected_count:
            raise ValueError('CSV does not contain the complete filtered result')
    return {'rows': count, 'columns': len(selected), 'all_values_exact': True,
            'full_object_identities_exact': True, 'ordered_identity_sha256': identities.hexdigest(),
            'sha256': _digest(exported), 'path': str(exported)}


def retained_scene_mapping(catalog):
    lesson = next(item for item in catalog['lessons'] if item['id'] == '34_database')
    canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if canonical != ENGLISH_SHA256 or tuple(s['narration'] for s in lesson['scenes']) != NARRATIONS:
        raise ValueError('The retained eight-sentence lesson changed; review before capture')
    return {'english_sha256': canonical, 'narration_changed': False,
            'existing_voices_reusable': True,
            'scenes': [{'scene': i, 'visual': frame, 'narration': text,
                        'speech_text': scene['speech_text']}
                       for i, (frame, text, scene) in enumerate(
                           zip(SCENE_FRAMES, NARRATIONS, lesson['scenes']), 1)]}


def _retire_database_jobs(screen, settle):
    """Pump real completion signals, including queued and winding-down jobs.

    An expired capture deadline is not a safe reason to destroy a live QThread.
    The caller's outer process watchdog, not forced thread termination or this
    possibly expired deadline, bounds a genuinely wedged database worker.
    """
    polls = 0
    if screen is None:
        return {'event_processing_polls': 0, 'active_jobs': 0, 'queued_jobs': 0,
                'busy': False, 'screen_opened': False, 'workers_forcibly_stopped': False}
    while screen.is_busy() or screen.active_jobs() or screen.queued_jobs():
        settle(.05)
        polls += 1
    return {'event_processing_polls': polls, 'active_jobs': screen.active_jobs(),
            'queued_jobs': screen.queued_jobs(), 'busy': screen.is_busy(),
            'screen_opened': True, 'workers_forcibly_stopped': False}


@contextmanager
def _database_job_lifecycle(screen_lookup, settle, report):
    """Retire jobs on every exit and retain the original capture exception."""
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            retirement = _retire_database_jobs(screen_lookup(), settle)
            report(retirement, failure)
        except BaseException as cleanup_error:
            if failure is None:
                raise
            failure.add_note(f'Database cleanup also failed: {cleanup_error!r}')


def _visible_header_section(columns, target):
    """Resolve a model-visible index, never the underlying schema index.

    PreviewModel.columnCount/headerData use its _visible projection. Column
    search resets that model; it does not hide schema-indexed QHeaderView
    sections. A unique cell_area search therefore has logical section zero.
    """
    if list(columns).count(target) != 1:
        raise ValueError('The requested visible header is absent or ambiguous')
    return list(columns).index(target)


def _native_replace_text(widget, value, qtest, qt):
    """Replace a visible editor using only ordinary keyboard gestures."""
    widget.setFocus()
    qtest.keyClick(widget, qt.Key_A, qt.ControlModifier)
    # keyClicks('') emits no event; it cannot delete the selected old text.
    qtest.keyClick(widget, qt.Key_Backspace)
    qtest.keyClicks(widget, str(value))
    qtest.keyClick(widget, qt.Key_Tab)


def _column_view_snapshot(search_text, visible_columns, columns, rows):
    """Bounded failure diagnostics: displayed names and full available keys."""
    if len(rows) > 1000:
        raise ValueError('Column diagnostic exceeds the bounded page budget')
    keys = [key for key in IDENTITY if key in columns]
    indices = [list(columns).index(key) for key in keys]
    return {'column_search_text': search_text, 'visible_columns': list(visible_columns),
            'loaded_rows': len(rows), 'identity_columns': keys,
            'row_identities': [[row[index] for index in indices] for row in rows],
            'snapshot_time': 'after actual database jobs retired'}


def record_database(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    """Record authentic interactions, then retire real jobs before returning."""
    def report(retirement, failure):
        evidence = {**retirement, 'capture_error': None if failure is None else str(failure)}
        write_json(Path(captures) / 'job_retirement.json', evidence)
        if failure is not None:
            write_json(Path(captures) / 'scientific_acceptance.json', {
                'accepted': False, 'reason': str(failure), 'error_cleanup': evidence,
                'published': False})
            current = window._screens.get('db_browser')
            if current is not None:
                write_json(Path(captures) / 'column_view_error.json', _column_view_snapshot(
                    current._col_search.text(), current.visible_columns(),
                    current.preview_columns(), current.preview_rows()))

    with _database_job_lifecycle(lambda: window._screens.get('db_browser'), settle, report):
        return _record_database(app, window, screen, stage, captures, capture,
                                settle, write_json, timeout)


def _record_database(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    """Use genuine Help, table/header controls and Open/Save file pickers."""
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QLineEdit
    from shiboken6 import isValid

    stage, captures = Path(stage), Path(captures)
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': False, 'reason': 'Actual Database capture not independently verified',
        'published': False})
    catalog = Path(__file__).resolve().parents[2] / 'docs/source/_extra/tutorials/catalog/lessons_en.json'
    mapping = retained_scene_mapping(json.loads(catalog.read_text()))
    write_json(captures / 'scene_mapping.json', mapping)

    def click(widget):
        if widget is None or not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The actual requested control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton)
        settle(.2)

    def fill(widget, text):
        if widget is None or not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The actual editor is not usable')
        _native_replace_text(widget, text, QTest, Qt)
        settle(.2)

    def choose(button, path, frame, *, save=False):
        accepted, errors, owned = [], [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            if isinstance(dialog, QDialog):
                owned.append(dialog)
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Expected the actual Qt file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                bounds = window.geometry()
                dialog.resize(min(1800, bounds.width() - 100), min(1200, bounds.height() - 100))
                dialog.move(bounds.center() - dialog.rect().center())
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                if box is None:
                    raise RuntimeError('The actual picker has no standard action buttons')
                click(box.button(QDialogButtonBox.Save if save else QDialogButtonBox.Open))
            except Exception as error:
                errors.append(str(error))
                if isinstance(dialog, QDialog) and isValid(dialog):
                    dialog.reject()

        def stalled():
            errors.append('The actual file picker did not complete')
            for dialog in owned:
                if isValid(dialog) and dialog.isVisible():
                    dialog.reject()

        timer.timeout.connect(handle)
        watchdog.timeout.connect(stalled)
        timer.start(400)
        watchdog.start(15000)
        try:
            click(button)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'The file picker was not accepted')

    # The parent may have instantiated the screen, but the recorded entry still
    # goes through Help; never substitute a call to the navigation slot here.
    help_actions = [a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help']
    if len(help_actions) != 1 or help_actions[0].menu() is None:
        raise RuntimeError('Expected one actual Help menu')
    menu = help_actions[0].menu()
    choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Database browser']
    if len(choices) != 1 or not choices[0].isEnabled():
        raise RuntimeError('The actual Help menu lacks Database browser')
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_actions[0]).center())
    settle(.3)
    if not menu.isVisible():
        raise RuntimeError('Help did not open')
    capture('00a_help_database_menu')
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
    deadline = time.monotonic() + timeout
    while window._screens.get('db_browser') is None:
        if time.monotonic() > deadline:
            raise TimeoutError('Database Browser did not open through Help')
        settle(.1)
    screen = window._screens['db_browser']
    settle(1)
    if not screen.isVisible() or not screen._threaded:
        raise RuntimeError('The visible Database Browser must use its real threaded workers')
    capture('01_overview')

    parent = stage / 'database_runs'
    parent.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='example-', dir=parent))
    copied = prepare_database_copy(SOURCE, run / 'measurements.db', expected_sha256=SOURCE_SHA256)
    database = Path(copied['database'])
    facts = inspect_database(database)
    if 'cell_channel_1_mean_intensity' not in facts['columns']:
        raise RuntimeError('The audited intensity feature is missing')
    write_json(captures / 'input_manifest.json', {**copied, 'inspection': facts,
               'area_units': 'pixels', 'biological_validation_claimed': False})
    checks, job_results = {}, []
    screen.job_finished.connect(job_results.append)

    def wait_jobs(label):
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs() or screen.queued_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError(f'{label}: {screen.status_text()}')
            settle(.1)
        settle(.3)
        if screen.last_error or any(result is not True for result in job_results):
            raise RuntimeError(f'{label}: actual query/export failed: {screen.last_error}')

    def prove(label, *, threshold=None, sort=None):
        wait_jobs(label)
        if (screen.edit_mode_enabled() or screen._edit_check.isChecked()
                or screen._view.editTriggers() != QAbstractItemView.NoEditTriggers
                or screen.row_count_is_estimate() or screen.current_table() != 'cell'
                or Path(screen.database_path()).resolve() != database):
            raise RuntimeError('Read-only private-table state or exact count was lost')
        if screen._linked_hidden:
            raise RuntimeError('A shared filter hides rows from the visible tutorial page')
        if screen._sort != sort:
            raise RuntimeError('The visible SQL sort differs from the requested state')
        proof = verify_preview(database, screen.preview_columns(), screen.preview_rows(),
                               screen.row_count(), threshold=threshold, sort=sort,
                               max_loaded=screen.page_size())
        if proof['loaded_rows'] >= proof['exact_total_rows']:
            raise RuntimeError('The narrated bounded preview must not load the full result')
        if screen._view.viewport().height() < 500:
            raise RuntimeError('The actual table viewport is not readable at this layout')
        proof.update({'status': screen.status_text(), 'rows_label': screen._rows_label.text(),
                      'sort_note': screen._sort_note.text(),
                      'visible_columns': screen.visible_columns()})
        checks[label] = proof
        write_json(captures / 'query_evidence.json', checks)

    choose(screen._btn_pick_db, database, '02_open')
    wait_jobs('Open database')
    if screen.tables() != facts['tables']:
        raise RuntimeError('The actual table list differs from the independent schema')
    capture('03_tables')
    items = screen._table_list.findItems('cell', Qt.MatchExactly)
    if len(items) != 1:
        raise RuntimeError('No unique actual cell-table item')
    screen._table_list.scrollToItem(items[0])
    QTest.mouseClick(screen._table_list.viewport(), Qt.LeftButton,
                     pos=screen._table_list.visualItemRect(items[0]).center())
    prove('unfiltered')
    capture('04_cell_table')
    original_rows = screen.preview_rows()
    fill(screen._col_search, 'cell_channel_1_mean_intensity')
    if (screen.visible_columns() != ['cell_channel_1_mean_intensity']
            or screen.preview_rows() != original_rows):
        raise RuntimeError('Column search changed rows or selected the wrong feature')
    prove('column_search')
    capture('05_column_search')
    fill(screen._col_search, '')
    if screen.visible_columns() != facts['columns'] or screen.preview_rows() != original_rows:
        raise RuntimeError('Clearing column search did not restore the exact view')
    capture('05b_columns_restored')

    # Extra evidence for the current footer's SQL sorting. None of these
    # frames adds a sentence to the retained eight-scene lesson.
    fill(screen._col_search, 'cell_area')
    # Search is a substring filter: the acquired table also contains
    # cell_area_filled and cell_area_bbox. Select the exact cell_area header
    # within those real matches, not an invented exact-match search mode.
    area_columns = [column for column in facts['columns'] if 'cell_area' in column.lower()]
    if screen.visible_columns() != area_columns:
        raise RuntimeError('The area column search differs from the actual schema matches')
    for label, order in [('06_sort_descending', ('cell_area', True)),
                         ('06b_sort_ascending', ('cell_area', False)),
                         ('06c_sort_cleared', None)]:
        header = screen._view.horizontalHeader()
        section = _visible_header_section(screen.visible_columns(), 'cell_area')
        if (screen._model.columnCount() != len(area_columns) or header.count() != len(area_columns)
                or screen._model.headerData(section, Qt.Horizontal, Qt.DisplayRole) != 'cell_area'
                or header.isSectionHidden(section)):
            raise RuntimeError('The actual model/header does not expose the exact cell_area section')
        point = QPoint(header.sectionViewportPosition(section) + header.sectionSize(section) // 2,
                       header.viewport().height() // 2)
        if (not header.viewport().rect().contains(point)
                or header.logicalIndexAt(point) != section):
            raise RuntimeError('The native header click does not target the visible cell_area section')
        QTest.mouseClick(header.viewport(), Qt.LeftButton, pos=point)
        prove(label, sort=order)
        capture(label)
    fill(screen._col_search, '')

    def select_combo(combo, text):
        index = combo.findText(text)
        if index < 0:
            raise RuntimeError(f'Actual filter option is unavailable: {text}')
        click(combo)
        popup = combo.view()
        if not popup.isVisible():
            raise RuntimeError('The actual filter choices did not open')
        QTest.keyClick(popup, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(popup, Qt.Key_Down)
        QTest.keyClick(popup, Qt.Key_Return)
        settle(.2)
        if combo.currentText() != text:
            raise RuntimeError('Native filter selection did not take effect')

    select_combo(screen._filter_col, 'cell_area')
    select_combo(screen._filter_op, '>=')
    fill(screen._filter_value, 10000)
    if screen._raw_toggle.isChecked():
        raise RuntimeError('Expected the genuine structured filter, not raw SQL')
    capture('08_filter')
    click(screen._btn_apply)
    prove('filtered', threshold=10000)
    if screen.where_clause() != '"cell_area" >= ?' or screen._params != (10000,):
        raise RuntimeError('The actual structured filter did not bind the numeric value')
    capture('09_filtered')
    click(screen._btn_clear)
    prove('filter_cleared')
    if screen.where_clause() is not None or screen.preview_rows() != original_rows:
        raise RuntimeError('Clearing the filter did not restore the original complete query')
    capture('09a_filter_restored')
    fill(screen._filter_value, 10000)
    click(screen._btn_apply)
    prove('filter_reapplied', threshold=10000)
    capture('09b_filter_reapplied')

    exported = run / 'cell_area_ge_10000.csv'
    if exported.exists() or screen.visible_columns() != facts['columns']:
        raise RuntimeError('Export must target a new file and the reviewed visible columns')
    choose(screen._btn_export, exported, '10a_export_picker', save=True)
    wait_jobs('Export')
    result = verify_export(database, exported, screen.visible_columns(), facts['filtered_rows'])
    if (not screen.status_text().startswith(f"Exported {facts['filtered_rows']:,} rows × {len(facts['columns'])} columns")
            or str(exported) not in screen.status_text()):
        raise RuntimeError('The visible export confirmation disagrees with the independent output')
    capture('10_export')
    if _digest(database) != copied['database_sha256']:
        raise RuntimeError('The browsed private database changed')
    require_unchanged_source(SOURCE, copied['source_bundle'])
    write_json(captures / 'output_evidence.json', {
        'accepted': True, 'export': result, 'query_checks': checks,
        'source_database_and_sidecars_unchanged': True,
        'private_database_bytes_unchanged': True, 'actual_file_pickers': True,
        'actual_threaded_workers': True, 'worker_results': job_results,
        'source_opened_in_gui': False, 'narration_changed': False})
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'reason': 'Actual read-only pages, full counts, SQL sort, restoration and complete CSV independently verified',
        'published': False, 'biological_validation_claimed': False,
        'source_database_and_sidecars_unchanged': True,
        'private_database_bytes_unchanged': True, 'exported_rows': result['rows'],
        'exported_columns': result['columns'], 'narration_changed': False})
    return screen
