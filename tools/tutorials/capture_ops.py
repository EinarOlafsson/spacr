"""Record Mask's actual OPS fold.

By default this records navigation only (the older introductory capture).
With ``SPACR_OPS_WALKTHROUGH=1`` it records the practical walkthrough: the
real Load test data download, the settings that matter, a complete native
run of the downloaded example and the written tables, opened through Help ->
Database browser. Nothing is injected: every value on screen is either the
test-data fill, a typed entry or the application's own output.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def record_screen(app, window, stage, captures, capture, settle, write_json):
    if os.environ.get('SPACR_OPS_WALKTHROUGH') == '1':
        return record_walkthrough(app, window, stage, captures, capture, settle, write_json,
                                  float(os.environ.get('SPACR_OPS_CAPTURE_TIMEOUT', '10800')))
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt.screens.mask import ops_page

    buttons = [w for w in window.findChildren(QAbstractButton) if w.isVisible()
               and w.property('moduleAppKey') == 'mask' and w.objectName() == 'AppTile']
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise ValueError('Expected one enabled Mask Home tile')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(2)
    screen = window._screens.get('mask')
    if screen is None or not screen.isVisible():
        raise ValueError('The actual Home click did not open Mask')
    switch = screen._ops_switch
    if not switch.isVisible() or not switch.isEnabled() or switch.isChecked():
        raise ValueError('Expected the visible, initially closed OPS toggle')
    capture('01_mask_host')
    QTest.mouseClick(switch, Qt.LeftButton)
    settle(2)
    manager = ops_page(screen)
    if (not switch.isChecked() or manager is None or manager.page is None
            or not manager.page.isVisible() or manager.page.app_key != 'ops'):
        raise ValueError('The real OPS toggle did not open its settings page')
    capture('02_ops_settings')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Visible Home to Mask to OPS navigation only',
        'route': ['mask', 'ops'], 'gui_workflow_completed': False,
        'run_clicked': False, 'inputs_injected': False,
        'segmentation_or_decoding_performed': False,
        'app_source_modified': False, 'published': False})


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def _readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def independent_tables(database):
    """Row counts and schemas read with stdlib SQLite, never the app's code."""
    with _readonly(database) as con:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        result = {}
        for table in tables:
            columns = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
            rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            result[table] = {'rows': rows, 'columns': columns}
    return result


def barcode_facts(database, library):
    """Recount assigned barcodes and exact library matches independently."""
    prefixes = set()
    import csv
    with Path(library).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        column = next(c for c in reader.fieldnames if c in ('prefix', 'barcode', 'sequence'))
        for row in reader:
            prefixes.add(row[column].strip().upper())
    with _readonly(database) as con:
        columns = [r[1] for r in con.execute('PRAGMA table_info("ops_barcodes")')]
        rows = con.execute('SELECT * FROM "ops_barcodes"').fetchall()
    frame = [dict(zip(columns, row)) for row in rows]
    code = next((c for c in ('barcode', 'barcode_0', 'called_barcode') if c in columns), None)
    facts = {'columns': columns, 'rows': len(frame), 'barcode_column': code,
             'library_prefixes': len(prefixes)}
    if code is not None:
        called = [str(r[code]).upper() for r in frame if r[code] not in (None, '')]
        facts['called'] = len(called)
        facts['library_exact_recounted'] = sum(c in prefixes for c in called)
    return facts


def record_walkthrough(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QDialog,
                                   QDialogButtonBox, QFileDialog, QLineEdit,
                                   QMessageBox)
    from shiboken6 import isValid
    from spacr.qt.screens.mask import ops_page
    from capture_acceptance import assess_pipeline
    from capture_geometry import capture_rect

    stage, captures = Path(stage), Path(captures)
    proof = {'lesson': '76_ops', 'accepted': False, 'app_source_modified': False,
             'inputs_injected': False, 'published': False}
    write_json(captures / 'scientific_acceptance.json', proof)

    def click(widget):
        if widget is None or not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('An actual OPS control is unavailable: ' +
                               (widget.objectName() if widget is not None else 'None'))
        QTest.mouseClick(widget, Qt.LeftButton,
                         pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    def type_into(widget, value):
        if isinstance(widget, QAbstractSpinBox):
            editor = widget.findChild(QLineEdit)
        elif isinstance(widget, QLineEdit):
            editor = widget
        else:
            editors = [e for e in widget.findChildren(QLineEdit) if e.isVisible()]
            if len(editors) != 1:
                raise RuntimeError('No unique visible editor for a typed OPS setting')
            editor = editors[0]
        editor.setFocus()
        QTest.mouseClick(editor, Qt.LeftButton)
        QTest.keyClick(editor, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(editor, Qt.Key_Backspace)
        QTest.keyClicks(editor, str(value))
        QTest.keyClick(editor, Qt.Key_Tab)
        settle(.3)

    # Home -> Mask -> OPS, through the real tile and the real actions-row switch.
    tiles = [w for w in window.findChildren(QAbstractButton) if w.isVisible()
             and w.property('moduleAppKey') == 'mask' and w.objectName() == 'AppTile']
    if len(tiles) != 1:
        raise RuntimeError('Expected one Mask Home tile')
    click(tiles[0])
    settle(2)
    mask = window._screens.get('mask')
    if mask is None or not mask.isVisible():
        raise RuntimeError('The Home tile did not open Mask')
    switch = mask._ops_switch
    if switch is None or not switch.isVisible() or switch.isChecked():
        raise RuntimeError('Expected the visible, initially closed OPS switch')
    capture('01_mask_host')
    click(switch)
    settle(2)
    manager = ops_page(mask)
    if manager is None or manager.page is None or not manager.page.isVisible():
        raise RuntimeError('The OPS switch did not open its page')
    page = manager.page
    if page.app_key != 'ops':
        raise RuntimeError('The OPS page is not the ops settings screen')
    deadline = time.monotonic() + 120
    while ('genotype_source' not in (getattr(getattr(page, '_settings_model', None), '_widgets', None) or {})
           or getattr(page, '_ops_example_button', None) is None
           or getattr(page, '_settings_scroll', None) is None):
        if time.monotonic() > deadline:
            raise RuntimeError('The OPS settings form was not built')
        settle(.3)
    model = page._settings_model
    bar = getattr(page, '_settings_search', None)
    proof['settings_search_bar'] = bar is not None
    if bar is not None:
        if bar.modified_only():
            click(bar._modified)
        if bar.level() != 'all':
            click(bar._disclosure)
        bar.set_query('')
    settle(.5)
    if getattr(page, '_console_folder', None) is not None and page._console_folder.shut:
        click(page._console_folder.heading)
    before = model.collect()
    proof['defaults_before_test_data'] = {k: before.get(k) for k in (
        'genotype_source', 'phenotype_source', 'dst_root', 'ops_library', 'plate',
        'cellpose_model', 'cellpose_diameter', 'ops_raster_overlap', 'ops_window_overlap',
        'ops_base_channels', 'ops_read_threshold', 'ops_footprint', 'ops_store_reads',
        'ops_gpu', 'n_workers')}
    capture('01_mask_ops')

    def show(keys):
        """Expand the real sections holding ``keys`` and scroll them into view."""
        rects = []
        fields = [model._widgets[k] for k in keys]
        for field in fields:
            parents, widget = [], field.parentWidget()
            while widget is not None and widget is not page:
                if callable(getattr(widget, 'is_expanded', None)) and callable(getattr(widget, 'header', None)):
                    parents.append(widget)
                widget = widget.parentWidget()
            for section in reversed(parents):
                if not section.is_expanded():
                    page._settings_scroll.ensureWidgetVisible(section.header())
                    settle(.2)
                    click(section.header())
        page._settings_scroll.ensureWidgetVisible(fields[-1], 50, 120)
        settle(.2)
        page._settings_scroll.ensureWidgetVisible(fields[0], 50, 120)
        settle(.3)
        for key, field in zip(keys, fields):
            if not field.isVisible() or field.visibleRegion().isEmpty():
                raise RuntimeError('The actual OPS setting is not visible: ' + key)
            rects.append(capture_rect(field, window))
        return rects

    # Load test data: the real button, the real download and the real fill.
    button = page._ops_example_button
    page._settings_scroll.ensureWidgetVisible(button, 50, 200)
    settle(.3)
    folder = Path.home() / '.cache/spacr/example_data/ops_screen'
    proof['cached_before_click'] = (folder / 'manifest.csv').is_file()
    QTimer.singleShot(2500, lambda: capture('02_downloading'))
    click(button)
    deadline = time.monotonic() + 3600
    while not model.collect().get('genotype_source') or not button.isEnabled():
        if time.monotonic() > deadline:
            raise TimeoutError('The OPS test data did not finish loading')
        settle(.5)
    settle(1)
    filled = model.collect()
    source = Path(filled['genotype_source'])
    library = Path(filled['ops_library'])
    destination = Path(filled['dst_root'])
    if (source != folder / 'sequencing' or library != folder / 'library/pool10_prefixes.csv'
            or destination != folder / 'ops_output' or filled.get('phenotype_source')
            or filled.get('plate') != '20200202_6W-LaC024A'):
        raise RuntimeError('Load test data filled unexpected settings: ' + repr(
            {k: filled.get(k) for k in ('genotype_source', 'ops_library', 'dst_root', 'phenotype_source', 'plate')}))
    if destination.exists():
        raise RuntimeError('Use a fresh stage: the example output folder already exists')
    import csv
    with (folder / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        manifest = list(csv.DictReader(stream))
    inputs = {}
    for row in manifest:
        path = folder / row['path']
        size = path.stat().st_size
        localized = row['path'].startswith('settings/')
        if not localized and row.get('bytes') and int(row['bytes']) != size:
            raise RuntimeError('A downloaded test-data file differs from its manifest size: ' + row['path'])
        digest = _digest(path)
        if not localized and row.get('sha256') and row['sha256'] != digest:
            raise RuntimeError('A downloaded test-data file differs from its manifest hash: ' + row['path'])
        inputs[row['path']] = {'bytes': size, 'sha256': digest, 'manifest_verified': not localized}
    sites = sorted({p.rsplit('Site-', 1)[1].split('.')[0] for p in inputs if 'Site-' in p})
    cycles = sorted({p.split('/')[1] for p in inputs if p.startswith('sequencing/')})
    proof['test_data'] = {'files': len(inputs), 'sites': sites, 'cycle_folders': cycles,
                          'library': str(library.relative_to(folder)),
                          'downloaded_in_this_recording': not proof['cached_before_click']}
    write_json(captures / 'input_manifest.json', {'folder': str(folder), 'files': inputs})
    write_json(captures / 'scientific_acceptance.json', proof)
    rects = show(['genotype_source', 'ops_library', 'dst_root'])
    page._settings_scroll.ensureWidgetVisible(button, 50, 200)
    settle(.3)
    rects.append(capture_rect(button, window))
    proof.setdefault('focus', {})['02_load_test_data'] = [r for r in rects if r]
    capture('02_load_test_data')

    tour = [('03_inputs', ['genotype_source', 'ops_library', 'dst_root']),
            ('04_phenotype', ['phenotype_source']),
            ('05_alignment', ['cellpose_model', 'cellpose_diameter', 'ops_raster_overlap']),
            ('06_window_overlap', ['ops_window_overlap']),
            ('07_base_channels', ['ops_base_channels']),
            ('08_read_settings', ['ops_read_threshold', 'ops_footprint']),
            ('09_store_reads', ['ops_store_reads'])]
    for name, keys in tour:
        proof['focus'][name] = show(keys)
        capture(name)
    # Two example fields: two decode workers are enough and keep memory low.
    type_into(model._widgets['n_workers'], 2)
    configured = model.collect()
    if configured.get('n_workers') != 2:
        raise RuntimeError('The typed Workers value did not reach the settings')
    unchanged = {k: v for k, v in filled.items() if k != 'n_workers'}
    if any(configured.get(k) != v for k, v in unchanged.items()):
        raise RuntimeError('The settings tour changed a test-data value')
    write_json(captures / 'configured_settings.json', configured)
    proof['focus']['10_run'] = show(['ops_gpu', 'n_workers'])
    proof['focus']['10_run'].append(capture_rect(page._btn_run, window))

    outcome = {'finished': False, 'ok': False, 'errors': [], 'prompts': []}
    guard = QTimer(window)

    def refuse_prompts():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                outcome['prompts'].append(box.windowTitle() + ': ' + box.text())
                capture('10_unexpected_prompt')
                box.reject()
    guard.timeout.connect(refuse_prompts)
    guard.start(500)
    started = time.monotonic()
    try:
        click(page._btn_run)
        worker = page._worker
        if worker is None:
            raise RuntimeError('Run did not start a pipeline worker: ' + repr(outcome['prompts']))
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda message: outcome['errors'].append(str(message)))
        proof['run_clicked'] = True

        def console_text():
            return '\n'.join(t for _, _, t in page._console._pipeline_console_blocks())
        wait_until = time.monotonic() + 600
        while 'OPS: ' not in console_text() and not outcome['finished']:
            if time.monotonic() > wait_until:
                break
            settle(.5)
        page._console.jump_to_the_end()
        settle(.5)
        capture('10_run')
        progress_captured = False
        deadline = time.monotonic() + timeout
        while not outcome['finished'] or page._worker_thread_is_running():
            if time.monotonic() > deadline:
                page._request_cooperative_stop()
                raise TimeoutError('The native OPS run exceeded its time limit')
            if not progress_captured and 'objects:' in console_text():
                page._console.jump_to_the_end()
                settle(.3)
                capture('10b_progress')
                progress_captured = True
            settle(1)
        settle(1)
    finally:
        guard.stop()
    proof['run_seconds'] = round(time.monotonic() - started, 1)
    blocks = [t for _, _, t in page._console._pipeline_console_blocks()]
    write_json(captures / 'run_console.json', blocks)
    proof['outcome'] = outcome
    proof['pipeline'] = assess_pipeline(outcome, blocks, page._figure_queue.count(),
                                        requires_figure=False)
    write_json(captures / 'scientific_acceptance.json', proof)
    if not proof['pipeline']['accepted']:
        page._console.jump_to_the_end()
        settle(.5)
        capture('11_failed_run')
        raise RuntimeError('The native OPS run did not complete: ' + repr(proof['pipeline']) +
                           ' ' + repr(outcome['errors'])[:2000])
    for block, _, _ in page._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    page._console.jump_to_the_end()
    settle(.5)
    capture('11_run_finished')

    database = destination / 'measurements.db'
    reports = sorted(destination.glob('*/ops_report.json'))
    if not database.is_file() or not reports:
        raise RuntimeError('The run wrote no measurements.db or OPS report')
    tables = independent_tables(database)
    report_values = {p.parent.name: json.loads(p.read_text()) for p in reports}
    facts = barcode_facts(database, library)
    proof['outputs'] = {'database': str(database), 'tables': tables,
                        'reports': {k: {'path': str(destination / k / 'ops_report.json'),
                                        'decode': v.get('decode'), 'objects': {
                                            key: v.get('objects', {}).get(key) for key in (
                                                'canvas', 'windows', 'objects', 'total_seconds')},
                                        'stitch': v.get('stitch'), 'phenotype': v.get('phenotype'),
                                        'seconds': v.get('seconds'), 'peak_rss_gb': v.get('peak_rss_gb')}
                                    for k, v in report_values.items()},
                        'barcodes': facts}
    for required in ('ops_objects', 'ops_barcodes'):
        if tables.get(required, {}).get('rows', 0) < 1:
            raise RuntimeError('The run wrote no rows to ' + required)
    proof['inputs_unchanged'] = all(_digest(folder / p) == v['sha256']
                                    for p, v in json.loads((captures / 'input_manifest.json').read_text())['files'].items())
    if not proof['inputs_unchanged']:
        raise RuntimeError('The run changed a downloaded input file')
    write_json(captures / 'scientific_acceptance.json', proof)

    # The written output, opened through Help -> Database browser.
    help_actions = [a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help']
    if len(help_actions) != 1 or help_actions[0].menu() is None:
        raise RuntimeError('Expected one actual Help menu')
    menu = help_actions[0].menu()
    choices = [a for a in menu.actions() if a.text().replace('&', '').rstrip('…').strip().lower() == 'database browser']
    if len(choices) != 1 or not choices[0].isEnabled():
        raise RuntimeError('The actual Help menu lacks Database browser')
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_actions[0]).center())
    settle(.4)
    if not menu.isVisible():
        raise RuntimeError('Help did not open')
    capture('11a_help_database_menu')
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
    deadline = time.monotonic() + 120
    while window._screens.get('db_browser') is None:
        if time.monotonic() > deadline:
            raise TimeoutError('Database browser did not open')
        settle(.1)
    browser = window._screens['db_browser']
    settle(1)

    def wait_jobs(label):
        deadline = time.monotonic() + 600
        while browser.is_busy() or browser.active_jobs() or browser.queued_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError(label + ': ' + browser.status_text())
            settle(.1)
        settle(.4)
        if browser.last_error:
            raise RuntimeError(label + ': ' + str(browser.last_error))

    accepted, errors, owned = [], [], []

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
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            edit.setFocus()
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClick(edit, Qt.Key_Backspace)
            QTest.keyClicks(edit, str(destination))
            QTest.keyClick(edit, Qt.Key_Return)
            settle(1.5)
            if Path(dialog.directory().absolutePath()) != destination:
                raise RuntimeError('The picker did not open the selected output folder')
            capture('11_output_folder')
            edit.setFocus()
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClick(edit, Qt.Key_Backspace)
            QTest.keyClicks(edit, 'measurements.db')
            settle(.3)
            box = dialog.findChild(QDialogButtonBox)
            click(box.button(QDialogButtonBox.Open))
        except Exception as error:
            errors.append(str(error))
            if isinstance(dialog, QDialog) and isValid(dialog):
                dialog.reject()

    def stalled():
        errors.append('The actual file picker did not complete')
        for dialog in owned:
            if isValid(dialog) and dialog.isVisible():
                dialog.reject()

    timer, watchdog = QTimer(window), QTimer(window)
    timer.setSingleShot(True)
    watchdog.setSingleShot(True)
    timer.timeout.connect(handle)
    watchdog.timeout.connect(stalled)
    timer.start(500)
    watchdog.start(30000)
    try:
        click(browser._btn_pick_db)
    finally:
        timer.stop()
        watchdog.stop()
    if errors or not accepted:
        raise RuntimeError('; '.join(errors) or 'The database picker was not accepted')
    wait_jobs('Open database')
    if Path(browser.database_path()).resolve() != database.resolve():
        raise RuntimeError('The browser opened a different database')
    if sorted(browser.tables()) != sorted(t for t in tables):
        raise RuntimeError('Browser table list differs from SQLite: %r' % (browser.tables(),))
    capture('11b_tables')
    shown = {}
    for table, frame in (('ops_barcodes', '12_barcode_table'), ('ops_objects', '12b_nucleus_table')):
        items = browser._table_list.findItems(table, Qt.MatchExactly)
        if len(items) != 1:
            raise RuntimeError('No unique table item ' + table)
        browser._table_list.scrollToItem(items[0])
        QTest.mouseClick(browser._table_list.viewport(), Qt.LeftButton,
                         pos=browser._table_list.visualItemRect(items[0]).center())
        wait_jobs(table)
        if browser.current_table() != table:
            raise RuntimeError('The table click did not select ' + table)
        columns, rows = browser.preview_columns(), browser.preview_rows()
        with _readonly(database) as con:
            expected = con.execute(f'SELECT * FROM "{table}" LIMIT ?', (len(rows),)).fetchall()
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if list(columns) != tables[table]['columns'] or [tuple(r) for r in rows] != expected[:len(rows)]:
            raise RuntimeError('The displayed page differs from SQLite for ' + table)
        if browser.row_count() != count:
            raise RuntimeError('The displayed row count differs for ' + table)
        shown[table] = {'columns': list(columns), 'loaded_rows': len(rows), 'total_rows': count,
                        'rows_label': browser._rows_label.text()}
        capture(frame)
    proof['database_browser'] = shown
    proof['accepted'] = True
    proof['scope'] = ('Real downloaded OPS example, complete native stitch/objects/decode run '
                      'and written tables checked with SQLite; not a biological validation')
    write_json(captures / 'scientific_acceptance.json', proof)
