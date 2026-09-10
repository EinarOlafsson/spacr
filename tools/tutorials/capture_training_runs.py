"""Capture Training Runs controls using the existing, disclosed synthetic logs.

No training, inference, checkpoint loading, data generation or app launch occurs
here. The caller owns the app, private capture stage and outer process timeout.
All curves, folds, configuration values and the fake checkpoint are examples,
not measurements of trained models or evidence of biological validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


AUTHOR = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
SOURCE = AUTHOR / 'synthetic/training_runs'
AUDIT = Path(__file__).parent / 'evidence/2026-09-09_training_runs_input_audit.json'
RUNS = {
    'baseline': 'baseline/model/maxvit_t/0_1/epochs_20',
    'tuned': 'tuned/model/maxvit_t/0_1/epochs_25',
    'cross_validated': 'cross_validated/model/resnet50/0_1/epochs_18',
    'incomplete': 'incomplete/model/maxvit_t/0_1/epochs_8',
}
SETTINGS = {
    'baseline': 'baseline/settings/train_test_maxvit_t_20.csv',
    'tuned': 'tuned/settings/train_test_maxvit_t_25.csv',
    'cross_validated': 'cross_validated/settings/train_test_resnet50_18.csv',
}
DISCLOSURE = (
    'All progress curves, folds and settings are synthetic demonstration data. '
    'The incomplete .pth is a text placeholder and is never loaded. '
    'Only the real CSV parser, comparison controls and plot are demonstrated; '
    'no model was trained or evaluated and no biological performance is validated.'
)


def _stat(path):
    value = Path(path).stat()
    return {key: int(getattr(value, key)) for key in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')}


def _fingerprint(path):
    before = _stat(path)
    if before['st_size'] > 2 * 1024 * 1024:
        raise RuntimeError(f'Unexpectedly large tutorial fixture: {path}')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if _stat(path) != before:
        raise RuntimeError(f'The source changed while being read: {path}')
    return {'stat': before, 'sha256': digest}


def _tree_state():
    """Bound the audit to this one small, existing synthetic fixture tree."""
    state = {'.': _stat(SOURCE)}
    for item in SOURCE.rglob('*'):
        if item.is_symlink() or len(state) >= 100:
            raise RuntimeError('Unexpected symlink or expanded tutorial input tree')
        state[str(item.relative_to(SOURCE))] = _stat(item)
    return state


def _read_curve(path):
    """Independently read this known numeric CSV schema; never write a log."""
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        values = {}
        for raw_key, value in row.items():
            key = raw_key.strip().lower()
            if not key or key.startswith('unnamed:') or key in values:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(f'Non-finite value in the frozen fixture: {path}')
            values[key] = number
        out.append(values)
    if not out or [r['epoch'] for r in out] != list(range(1, len(out) + 1)):
        raise RuntimeError(f'The fixture does not contain consecutive epoch rows: {path}')
    return out


def _same_rows(actual, expected, label):
    if len(actual) != len(expected):
        raise RuntimeError(f'Wrong returned row count: {label}')
    for got, wanted in zip(actual, expected):
        if set(got) != set(wanted):
            raise RuntimeError(f'Wrong returned columns: {label}')
        for key, value in wanted.items():
            if not math.isclose(float(got[key]), float(value), rel_tol=1e-11, abs_tol=1e-12):
                raise RuntimeError(f'Returned data differs from the source CSV: {label}/{key}')


def _require_visible_identifier(run_id, displayed_text, identifier_rect, viewport_rect):
    """Reject elided identifiers or glyph bounds clipped by the list viewport."""
    if not displayed_text.startswith(run_id + ' ·'):
        raise RuntimeError(f'The displayed run identifier is elided: {run_id}')
    x, y, width, height = identifier_rect
    left, top, available_width, available_height = viewport_rect
    values = (*identifier_rect, *viewport_rect)
    if (not all(math.isfinite(float(value)) for value in values)
            or width <= 0 or height <= 0 or available_width <= 0 or available_height <= 0
            or x < left or y < top or x + width > left + available_width
            or y + height > top + available_height):
        raise RuntimeError(f'The full run identifier is not visibly inside its row: {run_id}')


def record_training_runs(app, window, screen, stage, captures, capture, settle,
                         write_json, timeout):
    """Use actual visible controls and validate their results against the CSVs."""
    stage, captures = Path(stage).resolve(), Path(captures).resolve()
    if not captures.is_relative_to(stage) or not captures.is_dir():
        raise ValueError('Training Runs evidence requires an existing private capture directory')
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError('A positive finite lifecycle timeout is required')
    deadline = time.monotonic() + float(timeout)
    acceptance = captures / 'scientific_acceptance.json'
    evidence = {
        'lesson': '28_training_runs', 'accepted': False, 'published': False,
        'reason': 'The actual controls and parsed synthetic inputs are not yet verified',
        'acceptance_scope': 'Software workflow on disclosed synthetic fixtures only',
        'disclosure': DISCLOSURE, 'synthetic_inputs': True,
        'scientific_validation': False, 'held_out_validation': False,
        'training_performed': False, 'inference_performed': False,
        'checkpoint_loaded': False, 'inputs_generated_or_copied': False,
        'device': None, 'device_note': 'Fixture device settings do not describe an actual run',
        'core_classify_hold_unchanged': True,
    }
    write_json(acceptance, evidence)
    panel = None
    originals, expected, settings, identities = {}, {}, {}, {}
    jobs, picked_events, snapshots = [], [], []
    tree_before = None

    def job_finished(ok):
        jobs.append(bool(ok))

    def series_clicked(label):
        picked_events.append(str(label))

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Training Runs recorder exceeded its lifecycle timeout')
        settle(0)

    def preserve():
        for name, previous in originals.items():
            tick()
            if _fingerprint(Path(name)) != previous:
                raise RuntimeError(f'An original tutorial fixture changed: {name}')
        if tree_before is not None and _tree_state() != tree_before:
            raise RuntimeError('The original synthetic fixture tree changed')

    try:
        from PySide6.QtCore import QPoint, QRect, Qt, QTimer
        from PySide6.QtGui import QFontMetrics
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import (QFileDialog, QDialogButtonBox, QLineEdit, QScrollArea,
                                       QSplitter, QStyle, QStyleOptionViewItem)
        from spacr.qt.screens.train_compare import TrainCompareScreen
        from spacr.qt.widgets.fold_strip import FoldButton
        from spacr.train_compare import render_setting_value

        def expose(widget):
            tick()
            parent = widget.parentWidget()
            while parent is not None:
                if isinstance(parent, QScrollArea):
                    parent.ensureWidgetVisible(widget)
                parent = parent.parentWidget()
            settle(.1)
            if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
                raise RuntimeError('A requested Training Runs control is not visibly usable')

        def click(widget):
            expose(widget)
            QTest.mouseClick(widget, Qt.LeftButton,
                             pos=widget.visibleRegion().boundingRect().center())
            settle(.15)
            tick()

        def wait_scan():
            while panel.is_busy() or panel.active_jobs():
                tick()
                settle(.1)
            settle(.25)
            tick()
            if panel.last_error:
                raise RuntimeError(panel.last_error)

        def widen_existing_splitter():
            candidates = [widget for widget in panel.findChildren(QSplitter)
                          if widget.orientation() == Qt.Horizontal and widget.count() == 2
                          and widget.widget(0).isAncestorOf(panel._runs_list)
                          and widget.widget(1).isAncestorOf(panel._canvas)]
            if len(candidates) != 1:
                raise RuntimeError('Cannot identify the existing native run-list/plot splitter')
            splitter = candidates[0]
            expose(splitter)
            before = splitter.sizes()
            total = sum(before)
            if total < 2920:
                raise RuntimeError('The native window is too narrow for readable list and plot panes')
            # Resize only the existing user-adjustable divider; no widget,
            # font, layout minimum or size-policy changes are introduced.
            splitter.setSizes([920, total - 920])
            settle(.35)
            after = splitter.sizes()
            if not 850 <= after[0] <= 950 or panel._canvas.width() < 2000:
                raise RuntimeError('The native splitter did not leave readable list and plot widths')
            evidence['native_layout'] = {
                'splitter_sizes_before': before, 'splitter_sizes_after': after,
                'run_list_viewport_width': panel._runs_list.viewport().width(),
                'plot_width': panel._canvas.width(), 'minima_or_policies_changed': False,
            }

        def verify_visible_identifiers(names, phase):
            view = panel._runs_list
            wanted = {identities[name] for name in names}
            rows = []
            for index in range(view.count()):
                item = view.item(index)
                run_id = item.data(Qt.UserRole)
                if run_id not in wanted:
                    continue
                row_rect = view.visualItemRect(item)
                option = QStyleOptionViewItem()
                option.initFrom(view)
                option.rect = row_rect
                option.text = item.text()
                option.font = item.data(Qt.FontRole) or view.font()
                option.fontMetrics = QFontMetrics(option.font)
                option.displayAlignment = Qt.AlignLeft | Qt.AlignVCenter
                option.textElideMode = view.textElideMode()
                option.checkState = item.checkState()
                option.features = (QStyleOptionViewItem.ViewItemFeature.HasDisplay
                                   | QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator)
                text_rect = view.style().subElementRect(
                    QStyle.SubElement.SE_ItemViewItemText, option, view)
                metrics = option.fontMetrics
                glyphs = metrics.boundingRect(run_id)
                left_bearing = min(0, glyphs.left())
                identifier_rect = QRect(
                    text_rect.left() + left_bearing, text_rect.top(),
                    max(metrics.horizontalAdvance(run_id), glyphs.right() + 1) - left_bearing,
                    metrics.height())
                available = max(0, min(text_rect.width(), view.viewport().width() - text_rect.left()))
                displayed = metrics.elidedText(item.text(), view.textElideMode(), available)
                _require_visible_identifier(run_id, displayed, identifier_rect.getRect(),
                                            view.viewport().rect().getRect())
                if (not row_rect.contains(identifier_rect)
                        or not view.viewport().visibleRegion().contains(identifier_rect)):
                    raise RuntimeError(f'The run identifier is clipped by its visible item row: {run_id}')
                origin = view.viewport().mapToGlobal(identifier_rect.topLeft()) - window.mapToGlobal(QPoint(0, 0))
                rows.append({'run_id': run_id, 'displayed_text': displayed,
                             'identifier_rect_in_window': [origin.x(), origin.y(),
                                                           identifier_rect.width(), identifier_rect.height()],
                             'row_rect_in_viewport': list(row_rect.getRect())})
            if len(rows) != len(wanted):
                raise RuntimeError('Not every selected run has a visible identifier')
            evidence.setdefault('identifier_visibility_checks', []).append({'phase': phase, 'rows': rows})

        def choose_source():
            accepted, errors, timers = [], [], []

            def handle():
                dialog = app.activeModalWidget()
                try:
                    if not isinstance(dialog, QFileDialog):
                        raise RuntimeError('The actual source folder picker did not open')
                    dialog.accepted.connect(lambda: accepted.append(True))
                    timer = QTimer(dialog)
                    timer.setSingleShot(True)
                    timer.timeout.connect(dialog.reject)
                    timer.start(max(1, min(12000, int((deadline - time.monotonic()) * 1000))))
                    timers.append(timer)
                    dialog.resize(1400, 950)
                    edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                    if edit is None:
                        raise RuntimeError('The actual picker lacks its filename editor')
                    expose(edit)
                    edit.setFocus()
                    QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                    QTest.keyClicks(edit, str(SOURCE))
                    QTest.keyClick(edit, Qt.Key_Tab)
                    settle(.15)
                    capture('02_synthetic_source_folder_picker')
                    box = dialog.findChild(QDialogButtonBox)
                    button = None if box is None else box.button(QDialogButtonBox.Open)
                    if button is None:
                        raise RuntimeError('The actual folder picker lacks Open')
                    click(button)
                except Exception as exc:
                    errors.append(str(exc))
                    if dialog is not None:
                        dialog.reject()

            opener = QTimer(window)
            opener.setSingleShot(True)
            opener.timeout.connect(handle)
            opener.start(400)
            try:
                click(panel._btn_pick)
            finally:
                opener.stop()
                opener.deleteLater()
                for timer in timers:
                    try:
                        timer.stop()
                    except RuntimeError:
                        pass
            if errors or not accepted:
                raise RuntimeError('; '.join(errors) or 'The source picker was not accepted')
            wait_scan()  # Choosing the folder itself starts the normal scan.

        def choose_combo(combo, text, frame=None):
            diagnostic = {
                'requested': text, 'before': combo.currentText(),
                'items_before': [combo.itemText(i) for i in range(combo.count())],
                'interaction': 'Actual popup keyboard Home/Down/Enter',
            }
            evidence.setdefault('combo_interactions', []).append(diagnostic)
            try:
                index = combo.findText(text)
                if index < 0:
                    raise RuntimeError('The actual combo does not offer the requested value')
                click(combo)
                view = combo.view()
                if not view.isVisible():
                    raise RuntimeError('The actual combo popup did not open')
                if frame is not None:
                    capture(frame, desktop=True)
                # Popup coordinates can move after a metric redraw. Navigate
                # its real selection model through keyboard events instead.
                view.setFocus()
                QTest.keyClick(view, Qt.Key_Home)
                for _ in range(index):
                    QTest.keyClick(view, Qt.Key_Down)
                settle(.1)
                diagnostic['popup_selected'] = view.currentIndex().data(Qt.DisplayRole)
                if diagnostic['popup_selected'] != text:
                    raise RuntimeError('Keyboard navigation did not select the requested popup item')
                QTest.keyClick(view, Qt.Key_Return)
                settle(.25)
                tick()
                diagnostic['actual'] = combo.currentText()
                diagnostic['items_after'] = [combo.itemText(i) for i in range(combo.count())]
                if combo.currentText() != text:
                    raise RuntimeError('The visible combo selection did not change')
            except Exception as exc:
                diagnostic.update(actual=combo.currentText(), error=str(exc),
                                  items_after=[combo.itemText(i) for i in range(combo.count())])
                raise RuntimeError('Combo interaction failed: ' + json.dumps(diagnostic, ensure_ascii=False)) from exc

        def select_runs(names):
            wanted = {identities[name] for name in names}
            for index in range(panel._runs_list.count()):
                item = panel._runs_list.item(index)
                desired = item.data(Qt.UserRole) in wanted
                if (item.checkState() == Qt.Checked) == desired:
                    continue
                panel._runs_list.scrollToItem(item)
                expose(panel._runs_list)
                rect = panel._runs_list.visualItemRect(item)
                point = QPoint(rect.left() + 9, rect.center().y())
                if not panel._runs_list.viewport().rect().contains(point):
                    raise RuntimeError('The actual run checkbox is not visible')
                QTest.mouseClick(panel._runs_list.viewport(), Qt.LeftButton, pos=point)
                settle(.15)
                if (item.checkState() == Qt.Checked) != desired:
                    raise RuntimeError('The actual run checkbox did not change')
            if set(panel.selected_run_ids()) != wanted:
                raise RuntimeError('The selected run identities are wrong')
            verify_visible_identifiers(names, 'selected: ' + ', '.join(names))

        def expected_series(names, mode):
            result = {}
            for name in names:
                blocks = expected[name]
                folds = sorted({fold for split, fold in blocks if fold})
                if mode in ('per_fold', 'both') or not folds:
                    for (split, fold), rows in blocks.items():
                        result[(identities[name], split, fold)] = rows
                if folds and mode in ('mean', 'both'):
                    for split in ('train', 'val'):
                        by_epoch = {}
                        for fold in folds:
                            for row in blocks[(split, fold)]:
                                by_epoch.setdefault(row['epoch'], []).append(row)
                        rows = []
                        for epoch, observed in sorted(by_epoch.items()):
                            row = {'epoch': epoch, 'n_folds': len(observed)}
                            for key in observed[0]:
                                if key == 'epoch':
                                    continue
                                values = [item[key] for item in observed]
                                row[key] = statistics.mean(values)
                                row[key + '__sd'] = statistics.stdev(values)
                            rows.append(row)
                        result[(identities[name], split, 'mean')] = rows
            return result

        def verify_diff(comparison, names):
            diff = comparison.settings_diff
            ids = [r.run_id for r in comparison.runs]
            by_id = {identities[name]: settings[name] for name in names}
            shared = set.intersection(*(set(v) for v in by_id.values()))
            changed, env = {}, {}
            for key in shared:
                values = {rid: by_id[rid][key] for rid in ids}
                if len(set(values.values())) > 1:
                    (env if key in ('n_jobs', 'device') else changed)[key] = values
            if (diff['run_ids'] != ids or diff['shared'] != len(shared)
                    or diff['same'] != len(shared) - len(changed) - len(env)
                    or diff['identical'] != (not changed and not env)
                    or diff['no_settings'] or diff['drift'] or diff['env_manifest']):
                raise RuntimeError('The actual settings diff has unexpected provenance or counts')
            expected_rows = []
            for bucket, wanted in (('changed', changed), ('env', env)):
                actual = {entry['key']: entry['values'] for entry in diff[bucket]}
                if actual != wanted:
                    raise RuntimeError(f'The actual {bucket} settings values are wrong')
                for key, values in wanted.items():
                    expected_rows.append([bucket, key] + [render_setting_value(values[rid], 40) for rid in ids])
            if (panel.diff_headers() != ['bucket', 'setting', *ids]
                    or sorted(panel.diff_rows()) != sorted(expected_rows)):
                raise RuntimeError('The visible settings table differs from the actual settings')

        def verify_comparison(names, mode, metric, frame):
            tick()
            comparison = panel.comparison()
            wanted = expected_series(names, mode)
            if (comparison is None or comparison.fold_mode != mode
                    or {r.run_id for r in comparison.runs} != {identities[n] for n in names}
                    or len(comparison.series) != len(wanted)
                    or panel.selected_metric() != metric):
                raise RuntimeError('The real comparison has wrong runs, mode, metric or series count')
            seen = set()
            summary = []
            for series in comparison.series:
                key = (series.run_id, series.split, series.fold)
                if key not in wanted or key in seen:
                    raise RuntimeError('Unexpected or duplicate returned series identity')
                seen.add(key)
                kind = 'mean' if series.fold == 'mean' else ('fold' if series.fold else 'single')
                if series.kind != kind:
                    raise RuntimeError('A returned series has the wrong fold/mean kind')
                _same_rows(series.frame.to_dict(orient='records'), wanted[key], series.label)
                if kind == 'mean' and series.n_folds != max(r['n_folds'] for r in wanted[key]):
                    raise RuntimeError('A returned mean has the wrong contributing fold count')
                summary.append({'label': series.label, 'run_id': series.run_id,
                                'split': series.split, 'fold': series.fold, 'kind': series.kind,
                                'rows': len(wanted[key]), 'data': wanted[key]})
            figure = panel.figure()
            if len(figure.axes) != 1:
                raise RuntimeError('The actual comparison has no single plotted axes')
            lines = {line.get_label(): line for line in figure.axes[0].lines}
            if set(lines) != set(panel.series_labels()) or len(lines) != len(wanted):
                raise RuntimeError('The actual plotted series do not match the returned curves')
            mapping = getattr(figure, 'spacr_series_by_label', {})
            for series in comparison.series:
                rows = wanted[(series.run_id, series.split, series.fold)]
                line = lines[series.label]
                if (mapping.get(series.label) is not series
                        or len(line.get_xdata()) != len(rows) or len(line.get_ydata()) != len(rows)
                        or line.get_linestyle() != ('--' if series.split == 'train' else '-')):
                    raise RuntimeError('Plotted identities, lengths or train/validation styles are wrong')
                _same_rows([{'epoch': x, metric: y} for x, y in zip(line.get_xdata(), line.get_ydata())],
                           [{'epoch': r['epoch'], metric: r[metric]} for r in rows], series.label)
            means = [s for s in comparison.series if s.kind == 'mean']
            bands = figure.axes[0].collections
            if len(bands) != len(means):
                raise RuntimeError('The actual plot is missing or adding a fold-SD band')
            for series, band in zip(means, bands):
                rows = wanted[(series.run_id, series.split, series.fold)]
                expected_vertices = [(r['epoch'], r[metric] + sign * r[metric + '__sd'])
                                     for r in rows for sign in (-1, 1)]
                vertices = [tuple(v) for path in band.get_paths() for v in path.vertices]

                def close_point(left, right):
                    return all(math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)
                               for a, b in zip(left, right))

                if (not vertices
                        or any(not any(close_point(v, w) for w in expected_vertices) for v in vertices)
                        or any(not any(close_point(w, v) for v in vertices) for w in expected_vertices)):
                    raise RuntimeError('The visible fold-SD band differs from the independent CSV calculation')
            verify_diff(comparison, names)
            expose(panel._canvas)
            verify_visible_identifiers(names, frame)
            capture(frame)
            snapshots.append({'frame': frame, 'synthetic': True, 'fold_mode': mode,
                              'metric': metric, 'series_count': len(summary), 'series': summary,
                              'status': panel.status_text(), 'settings_summary': panel.summary_text(),
                              'diff_headers': panel.diff_headers(), 'diff_rows': panel.diff_rows()})

        # Every input is pre-existing. No recipe, checkpoint loader or trainer is called.
        audit = json.loads(AUDIT.read_text())
        tree_before = _tree_state()
        for relative in tree_before:
            path = SOURCE / relative
            if path.is_file():
                originals[str(path)] = _fingerprint(path)
        for entry in audit['original_log_files']:
            path = SOURCE / entry['path_relative_to_synthetic_logs']
            if originals[str(path)]['sha256'] != entry['sha256']:
                raise RuntimeError(f'A frozen synthetic input has changed: {path}')
        manifest_path = SOURCE / 'manifest.json'
        if originals[str(manifest_path)]['sha256'] != audit['original_capture_provenance']['input_manifest']['sha256']:
            raise RuntimeError('The original synthetic-data disclosure manifest changed')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('disclosure') != 'Synthetic progress logs; no model was trained.':
            raise RuntimeError('The original synthetic-data disclosure is missing')
        for name, relative in RUNS.items():
            if not (SOURCE / relative).is_dir() or not manifest['runs'][name].endswith('/' + relative):
                raise RuntimeError('The exact synthetic run identity is missing')
            expected[name] = {}
            settings[name] = {}
            if name in SETTINGS:
                with (SOURCE / SETTINGS[name]).open(newline='') as handle:
                    settings[name] = {row['Key']: row['Value'] for row in csv.DictReader(handle)}
            folder = SOURCE / relative
            for path in sorted(folder.rglob('*.csv')):
                if path.name not in ('train.csv', 'validation.csv'):
                    raise RuntimeError('An unexpected progress CSV is present')
                fold = '' if path.parent == folder else path.parent.name
                split = 'train' if path.name == 'train.csv' else 'val'
                expected[name][(split, fold)] = _read_curve(path)
        placeholder = SOURCE / RUNS['incomplete'] / 'maxvit_t_epoch_8_channels_0_1.pth'
        if placeholder.read_bytes() != b'synthetic tutorial checkpoint placeholder':
            raise RuntimeError('The incomplete fixture is not the disclosed text placeholder')
        evidence['source'] = str(SOURCE)
        evidence['input_fingerprints'] = originals
        evidence['input_manifest'] = manifest
        write_json(captures / 'synthetic_input_provenance.json', evidence)

        buttons = [b for b in screen.findChildren(FoldButton)
                   if b.app_key == 'train_compare' and b.isVisible()]
        if len(buttons) != 1:
            raise RuntimeError('Classify must expose one visible Training Runs fold')
        click(buttons[0])
        visible = [w for w in window.findChildren(TrainCompareScreen) if w.isVisible()]
        if len(visible) != 1:
            raise RuntimeError('The actual Training Runs fold did not open visibly')
        panel = visible[0]
        if panel.runs() or panel.comparison() is not None:
            raise RuntimeError('A fresh current Training Runs panel is required')
        widen_existing_splitter()
        panel.job_finished.connect(job_finished)
        panel.series_clicked.connect(series_clicked)
        capture('01_current_training_runs_fold')
        choose_source()
        click(panel._btn_scan)  # Show the actual Scan action as well as the picker.
        wait_scan()
        if not jobs or not all(jobs) or Path(panel.root()).resolve() != SOURCE.resolve():
            raise RuntimeError('The real Scan did not succeed on the exact synthetic source')
        runs = panel.runs()
        if len(runs) != len(RUNS) or len(panel.run_ids()) != len(set(panel.run_ids())):
            raise RuntimeError('The real Scan returned unexpected run identities/counts')
        for run in runs:
            matches = [name for name, relative in RUNS.items() if run.path.resolve() == (SOURCE / relative).resolve()]
            if len(matches) != 1:
                raise RuntimeError('The real Scan returned a different run path')
            name = matches[0]
            identities[name] = run.run_id
            if run.settings != settings[name]:
                raise RuntimeError('The actual recovered settings differ from the source CSV')
            if name in SETTINGS and Path(run.settings_path).resolve() != (SOURCE / SETTINGS[name]).resolve():
                raise RuntimeError('The settings were recovered from the wrong project')
            blocks = expected[name]
            actual_blocks = set(zip(run.curves['split'], run.curves['fold']))
            if actual_blocks != set(blocks) or sorted(run.folds) != sorted({f for s, f in blocks if f}):
                raise RuntimeError('The loaded split/fold identities differ from the CSVs')
            for (split, fold), rows in blocks.items():
                actual = run.curves[(run.curves['split'] == split) & (run.curves['fold'] == fold)]
                if set(actual['run_id']) != {run.run_id}:
                    raise RuntimeError('Returned curve rows have the wrong run identity')
                _same_rows(actual.drop(columns=['run_id', 'split', 'fold']).to_dict(orient='records'), rows, name)
            if name == 'incomplete' and (run.has_curves or run.settings or not run.notes):
                raise RuntimeError('The actual incomplete-run warning is missing')
        if set(identities) != set(RUNS) or identities['incomplete'] not in panel.problem_text():
            raise RuntimeError('The actual discovered-run warnings are incomplete')
        evidence['discovered_runs'] = [{'name': name, 'run_id': identities[name],
                                       'path': str(SOURCE / RUNS[name]),
                                       'csv_rows': sum(len(v) for v in expected[name].values())}
                                      for name in RUNS]
        evidence['run_rows'] = panel.run_rows()
        evidence['problem_text'] = panel.problem_text()
        verify_visible_identifiers(list(RUNS), 'discovery')
        capture('03_synthetic_runs_and_real_warnings')

        select_runs(['baseline', 'tuned'])
        capture('04_select_two_synthetic_runs')
        choose_combo(panel._metric_combo, 'accuracy', '04b_actual_metric_choices')
        choose_combo(panel._fold_combo, 'per fold')
        click(panel._btn_overlay)
        verify_comparison(['baseline', 'tuned'], 'per_fold', 'accuracy', '05_actual_synthetic_accuracy_overlay')
        expose(panel._diff_table)
        capture('06_actual_synthetic_settings_diff')

        # A real mouse pick is optional: never inject an event or call identify_series.
        series = next(s for s in panel.comparison().series
                      if s.run_id == identities['baseline'] and s.split == 'val')
        axes, canvas = panel.figure().axes[0], panel._canvas
        expose(canvas)
        rows = expected['baseline'][('val', '')]
        picked = False
        for index in (len(rows) // 2, len(rows) // 3, len(rows) - 2):
            x, y = axes.transData.transform((rows[index]['epoch'], rows[index]['accuracy']))
            ratio = float(canvas.device_pixel_ratio)
            point = QPoint(round(x / ratio), round((panel.figure().bbox.height - y) / ratio))
            if not canvas.visibleRegion().contains(point):
                continue
            QTest.mouseClick(canvas, Qt.LeftButton, pos=point)
            settle(.25)
            tick()
            if picked_events and picked_events[-1] == series.label and panel.picked_text().startswith(series.label):
                best = max(rows, key=lambda row: row['accuracy'])
                last = rows[-1]
                text = panel.picked_text()
                if (f"last accuracy {last['accuracy']:.4f} @ {int(last['epoch'])}" not in text
                        or f"best accuracy {best['accuracy']:.4f} @ {int(best['epoch'])}" not in text
                        or 'chosen on this same curve, so optimistic' not in text
                        or str(SOURCE / RUNS['baseline']) not in text):
                    raise RuntimeError('The picked synthetic series detail disagrees with its CSV')
                picked = True
                capture('07_actual_series_pick_best_and_last_epochs')
                break
        evidence['series_pick'] = {'performed_and_verified': picked,
                                   'text': panel.picked_text() if picked else None,
                                   'limitation': None if picked else 'Actual mouse picking did not identify the requested series; no injected fallback or detail capture.'}

        choose_combo(panel._metric_combo, 'loss')
        verify_comparison(['baseline', 'tuned'], 'per_fold', 'loss', '08_actual_synthetic_loss_overlay')
        select_runs(['tuned', 'cross_validated'])
        choose_combo(panel._metric_combo, 'accuracy')
        click(panel._btn_overlay)
        choose_combo(panel._fold_combo, 'mean ± sd', '08b_actual_fold_choices')
        verify_comparison(['tuned', 'cross_validated'], 'mean', 'accuracy', '09_actual_synthetic_fold_mean_sd')
        choose_combo(panel._fold_combo, 'both')
        verify_comparison(['tuned', 'cross_validated'], 'both', 'accuracy', '10_actual_synthetic_folds_and_mean')
        choose_combo(panel._fold_combo, 'per fold')
        verify_comparison(['tuned', 'cross_validated'], 'per_fold', 'accuracy', '11_actual_synthetic_per_fold_curves')
        wait_scan()
        preserve()
        evidence.update(accepted=True, reason='Actual current controls, returned curves, plotted values and settings match the disclosed existing synthetic inputs; originals unchanged',
                        source_preserved=True, active_jobs=panel.active_jobs(),
                        job_results=jobs, comparisons=snapshots)
        write_json(captures / 'training_runs_results.json', evidence)
        write_json(acceptance, evidence)
    except Exception as exc:
        evidence.update(accepted=False, reason=str(exc), comparisons=snapshots,
                        job_results=jobs,
                        active_jobs=None if panel is None else panel.active_jobs())
        write_json(acceptance, evidence)
        raise
    finally:
        if panel is not None:
            try:
                panel.job_finished.disconnect(job_finished)
                panel.series_clicked.disconnect(series_clicked)
            except (RuntimeError, TypeError):
                pass
        # Do not close or destroy a still-running scan. The caller owns timeout cleanup.
