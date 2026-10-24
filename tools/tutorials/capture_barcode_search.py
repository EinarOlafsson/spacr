"""Drive the genuine live barcode search, then explicitly apply its proposal."""
from dataclasses import asdict
from pathlib import Path
import time
import shutil

import capture_ui
from capture_ui_drv_a import Rec, table_row_to_top
from map_barcodes_data import prepare_references, digest



def select_reference_file(capture, screen, key, value, settle):
    """Select a single CSV using its actual Choose file button and file dialog."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QDialogButtonBox, QDialog
    from shiboken6 import isValid
    from spacr.qt.widgets.file_list import FilePathListWidget

    field = screen._settings_model._widgets[key]
    if not isinstance(field, FilePathListWidget) or not field._single:
        raise RuntimeError('The current reference does not have its single-file picker: ' + key)
    bar = screen._settings_search
    old = (bar.query(), bar.level(), bar.modified_only())
    errors = []
    state = dict(active=True, selected=False)
    watchdog = QTimer(screen)
    watchdog.setSingleShot(True)

    def fail(message):
        errors.append(message)
        state['active'] = False
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog):
            dialog.reject()  # Failure cleanup only; never accept an unobserved selection.

    def choose():
        if not state['active'] or not isValid(field):
            return
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QFileDialog):
            fail('The actual reference file dialog did not open: ' + key)
            return
        try:
            capture_ui.choose_file(capture, dialog, value)
            if isValid(dialog) and dialog.isVisible():
                box = dialog.findChild(QDialogButtonBox)
                button = box.button(QDialogButtonBox.Open) if box is not None else None
                if button is None or not button.isVisible() or not button.isEnabled():
                    raise RuntimeError('The file dialog has no available Open button')
                capture_ui.press(capture, button)
            state['selected'] = True
        except Exception as exc:
            fail('The reference file selection failed: ' + str(exc))

    watchdog.timeout.connect(lambda: fail('The reference file selection timed out: ' + key))
    try:
        capture_ui.settings_modified_only(capture, bar, False)
        capture_ui.settings_level(capture, bar, 'all')
        capture_ui.search_settings(capture, bar, key)
        settle(.2)
        button = field._add_files_button
        capture_ui.scroll_to(capture, screen._settings_scroll, button)
        if not button.isVisible() or not button.isEnabled():
            raise RuntimeError('The actual Choose file button is unavailable: ' + key)
        watchdog.start(90000)
        capture_ui.when_modal(capture, 150, choose, timeout=30)
        capture_ui.press(capture, button)
        settle(.2)
        if errors or not state['selected'] or screen._settings_model.collect().get(key) != str(value):
            raise RuntimeError('; '.join(errors) or 'The selected reference did not reach its setting: ' + key)
    finally:
        state['active'] = False
        watchdog.stop()
        watchdog.deleteLater()
        capture_ui.search_settings(capture, bar, old[0])
        capture_ui.settings_level(capture, bar, old[1])
        capture_ui.settings_modified_only(capture, bar, old[2])


def prepare_default_references(screen, stage, capture, settle):
    """Select byte-identical private copies of the three actual default CSVs."""
    bundled = Path(__file__).resolve().parents[2] / 'spacr/resources/data'
    names = {'column_csv': 'barcodes_column.csv', 'row_csv': 'barcodes_row.csv',
             'grna_csv': 'barcodes_grna.csv'}
    before = screen._settings_model.collect()
    sources = {}
    for key, name in names.items():
        source = Path(before[key])
        if source.resolve() != (bundled / name).resolve() or not source.is_file():
            raise ValueError('The fresh Map Barcodes default reference differs: ' + key)
        sources[key] = source
    destination = Path(stage) / 'bundled_barcode_references'
    if not destination.parent.resolve().is_relative_to(Path(stage).resolve()):
        raise ValueError('The private default references must stay inside the capture stage')
    destination.mkdir(exist_ok=False)
    files = []
    for key, source in sources.items():
        target = destination / source.name
        expected = digest(source)
        with source.open('rb') as original, target.open('xb') as copied:
            shutil.copyfileobj(original, copied)
        if digest(source) != expected or digest(target) != expected:
            raise ValueError('A bundled barcode reference changed during private copying')
        files.append(dict(key=key, source=str(source), copy=str(target),
                          sha256=expected, bytes=target.stat().st_size))
    for row in files:
        select_reference_file(capture, screen, row['key'], row['copy'], settle)
    after = screen._settings_model.collect()
    expected_settings = dict(before)
    expected_settings.update({row['key']: row['copy'] for row in files})
    if after != expected_settings:
        mismatch = {key: (expected_settings.get(key), after.get(key))
                    for key in expected_settings.keys() | after.keys()
                    if expected_settings.get(key) != after.get(key)}
        raise ValueError('Private reference selection differs: ' + str(mismatch))
    for row in files:
        if digest(row['source']) != row['sha256'] or digest(row['copy']) != row['sha256']:
            raise ValueError('A bundled barcode reference changed during visible selection')
    return dict(files=files, other_settings_unchanged=True,
                initial_default_bytes_preserved=True, matching_library_references_selected_later=True)



def show_findings_rows(capture, table, rows, settle):
    """Finish the actual table scroll until every requested row is fully visible."""
    from PySide6.QtWidgets import QAbstractItemView
    table_row_to_top(capture, table, rows[0])
    settle(.2)
    seen = set()
    for _ in range(table.rowCount() + 1):
        viewport = table.viewport().rect()
        rectangles = [table.visualItemRect(table.item(row, 0)) for row in rows]
        if all(viewport.contains(rectangle) for rectangle in rectangles):
            return
        if not capture_ui.live(capture):
            break
        top, bottom = min(rect.top() for rect in rectangles), max(rect.bottom() for rect in rectangles)
        if bottom - top + 1 > viewport.height():
            raise RuntimeError('The findings viewport cannot contain the requested rows')
        bar = table.verticalScrollBar()
        before = bar.value()
        seen.add(before)
        # Real wheel input first. A notch moves several rows, so on a short live viewport it can overshoot the
        # rows and back (the value comes round to one already seen); only then finish with an exact real drag
        # of the scroll bar by the clipped amount.
        direction = 1 if bottom > viewport.bottom() else -1
        capture_ui._h().wheel(capture, table.viewport(), direction, viewport.center())
        settle(.2)
        if bar.value() in seen or bar.value() == before:
            if table.verticalScrollMode() == QAbstractItemView.ScrollPerItem:
                step = 1 if bottom > viewport.bottom() else -1
            else:
                step = (bottom - viewport.bottom()) if bottom > viewport.bottom() else min(top - viewport.top(), -1)
            target = max(bar.minimum(), min(bar.maximum(), before + step))
            if capture_ui.live(capture) and target != bar.value():
                capture_ui._h().drag_scroll_bar(capture, bar, target)
                settle(.2)
        if table.verticalScrollBar().value() == before:
            break
    raise RuntimeError('The recorded findings still clip their requested rows (viewport=%s, rows=%s, scroll=%s/%s, per_item=%s, dpr=%s)' % (
        table.viewport().rect().getRect(), [table.visualItemRect(table.item(r, 0)).getRect() for r in rows],
        table.verticalScrollBar().value(), table.verticalScrollBar().maximum(),
        table.verticalScrollMode() == QAbstractItemView.ScrollPerItem, table.devicePixelRatioF()))



def record_apply_settings(panel, model, capture, settle, *, splitters=()):
    """Keep the real Apply click and verified result in its selected scene."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    with Rec(capture, '09_proposed_settings', settle) as proposed:
        for split in splitters:
            capture_ui.splitter_sizes(capture, split, [350, 220, 830])
        settle(.4)
        proposed.point(panel.apply_button, dwell=1.5)
        if not panel.apply_button.isEnabled():
            raise RuntimeError('There are no settings to apply in this demonstration')
        changes = panel.proposed_changes()
        QTest.mouseClick(panel.apply_button, Qt.LeftButton)
        settle(.4)
        after = model.collect()
        for key, _, value in changes:
            if after.get(key) != value:
                raise RuntimeError('Apply did not write the displayed proposed value')
    return changes, after



def record_live_search(panel, capture, settle, timeout):
    """End progress only after returning from the search's Qt signal delivery."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    reports, partial_recorded, finished = [], [], []
    finished_slot = finished.append

    def updated(report):
        reports.append({'reads': report.reads, 'complete': report.complete})
        if not report.complete and not partial_recorded:
            partial_recorded.append(True)

    panel.search_finished.connect(finished_slot)
    panel.search_updated.connect(updated)
    progress = Rec(capture, '06_live_search_progress', settle).start()
    stopped = False

    def finish_partial():
        nonlocal stopped
        if partial_recorded and not stopped:
            stopped = True
            progress.stop()

    try:
        QTest.mouseClick(panel.search_button, Qt.LeftButton)
        deadline = time.monotonic() + timeout
        while panel.is_searching():
            finish_partial()
            if time.monotonic() > deadline:
                QTest.mouseClick(panel.cancel_button, Qt.LeftButton)
                raise TimeoutError('The real barcode search did not complete')
            settle(.1)
        finish_partial()  # Also covers synchronous or final-pump completion.
        settle(1)
    finally:
        panel.search_updated.disconnect(updated)
        panel.search_finished.disconnect(finished_slot)
        progress.abort()
    return reports, partial_recorded, finished


def record_search(app, screen, stage, captures, capture, settle, write_json, timeout,
                  *, reference_source=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractItemView, QSplitter
    from spacr.qt.widgets.card import Card

    references = prepare_references(reference_source or Path.home() / 'Documents/barcodes',
                                    stage / 'map_references' / captures.name)
    write_json(captures / 'references.json', references)
    model = screen._settings_model
    bar = screen._settings_search
    # Host: the clip starts before any field is touched, so the viewer sees
    # the reference tables typed in and the search that lists them.
    with Rec(capture, '04_reference_tables', settle):
        for role, versions in references.items():
            select_reference_file(capture, screen, role + '_csv', versions['plain']['path'], settle)
        # This is an explicit demonstration starting value, not a claimed default.
        if not capture_ui.fill_setting(capture, screen, 'window_length', 120):
            raise RuntimeError('Cannot configure the visible example window')
        capture_ui.settings_level(capture, bar, 'all')
        capture_ui.search_settings(capture, bar, 'csv')
        settle(.4)
        capture_ui.scroll_to_value(capture, screen._settings_scroll, 0)
    capture_ui.search_settings(capture, bar, '')
    if screen._console.isVisible():
        QTest.mouseClick(screen._console_header, Qt.LeftButton)
    for card in screen.findChildren(Card):
        if (card.folder is not None and card.title_label is not None
                and card.body.isVisible() and card.title_label.text().lstrip('▼▾ ') in {'Console', 'System'}):
            QTest.mouseClick(card.title_label, Qt.LeftButton)
    settle(.3)
    panel = screen._barcode_search
    toggle = screen._barcode_search_toggle
    if not toggle.isVisible() or not toggle.isEnabled():
        raise RuntimeError('The real Find barcodes toggle is unavailable')
    with Rec(capture, '05_find_barcodes', settle):
        QTest.mouseClick(toggle, Qt.LeftButton)
        settle(.5)
        if not panel.isVisible():
            raise RuntimeError('Find barcodes did not expose its real panel')
        for split in panel.findChildren(QSplitter):
            capture_ui.splitter_sizes(capture, split, [650, 430, 300])
    before = model.collect()
    # The first partial result requests a stop; the polling driver performs it
    # after signal delivery returns, never inside the search worker's callback.
    reports, partial_recorded, finished = record_live_search(panel, capture, settle, timeout)
    report, proposal = panel.report(), panel.proposal()
    write_json(captures / 'search_outcome.json', {
        'report': asdict(report) if report else None,
        'proposal': asdict(proposal) if proposal else None,
        'status': panel.status.text(), 'finished_signal': bool(finished and finished[-1] is report)})
    # At exact chunk-sized EOF, the current iterator leaves complete=False.
    # Require the genuine completion signal AND every downloaded record, not
    # a fabricated completion flag or a smaller-than-requested result.
    if (report is None or not finished or finished[-1] is not report
            or dict(report.reads_by_file) != {'R1': 10000, 'R2': 10000}
            or proposal is None or proposal.unresolved_roles):
        raise RuntimeError('Search did not establish every requested barcode')
    if not partial_recorded or len(reports) < 2:
        raise RuntimeError('No actual live refinement was observed')
    if model.collect() != before:
        raise RuntimeError('The search silently rewrote settings before Apply')
    with Rec(capture, '07_search_verdicts', settle) as verdicts:
        verdicts.point(panel.findings, panel.status, dwell=1.5)
    visible_roles = {}
    for role, name in [('grna', '07b_guide_verdicts'), ('row', '07c_row_verdicts')]:
        rows = [row for row in range(panel.findings.rowCount())
                if panel.findings.item(row, 0).text() == role]
        if len(rows) != 4:
            raise RuntimeError('Expected both mates and orientations for ' + role)
        with Rec(capture, name, settle):
            show_findings_rows(capture, panel.findings, rows, settle)
            settle(.4)
            viewport = panel.findings.viewport().rect()
            if not all(viewport.contains(panel.findings.visualItemRect(
                    panel.findings.item(row, 0))) for row in rows):
                raise RuntimeError('The recorded findings still clip ' + role)
            visible_roles[role] = [[panel.findings.item(row, column).text()
                                    for column in range(panel.findings.columnCount())]
                                   for row in rows]
    splits = panel.findChildren(QSplitter)
    with Rec(capture, '08_coloured_reads', settle):
        for split in splits:
            capture_ui.splitter_sizes(capture, split, [300, 900, 200])
        settle(.4)
    proposed_changes, after = record_apply_settings(panel, model, capture, settle, splitters=splits)
    # Older callers may select this follow-up view; the actual click is already
    # inside canonical scene09, which remains open through the result readback.
    with Rec(capture, '10_settings_applied', settle):
        settle(.4)
    write_json(captures / 'barcode_search.json', {
        'report': asdict(report), 'proposal': asdict(proposal),
        'live_updates': reports, 'settings_before_apply': before,
        'settings_after_apply': after, 'proposed_changes': proposed_changes,
        'no_automatic_settings_change': True,
        'demonstration_initial_window_length': 120,
        'applied_through_visible_button': True,
        'visible_role_rows': visible_roles,
        'application_functions_replaced': False})
    # The proposal explicitly requires reversed reference copies; Apply does
    # not create those files. Show this separate step without claiming it does.
    with Rec(capture, '11_oriented_reference_copies', settle):
        QTest.mouseClick(toggle, Qt.LeftButton)
        for role in proposal.reverse_complement_needed:
            if role not in references:
                raise RuntimeError('Unexpected reference role needs reversing')
            value = references[role]['reverse_complement']['path']
            select_reference_file(capture, screen, role + '_csv', value, settle)
        # Arbitrary sets are supported by settings/API; the current form does not
        # render that key. Its separate recorded API example must say so.
        capture_ui.search_settings(capture, bar, 'csv')
        settle(.4)
    capture_ui.search_settings(capture, bar, '')
    if not screen._console.isVisible():
        QTest.mouseClick(screen._console_header, Qt.LeftButton)
    for card in screen.findChildren(Card):
        if (card.folder is not None and card.title_label is not None
                and not card.body.isVisible() and card.title_label.text().lstrip('▶▸ ') == 'Console'):
            QTest.mouseClick(card.title_label, Qt.LeftButton)
    for role, versions in references.items():
        for row in versions.values():
            if digest(row['source']) != row['sha256'] or digest(row['path']) != row['sha256']:
                raise RuntimeError('A reference changed during the recording')
    write_json(captures / 'mapping_reference_selection.json', {
        role: model.collect()[role + '_csv'] for role in references})
