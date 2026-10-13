"""Record Report on a byte-verified private copy of an accepted project.

The caller owns the application, Help navigation and process timeout. This
module never launches an app/browser or fabricates a run stamp. Generation is
through the real GUI into a fresh private directory. Browser viewing is a
separate acceptance step; the driver deliberately does not click Open.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time


SOURCE_RELATIVE = 'external_mask_runs/example-pj_qwr_5/project'
CAPTURE_RELATIVE = 'captures/external_masks_final_console'
DATABASE_PATHS = ('artifacts.db', 'measurements/measurements.db')
CORE_SECTIONS = ['run_status', 'provenance', 'segmentation_qc', 'plate_qc',
                 'figures', 'statistics', 'settings', 'appendix']


def _picker_timeout(window, dialog, timer_type, interval_ms):
    """Keep the timer alive after a static QFileDialog destroys its dialog."""
    timer = timer_type(window)
    timer.setSingleShot(True)
    timer.timeout.connect(dialog.reject)
    timer.start(interval_ms)
    return timer


def _dispose_timers(timers):
    """Stop callbacks before deferred deletion; do not hide lifetime errors."""
    for timer in timers:
        timer.stop()
        timer.deleteLater()


def _retire_report_jobs(screen, settle):
    """Allow queued completion and thread-retirement signals to reach Report.

    A picker/capture failure can occur after folder selection starts a scan.
    Unwinding then would destroy the window while its QThread is still running.
    Do not reuse the potentially expired capture deadline or kill a worker:
    keep the caller's Qt event loop moving until both states are clear. The
    caller's outer process watchdog bounds a genuinely wedged worker.
    """
    polls = 0
    while screen.is_busy() or screen.active_jobs():
        settle(.05)
        polls += 1
    return {'event_processing_polls': polls, 'active_jobs': screen.active_jobs(),
            'busy': screen.is_busy(), 'workers_forcibly_stopped': False}


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot_source(source):
    """Hash only this bounded private project, including its actual file set."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError('Report input must be an existing non-symlink directory')
    result, total, entries = {}, 0, 0
    for path in sorted(source.rglob('*')):
        entries += 1
        if path.is_symlink() or entries > 350:
            raise RuntimeError('Unexpected symlink or enlarged Report source tree')
        if path.is_file():
            total += path.stat().st_size
            if total > 128 * 1024 * 1024:
                raise RuntimeError('Report source exceeds the bounded example size')
            result[str(path.relative_to(source))] = {
                'bytes': path.stat().st_size, 'sha256': _digest(path)}
    if not result:
        raise RuntimeError('The Report source is empty')
    return result


def require_unchanged(before, after):
    if before != after:
        raise RuntimeError('The original Report source file set or bytes changed')


def copy_private_source(original, destination):
    """Copy every bounded source file, without excluding pre-existing sidecars."""
    original, destination = Path(original).resolve(), Path(destination).resolve()
    if (destination.exists() or destination.is_relative_to(original)
            or original.is_relative_to(destination)):
        raise RuntimeError('Report requires a fresh private copy outside the original project')
    before = snapshot_source(original)
    shutil.copytree(original, destination, symlinks=True, copy_function=shutil.copy2)
    require_unchanged(before, snapshot_source(original))
    copied = snapshot_source(destination)
    require_unchanged(before, copied)
    for name in before:
        left, right = (original / name).stat(), (destination / name).stat()
        if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
            raise RuntimeError('A private Report source file must not be a hard link to the original')
    return before, copied


def verify_private_sqlite_changes(source, before, after):
    """Allow only documented empty-WAL/32-KiB-SHM changes in the private copy.

    The two base databases must be pre-existing, unchanged SQLite files. Other
    similarly named files are material, not exempt. The recorder itself never
    deletes a sidecar; app-created, changed or removed sidecars are reported.
    This allowance never applies to the accepted original project.
    """
    source = Path(source)
    allowed = {}
    for name in DATABASE_PATHS:
        if name not in before or before.get(name) != after.get(name):
            raise RuntimeError('A private Report base database changed or disappeared')
        with (source / name).open('rb') as handle:
            if handle.read(16) != b'SQLite format 3\x00':
                raise RuntimeError('Sidecar allowances require an actual existing SQLite database')
        allowed[name + '-wal'] = 0
        allowed[name + '-shm'] = 32768
    material_before = {name: value for name, value in before.items() if name not in allowed}
    material_after = {name: value for name, value in after.items() if name not in allowed}
    if material_before != material_after:
        raise RuntimeError('The private Report material file set or bytes changed')
    changes = []
    for name, expected_size in sorted(allowed.items()):
        previous, current = before.get(name), after.get(name)
        for value in (previous, current):
            if value is not None and (value['bytes'] != expected_size or
                    (expected_size == 0 and value['sha256'] != hashlib.sha256(b'').hexdigest())):
                raise RuntimeError('Private SQLite WAL must be empty and SHM must be exactly 32768 bytes')
        if previous != current:
            changes.append({'path': name, 'change': 'added' if previous is None else
                            'removed' if current is None else 'changed',
                            'before': previous, 'after': current})
    return changes


def read_source_facts(source):
    """Independently read small SQLite counts/stamps, never spaCR helpers.

    Immutable mode avoids creating read-lock sidecars. A live/nonempty WAL is
    rejected rather than ignored; this profile requires settled database files.
    """
    source = Path(source)
    counts, stamps = {}, []
    for relative in DATABASE_PATHS:
        path = source / relative
        wal = Path(str(path) + '-wal')
        if not path.is_file() or (wal.exists() and wal.stat().st_size):
            raise RuntimeError('Expected a settled existing Report database')
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            tables = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            if len(tables) > 40:
                raise RuntimeError('Unexpectedly many Report source tables')
            counts[relative] = {}
            for table in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                count = connection.execute('SELECT COUNT(*) FROM ' + quoted).fetchone()[0]
                if count > 10000:
                    raise RuntimeError('Unexpectedly large Report source table')
                counts[relative][table] = count
            if 'run_status' in tables:
                connection.row_factory = sqlite3.Row
                for row in connection.execute('SELECT * FROM run_status ORDER BY rowid LIMIT 9'):
                    stamps.append({'artifact': Path(relative).name, **dict(row)})
        finally:
            connection.close()
    if (len(stamps) != 1 or stamps[0]['name'] != 'measure_crop'
            or stamps[0]['status'] != 'complete'
            or [stamps[0][key] for key in ('n_attempted', 'n_succeeded', 'n_failed')] != [2, 2, 0]):
        raise RuntimeError('The actual measurement stamp differs from the approved two-field example')
    measured = counts['measurements/measurements.db']
    if any(measured.get(table) != 103 for table in ('cell', 'cytoplasm', 'png_list')):
        raise RuntimeError('The accepted 103-cell/duplicate-cytoplasm/PNG counts changed')
    figures = sorted(str(p.relative_to(source)) for p in (source / 'results').rglob('*') if p.is_file())
    expected = sorted(f'results/plate1_A01_{field}/{name}.pdf'
                      for field in (1, 2) for name in ('after_filtration', 'before_filtration', 'pngs'))
    if figures != expected or (source / 'qc').exists():
        raise RuntimeError('The approved six-PDF, missing-QC source profile changed')
    return {'database_counts': counts, 'stamps': stamps, 'vector_figures': figures,
            'conversion_stamp_note': 'The real conversion stamp under images/ is outside the current Report artifact scan; no claim it appears in this report.'}


def verify_report_summary(summary, source, facts):
    """Check returned data identities against the independent source facts."""
    if Path(summary['src']).resolve() != Path(source).resolve():
        raise RuntimeError('Report refers to a different source project')
    sections = summary['sections']
    if [section['key'] for section in sections] != CORE_SECTIONS:
        raise RuntimeError('Missing, duplicated, reordered or unexpected Report sections')
    by_key = {section['key']: section for section in sections}
    if (summary['status'] != 'complete'
            or any(by_key[key]['status'] != 'missing' for key in ('segmentation_qc', 'plate_qc'))):
        raise RuntimeError('Report run status or unavailable-QC disclosure differs')
    rows = by_key['run_status']['rows']
    stamp = facts['stamps'][0]
    expected = [stamp['artifact'], stamp['name'], stamp['status'],
                *[str(stamp[key]) for key in ('n_attempted', 'n_succeeded', 'n_failed')]]
    if len(rows) != 1 or rows[0][:6] != expected:
        raise RuntimeError('Report processing-stamp identity or counts differ from SQLite')
    if summary['figures_found'] != 6 or summary['figures_embedded'] != 0:
        raise RuntimeError('The six vector PDFs must be listed, not embedded')
    if by_key['figures']['rows'] != [[name, 'vector — not embeddable'] for name in facts['vector_figures']]:
        raise RuntimeError('The exact six source PDF identities were not retained')
    expected_counts = {name: sum(tables.values()) for name, tables in facts['database_counts'].items()}
    actual_counts = {}
    for row in by_key['statistics']['rows']:
        if len(row) != 4 or row[1] != 'sqlite' or row[0] in actual_counts:
            raise RuntimeError('Unexpected or duplicate Report database inventory row')
        actual_counts[row[0]] = int(row[2])
    if actual_counts != expected_counts:
        raise RuntimeError('Report database identities or aggregate row counts differ')
    return True


class _Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections, self.current, self.text, self.images = [], None, [], []
        self.row, self.cell, self.forbidden = None, None, []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'section':
            self.current = {'key': attrs.get('id'), 'classes': attrs.get('class', '').split(), 'rows': []}
            self.sections.append(self.current)
        if tag == 'tr' and self.current is not None:
            self.row = []
        if tag in ('td', 'th') and self.row is not None:
            self.cell = []
        if tag == 'img':
            self.images.append(attrs.get('src', ''))
        if tag in ('script', 'iframe', 'link') or any(
                value.startswith(('http:', 'https:', '//')) for key, value in attrs.items()
                if key in ('src', 'href')):
            self.forbidden.append(tag)

    def handle_data(self, text):
        self.text.append(text)
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.current['rows'].append(self.row)
            self.row = None
        if tag == 'section':
            self.current = None


def verify_html(text, summary, facts):
    """Check actual generated HTML, without rendering or rebuilding it."""
    document = _Document()
    document.feed(text)
    if document.forbidden or document.images:
        raise RuntimeError('Unexpected active/external/embedded content in this vector-only HTML example')
    if [section['key'] for section in document.sections] != CORE_SECTIONS:
        raise RuntimeError('Generated HTML dropped or duplicated Report sections')
    plain = ''.join(document.text)
    if summary['src'] not in plain or summary['status_detail'] not in plain:
        raise RuntimeError('Generated HTML omitted the source identity or actual run status')
    for actual, expected in zip(document.sections, summary['sections']):
        if ('missing' in actual['classes']) != (expected['status'] == 'missing'):
            raise RuntimeError('Generated HTML changed a missing-section disclosure')
        for row in expected['rows']:
            if actual['rows'].count(row) != 1:
                raise RuntimeError('Generated HTML altered a returned table identity or value')
    statistics_rows = document.sections[5]['rows']
    for counts in facts['database_counts'].values():
        for table, count in counts.items():
            if [table, str(count)] not in statistics_rows:
                raise RuntimeError('Generated HTML omitted an actual database table count')
    return {'accepted': True, 'section_count': len(document.sections),
            'embedded_images': len(document.images), 'external_dependencies': False,
            'browser_rendering_reviewed': False}


def record_report(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    """Capture real Scan/Generate controls; leave external viewing to the caller."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QListWidget

    stage, captures = Path(stage).resolve(), Path(captures).resolve()
    if not captures.is_relative_to(stage / 'captures') or not captures.is_dir():
        raise ValueError('Report needs an existing private capture directory')
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError('Report needs a positive finite timeout')
    deadline = time.monotonic() + timeout
    evidence = {'accepted': False, 'published': False, 'lesson': '29_report',
                'reason': 'Actual Report scan/output has not been verified',
                'acceptance_scope': 'Report collection and HTML generation only; not QC or scientific validation',
                'browser_reviewed': False, 'open_control_clicked': False,
                'synthetic_status_created': False, 'source_pipeline_rerun': False}
    acceptance = captures / 'scientific_acceptance.json'
    write_json(acceptance, evidence)
    original, original_before = None, None
    jobs = []
    job_finished = jobs.append
    screen.job_finished.connect(job_finished)

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('Report capture exceeded its lifecycle timeout')
        settle(.02)

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise RuntimeError('A requested Report control is not visibly usable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def wait_job():
        while screen.is_busy() or screen.active_jobs():
            tick()
        if screen.last_error or not jobs or not all(jobs):
            raise RuntimeError(screen.last_error or 'An actual Report job failed')
        settle(.2)

    def check_copies():
        original_after = snapshot_source(original)
        require_unchanged(original_before, original_after)
        private_after = snapshot_source(source)
        changes = verify_private_sqlite_changes(source, before, private_after)
        evidence.update(original_preserved=True, original_source_files_after=original_after,
                        source_files_after=private_after, private_material_files_unchanged=True,
                        private_sqlite_sidecar_changes=changes)
        return changes

    def choose_path(button, path, frame, save=False):
        errors, accepted, timers = [], [], []

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The actual Report file picker did not open')
                dialog.accepted.connect(lambda: accepted.append(True))
                timer = _picker_timeout(
                    window, dialog, QTimer,
                    max(1, min(12000, int((deadline - time.monotonic()) * 1000))))
                timers.append(timer)
                dialog.resize(1500, 950)
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                if edit is None:
                    raise RuntimeError('The actual file picker has no path editor')
                edit.setFocus()
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(path))
                QTest.keyClick(edit, Qt.Key_Tab)
                settle(.15)
                capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                target = None if box is None else box.button(QDialogButtonBox.Save if save else QDialogButtonBox.Open)
                if target is None:
                    raise RuntimeError('The actual picker lacks its confirmation control')
                click(target)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        opener = QTimer(window)
        opener.setSingleShot(True)
        opener.timeout.connect(handle)
        opener.start(350)
        try:
            click(button)
        finally:
            _dispose_timers([opener, *timers])
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'The actual Report picker was not accepted')

    try:
        previous = stage / CAPTURE_RELATIVE
        documents = {name: json.loads((previous / name).read_text()) for name in
                     ('scientific_acceptance.json', 'provenance.json', 'output_evidence.json')}
        if (documents['scientific_acceptance.json'].get('accepted') is not True
                or documents['provenance.json'].get('completed_capture') is not True
                or documents['output_evidence.json'].get('accepted') is not True):
            raise RuntimeError('The exact reused External Masks capture is not complete and accepted')
        original = stage / SOURCE_RELATIVE
        if Path(documents['scientific_acceptance.json']['destination']).resolve() != original:
            raise RuntimeError('The accepted External Masks receipt names a different project')
        runs = stage / 'report_runs'
        runs.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='example-', dir=runs))
        source = work / 'source_project'
        original_before, before = copy_private_source(original, source)
        facts = read_source_facts(source)
        verify_private_sqlite_changes(source, before, snapshot_source(source))
        require_unchanged(original_before, snapshot_source(original))
        evidence.update(source=str(source), original_source=str(original),
                        source_capture=str(previous), source_facts=facts,
                        prior_evidence_sha256={name: _digest(previous / name) for name in documents},
                        source_files_before=before, original_source_files_before=original_before,
                        full_copy_byte_identical=True, copied_file_count=len(before),
                        original_preserved=True, private_sqlite_sidecar_changes=[],
                        private_copy_note='All original files, including existing SQLite sidecars, were copied byte-for-byte. Only this copy is submitted to Report; no run stamp or data is fabricated.')
        write_json(captures / 'input_manifest.json', evidence)
        destination = work / 'reports'
        destination.mkdir()
        output = destination / 'external_masks_report.html'
        if screen.report is not None or screen.written:
            raise RuntimeError('A fresh Report panel is required')
        capture('01_current_report_controls')
        choose_path(screen._btn_pick_src, source, '02_existing_project_picker')
        wait_job()
        click(screen._btn_scan)
        wait_job()
        report = screen.report
        summary = {'src': str(report.src), 'title': report.title, 'status': report.status,
                   'status_detail': report.status_detail, 'figures_found': report.n_figures_found,
                   'figures_embedded': report.n_figures_embedded,
                   'sections': [{'key': section.key, 'title': section.title, 'status': section.status,
                                 'rows': section.table.rows if section.table is not None else []}
                                for section in report.sections]}
        verify_report_summary(summary, source, facts)
        if (screen._sections.selectionMode() != QListWidget.NoSelection
                or any(screen._sections.item(i).flags() & Qt.ItemIsSelectable
                       for i in range(screen._sections.count()))):
            raise RuntimeError('Expected the actual read-only Report section inventory')
        if list(destination.iterdir()):
            raise RuntimeError('Report Scan unexpectedly wrote output')
        check_copies()
        write_json(captures / 'scan_evidence.json', {**summary, 'no_report_output_written': True,
                    'original_preserved': True, 'private_material_files_unchanged': True,
                    'private_sqlite_sidecar_changes': evidence['private_sqlite_sidecar_changes'],
                    'sections_selectable': False})
        capture('03_actual_completion_not_qc_pass')
        capture('04_read_only_missing_sections')
        click(screen._format)
        capture('05_actual_output_format_choices', desktop=True)
        QTest.keyClick(screen._format, Qt.Key_Escape)
        if screen.output_format() != 'html':
            raise RuntimeError('The fresh Report format must be HTML')
        screen._figure_cap.setFocus()
        QTest.keyClick(screen._figure_cap, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(screen._figure_cap, '3')
        QTest.keyClick(screen._figure_cap, Qt.Key_Tab)
        if screen.figure_cap() != 3:
            raise RuntimeError('The actual figure-cap editor did not retain 3')
        capture('06_actual_figure_cap')
        choose_path(screen._btn_pick_out, output, '07_separate_output_picker', save=True)
        if Path(screen._out_edit.text()).resolve() != output:
            raise RuntimeError('The actual output picker retained the wrong path')
        capture('08_generate_report')
        click(screen._btn_generate)
        wait_job()
        if screen.written != [str(output)] or not output.is_file():
            raise RuntimeError('Report generation did not write exactly the selected HTML file')
        if sorted(destination.iterdir()) != [output]:
            raise RuntimeError('Unexpected output files were generated')
        html_evidence = verify_html(output.read_text(), summary, facts)
        check_copies()
        capture('09_actual_report_written')
        capture('10_open_control_not_yet_invoked')
        evidence.update(accepted=True, reason='Real Scan and Generate returned the exact source stamp, six PDF identities and database counts; generated HTML retains sections and missing-QC disclosure; accepted original preserved and private-copy material unchanged, with any allowed SQLite sidecar changes explicitly recorded',
                        destination=str(destination), output=str(output), output_sha256=_digest(output),
                        report=summary, html=html_evidence,
                        job_results=jobs, active_jobs=screen.active_jobs(), figure_cap=3)
        write_json(captures / 'report_output_evidence.json', evidence)
        write_json(acceptance, evidence)
        return output
    except Exception as exc:
        evidence.update(accepted=False, reason=str(exc), job_results=jobs)
        write_json(acceptance, evidence)
        raise
    finally:
        retirement = _retire_report_jobs(screen, settle)
        screen.job_finished.disconnect(job_finished)
        if original_before is not None:
            original_after = snapshot_source(original)
            evidence['original_preserved'] = original_before == original_after
            evidence['original_source_files_after'] = original_after
            if not evidence['original_preserved']:
                evidence.update(accepted=False, reason='The accepted original changed during Report capture')
                write_json(acceptance, evidence)
                require_unchanged(original_before, original_after)
        if evidence['accepted'] is False:
            evidence.update(error_cleanup=retirement, active_jobs=retirement['active_jobs'],
                            job_results=jobs)
            write_json(acceptance, evidence)
