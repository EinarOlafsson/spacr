"""Report guards and QtCore timer lifetimes; no GUI or real pipeline is run."""
from copy import deepcopy
import html
import importlib.util
from pathlib import Path
import sqlite3

import pytest


MODULE = Path(__file__).resolve().parents[1] / 'capture_report.py'
spec = importlib.util.spec_from_file_location('capture_report', MODULE)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'project'
    (root / 'measurements').mkdir(parents=True)
    connection = sqlite3.connect(root / 'artifacts.db')
    connection.execute('CREATE TABLE artifacts (id INTEGER)')
    connection.execute('INSERT INTO artifacts VALUES (1)')
    connection.commit()
    connection.close()
    connection = sqlite3.connect(root / 'measurements/measurements.db')
    for table in ('cell', 'cytoplasm', 'png_list'):
        connection.execute(f'CREATE TABLE {table} (id INTEGER)')
        connection.executemany(f'INSERT INTO {table} VALUES (?)', [(i,) for i in range(103)])
    connection.execute('CREATE TABLE run_status (name TEXT, status TEXT, n_attempted INTEGER, n_succeeded INTEGER, n_failed INTEGER)')
    connection.execute("INSERT INTO run_status VALUES ('measure_crop','complete',2,2,0)")
    connection.commit()
    connection.close()
    for field in (1, 2):
        target = root / f'results/plate1_A01_{field}'
        target.mkdir(parents=True)
        for name in ('after_filtration', 'before_filtration', 'pngs'):
            (target / f'{name}.pdf').write_bytes(b'fixture only; not a rendered PDF')
    return root


def summary_for(source, facts):
    sections = [{'key': key, 'title': key, 'status': 'missing' if key in
                 ('segmentation_qc', 'plate_qc', 'provenance') else 'ok', 'rows': []}
                for key in report.CORE_SECTIONS]
    sections[0]['rows'] = [['measurements.db', 'measure_crop', 'complete', '2', '2', '0', '2026-09-09']]
    sections[4]['rows'] = [[name, 'vector — not embeddable'] for name in facts['vector_figures']]
    sections[5]['rows'] = [[name, 'sqlite', str(sum(counts.values())), 'size']
                           for name, counts in facts['database_counts'].items()]
    return {'src': str(source), 'title': 'Example', 'status': 'complete',
            'status_detail': 'Complete — every stamped step processed every item.',
            'figures_found': 6, 'figures_embedded': 0, 'sections': sections}


def html_for(summary, facts):
    def table(rows):
        return '<table>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(cell)) + '</td>'
                                                   for cell in row) + '</tr>' for row in rows) + '</table>'
    parts = ['<html><body>', html.escape(summary['src']), html.escape(summary['status_detail'])]
    for section in summary['sections']:
        classes = 'chapter missing' if section['status'] == 'missing' else 'chapter'
        parts.append(f'<section class="{classes}" id="{section["key"]}">')
        parts.append(table(section['rows']))
        if section['key'] == 'statistics':
            for counts in facts['database_counts'].values():
                parts.append(table([[name, str(count)] for name, count in counts.items()]))
        parts.append('</section>')
    return ''.join(parts) + '</body></html>'


def test_positive_source_summary_and_real_html_structure(source):
    before = report.snapshot_source(source)
    facts = report.read_source_facts(source)
    summary = summary_for(source, facts)
    assert report.verify_report_summary(summary, source, facts) is True
    result = report.verify_html(html_for(summary, facts), summary, facts)
    assert result['accepted'] is True
    assert result['section_count'] == 8
    assert result['browser_rendering_reviewed'] is False
    report.require_unchanged(before, report.snapshot_source(source))


@pytest.mark.parametrize('column,value', [('name', 'convert'), ('status', 'partial'),
                                        ('n_attempted', 3), ('n_succeeded', 1), ('n_failed', 1)])
def test_rejects_changed_actual_stamp(source, column, value):
    connection = sqlite3.connect(source / 'measurements/measurements.db')
    connection.execute(f'UPDATE run_status SET {column}=?', (value,))
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match='stamp'):
        report.read_source_facts(source)


@pytest.mark.parametrize('table', ['cell', 'cytoplasm', 'png_list'])
def test_rejects_changed_accepted_counts(source, table):
    connection = sqlite3.connect(source / 'measurements/measurements.db')
    connection.execute(f'DELETE FROM {table} WHERE id=0')
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match='counts changed'):
        report.read_source_facts(source)


def test_rejects_unsettled_database(source):
    (source / 'measurements/measurements.db-wal').write_bytes(b'uncheckpointed data')
    with pytest.raises(RuntimeError, match='settled'):
        report.read_source_facts(source)


def test_rejects_duplicate_stamp(source):
    connection = sqlite3.connect(source / 'measurements/measurements.db')
    connection.execute('INSERT INTO run_status SELECT * FROM run_status')
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match='stamp'):
        report.read_source_facts(source)


@pytest.mark.parametrize('change', ['qc', 'missing_pdf', 'unexpected_raster'])
def test_rejects_changed_figure_or_qc_profile(source, change):
    if change == 'qc':
        (source / 'qc').mkdir()
    elif change == 'missing_pdf':
        (source / 'results/plate1_A01_1/pngs.pdf').unlink()
    else:
        (source / 'results/unexpected.png').write_bytes(b'fixture')
    with pytest.raises(RuntimeError, match='profile changed'):
        report.read_source_facts(source)


@pytest.mark.parametrize('change', ['source', 'stamp_identity', 'stamp_count', 'missing_section',
                                  'duplicate_section', 'qc_pass', 'embedded', 'figure_identity',
                                  'db_identity', 'db_count', 'db_duplicate'])
def test_rejects_incorrect_returned_summary(source, change):
    facts = report.read_source_facts(source)
    summary = summary_for(source, facts)
    if change == 'source':
        summary['src'] += '_wrong'
    elif change == 'stamp_identity':
        summary['sections'][0]['rows'][0][0] = 'wrong.db'
    elif change == 'stamp_count':
        summary['sections'][0]['rows'][0][4] = '1'
    elif change == 'missing_section':
        summary['sections'].pop(2)
    elif change == 'duplicate_section':
        summary['sections'][2] = deepcopy(summary['sections'][1])
    elif change == 'qc_pass':
        summary['sections'][2]['status'] = 'ok'
    elif change == 'embedded':
        summary['figures_embedded'] = 1
    elif change == 'figure_identity':
        summary['sections'][4]['rows'][0][0] = 'other.pdf'
    elif change == 'db_identity':
        summary['sections'][5]['rows'][0][0] = 'other.db'
    elif change == 'db_count':
        summary['sections'][5]['rows'][0][2] = '999'
    else:
        summary['sections'][5]['rows'].append(deepcopy(summary['sections'][5]['rows'][0]))
    with pytest.raises(RuntimeError):
        report.verify_report_summary(summary, source, facts)


@pytest.mark.parametrize('change', ['source', 'stamp', 'missing_tag', 'section', 'count',
                                  'script', 'remote', 'embedded'])
def test_rejects_incorrect_generated_html(source, change):
    facts = report.read_source_facts(source)
    summary = summary_for(source, facts)
    text = html_for(summary, facts)
    if change == 'source':
        text = text.replace(str(source), '/wrong/source')
    elif change == 'stamp':
        text = text.replace('<td>measure_crop</td>', '<td>not_the_run</td>')
    elif change == 'missing_tag':
        text = text.replace('chapter missing', 'chapter', 1)
    elif change == 'section':
        text = text.replace('id="plate_qc"', 'id="wrong"')
    elif change == 'count':
        text = text.replace('<td>cell</td><td>103</td>', '<td>cell</td><td>102</td>')
    elif change == 'script':
        text += '<script>bad()</script>'
    elif change == 'remote':
        text += '<img src="https://example.invalid/figure.png">'
    else:
        text += '<img src="data:image/png;base64,aA==">'
    with pytest.raises(RuntimeError):
        report.verify_html(text, summary, facts)


def test_rejects_source_bytes_changed(source):
    before = report.snapshot_source(source)
    (source / 'results/plate1_A01_1/pngs.pdf').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='source file set or bytes changed'):
        report.require_unchanged(before, report.snapshot_source(source))


def test_rejects_source_file_added(source):
    before = report.snapshot_source(source)
    (source / 'extra.txt').write_text('unexpected')
    with pytest.raises(RuntimeError, match='source file set or bytes changed'):
        report.require_unchanged(before, report.snapshot_source(source))


def test_rejects_symlink_source_member(source):
    (source / 'linked').symlink_to(source / 'artifacts.db')
    with pytest.raises(RuntimeError, match='symlink'):
        report.snapshot_source(source)


@pytest.fixture
def core_timers():
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QTimer
    from shiboken6 import delete, isValid

    class Picker(QObject):
        rejected = 0

        def reject(self):
            self.rejected += 1

    app = QCoreApplication.instance() or QCoreApplication([])
    owner = QObject()
    dialog = Picker()
    yield app, owner, dialog, QTimer, delete, isValid
    if isValid(dialog):
        delete(dialog)
    if isValid(owner):
        delete(owner)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_picker_timer_survives_dialog_deletion_and_is_cleaned(core_timers):
    from PySide6.QtCore import QCoreApplication, QEvent

    _, owner, dialog, timer_type, delete, is_valid = core_timers
    timer = report._picker_timeout(owner, dialog, timer_type, 60000)
    assert timer.parent() is owner
    assert timer.isActive()
    delete(dialog)  # The static picker has returned and destroyed its widget.
    assert is_valid(timer)
    report._dispose_timers([timer])
    assert not timer.isActive()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not is_valid(timer)


def test_dialog_owned_timer_reproduces_deleted_cpp_object_failure(core_timers):
    _, _, dialog, timer_type, delete, is_valid = core_timers
    timer = timer_type(dialog)  # The old, incorrect ownership.
    timer.start(60000)
    delete(dialog)
    assert not is_valid(timer)
    with pytest.raises(RuntimeError, match='deleted'):
        report._dispose_timers([timer])


def test_live_picker_timeout_still_rejects_once(core_timers):
    app, owner, dialog, timer_type, _, _ = core_timers
    timer = report._picker_timeout(owner, dialog, timer_type, 0)
    app.processEvents()
    app.processEvents()
    assert dialog.rejected == 1
    assert not timer.isActive()
    report._dispose_timers([timer])


class _JobState:
    def __init__(self, states):
        self.states = list(states)
        self.polls = []

    def is_busy(self):
        return self.states[0][0]

    def active_jobs(self):
        return self.states[0][1]

    def settle(self, seconds):
        self.polls.append(seconds)
        assert len(self.states) > 1, 'Unexpected extra wait after retirement'
        self.states.pop(0)

    def close(self):
        raise AssertionError('Do not close a screen to stop its worker')


def test_error_cleanup_waits_for_queued_thread_retirement_after_busy_clears():
    screen = _JobState([(True, 1), (False, 1), (False, 0)])
    result = report._retire_report_jobs(screen, screen.settle)
    assert screen.polls == [.05, .05]
    assert result == {'event_processing_polls': 2, 'active_jobs': 0,
                      'busy': False, 'workers_forcibly_stopped': False}


def test_idle_error_cleanup_does_not_wait():
    screen = _JobState([(False, 0)])
    assert report._retire_report_jobs(screen, screen.settle)['event_processing_polls'] == 0


def test_error_cleanup_does_not_swallow_event_processing_errors():
    screen = _JobState([(True, 1)])

    def broken_settle(_seconds):
        raise RuntimeError('event processing failed')

    with pytest.raises(RuntimeError, match='event processing failed'):
        report._retire_report_jobs(screen, broken_settle)
