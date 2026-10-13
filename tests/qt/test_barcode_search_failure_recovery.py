"""A failed read or settings write leaves an honest, usable search panel."""
from types import SimpleNamespace

import pytest

from spacr.qt.screens import map_barcodes as mb


@pytest.mark.parametrize('name,sample,mate', [
    ('sample_R1.fastq.gz', 'sample', 'R1'),
    ('sample_R2.fastq.gz', 'sample', 'R2'),
    ('sample_1.fastq.gz', 'sample', 'R1'),
    ('sample_2.fastq.gz', 'sample', 'R2'),
    ('single.fastq.gz', 'single.fastq.gz', 'R1'),
])
def test_a_single_read_file_keeps_its_sample_and_mate(tmp_path, name, sample, mate):
    path = tmp_path / name
    path.touch()
    assert mb._fastq_files_under(path) == {sample: {mate: str(path)}}


def test_unreadable_directory_does_not_invent_a_sequencing_sample(tmp_path, monkeypatch):
    from spacr import io

    def unavailable(path):
        raise OSError('directory disappeared')
    monkeypatch.setattr(io, 'parse_gz_files', unavailable)
    assert mb._fastq_files_under(tmp_path) == {}


def test_named_barcode_set_takes_priority_and_missing_tables_fall_back(tmp_path, monkeypatch):
    from spacr import settings

    custom, shipped = tmp_path / 'custom.csv', tmp_path / 'shipped.csv'
    custom.touch()
    shipped.touch()
    entries = [SimpleNamespace(name='custom', csv=str(custom))]
    monkeypatch.setattr(settings, 'barcode_set_from_settings', lambda values: entries)
    values = {'grna_csv': str(shipped)}
    assert mb._planned_reference_tables(values) == (('custom', str(custom), 'custom'),)
    custom.unlink()
    assert mb._planned_reference_tables(values) == (('shipped', str(shipped), 'grna'),)
    def broken(values):
        raise ValueError('malformed set')
    monkeypatch.setattr(settings, 'barcode_set_from_settings', broken)
    assert mb._planned_reference_tables(values) == (('shipped', str(shipped), 'grna'),)


@pytest.mark.parametrize('message', ['cannot read references', ''])
def test_preparation_errors_are_returned_without_a_partial_recommendation(monkeypatch, message):
    def fail(settings):
        raise OSError(message)
    monkeypatch.setattr(mb, 'plan_barcode_search', fail)
    result = mb._prepare_barcode_search({}, 10, 5)
    assert result['error'] == (message or 'OSError')
    assert result['report'] is None and result['iterator'] is None


def test_chunk_and_read_sample_failures_are_explicit(monkeypatch):
    from spacr import barcode_search

    def broken():
        raise OSError('truncated reads')
        yield
    result = mb._advance_barcode_search(broken())
    assert result == {'report': None, 'error': 'truncated reads'}
    monkeypatch.setattr(barcode_search, 'iter_annotated_reads', lambda *a, **kw: broken())
    result = mb._sample_annotated_reads('unreadable.fastq.gz', (), '', 2)
    assert result == {'rows': (), 'error': 'truncated reads'}


@pytest.fixture
def panel(qtbot):
    value = mb.BarcodeSearchPanel(None, threaded=False)
    qtbot.addWidget(value)
    yield value
    value.shutdown()


@pytest.mark.parametrize('result,reason', [
    ({'error': 'reference unreadable'}, 'reference unreadable'),
    ({}, 'could not work out what to read'),
    ({'plan': mb.BarcodeSearchPlan({}, (), '', '', 0, '')}, 'hold no reads'),
])
def test_unusable_preparation_restores_controls_and_proposes_nothing(panel, result, reason):
    ended = []
    panel.search_finished.connect(ended.append)
    panel._running = True
    panel._on_prepared(result)
    assert not panel.is_searching() and panel.search_button.isEnabled()
    assert not panel.apply_button.isEnabled() and panel.proposal() is None
    assert ended == [None] and reason in panel.status.text()


def test_cancelled_search_ignores_every_late_callback(panel):
    ended = []
    panel.search_finished.connect(ended.append)
    panel._running = True
    assert panel.cancel_search()
    message = panel.status.text()
    panel._on_prepared({'error': 'late preparation'})
    panel._on_reads({'rows': [('ACGT', ())]})
    panel._on_chunk({'error': 'late chunk'})
    assert ended == [None] and panel.status.text() == message
    assert not panel.cancel_search()


def test_a_failed_chunk_ends_the_search_without_a_proposal(panel):
    panel._running = True
    panel._on_chunk({'error': 'disk disconnected'})
    assert not panel.is_searching() and panel.proposal() is None
    assert 'disk disconnected' in panel.status.text()


def test_apply_reports_only_settings_that_were_actually_written(panel):
    written, notifications = {}, []
    def set_value(key, value):
        if key == 'offset_start':
            raise ValueError('field no longer exists')
        written[key] = value
        return True
    panel._screen = SimpleNamespace(_settings_model=SimpleNamespace(
        collect=lambda: dict(written), set_value_for_key=set_value))
    panel._changes = (('offset_start', 0, 2), ('offset_end', 5, 9))
    panel.settings_applied.connect(notifications.append)
    assert panel.apply_proposal() == ('offset_end',)
    assert notifications == [written] == [{'offset_end': 9}]
    assert panel.proposed_changes() == () and not panel.apply_button.isEnabled()
    assert not panel._live_timer.isActive()


def test_unreadable_settings_do_not_escape_into_the_gui(panel):
    def unavailable():
        raise RuntimeError('form was deleted')
    panel._screen = SimpleNamespace(_settings_model=SimpleNamespace(collect=unavailable))
    assert panel.current_settings() == {}


def test_one_stack_watcher_follows_current_pages_and_survives_stack_removal(qtbot, monkeypatch):
    from PySide6.QtWidgets import QStackedWidget, QWidget

    window = QWidget()
    qtbot.addWidget(window)
    assert mb.install_window_hooks(window) is None
    stack = window._stack = QStackedWidget(window)
    pages = [QWidget(), QWidget()]
    for page in pages:
        stack.addWidget(page)
    installed = []
    monkeypatch.setattr(mb, 'install_folds_on', installed.append)
    watcher = mb.install_window_hooks(window)
    assert mb.install_window_hooks(window) is watcher
    qtbot.waitUntil(lambda: installed == [pages[0]])
    stack.setCurrentWidget(pages[1])
    assert installed == pages
    stack.removeWidget(pages[0])
    stack.removeWidget(pages[1])
    watcher.install_current()
    assert installed == pages
    del window._stack
    assert watcher.install_current() is None


def test_hiding_a_fold_keeps_the_page_reusable_and_never_hides_the_host(qtbot):
    from PySide6.QtWidgets import QTabWidget, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    tabs = host._fold_pages = QTabWidget(host)
    primary, folded = QWidget(), QWidget()
    tabs.addTab(primary, 'Host')
    tabs.addTab(folded, 'Fold')
    assert not mb.hide_as_page(primary, host)
    assert mb.hide_as_page(folded, host)
    assert tabs.count() == 1 and tabs.widget(0) is primary
    assert not mb.hide_as_page(folded, host)
    assert tabs.addTab(folded, 'Fold again') == 1


def test_registered_factory_failure_falls_back_to_a_usable_settings_screen(monkeypatch):
    calls, fallback = [], object()
    def fail(key):
        raise RuntimeError('factory unavailable')
    owner = SimpleNamespace(_build_screen=fail)
    def build(key, host):
        calls.append((key, host))
        return fallback
    monkeypatch.setattr(mb, 'build_settings_screen', build)
    assert mb.build_registered_screen('barcode_qc', owner) is fallback
    assert calls == [('barcode_qc', owner)]
