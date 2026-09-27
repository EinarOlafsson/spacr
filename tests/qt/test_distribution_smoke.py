"""Behavioral guards for real installed-application acceptance, not native receipts."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('PySide6')
from spacr.qt import startup_benchmark as benchmark

Controller = benchmark._DistributionSmokeController


def test_artifact_origin_rejects_source_and_symlink_escape(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    source = tmp_path / 'checkout.py'
    source.write_text('')
    Controller._require_installed_origin(bundle / 'spacr' / '__init__.py', bundle)
    with pytest.raises(RuntimeError, match='outside the installed artifact'):
        Controller._require_installed_origin(source, bundle)
    link = bundle / 'borrowed.py'
    try:
        link.symlink_to(source)
    except OSError:
        return
    with pytest.raises(RuntimeError, match='outside the installed artifact'):
        Controller._require_installed_origin(link, bundle)


def test_database_proof_requires_real_rows_and_releases_windows_handle(tmp_path):
    database = tmp_path / 'measurements.db'
    with pytest.raises(RuntimeError, match='did not produce'):
        Controller._read_result(database)
    connection = sqlite3.connect(database)
    connection.executescript('''
        CREATE TABLE run_status (status TEXT, n_succeeded INTEGER, n_failed INTEGER);
        CREATE TABLE cell (object_id INTEGER);
        INSERT INTO run_status VALUES ('complete',1,0);
        INSERT INTO cell VALUES (1);
    ''')
    connection.close()
    assert Controller._read_result(database) == (('complete', 1, 0), 1)
    database.unlink()
    assert not database.exists()


def test_macos_bundle_allows_resources_sibling_but_not_checkout_symlink(tmp_path):
    contents = tmp_path / 'spaCR.app' / 'Contents'
    macos, frameworks, resources = (contents / name for name in ('MacOS', 'Frameworks', 'Resources'))
    for folder in (macos, frameworks, resources):
        folder.mkdir(parents=True)
    executable = macos / 'spacr'
    executable.write_text('')
    root = Controller._installed_root(executable, frameworks, 'darwin')
    assert root == contents.parent
    Controller._require_installed_origin(resources / 'spacr' / '__init__.py', root)
    outside = tmp_path / 'checkout'
    outside.mkdir()
    good = frameworks / 'resources'
    bad = frameworks / 'borrowed'
    try:
        good.symlink_to(resources, target_is_directory=True)
        bad.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    Controller._require_installed_origin(good / 'spacr.py', root)
    with pytest.raises(RuntimeError, match='outside the installed artifact'):
        Controller._require_installed_origin(bad / 'spacr.py', root)
    with pytest.raises(RuntimeError, match='outside the installed artifact'):
        Controller._installed_root(executable, outside, 'darwin')
    with pytest.raises(RuntimeError, match='not inside its installed app'):
        Controller._installed_root(tmp_path / 'spacr', tmp_path, 'darwin')


def test_smoke_mode_is_explicit_and_keeps_normal_registry_sweep(monkeypatch):
    calls = []
    normal = object()
    smoke = object()
    monkeypatch.setenv(benchmark.OUTPUT_ENV, 'receipt.json')
    monkeypatch.delenv('SPACR_DISTRIBUTION_SMOKE', raising=False)
    monkeypatch.setattr(benchmark, 'BenchmarkController', lambda *a, **kw: normal)
    monkeypatch.setattr(benchmark, '_DistributionSmokeController',
                        lambda *a: calls.append(a) or smoke)
    assert benchmark.maybe_start('app', 'window') is normal
    assert not calls
    monkeypatch.setenv('SPACR_DISTRIBUTION_SMOKE', '1')
    assert benchmark.maybe_start('app', 'window') is smoke
    assert calls == [('app', 'window', 'receipt.json')]
    monkeypatch.delenv(benchmark.OUTPUT_ENV)
    assert benchmark.maybe_start('app', 'window') is None


def test_terminal_success_waits_for_worker_and_requires_complete_database(tmp_path):
    calls = []
    fake = SimpleNamespace(
        started=benchmark.time.monotonic(), phase='running',
        screen=SimpleNamespace(_thread=object()), record={},
        window=SimpleNamespace(findChildren=lambda kind: []),
        _pipeline_failed=lambda error: calls.append(('failed', error)),
        _finish=lambda code: calls.append(('finished', code)),
    )
    Controller._advance(fake)
    assert not calls
    fake.screen._thread = None
    fake.database = tmp_path / 'measurements.db'
    fake._read_result = lambda path: (('failed', 0, 1), 0)
    Controller._advance(fake)
    assert calls and calls[0][0] == 'failed'
    assert not any(kind == 'finished' for kind, _ in calls)


def test_premature_quit_records_failure_instead_of_green(tmp_path):
    target = tmp_path / 'receipt.json'
    fake = SimpleNamespace(phase='running', record={'status': 'running'}, output=target)
    fake._write = lambda: Controller._write(fake)
    Controller._quitting(fake)
    assert json.loads(target.read_text())['status'] == 'failed'


def test_existing_successful_experiment_cannot_be_reused(tmp_path):
    root = tmp_path / 'experiment'
    root.mkdir()
    database = root / 'measurements.db'
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE run_status (status TEXT)')
    connection.execute("INSERT INTO run_status VALUES ('complete')")
    connection.commit()
    connection.close()
    calls = []
    fake = SimpleNamespace(
        started=benchmark.time.monotonic(), phase='launch', root=root,
        window=SimpleNamespace(findChildren=lambda kind: []),
        _provenance=lambda: calls.append('unexpected provenance'),
        _pipeline_failed=lambda error: calls.append(error))
    before = database.read_bytes()
    Controller._advance(fake)
    assert len(calls) == 1 and 'refusing stale analysis output' in calls[0]
    assert database.read_bytes() == before


def test_source_ui_integration_runs_real_measurement(qapp, qtbot, monkeypatch, tmp_path):
    """Exercise controller/Run/SQLite locally; this is not frozen acceptance."""
    import spacr.qt
    from spacr.qt.app import MainWindow
    from spacr.qt.preferences import set_preload_policy

    spacr.qt.register_self_registering_modules()
    set_preload_policy('on_demand')
    completed = []

    def source_scope(self):
        """Permit the test interpreter only in this explicitly source-level test."""
        self.record['test_scope'] = 'source UI integration, not native artifact acceptance'

    def finish(self, code):
        """Retain the test QApplication after the real pipeline has terminated."""
        self.phase = 'finished'
        self.timer.stop()
        self._write()
        completed.append(code)

    monkeypatch.setattr(Controller, '_provenance', source_scope)
    monkeypatch.setattr(Controller, '_finish', finish)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show()
    controller = Controller(qapp, window, str(tmp_path / 'source-unit.json'))
    try:
        qtbot.waitUntil(lambda: bool(completed), timeout=300000)
        record = json.loads((tmp_path / 'source-unit.json').read_text())
        assert completed == [0], record
        assert record['module_constructed'] and record['real_run_clicked']
        assert record['run_status'] == ['complete', 1, 0] and record['cells'] > 0
        assert record['worker_finished']
        assert (window.width(), window.height()) == (1280, 720)
        assert record['visual_layout_status'] == 'passed'
        assert record['layout']['scroll_origin_restored'] is True
    finally:
        controller.timer.stop()
        window.close()


def test_layout_snapshot_retains_actual_clipping_without_resizing(qtbot):
    from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QSpinBox, QWidget

    window = QWidget()
    qtbot.addWidget(window)
    window.setFixedSize(1280, 720)
    settings = QWidget(window)
    settings.setGeometry(0, 0, 400, 720)
    panes = []
    for index, name in enumerate(('Console', 'System', 'Actions')):
        pane = QWidget(window)
        pane.setObjectName(name)
        pane.setGeometry(400, index * 200, 880, 180)
        label = QLabel(name, pane)
        label.setGeometry(10, 10, 100, 30)
        panes.append(pane)
    clipped = QPushButton('Run', panes[2])
    clipped.setObjectName('deliberately_clipped')
    clipped.setGeometry(850, 40, 100, 30)
    spin = QSpinBox(panes[1])
    spin.setObjectName('measured_spin')
    spin.setGeometry(20, 50, 100, 30)
    narrow = QLineEdit(panes[1])
    narrow.setObjectName('undersized_standalone_editor')
    narrow.setGeometry(20, 90, 5, 30)
    window.show()
    qtbot.waitUntil(window.isVisible)
    runtime = SimpleNamespace(
        count=lambda: len(panes), widget=lambda index: panes[index],
        _pane_of=lambda pane: SimpleNamespace(name=pane.objectName()),
        sizes=lambda: [pane.height() for pane in panes])
    fake = SimpleNamespace(window=window, screen=SimpleNamespace(
        _runtime_splitter=runtime, _settings_panel=settings,
        _body_splitter=SimpleNamespace(sizes=lambda: [400, 880])))
    measured = Controller._layout_snapshot(fake)
    row = next(control for control in measured['controls']
               if control['object_name'] == 'deliberately_clipped')
    assert row['rect'] == [1250, 440, 100, 30]
    assert row['clipped'] is True
    assert row['visible_clip'][0] + row['visible_clip'][2] == 1280
    internal = next(control for control in measured['controls']
                    if control['embedded_editor'])
    assert internal['class'] == 'QLineEdit'
    assert internal['acceptance_control'] is False
    assert internal['undersized'] is False
    outer = next(control for control in measured['controls']
                 if control['object_name'] == 'measured_spin')
    assert outer['acceptance_control'] is True
    standalone = next(control for control in measured['controls']
                      if control['object_name'] == 'undersized_standalone_editor')
    assert standalone['acceptance_control'] is True
    assert standalone['undersized'] is True
    assert measured['window_size'] == [1280, 720]
    assert window.width() == 1280 and window.height() == 720


def _layout_controller(snapshot, monkeypatch):
    calls = []
    fake = SimpleNamespace(record={'analysis_status': 'passed'}, screen=SimpleNamespace(),
                           _layout_snapshot=lambda: snapshot,
                           _write=lambda: calls.append('written'),
                           _save_layout_image=lambda name: calls.append(name) or [1280, 720],
                           _finish=lambda code: calls.append(('finished', code)))
    fake._finish_layout_check = lambda: Controller._finish_layout_check(fake)
    monkeypatch.delenv('SPACR_NATIVE_MENU_SMOKE', raising=False)
    Controller._start_layout_check(fake)
    return fake, calls


def _contained_layout():
    names = ('Console', 'System', 'Actions')
    return {'window_size': [1280, 720],
            'panes': [{'name': name} for name in names],
            'controls': [{'pane': name, 'clipped': False, 'undersized': False}
                         for name in names]}


def test_layout_acceptance_observes_three_samples_and_preserves_two_images(monkeypatch):
    fake, calls = _layout_controller(_contained_layout(), monkeypatch)
    Controller._poll_layout_check(fake)
    assert not any(isinstance(call, tuple) for call in calls)
    Controller._poll_layout_check(fake)
    assert calls.count('measure-complete.png') == 1
    assert calls.count('measure-settled.png') == 1
    assert ('finished', 0) in calls
    assert fake.record['layout']['window_size_unchanged'] is True
    assert len(fake.record['layout']['samples']) == 3
    assert fake.record['visual_layout_status'] == 'passed'


@pytest.mark.parametrize('defect', ['clipped', 'undersized', 'missing-pane', 'empty-controls'])
def test_layout_acceptance_cannot_hide_missing_or_clipped_controls(monkeypatch, defect):
    snapshot = _contained_layout()
    if defect == 'missing-pane':
        snapshot['panes'].pop()
    elif defect == 'empty-controls':
        snapshot['controls'] = []
    else:
        snapshot['controls'][0][defect] = True
    fake, calls = _layout_controller(snapshot, monkeypatch)
    Controller._poll_layout_check(fake)
    with pytest.raises(RuntimeError, match='fully contained controls'):
        Controller._poll_layout_check(fake)
    assert fake.record['analysis_status'] == 'passed'
    assert fake.record['visual_layout_status'] == 'failed'
    assert 'measure-settled.png' in calls
    assert ('finished', 0) not in calls


def test_layout_timeout_retains_failure_instead_of_forcing_geometry(monkeypatch):
    fake, calls = _layout_controller(_contained_layout(), monkeypatch)
    fake._layout_started -= 6
    fake._layout_snapshot = lambda: {**_contained_layout(), 'runtime_sizes': [1, 2, 3]}
    with pytest.raises(RuntimeError, match='fully contained controls'):
        Controller._poll_layout_check(fake)
    assert fake.record['layout']['settled'] is False
    assert fake.record['layout']['window_resized_by_witness'] is False
    assert 'measure-settled.png' in calls
    assert ('finished', 0) not in calls


@pytest.mark.parametrize('unreachable', [False, True])
def test_scroll_witness_measures_real_reachability_and_retains_initial_clipping(
        qtbot, tmp_path, monkeypatch, unreachable):
    from types import MethodType
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QScrollArea, QWidget

    window = QWidget()
    qtbot.addWidget(window)
    window.setFixedSize(1280, 720)
    settings = QWidget(window)
    settings.setGeometry(0, 0, 400, 720)
    scroll = QScrollArea(window)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setGeometry(400, 0, 880, 720)
    scroll.setWidgetResizable(True)
    content = QWidget()
    content.setMinimumHeight(1200)
    scroll.setWidget(content)
    panes = []
    for index, name in enumerate(('Console', 'System', 'Actions')):
        pane = QWidget(content)
        pane.setObjectName(name)
        pane.setGeometry(0, index * 400, 850, 380)
        label = QLabel(name, pane)
        label.setGeometry(10, 10, 100, 30)
        panes.append(pane)
    button = QPushButton('Run', panes[-1])
    button.setObjectName('actual_run')
    button.setGeometry(20, 100, 100, 30)
    if unreachable:
        button.setMinimumWidth(1500)
    nested = QScrollArea(panes[0])
    nested.setObjectName('actual_nested_console')
    nested.setGeometry(10, 60, 800, 100)
    document = QPlainTextEdit()
    document.setObjectName('actual_read_only_log')
    document.setReadOnly(True)
    document.setPlainText('\n'.join(f'Actual log line {index}' for index in range(50)))
    document.setFixedSize(780, 300)
    cursor = document.textCursor()
    cursor.setPosition(3)
    cursor.setPosition(7, QTextCursor.KeepAnchor)
    document.setTextCursor(cursor)
    nested.setWidget(document)
    window.show()
    qtbot.waitUntil(window.isVisible)
    finished = []
    runtime = SimpleNamespace(
        count=lambda: len(panes), widget=lambda index: panes[index],
        _pane_of=lambda pane: SimpleNamespace(name=pane.objectName()),
        sizes=lambda: [pane.height() for pane in panes])
    fake = SimpleNamespace(window=window, screen=SimpleNamespace(
        _runtime_splitter=runtime, _settings_panel=settings,
        _body_splitter=SimpleNamespace(sizes=lambda: [400, 880]),
        _runtime_viewport=scroll),
        record={'analysis_status': 'passed'}, output=tmp_path / 'receipt.json',
        _write=lambda: None, _finish=lambda code: finished.append(code))
    for method in ('_layout_snapshot', '_save_layout_image', '_start_layout_check',
                   '_poll_layout_check', '_control_for_layout_record',
                   '_scroll_layout_target', '_poll_layout_reachability', '_finish_layout_check'):
        setattr(fake, method, MethodType(getattr(Controller, method), fake))
    monkeypatch.delenv('SPACR_NATIVE_MENU_SMOKE', raising=False)
    fake._start_layout_check()
    errors = []

    def advance():
        try:
            if fake.phase == 'settling-layout':
                fake._poll_layout_check()
            else:
                fake._poll_layout_reachability()
        except RuntimeError as exc:
            errors.append(str(exc))
        return bool(finished or errors)

    qtbot.waitUntil(advance, timeout=10000)
    layout = fake.record['layout']
    assert layout['raw_viewport_status'] == 'failed'
    assert layout['violations']
    run = next(row for row in layout['reachability']
               if row['before']['object_name'] == 'actual_run')
    assert run['before']['clipped'] is True
    assert run['reachable'] is (not unreachable)
    assert run['scroll_position'][1] > 0
    assert (tmp_path / run['image']).is_file()
    document_rows = [row for row in layout['reachability']
                     if row['before']['object_name'] == 'actual_read_only_log']
    assert {row['before']['document_edge'] for row in document_rows} == {'start', 'end'}
    assert all(row['reachable'] for row in document_rows)
    assert any(row['scroll_positions'][1][1] > 0 for row in document_rows)
    document_scroll = next(row['index'] for row in layout['scroll_origins']
                           if row['name'] == 'actual_read_only_log')
    assert any(row['scroll_positions'][document_scroll][1] > 0 for row in document_rows)
    assert layout['scroll_origin_restored'] is True
    assert scroll.verticalScrollBar().value() == 0
    assert nested.verticalScrollBar().value() == 0
    assert document.verticalScrollBar().value() == 0
    assert (document.textCursor().position(), document.textCursor().anchor()) == (7, 3)
    assert layout['document_cursors_restored'] is True
    assert (window.width(), window.height()) == (1280, 720)
    assert fake.record['analysis_status'] == 'passed'
    if unreachable:
        assert errors and not finished
        assert fake.record['visual_layout_status'] == 'failed'
    else:
        assert finished == [0] and not errors
        assert fake.record['visual_layout_status'] == 'passed'
