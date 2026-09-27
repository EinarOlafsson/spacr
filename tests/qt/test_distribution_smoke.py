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
    window.show()
    controller = Controller(qapp, window, str(tmp_path / 'source-unit.json'))
    try:
        qtbot.waitUntil(lambda: bool(completed), timeout=180000)
        record = json.loads((tmp_path / 'source-unit.json').read_text())
        assert completed == [0], record
        assert record['module_constructed'] and record['real_run_clicked']
        assert record['run_status'] == ['complete', 1, 0] and record['cells'] > 0
        assert record['worker_finished']
    finally:
        controller.timer.stop()
        window.close()
