"""Annotate checks writable projects and explains SQLite read-only failures."""

import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from spacr.qt import annotate_engine as engine


def _database(path):
    """Create an annotation database whose schema already needs no migration."""
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE png_list(png_path TEXT, annotate INTEGER)')
        connection.execute("INSERT INTO png_list VALUES ('cell.png', NULL)")
        connection.execute('CREATE INDEX idx_png_path ON png_list(png_path)')
    return path


@pytest.fixture(params=['database', 'folder'])
def readonly_project(tmp_path, request):
    """Restrict either the database or its folder, restoring cleanup access."""
    database = _database(tmp_path / 'experiment' / 'measurements' / 'measurements.db')
    target = database if request.param == 'database' else database.parent
    original = target.stat().st_mode
    target.chmod(original & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        if os.access(target, os.W_OK):
            pytest.skip('This account bypasses filesystem write permissions')
        yield database
    finally:
        target.chmod(original)


def test_source_refuses_readonly_database_or_parent_before_schema_or_writer(
        readonly_project, monkeypatch):
    """Even an already prepared schema must not admit an unwritable project."""
    from spacr.qt.screens import annotate

    warnings = []
    screen = SimpleNamespace(
        _settings=engine.AnnotateSettings(),
        _worker=None,
        _blind_warning=lambda title, text: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        annotate, 'ensure_annotation_column',
        lambda *args, **kwargs: pytest.fail('Schema setup preceded the access check'))

    annotate.AnnotateScreen._open_source(screen, str(readonly_project.parent.parent))

    assert len(warnings) == 1
    assert str(readonly_project) in warnings[0][1]
    assert 'copy the entire experiment folder' in warnings[0][1]
    assert 'writable' in warnings[0][1]
    assert screen._settings.db_path == ''
    assert screen._worker is None


def test_writer_refuses_readonly_location_before_consuming_labels(readonly_project):
    """A startup permission failure leaves the original pending queue intact."""
    worker = engine.SaveWorker(str(readonly_project), 'annotate')
    worker.submit({'cell.png': 2})
    worker.start()
    worker.stop(wait=True)

    assert worker.pending_batches == 1
    assert worker._q.get_nowait() == {'cell.png': 2}
    assert str(readonly_project) in worker.last_error
    assert 'copy the entire experiment folder' in worker.last_error
    with sqlite3.connect(f'{readonly_project.as_uri()}?mode=ro', uri=True) as connection:
        assert connection.execute('SELECT annotate FROM png_list').fetchone() == (None,)


def test_readonly_database_still_allows_a_clean_row_count(readonly_project):
    """A normal SELECT remains read-only and requires no writable location."""
    original = readonly_project.read_bytes()
    siblings = {path.name for path in readonly_project.parent.iterdir()}
    assert engine.count_rows(str(readonly_project)) == 1
    assert readonly_project.read_bytes() == original
    assert {path.name for path in readonly_project.parent.iterdir()} == siblings


def _leave_hot_journal(path):
    """Crash a disposable SQLite writer after dirty pages spill to disk."""
    subprocess.run(
        [sys.executable, '-c', '''
import os
import sqlite3
import sys
connection = sqlite3.connect(sys.argv[1])
connection.execute('PRAGMA journal_mode=DELETE')
connection.execute('PRAGMA cache_size=1')
connection.execute('CREATE TABLE padding(value BLOB)')
connection.executemany('INSERT INTO padding VALUES (?)', [(b'a' * 4096,)] * 20)
connection.commit()
connection.execute('BEGIN IMMEDIATE')
connection.execute('UPDATE padding SET value=?', (b'b' * 4096,))
os._exit(0)
''', str(path)], check=True, timeout=10)
    assert Path(f'{path}-journal').is_file()


def test_row_count_explains_a_real_sqlite_readonly_recovery_failure(tmp_path):
    """A hot rollback journal reproduces one possible reason a SELECT needs writes."""
    database = _database(tmp_path / 'experiment' / 'measurements' / 'measurements.db')
    _leave_hot_journal(database)
    original = database.read_bytes()
    journal = Path(f'{database}-journal')
    journal_bytes = journal.read_bytes()

    with pytest.raises(PermissionError) as caught:
        engine.count_rows(str(database))

    assert str(database) in str(caught.value)
    assert 'copy the entire experiment folder' in str(caught.value)
    assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    assert caught.value.__cause__.sqlite_errorcode & 0xff == sqlite3.SQLITE_READONLY
    assert database.read_bytes() == original
    assert journal.read_bytes() == journal_bytes


@pytest.mark.parametrize('table', ['absent', 'readonly database'])
def test_unrelated_row_count_errors_keep_their_original_cause(tmp_path, table):
    """A missing crop table is a schema error, never a read-only diagnosis."""
    database = _database(tmp_path / 'experiment' / 'measurements' / 'measurements.db')
    with pytest.raises(sqlite3.OperationalError, match='no such table'):
        engine.count_rows(str(database), table=table)


@pytest.mark.parametrize('readonly', [True, False])
def test_source_handles_only_readonly_errors_from_schema_preparation(
        tmp_path, monkeypatch, readonly):
    """SQLite may refuse access even when the filesystem access probe succeeds."""
    from spacr.qt.screens import annotate

    database = _database(tmp_path / 'experiment' / 'measurements' / 'measurements.db')
    warnings = []
    screen = SimpleNamespace(
        _settings=engine.AnnotateSettings(),
        _worker=None,
        _blind_warning=lambda title, text: warnings.append((title, text)),
    )
    error = sqlite3.OperationalError(
        'attempt to write a readonly database' if readonly else 'no such table: png_list')

    def reject_schema(*args, **kwargs):
        """Expose a SQLite error after the filesystem preflight passes."""
        raise error

    monkeypatch.setattr(annotate, 'ensure_annotation_column', reject_schema)
    if readonly:
        annotate.AnnotateScreen._open_source(screen, str(database.parent.parent))
        assert len(warnings) == 1
        assert 'copy the entire experiment folder' in warnings[0][1]
        assert screen._settings.db_path == ''
    else:
        with pytest.raises(sqlite3.OperationalError) as caught:
            annotate.AnnotateScreen._open_source(screen, str(database.parent.parent))
        assert caught.value is error
        assert warnings == []
