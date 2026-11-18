"""Annotate refuses missing databases and anchors relative paths before startup."""

import sqlite3
from types import SimpleNamespace

import pytest

from spacr.database_concurrency import DatabaseBusy
from spacr.qt import annotate_engine as engine


def _database(path):
    """Create one existing annotation database with a label to update."""
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE png_list(png_path TEXT, annotate INTEGER)')
        connection.execute("INSERT INTO png_list VALUES ('cell.png', NULL)")
    return path


def _screen(monkeypatch):
    """Capture source-opening warnings before any Qt widgets or workers start."""
    from spacr.qt.screens import annotate

    warnings = []
    screen = SimpleNamespace(
        _settings=engine.AnnotateSettings(),
        _worker=None,
        _blind_warning=lambda title, message: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        annotate.QMessageBox, 'question',
        lambda *args, **kwargs: pytest.fail('A missing database cannot be opened anyway'))
    return annotate, screen, warnings


@pytest.mark.parametrize('parent_exists', [False, True])
def test_missing_source_is_refused_before_a_writer_starts(
        tmp_path, monkeypatch, parent_exists):
    """Missing measurements or source folders never become empty databases."""
    annotate, screen, warnings = _screen(monkeypatch)
    source = tmp_path / 'moved-experiment'
    if parent_exists:
        (source / 'measurements').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    annotate.AnnotateScreen._open_source(screen, 'moved-experiment')

    assert len(warnings) == 1
    assert str(source / 'measurements' / 'measurements.db') in warnings[0][1]
    assert 'Select the experiment folder' in warnings[0][1]
    assert 'Measure' in warnings[0][1]
    assert screen._settings.src == screen._settings.db_path == ''
    assert screen._worker is None
    assert not (source / 'measurements' / 'measurements.db').exists()
    assert (source / 'measurements').exists() is parent_exists


@pytest.mark.parametrize('source', ['experiment', '~/experiment'])
def test_source_path_is_resolved_before_schema_checks(tmp_path, monkeypatch, source):
    """Relative and home-relative sources reach every reader as one absolute path."""
    annotate, screen, warnings = _screen(monkeypatch)
    database = _database(tmp_path / 'experiment' / 'measurements' / 'measurements.db')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    checked = []

    def inspect_path(path, *args, **kwargs):
        """Stop after observing the path handed to schema preparation."""
        checked.append(path)
        raise DatabaseBusy('database is locked')

    monkeypatch.setattr(annotate, 'ensure_annotation_column', inspect_path)
    annotate.AnnotateScreen._open_source(screen, source)
    assert checked == [str(database)]
    assert len(warnings) == 1


def test_relative_writer_path_is_anchored_before_the_working_directory_changes(
        tmp_path, monkeypatch):
    """A queued label must reach the selected project even after a cwd change."""
    database = _database(tmp_path / 'project' / 'measurements.db')
    monkeypatch.chdir(tmp_path)
    worker = engine.SaveWorker('project/measurements.db', 'annotate')
    worker.submit({'cell.png': 2})
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    worker.start()
    worker.stop(wait=True)

    assert worker.last_error is None
    assert worker.db_path == str(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT annotate FROM png_list').fetchone() == (2,)
    assert not (elsewhere / 'project').exists()


@pytest.mark.parametrize('parent_exists', [False, True])
def test_writer_refuses_a_missing_database_and_retains_queued_labels(
        tmp_path, monkeypatch, parent_exists):
    """An absent database is never created by the annotation writer."""
    target = tmp_path / 'missing' / 'measurements.db'
    if parent_exists:
        target.parent.mkdir()
    monkeypatch.chdir(tmp_path)
    worker = engine.SaveWorker('missing/measurements.db', 'annotate')
    worker.submit({'cell.png': 2})
    worker.start()
    worker.stop(wait=True)

    assert not target.exists()
    assert target.parent.exists() is parent_exists
    assert worker.pending_batches == 1
    assert worker._q.get_nowait() == {'cell.png': 2}
    assert str(target) in worker.last_error
    assert 'Select the experiment folder' in worker.last_error
