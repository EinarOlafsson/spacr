"""Edge paths of prediction merges into DuckDB, Parquet and PostgreSQL stores.

Covers verbose reporting, refusal of unsafe merges (join-key overwrite,
unsupported SQL type, invalid or missing Parquet table), transaction
rollback on failure, and the Parquet guard against a store changed while a
merge was being staged.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest


PRCFO = ['p1_r1_c1_f1_o1', 'p1_r1_c1_f1_o2']


def _store(tmp_path, backend, frame, table='png_list'):
    from spacr import tabular

    if backend == 'duckdb':
        pytest.importorskip('duckdb')
    else:
        pytest.importorskip('pyarrow')
    suffix = 'duckdb' if backend == 'duckdb' else 'parquetdb'
    target = str(tmp_path / f'measurements.{suffix}')
    tabular.write_database(frame, target, table, if_exists='replace',
                           canonicalise=False)
    return target


def _read(target, table='png_list'):
    from spacr import tabular

    return tabular.read_database(target, table, canonicalise=False,
                                 report=None)[0]


def _legacy_frame():
    return pd.DataFrame({'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png'],
                         'predictions': [2, 1]})


@pytest.mark.parametrize('backend', ['duckdb', 'parquet'])
def test_verbose_migration_prints_each_repair(tmp_path, backend, capsys):
    from spacr.predictions import migrate_prediction_columns

    target = _store(tmp_path, backend, _legacy_frame())

    repaired = migrate_prediction_columns(target, verbose=True)

    assert repaired == [('png_list', 'predictions', 1)]
    out = capsys.readouterr().out
    assert 'Repaired 1 row(s) of `png_list`.`predictions`' in out
    assert _read(target)['predictions'].tolist() == [0, 1]


@pytest.mark.parametrize('backend', ['duckdb', 'parquet'])
def test_verbose_merge_prints_summary(tmp_path, backend, capsys):
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, backend, pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))

    report = merge_prediction_results(
        pd.DataFrame({'prcfo': PRCFO, 'score': [0.25, 0.75]}), target,
        {'pred': ('score', 'REAL')}, verbose=True)

    assert report.matched_rows == 2
    assert 'Merged pred into png_list' in capsys.readouterr().out
    assert _read(target)['pred'].tolist() == pytest.approx([0.25, 0.75])


def test_duckdb_migration_of_absent_table_is_a_noop(tmp_path):
    from spacr.predictions import migrate_prediction_columns

    target = _store(tmp_path, 'duckdb', _legacy_frame())

    assert migrate_prediction_columns(target, table='absent',
                                      verbose=True) == []
    assert _read(target)['predictions'].tolist() == [2, 1]


def test_duckdb_migration_rolls_back_on_failure(tmp_path, monkeypatch):
    from spacr import predictions

    target = _store(tmp_path, 'duckdb', _legacy_frame())
    original = predictions._repair_sql_columns

    def fail_after_repair(conn, table, columns):
        original(conn, table, columns)
        raise RuntimeError('interrupted migration')

    monkeypatch.setattr(predictions, '_repair_sql_columns', fail_after_repair)
    with pytest.raises(RuntimeError, match='interrupted migration'):
        predictions.migrate_prediction_columns(target, verbose=False)
    monkeypatch.undo()

    assert _read(target)['predictions'].tolist() == [2, 1]


class _FailingPostgres:
    """A PostgreSQL connection that fails every statement but the lock."""

    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=()):
        self.log.append(sql)
        if sql.startswith('LOCK TABLE'):
            return None
        raise RuntimeError('server went away')

    def commit(self):
        self.log.append('COMMIT')

    def rollback(self):
        self.log.append('ROLLBACK')

    def close(self):
        self.log.append('CLOSE')


def test_postgres_migration_rolls_back_on_failure(monkeypatch):
    from spacr import tabular
    from spacr.predictions import migrate_prediction_columns

    log = []
    monkeypatch.setattr(tabular, '_postgres_connect',
                        lambda _dsn: _FailingPostgres(log))

    with pytest.raises(RuntimeError, match='server went away'):
        migrate_prediction_columns('postgresql://reader@localhost/m',
                                   verbose=False)

    assert log[-2:] == ['ROLLBACK', 'CLOSE']
    assert 'COMMIT' not in log


def test_postgres_migration_of_absent_table_commits(monkeypatch):
    from spacr import tabular
    from spacr.predictions import migrate_prediction_columns

    log = []

    class Cursor:
        def fetchall(self):
            return [('other',)]

    class Connection(_FailingPostgres):
        def execute(self, sql, params=()):
            self.log.append(sql)
            return Cursor()

    monkeypatch.setattr(tabular, '_postgres_connect',
                        lambda _dsn: Connection(log))

    assert migrate_prediction_columns('postgresql://reader@localhost/m',
                                      verbose=True) == []
    assert log[-2:] == ['COMMIT', 'CLOSE']
    assert not any(sql.startswith('LOCK TABLE') for sql in log)


def test_postgres_merge_rolls_back_on_failure(monkeypatch):
    from spacr import tabular
    from spacr.predictions import merge_prediction_results

    log = []
    monkeypatch.setattr(tabular, '_postgres_connect',
                        lambda _dsn: _FailingPostgres(log))

    with pytest.raises(RuntimeError, match='server went away'):
        merge_prediction_results(
            pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]}),
            'postgresql://reader@localhost/m', {'pred': ('score', 'REAL')},
            verbose=False)

    assert log[0].startswith('LOCK TABLE')
    assert log[-2:] == ['ROLLBACK', 'CLOSE']


@pytest.mark.parametrize('backend', ['duckdb', 'parquet'])
def test_merge_refuses_to_overwrite_a_join_key(tmp_path, backend):
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, backend, pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))

    with pytest.raises(ValueError, match='cannot replace a join-key'):
        merge_prediction_results(
            pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]}), target,
            {'png_path': ('score', 'TEXT')}, verbose=False)

    assert _read(target)['png_path'].tolist() == ['/a.png', '/b.png']


@pytest.mark.parametrize('backend', ['duckdb', 'parquet'])
def test_merge_refuses_unsupported_sql_type(tmp_path, backend):
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, backend, pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))

    with pytest.raises(ValueError, match='Unsupported prediction SQL type'):
        merge_prediction_results(
            pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]}), target,
            {'pred': ('score', 'BLOB')}, verbose=False)

    assert 'pred' not in _read(target).columns


def test_parquet_merge_rejects_invalid_and_missing_tables(tmp_path):
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, 'parquet', pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))
    results = pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]})

    with pytest.raises(ValueError, match='Invalid Parquet table name'):
        merge_prediction_results(results, target, {'pred': ('score', 'REAL')},
                                 table='../png_list', verbose=False)
    with pytest.raises(ValueError, match='Table not found'):
        merge_prediction_results(results, target, {'pred': ('score', 'REAL')},
                                 table='absent', verbose=False)


def test_parquet_merge_into_legacy_class_column_repairs_and_overwrites(
        tmp_path):
    from spacr.predictions import merge_prediction_results

    frame = _legacy_frame()
    frame = pd.concat([frame, pd.DataFrame({
        'prcfo': ['p1_r1_c1_f1_o3'], 'png_path': ['/c.png'],
        'predictions': [2]})], ignore_index=True)
    target = _store(tmp_path, 'parquet', frame)

    report = merge_prediction_results(
        pd.DataFrame({'prcfo': [PRCFO[1]], 'cls': [1]}), target,
        {'predictions': ('cls', 'INTEGER')}, verbose=False)

    assert report.repaired == (('png_list', 'predictions', 2),)
    assert report.matched_rows == 1
    assert _read(target)['predictions'].tolist() == [0, 1, 0]


def test_parquet_merge_overwrites_an_existing_score_column(tmp_path):
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, 'parquet', pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png'],
         'pred': [0.0, 0.0]}))

    report = merge_prediction_results(
        pd.DataFrame({'prcfo': [PRCFO[0]], 'score': [0.5]}), target,
        {'pred': ('score', 'REAL')}, verbose=False)

    assert report.added_columns == ()
    assert _read(target)['pred'].tolist() == pytest.approx([0.5, 0.0])


def _changing_manifest(monkeypatch, tabular):
    original = tabular._parquet_manifest
    calls = []

    def manifest(folder):
        calls.append(folder)
        value = original(folder)
        if len(calls) > 1:
            return {'active': [], 'retired': ['concurrent writer']}
        return value

    monkeypatch.setattr(tabular, '_parquet_manifest', manifest)


def test_parquet_merge_detects_concurrent_change_and_drops_staged_parts(
        tmp_path, monkeypatch):
    from spacr import tabular
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, 'parquet', pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))
    folder = os.path.join(target, 'png_list')
    before = sorted(os.listdir(folder))
    _changing_manifest(monkeypatch, tabular)

    with pytest.raises(RuntimeError, match='changed while scoring'):
        merge_prediction_results(
            pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]}), target,
            {'pred': ('score', 'REAL')}, verbose=False)
    monkeypatch.undo()

    assert sorted(os.listdir(folder)) == before
    assert 'pred' not in _read(target).columns


def test_parquet_merge_cleanup_tolerates_a_vanished_staged_part(
        tmp_path, monkeypatch):
    from spacr import tabular
    from spacr.predictions import merge_prediction_results

    target = _store(tmp_path, 'parquet', pd.DataFrame(
        {'prcfo': PRCFO, 'png_path': ['/a.png', '/b.png']}))
    folder = os.path.join(target, 'png_list')
    before = sorted(os.listdir(folder))
    _changing_manifest(monkeypatch, tabular)
    monkeypatch.setattr(tabular, '_publish', lambda path, write: None)

    with pytest.raises(RuntimeError, match='changed while scoring'):
        merge_prediction_results(
            pd.DataFrame({'prcfo': PRCFO, 'score': [0.1, 0.2]}), target,
            {'pred': ('score', 'REAL')}, verbose=False)
    monkeypatch.undo()

    assert sorted(os.listdir(folder)) == before
