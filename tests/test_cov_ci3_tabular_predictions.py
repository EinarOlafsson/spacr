"""Two store paths the coverage gate found unexercised (CI run 38059494502).

* ``tabular.table_columns`` on a Feather file: the reader's last branch,
  which reads the whole table and keeps only its header.
* ``predictions.migrate_prediction_columns`` on a PostgreSQL store whose
  table exists: the table is locked before its columns are read and the
  Annotate app's 1/2 encoding is rewritten to 0/1 in the same transaction.
"""
from __future__ import annotations

import pandas as pd
import pytest


def test_table_columns_reads_a_feather_header(tmp_path):
    pytest.importorskip('pyarrow')
    from spacr.tabular import table_columns

    path = tmp_path / 'scores.feather'
    pd.DataFrame({'prcfo': ['p1_r1_c1_f1_o1'], 'score': [0.5],
                  'gene': ['ABC']}).to_feather(path)

    assert tuple(table_columns(str(path), canonicalise=False)) == (
        'prcfo', 'score', 'gene')


class _LockingPostgres:
    """A PostgreSQL connection played by an in-memory SQLite database.

    SQLite answers the column and update statements the migration sends.
    The catalogue query is answered from ``sqlite_master`` and the
    PostgreSQL-only table lock is recorded instead of run.
    """

    def __init__(self, database, log):
        self.database = database
        self.log = log

    def execute(self, sql, params=()):
        self.log.append(sql)
        if sql.startswith('LOCK TABLE'):
            return None
        if 'information_schema.tables' in sql:
            return self.database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")
        return self.database.execute(sql)

    def commit(self):
        self.log.append('COMMIT')
        self.database.commit()

    def rollback(self):
        self.log.append('ROLLBACK')
        self.database.rollback()

    def close(self):
        self.log.append('CLOSE')


def test_postgres_migration_locks_the_table_and_rewrites_the_encoding(
        monkeypatch, capsys):
    import sqlite3

    from spacr import tabular
    from spacr.predictions import migrate_prediction_columns

    database = sqlite3.connect(':memory:')
    database.execute('CREATE TABLE png_list (prcfo TEXT, predictions INTEGER)')
    database.execute("INSERT INTO png_list VALUES ('a', 1), ('b', 2), "
                     "('c', 2), ('d', NULL)")
    database.commit()
    log = []
    monkeypatch.setattr(tabular, '_postgres_connect',
                        lambda _dsn: _LockingPostgres(database, log))

    repaired = migrate_prediction_columns('postgresql://reader@localhost/m',
                                          verbose=True)

    assert repaired == [('png_list', 'predictions', 2)]
    lock = [sql for sql in log if sql.startswith('LOCK TABLE')]
    assert lock == ['LOCK TABLE "png_list" IN EXCLUSIVE MODE']
    assert log.index(lock[0]) < next(
        index for index, sql in enumerate(log) if 'LIMIT 0' in sql)
    assert log[-2:] == ['COMMIT', 'CLOSE']
    rows = database.execute(
        'SELECT prcfo, predictions FROM png_list ORDER BY prcfo').fetchall()
    assert rows == [('a', 1), ('b', 0), ('c', 0), ('d', None)]
    assert 'Repaired 2 row(s) of `png_list`.`predictions`' in capsys.readouterr().out
