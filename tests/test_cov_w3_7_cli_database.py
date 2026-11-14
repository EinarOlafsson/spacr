"""``spacr-db-audit``: the SQLite health and concurrency command line.

The probe-failure reporting and the parser defaults.  The real-database
reports and exit codes are table-driven in ``tests/test_cli_audit_commands.py``.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr import cli_database


@pytest.fixture
def healthy_db(tmp_path):
    """A small, valid spaCR-shaped database in DELETE journal mode."""
    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute(
        "CREATE TABLE object (rowid_ INTEGER PRIMARY KEY, plate TEXT)")
    connection.executemany(
        "INSERT INTO object (plate) VALUES (?)",
        [(f"plate{i}",) for i in range(20)])
    connection.commit()
    connection.close()
    return path


def test_probe_errors_are_listed_under_the_summary_line(tmp_path, capsys,
                                                        monkeypatch):
    """Every thread failure the probe collected is printed on its own line."""
    from spacr.database_concurrency import ConcurrencyProbeResult

    failed = ConcurrencyProbeResult(
        path=str(tmp_path / "probe.db"), journal_mode="WAL", writers=2,
        readers=1, writes_per_writer=3, expected_rows=6, actual_rows=4,
        reader_queries=11, duration_seconds=0.25,
        errors=("writer-0: database is locked", "reader-0: disk I/O error"),
    )
    monkeypatch.setattr(cli_database, "run_concurrency_probe",
                        lambda *a, **k: failed)

    assert cli_database.main(["--probe"]) == 1
    out = capsys.readouterr().out
    assert "Probe: FAIL 4/6 rows" in out
    assert "  ERROR: writer-0: database is locked" in out
    assert "  ERROR: reader-0: disk I/O error" in out


def test_a_failed_probe_makes_the_whole_audit_fail(healthy_db, tmp_path,
                                                   capsys, monkeypatch):
    """A healthy database does not rescue an audit whose probe failed."""
    from spacr.database_concurrency import ConcurrencyProbeResult

    monkeypatch.setattr(
        cli_database, "run_concurrency_probe",
        lambda *a, **k: ConcurrencyProbeResult(
            path=str(tmp_path / "p.db"), journal_mode="WAL", writers=1,
            readers=1, writes_per_writer=1, expected_rows=1, actual_rows=0,
            reader_queries=1, duration_seconds=0.1))

    assert cli_database.main([str(healthy_db), "--probe"]) == 1
    out = capsys.readouterr().out
    assert "Database:" in out
    assert "Probe: FAIL" in out


def test_the_parser_defaults_match_the_documented_command(healthy_db):
    """The shipped defaults are the ones the help text promises."""
    args = cli_database.build_parser().parse_args([str(healthy_db)])
    assert args.writers == 4
    assert args.readers == 3
    assert args.writes == 50
    assert args.journal_mode == "WAL"
    assert args.probe is False
    assert args.quick_check is False
    assert args.json is False
    assert args.scratch is None
