"""``spacr-db-audit``: the SQLite health and concurrency command line.

Every test drives :func:`spacr.cli_database.main` against a real SQLite file
-- including a deliberately corrupted one -- and reads the printed report and
the exit code back. Nothing about ``inspect_database`` or the concurrency
probe is stubbed, so the exit code really is a verdict on a real database.
"""
from __future__ import annotations

import json
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


@pytest.fixture
def corrupt_db(tmp_path):
    """A real database whose pages have been overwritten mid-file.

    ``PRAGMA quick_check`` reports the damage as text rather than raising, so
    this is the fixture that exercises the "checks did not pass" exit code.
    """
    path = tmp_path / "damaged.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
    connection.executemany("INSERT INTO t (b) VALUES (?)",
                           [("x" * 200,) for _ in range(500)])
    connection.commit()
    connection.close()
    size = path.stat().st_size
    with open(path, "r+b") as handle:
        handle.seek(size // 2)
        handle.write(b"\xff" * 300)
    return path


def test_no_database_and_no_probe_is_a_usage_error(capsys):
    """The command needs something to do: exit 2 with the argparse usage text."""
    with pytest.raises(SystemExit) as excinfo:
        cli_database.main([])
    assert excinfo.value.code == 2
    assert "provide DATABASE, --probe, or both" in capsys.readouterr().err


def test_a_healthy_database_reports_its_journal_mode_and_exits_zero(
        healthy_db, capsys):
    """The default human report names the path, journal mode and busy timeout."""
    assert cli_database.main([str(healthy_db)]) == 0
    out = capsys.readouterr().out
    assert f"Database: {healthy_db}" in out
    assert "journal=DELETE" in out
    assert "busy_timeout=" in out and " ms" in out
    assert "quick_check=" not in out
    assert "WARNING:" not in out


def test_quick_check_is_printed_only_when_it_was_asked_for(healthy_db, capsys):
    """``--quick-check`` adds the integrity line and still exits zero when ok."""
    assert cli_database.main([str(healthy_db), "--quick-check"]) == 0
    out = capsys.readouterr().out
    assert "quick_check=ok" in out
    assert "WARNING:" not in out


def test_a_corrupt_database_warns_and_exits_nonzero(corrupt_db, capsys):
    """A quick_check that is not ``ok`` is a failed audit, not a passing one."""
    assert cli_database.main([str(corrupt_db), "--quick-check"]) == 1
    out = capsys.readouterr().out
    assert "quick_check=" in out
    assert "WARNING: SQLite quick_check reported:" in out
    assert "malformed" in out


def test_json_output_is_one_document_and_prints_no_table(healthy_db, capsys):
    """``--json`` replaces the human report entirely."""
    assert cli_database.main([str(healthy_db), "--json"]) == 0
    out = capsys.readouterr().out
    assert "Database:" not in out
    payload = json.loads(out)
    assert payload["database"]["journal_mode"] == "DELETE"
    assert payload["database"]["path"].endswith("measurements.db")
    assert payload["database"]["quick_check"] is None


def test_a_missing_database_is_reported_on_stderr(tmp_path, capsys):
    """An unreadable path is an error sentence and a nonzero exit, not a crash."""
    assert cli_database.main([str(tmp_path / "absent.db")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ERROR: FileNotFoundError")


def test_a_missing_database_in_json_mode_carries_the_error_key(tmp_path,
                                                               capsys):
    """``--json`` puts the failure in the document rather than on stderr."""
    assert cli_database.main([str(tmp_path / "absent.db"), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"].startswith("FileNotFoundError")
    assert "database" not in payload


def test_a_probe_runs_in_its_own_scratch_database(tmp_path, capsys):
    """``--probe`` stresses a new file and reports PASS with the row counts."""
    scratch = tmp_path / "probe.db"
    assert cli_database.main([
        "--probe", "--scratch", str(scratch),
        "--writers", "2", "--readers", "1", "--writes", "3",
        "--journal-mode", "DELETE",
    ]) == 0
    out = capsys.readouterr().out
    assert "Probe: PASS 6/6 rows" in out
    assert "concurrent reads" in out
    assert "journal=DELETE" in out
    assert "ERROR:" not in out
    assert scratch.exists(), "the named scratch database was not created"


def test_the_probe_never_touches_an_existing_database(healthy_db, capsys):
    """A ``--scratch`` path that already exists is refused, and the audit fails."""
    before = healthy_db.read_bytes()
    assert cli_database.main([
        "--probe", "--scratch", str(healthy_db),
        "--writers", "1", "--readers", "1", "--writes", "1",
    ]) == 1
    assert "ERROR:" in capsys.readouterr().err
    assert healthy_db.read_bytes() == before


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


def test_a_database_and_a_probe_are_reported_together(healthy_db, tmp_path,
                                                      capsys):
    """Both audits run in one invocation and both appear in the JSON document."""
    assert cli_database.main([
        str(healthy_db), "--probe", "--scratch", str(tmp_path / "p.db"),
        "--writers", "1", "--readers", "1", "--writes", "2", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["database"]["journal_mode"] == "DELETE"
    assert payload["probe"]["ok"] is True
    assert payload["probe"]["expected_rows"] == 2


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
