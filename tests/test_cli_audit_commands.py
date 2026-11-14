"""Table-driven tests of the two audit command lines: ``spacr-db-audit`` and ``spacr-leakage``.

``spacr.cli_database.main`` runs against real SQLite files -- a healthy one, a
deliberately corrupted one, a missing path and a disposable probe database --
and ``spacr.cli_leakage.main`` against real train/test crop folders.  Nothing
about the database inspection, the concurrency probe or the leakage auditor is
stubbed, so each exit code is the verdict on real data.  One general test per
command, walking a case table with ``subtests``.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from spacr import cli_database


def _cli_leakage():
    """Import ``spacr.cli_leakage`` when a test runs, so an import-time failure fails that test."""
    import importlib
    return importlib.import_module("spacr.cli_leakage")


# ---------------------------------------------------------------------------
# spacr-db-audit
# ---------------------------------------------------------------------------

def _healthy_db(path):
    """A small, valid spaCR-shaped database in DELETE journal mode."""
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE object (rowid_ INTEGER PRIMARY KEY, plate TEXT)")
    connection.executemany("INSERT INTO object (plate) VALUES (?)",
                           [(f"plate{i}",) for i in range(20)])
    connection.commit()
    connection.close()
    return path


def _corrupt_db(path):
    """A real database whose pages were overwritten mid-file.

    ``PRAGMA quick_check`` reports the damage as text rather than raising, so
    this is the database that exercises the "checks did not pass" exit code.
    """
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
    connection.executemany("INSERT INTO t (b) VALUES (?)", [("x" * 200,) for _ in range(500)])
    connection.commit()
    connection.close()
    size = path.stat().st_size
    with open(path, "r+b") as handle:
        handle.seek(size // 2)
        handle.write(b"\xff" * 300)
    return path


def _health_lines(path, journal="DELETE"):
    """The exact two-line human health header, filesystem left open."""
    return (rf"Database: {re.escape(str(path))}\n"
            rf"  journal={journal} filesystem=\S+ busy_timeout=5000 ms\n")


def _probe_line(ok, rows, journal):
    """The exact probe summary line, timing and read count left open."""
    return (rf"Probe: {'PASS' if ok else 'FAIL'} {rows} rows, \d+ concurrent reads, "
            rf"\d+\.\d{{3}} s, journal={journal}\n")


# (case, argv builder (db, corrupt, folder) -> argv, exit code,
#  stdout: regex (fullmatch) or JSON check, exact stderr or stderr regex,
#  follow-up check (db, folder) or None)
DATABASE_CASES = [
    ("a healthy database reports journal mode and busy timeout",
     lambda db, bad, d: [str(db)], 0, lambda db, d: _health_lines(db), "", None),
    ("--quick-check adds the integrity line when it was asked for",
     lambda db, bad, d: [str(db), "--quick-check"], 0,
     lambda db, d: _health_lines(db) + r"  quick_check=ok\n", "", None),
    ("a corrupt database warns and exits nonzero",
     lambda db, bad, d: [str(bad), "--quick-check"], 1,
     lambda db, d: (_health_lines(d / "damaged.db")
                    + r"  quick_check=(?s:.*malformed.*)\n"
                    + r"  WARNING: SQLite quick_check reported: (?s:.*malformed.*)\n"),
     "", None),
    ("--json replaces the report with one document",
     lambda db, bad, d: [str(db), "--json"], 0,
     lambda db, d: (lambda doc: (sorted(doc), doc["database"]["journal_mode"],
                                 doc["database"]["path"], doc["database"]["quick_check"],
                                 doc["database"]["warnings"], doc["database"]["busy_timeout_ms"])
                    == (["database"], "DELETE", str(db), None, [], 5000)),
     "", None),
    ("a missing database is an error sentence on stderr",
     lambda db, bad, d: [str(d / "absent.db")], 1, lambda db, d: "",
     lambda d: f"ERROR: FileNotFoundError: {d / 'absent.db'}\n", None),
    ("--json carries a missing database's error in the document",
     lambda db, bad, d: [str(d / "absent.db"), "--json"], 1,
     lambda db, d: (lambda doc: doc == {"error": f"FileNotFoundError: {d / 'absent.db'}"}),
     "", None),
    ("a probe runs in its own new scratch database",
     lambda db, bad, d: ["--probe", "--scratch", str(d / "probe.db"), "--writers", "2",
                         "--readers", "1", "--writes", "3", "--journal-mode", "DELETE"], 0,
     lambda db, d: _probe_line(True, "6/6", "DELETE"), "",
     lambda db, d: (d / "probe.db").exists()),
    ("the probe refuses an existing database and leaves it alone",
     lambda db, bad, d: ["--probe", "--scratch", str(db), "--writers", "1",
                         "--readers", "1", "--writes", "1"], 1, lambda db, d: "",
     lambda d: (f"ERROR: FileExistsError: Concurrency probe refuses existing database "
                f"{d / 'measurements.db'}; choose a new scratch path.\n"),
     "unchanged"),
    ("a database and a probe are reported together",
     lambda db, bad, d: [str(db), "--probe", "--scratch", str(d / "p.db"), "--writers", "1",
                         "--readers", "1", "--writes", "2", "--json"], 0,
     lambda db, d: (lambda doc: (sorted(doc), doc["database"]["journal_mode"],
                                 doc["probe"]["ok"], doc["probe"]["expected_rows"],
                                 doc["probe"]["actual_rows"], doc["probe"]["errors"])
                    == (["database", "probe"], "DELETE", True, 2, 2, [])),
     "", None),
]


def test_db_audit_reports_and_exit_codes(subtests, tmp_path, capsys):
    """``spacr-db-audit`` prints the exact report and exits 0 only when every check passed."""
    for index, (case_id, argv, code, out, err, after) in enumerate(DATABASE_CASES):
        with subtests.test(msg=case_id):
            folder = tmp_path / f"db{index}"
            folder.mkdir()
            db = _healthy_db(folder / "measurements.db")
            bad = _corrupt_db(folder / "damaged.db")
            before = db.read_bytes()
            assert cli_database.main(argv(db, bad, folder)) == code
            captured = capsys.readouterr()
            expected_out = out(db, folder)
            if callable(expected_out):
                assert expected_out(json.loads(captured.out)), captured.out
            else:
                assert re.fullmatch(expected_out, captured.out), captured.out
            assert captured.err == (err(folder) if callable(err) else err)
            if after == "unchanged":
                assert db.read_bytes() == before
            elif after is not None:
                assert after(db, folder)

    with subtests.test(msg="no database and no probe is a usage error"):
        with pytest.raises(SystemExit) as excinfo:
            cli_database.main([])
        assert excinfo.value.code == 2
        assert capsys.readouterr().err.endswith(
            "spacr-db-audit: error: provide DATABASE, --probe, or both\n")


# ---------------------------------------------------------------------------
# spacr-leakage
# ---------------------------------------------------------------------------

def _crop(root, split, cls, name, payload):
    """Write one crop file at ``root/split/cls/name``."""
    path = root / split / cls / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


CLEAN = [("train", "plate1_A01_f1_o1.png", b"one"), ("test", "plate2_B02_f2_o2.png", b"two")]
SHARED_WELL = [("train", "plate1_A01_f1_o1.png", b"one"), ("test", "plate1_A01_f1_o2.png", b"two")]
RENAMED_COPY = [("train", "plate1_A01_f1_o1.png", b"same pixels"),
                ("test", "plate9_Z99_f9_o9.png", b"same pixels")]
UNPARSEABLE = [("train", "unparseable.png", b"one"), ("test", "also-unparseable.png", b"two")]

# (case, crops, extra argv, exit code, check of the printed JSON report)
LEAKAGE_CASES = [
    ("a clean split exits 0 and prints the report", CLEAN, ["--group-by", "well"], 0,
     lambda r: (r["passed"], r["split_name"], r["critical_levels"]) == (True, "train_vs_test", [])),
    ("a well on both sides is leakage, exit 1", SHARED_WELL, ["--group-by", "well"], 1,
     lambda r: (r["passed"], r["critical_levels"], r["overlap_counts"]["well"],
                r["examples"]["well"]) == (False, ["well"], 1, ["plate1_A01"])),
    ("content hashing finds a renamed byte-identical crop", RENAMED_COPY,
     ["--group-by", "none"], 1,
     lambda r: (r["passed"], r["overlap_counts"]["content_sha256"]) == (False, 1)),
    ("--no-content-hash skips the byte-identical check", RENAMED_COPY,
     ["--group-by", "none", "--no-content-hash"], 0,
     lambda r: (r["passed"], r["overlap_counts"]["content_sha256"]) == (True, 0)),
    ("an unparseable name is a critical identity gap", UNPARSEABLE, ["--group-by", "well"], 1,
     lambda r: (r["passed"], "unverifiable_well" in r["critical_levels"]) == (False, True)),
    ("--allow-unverifiable downgrades the gap to a warning", UNPARSEABLE,
     ["--group-by", "well", "--allow-unverifiable"], 0,
     lambda r: (r["passed"], r["critical_levels"]) == (True, [])),
]


def test_leakage_exit_codes_and_reports(subtests, tmp_path, capsys):
    """``spacr-leakage`` exits 0 clean, 1 leaking, 2 unusable, and ``--output`` mirrors stdout."""
    for index, (case_id, crops, extra, code, check) in enumerate(LEAKAGE_CASES):
        with subtests.test(msg=case_id):
            root = tmp_path / f"leak{index}"
            for split, name, payload in crops:
                _crop(root, split, "neg", name, payload)
            assert _cli_leakage().main([str(root), *extra]) == code
            captured = capsys.readouterr()
            assert captured.err == ""
            assert check(json.loads(captured.out)), captured.out

    with subtests.test(msg="--output writes the same payload that was printed"):
        root = tmp_path / "output"
        for split, name, payload in CLEAN:
            _crop(root, split, "neg", name, payload)
        report = root / "reports" / "audit.json"
        assert _cli_leakage().main([str(root), "--group-by", "well", "--output", str(report)]) == 0
        assert json.loads(report.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)

    with subtests.test(msg="a folder without train/ exits 2 and says why on stderr"):
        root = tmp_path / "unusable"
        _crop(root, "test", "neg", "plate1_A01_f1_o1.png", b"one")
        assert _cli_leakage().main([str(root)]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ("spacr-leakage: Cannot audit dataset leakage: no images "
                                f"found in train under {root}.\n")

    with subtests.test(msg="--group-by is a closed choice list"):
        with pytest.raises(SystemExit) as excinfo:
            _cli_leakage().main([str(tmp_path), "--group-by", "quadrant"])
        assert excinfo.value.code == 2
        assert "invalid choice: 'quadrant'" in capsys.readouterr().err
