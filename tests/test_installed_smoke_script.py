"""The native installer workflow's clean-environment smoke run."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _smoke_module():
    path = Path(__file__).parents[1] / "packaging" / "online" / "smoke_installed.py"
    spec = importlib.util.spec_from_file_location("spacr_installed_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_measure_result_releases_database_handle(monkeypatch):
    """Windows cannot delete the smoke tree while SQLite remains open."""
    module = _smoke_module()

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        closed = False

        def execute(self, query):
            if "run_status" in query:
                return Cursor(("complete", 1, 0))
            return Cursor((3,))

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(module.sqlite3, "connect", lambda _path: connection)

    assert module._read_measure_result(Path("measurements.db")) == (
        ("complete", 1, 0),
        3,
    )
    assert connection.closed is True


def test_installed_smoke_run_completes(tmp_path, monkeypatch):
    module = _smoke_module()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert module.main() == 0
