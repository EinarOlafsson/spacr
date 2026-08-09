"""Home's run counters must not re-read the whole journal every launch.

`journal_totals` aggregates over ALL runs, so unlike `recent_runs` it
cannot be bounded -- the answer depends on every folder. But a journal is
append-only in practice, so it does not have to read them all twice.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from spacr import run_journal


def _run(root, i, app="mask", models=None):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    folder = root / f"{when:%Y-%m-%d_%H%M%S}_tag{i:04d}__{app}"
    folder.mkdir()
    payload = {"app_key": app, "status": "ok", "start_utc": when.isoformat()}
    if models:
        payload["model_hashes"] = models
    (folder / "manifest.json").write_text(json.dumps(payload))
    return folder


@pytest.fixture
def journal(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    for i in range(30):
        _run(root, i, "mask" if i % 2 else "measure")
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


class TestTheAnswerIsUnchanged:

    def test_the_cached_answer_matches_the_cold_one(self, journal):
        run_journal._totals_cache_path().unlink(missing_ok=True)
        cold = run_journal.journal_totals()
        warm = run_journal.journal_totals()
        assert cold == warm
        assert cold["total_runs"] == 30

    def test_a_new_run_is_counted(self, journal):
        run_journal.journal_totals()
        _run(journal, 99, "mask")
        assert run_journal.journal_totals()["total_runs"] == 31

    def test_a_deleted_run_forces_a_full_recount(self, journal):
        """Nothing records what a removed folder contributed, so the
        incremental path cannot undo it. Recounting is correct and rare."""
        run_journal.journal_totals()
        import shutil
        # DIRECTORIES only -- the cache file lives in this folder and sorts
        # first, being a dotfile.
        shutil.rmtree(sorted(d for d in journal.iterdir() if d.is_dir())[0])
        assert run_journal.journal_totals()["total_runs"] == 29

    def test_distinct_models_are_not_double_counted(self, journal):
        """models_recorded is a SET across runs, so the incremental path
        has to carry the set, not a count."""
        run_journal._totals_cache_path().unlink(missing_ok=True)
        _run(journal, 100, "mask", models={"cell": "file:aaa"})
        first = run_journal.journal_totals()["models_recorded"]
        _run(journal, 101, "mask", models={"cell": "file:aaa"})
        assert run_journal.journal_totals()["models_recorded"] == first

    def test_a_new_distinct_model_increments(self, journal):
        run_journal._totals_cache_path().unlink(missing_ok=True)
        _run(journal, 102, "mask", models={"cell": "file:aaa"})
        before = run_journal.journal_totals()["models_recorded"]
        _run(journal, 103, "mask", models={"cell": "file:bbb"})
        assert run_journal.journal_totals()["models_recorded"] == before + 1


class TestTheCacheNeverLies:
    """A wrong run count is worse than a slow one."""

    def test_a_corrupt_cache_is_ignored(self, journal):
        run_journal.journal_totals()
        run_journal._totals_cache_path().write_text("{ not json")
        assert run_journal.journal_totals()["total_runs"] == 30

    def test_an_old_cache_version_is_ignored(self, journal):
        run_journal.journal_totals()
        path = run_journal._totals_cache_path()
        raw = json.loads(path.read_text())
        raw["version"] = 0
        path.write_text(json.dumps(raw))
        assert run_journal.journal_totals()["total_runs"] == 30

    def test_the_cache_file_is_not_counted_as_a_run(self, journal):
        """It lives inside the runs root; only directories are counted."""
        run_journal.journal_totals()
        assert run_journal._totals_cache_path().exists()
        assert run_journal.journal_totals()["total_runs"] == 30

    def test_an_unwritable_cache_still_returns_totals(self, journal,
                                                      monkeypatch):
        """Storing the cache is an optimisation; failing to store it must
        not fail the count. Broken by making the WRITE fail for real rather
        than by stubbing the function, which would only test the stub."""
        real_replace = run_journal.os.replace

        def refuse(src, dst):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(run_journal.os, "replace", refuse)
        assert run_journal.journal_totals()["total_runs"] == 30
        monkeypatch.setattr(run_journal.os, "replace", real_replace)

    def test_no_part_files_are_left_behind(self, journal):
        run_journal.journal_totals()
        assert list(run_journal.runs_root().glob("*.part")) == []
