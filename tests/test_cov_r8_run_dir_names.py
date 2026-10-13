"""`_run_dir_names` — one directory read where there were four walks.

WHY IT WAS WRITTEN. `recent_runs` and `journal_totals` between them made
four `iterdir()` passes with an `is_dir()` on every entry. On a journal
that had grown to 10,192 run folders that was 30,832 filesystem calls on
the GUI thread, every time Home refreshed, to produce a ten-row list and
five integers. One `os.scandir` answers both for 173.

The risk in that change is not speed, it is TRUTH: the two callers must
still see exactly what they saw, and `journal_totals` in particular
decides between an incremental update and a full recount by comparing
the folder names it finds against the names it has already counted.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr.run_journal import _run_dir_names


def test_it_reports_the_directories(tmp_path):
    for name in ("2026-08-30_120000_aaaa__mask",
                 "2026-08-30_130000_bbbb__measure"):
        (tmp_path / name).mkdir()
    assert sorted(_run_dir_names(tmp_path)) == [
        "2026-08-30_120000_aaaa__mask",
        "2026-08-30_130000_bbbb__measure",
    ]


def test_a_loose_file_is_not_a_run(tmp_path):
    """The `is_dir()` half of the old pass still has to happen."""
    (tmp_path / "2026-08-30_120000_aaaa__mask").mkdir()
    (tmp_path / ".journal_totals.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("hello")
    assert _run_dir_names(tmp_path) == ["2026-08-30_120000_aaaa__mask"]


def test_an_empty_journal_is_an_empty_list(tmp_path):
    assert _run_dir_names(tmp_path) == []


def test_a_journal_that_is_not_there_is_not_an_error(tmp_path):
    """Answering [] is what lets the callers return zeros for a new user."""
    assert _run_dir_names(tmp_path / "no-such-directory") == []


def test_a_journal_that_cannot_be_read_is_not_an_error(tmp_path):
    """A permission problem must not take the Home screen down with it."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "2026-08-30_120000_aaaa__mask").mkdir()
    os.chmod(blocked, 0o000)
    try:
        if os.access(blocked, os.R_OK):        # root ignores the mode
            pytest.skip("this user can read a 000 directory")
        assert _run_dir_names(blocked) == []
    finally:
        os.chmod(blocked, 0o755)


def test_a_symlink_to_a_directory_still_counts(tmp_path):
    """`is_dir()` follows links, and the old `Path.is_dir()` did too.

    scandir's entry `is_dir()` follows symlinks by default, which is the
    behaviour being preserved -- a run folder reached through a link is
    still a run folder.
    """
    real = tmp_path / "elsewhere" / "2026-08-30_120000_aaaa__mask"
    real.mkdir(parents=True)
    link = tmp_path / "journal"
    link.mkdir()
    try:
        (link / "2026-08-30_120000_aaaa__mask").symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem will not make symlinks")
    assert _run_dir_names(link) == ["2026-08-30_120000_aaaa__mask"]


class TestTheCallersStillAgreeWithThemselves:
    """The optimisation is only correct if the answers did not move."""

    @staticmethod
    def _a_run(root, name, app_key, models=()):
        d = root / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "app_key": app_key,
            "status": "done",
            "start_utc": f"2026-08-30T{name[16:18]}:00:00",
            "elapsed_s": 1.0,
            "models": list(models),
        }))
        return d

    def test_recent_runs_orders_newest_first_and_bounds_the_read(
            self, tmp_path, monkeypatch):
        import spacr.run_journal as J

        monkeypatch.setattr(J, "runs_root", lambda: tmp_path)
        for hour in range(5):
            self._a_run(tmp_path, f"2026-08-30_{hour:02d}0000_x{hour}__mask",
                        "mask")
        got = [r["dir"].name for r in J.recent_runs(3)]
        assert got == ["2026-08-30_040000_x4__mask",
                       "2026-08-30_030000_x3__mask",
                       "2026-08-30_020000_x2__mask"]

    def test_journal_totals_counts_every_run(self, tmp_path, monkeypatch):
        import spacr.run_journal as J

        monkeypatch.setattr(J, "runs_root", lambda: tmp_path)
        self._a_run(tmp_path, "2026-08-30_010000_a__mask", "mask")
        self._a_run(tmp_path, "2026-08-30_020000_b__mask", "mask")
        self._a_run(tmp_path, "2026-08-30_030000_c__measure", "measure")
        totals = J.journal_totals()
        assert totals["total_runs"] == 3
        assert totals["mask_runs"] == 2
        assert totals["measure_runs"] == 1

    def test_a_second_call_reuses_the_cache_and_agrees(self, tmp_path,
                                                      monkeypatch):
        """The incremental path is the one the name list feeds."""
        import spacr.run_journal as J

        monkeypatch.setattr(J, "runs_root", lambda: tmp_path)
        self._a_run(tmp_path, "2026-08-30_010000_a__mask", "mask")
        first = J.journal_totals()
        self._a_run(tmp_path, "2026-08-30_020000_b__measure", "measure")
        second = J.journal_totals()
        assert first["total_runs"] == 1
        assert second["total_runs"] == 2
        assert second["measure_runs"] == 1

    def test_a_deleted_run_forces_an_honest_recount(self, tmp_path,
                                                   monkeypatch):
        """A folder that has gone cannot be subtracted incrementally.

        The name list is what detects it: the cached counted-set is no
        longer a subset of what is present, so the totals are rebuilt
        rather than left overstating the journal for ever.
        """
        import spacr.run_journal as J

        monkeypatch.setattr(J, "runs_root", lambda: tmp_path)
        a = self._a_run(tmp_path, "2026-08-30_010000_a__mask", "mask")
        self._a_run(tmp_path, "2026-08-30_020000_b__mask", "mask")
        assert J.journal_totals()["total_runs"] == 2
        import shutil
        shutil.rmtree(a)
        assert J.journal_totals()["total_runs"] == 1
