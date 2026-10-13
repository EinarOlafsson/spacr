"""`recent_runs` must not read the whole journal to show ten rows.

It used to. On a real machine that was 3521 run folders and ~0.85 s of
json.loads on the GUI thread at startup -- and it grew with every run the
user had ever done, so the launch got slower the more the tool was used.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from spacr import run_journal


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """A journal of 200 runs, named the way run_journal names them."""
    root = tmp_path / "runs"
    root.mkdir()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(200):
        when = base + timedelta(minutes=i)
        folder = root / f"{when:%Y-%m-%d_%H%M%S}_tag{i:04d}__mask"
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({
            "app_key": "mask", "status": "ok",
            "start_utc": when.isoformat(), "elapsed_s": 1.0,
        }))
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


class TestItReadsOnlyWhatItNeeds:

    def test_it_does_not_open_every_manifest(self, journal, monkeypatch):
        """The whole point. Counting reads is the assertion -- timing would
        be flaky, and the cost is the read count either way."""
        opened = []
        real = run_journal.Path.read_text if hasattr(run_journal, "Path") else None
        import pathlib
        original = pathlib.Path.read_text

        def counting(self, *a, **k):
            if self.name == "manifest.json":
                opened.append(self)
            return original(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", counting)
        run_journal.recent_runs(10)
        assert len(opened) < 200, "it read the entire journal"
        assert len(opened) >= 10, "it must read at least what it returns"


class TestTheAnswerIsUnchanged:

    def test_it_returns_the_newest_runs_newest_first(self, journal):
        rows = run_journal.recent_runs(10)
        assert len(rows) == 10
        stamps = [r["start_utc"] for r in rows]
        assert stamps == sorted(stamps, reverse=True)

    def test_it_matches_an_exhaustive_scan(self, journal):
        """Bounding the read must not change WHICH runs come back."""
        every = []
        for folder in journal.iterdir():
            manifest = json.loads((folder / "manifest.json").read_text())
            every.append((manifest["start_utc"], folder.name))
        every.sort(reverse=True)
        for limit in (1, 5, 10, 50):
            got = [r["dir"].name for r in run_journal.recent_runs(limit)]
            assert got == [name for _stamp, name in every[:limit]]

    def test_a_limit_larger_than_the_journal_returns_everything(self, journal):
        assert len(run_journal.recent_runs(10_000)) == 200

    def test_an_empty_journal_is_empty_not_an_error(self, tmp_path,
                                                    monkeypatch):
        empty = tmp_path / "none"
        empty.mkdir()
        monkeypatch.setattr(run_journal, "runs_root", lambda: empty)
        assert run_journal.recent_runs(10) == []

    def test_a_corrupt_manifest_is_skipped_not_fatal(self, journal):
        """One unreadable folder must not empty Run History."""
        victim = sorted(journal.iterdir())[-1]
        (victim / "manifest.json").write_text("{ not json")
        rows = run_journal.recent_runs(10)
        assert len(rows) == 10
        assert victim.name not in [r["dir"].name for r in rows]

    def test_a_folder_with_no_manifest_is_skipped(self, journal):
        (journal / "9999-12-31_235959_zzz__mask").mkdir()
        rows = run_journal.recent_runs(5)
        assert all("9999" not in r["dir"].name for r in rows)
