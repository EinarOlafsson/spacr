"""Restart-record edges: bad input, a failed write-back, and a stale stamp.

Every test points ``SPACR_HOME`` at its own tmp_path, so the record under
test is a real file on disk and ``save``/``peek``/``take``/``discard``
run unmocked against it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture()
def spacr_home(tmp_path, monkeypatch):
    """Redirect the restart record into an empty per-test directory."""
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_a_failed_preference_write_does_not_claim_the_drafts_were_saved(
        monkeypatch, caplog):
    """A refused preference write must let the caller keep its unsaved state."""
    from spacr import restart_state

    class RefusingStore:
        def setValue(self, key, value):
            raise OSError("preference store is read-only")

    monkeypatch.setattr(restart_state, "_store", RefusingStore)
    with caplog.at_level("WARNING", logger="spacr.restart_state"):
        assert restart_state._save_drafts({"measure": {"src": "/data"}}) is False
    assert "could not store session/drafts" in caplog.text


def test_saving_no_drafts_clears_an_earlier_unsaved_form(monkeypatch):
    """Closing the last draft must not offer its old settings on next start."""
    from spacr import restart_state

    class Store:
        def __init__(self):
            self.values = {}

        def setValue(self, key, value):
            self.values[key] = value

        def value(self, key, default):
            return self.values.get(key, default)

        def remove(self, key):
            self.values.pop(key, None)

        def sync(self):
            pass

        def status(self):
            return 0

    store = Store()
    monkeypatch.setattr(restart_state, "_store", lambda: store)
    assert restart_state._save_drafts({"mask": {"src": "/old"}})
    assert restart_state._save_drafts({})
    assert restart_state._take_drafts() == {}
    assert restart_state._DRAFTS_KEY not in store.values


def test_the_warning_names_the_runs_the_folders_and_the_promise():
    """All three paragraphs appear when there are runs and folders."""
    from spacr.restart_state import warning_text

    text = warning_text([{"module": "measure", "seconds": 90}],
                        ["/data/run1", "", "/data/run2"])

    assert "measure (running 1 min)" in text
    assert "/data/run1, /data/run2" in text
    assert "+" not in text.split("still in:")[1].split("\n")[0]
    assert "reopen this module" in text


def test_more_than_four_folders_are_summarised():
    """Only the first four folders are listed; the rest are counted."""
    from spacr.restart_state import warning_text

    text = warning_text([], [f"/data/run{i}" for i in range(7)])

    assert "No other module is running." in text
    assert "/data/run4" not in text
    assert "(+3 more)" in text


def test_peek_returns_none_when_nothing_was_saved(spacr_home):
    """A missing record is not an error."""
    from spacr import restart_state

    assert restart_state.peek() is None


def test_take_consumes_the_record(spacr_home):
    """The file is gone after ``take`` so state is not restored twice."""
    from spacr import restart_state

    restart_state.save(module="measure")

    assert restart_state.take()["module"] == "measure"
    assert not restart_state.state_path().exists()
    assert restart_state.take() is None


def test_take_discards_a_stale_record(spacr_home, caplog):
    """A record older than MAX_AGE_SECONDS is removed and not returned."""
    import logging

    from spacr import restart_state

    old = datetime.now(timezone.utc) - timedelta(
        seconds=restart_state.MAX_AGE_SECONDS + 60)
    restart_state.save(module="measure",
                       saved=old.isoformat(timespec="seconds"))

    with caplog.at_level(logging.INFO, logger="spacr.restart_state"):
        assert restart_state.take() is None

    assert not restart_state.state_path().exists()
    assert "older than" in caplog.text


@pytest.mark.parametrize("stamp", ["", "yesterday afternoon", None])
def test_a_record_without_a_usable_stamp_is_not_called_stale(spacr_home, stamp):
    """An unreadable timestamp must not silently discard live state."""
    from spacr import restart_state

    assert restart_state._too_old({"saved": stamp}) is False


def test_a_naive_stamp_is_read_as_utc(spacr_home):
    """A timestamp without a zone is assumed UTC rather than raising."""
    from spacr import restart_state

    fresh = datetime.now(timezone.utc).replace(tzinfo=None)
    stale = fresh - timedelta(seconds=restart_state.MAX_AGE_SECONDS + 60)

    assert restart_state._too_old({"saved": fresh.isoformat()}) is False
    assert restart_state._too_old({"saved": stale.isoformat()}) is True


def test_discard_reports_whether_it_removed_anything(spacr_home):
    """True when a record was there, False when there was nothing to remove."""
    from spacr import restart_state

    assert restart_state.discard() is False
    restart_state.save(module="measure")
    assert restart_state.discard() is True
    assert restart_state.discard() is False


def test_the_restart_command_names_this_interpreter_and_entry_point():
    """The relaunch never depends on a ``spacr`` executable found on PATH."""
    import sys

    from spacr.restart_state import command

    assert command() == [sys.executable, "-m", "spacr.qt", "--resume-restart"]
