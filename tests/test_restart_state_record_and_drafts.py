"""Table-driven tests of the forced-restart record and the settings drafts in ``spacr.restart_state``.

One general test per unit: the running-runs summary, the background
writer's flush, the draft read-back from the preference store, the record
file's location and its unreadable forms, ``save`` with its verify step, and
the JSON encoder that makes settings storable.  ``SPACR_HOME`` points at a
per-test directory, so every record is a real file on disk; the preference
store is replaced by small in-memory doubles.
"""
from __future__ import annotations

import json
import logging
import threading
import types
from pathlib import Path

import pytest

from spacr import restart_state as rs

LOGGER = "spacr.restart_state"


@pytest.fixture()
def spacr_home(tmp_path, monkeypatch):
    """Redirect the restart record into an empty per-test directory."""
    home = tmp_path / "home"
    monkeypatch.setenv("SPACR_HOME", str(home))
    return home


class _DraftStore:
    """A preference store that hands back one raw value and records removals."""

    def __init__(self, raw=None, read_error=None):
        self.raw = raw
        self.read_error = read_error
        self.removed = []

    def value(self, key, default=""):
        if self.read_error is not None:
            raise self.read_error
        return default if self.raw is None else self.raw

    def remove(self, key):
        self.removed.append(key)

    def sync(self):
        pass

    def status(self):
        return 0


def test_describe_running_summarises_named_runs_with_their_elapsed_time():
    """Only mappings with a module or name count; elapsed reads in the largest unit that fits."""
    cases = [
        ("non-mappings-skipped", [None, "measure", 7, {"module": "mask"}],
         "mask"),
        ("nameless-entries-add-nothing",
         [{"seconds": 90}, {"name": "   "}, {"module": "", "name": ""}], ""),
        ("name-is-the-fallback", [{"name": "classify"}], "classify"),
        ("module-beats-name", [{"module": "mask", "name": "other"}], "mask"),
        ("no-runs", None, ""),
        ("two-runs-joined", [{"module": "a", "seconds": 5},
                             {"name": "b"}], "a (running 5 s), b"),
        ("0-s", [{"module": "m", "seconds": 0}], "m (running 0 s)"),
        ("59-s", [{"module": "m", "seconds": 59}], "m (running 59 s)"),
        ("60-s", [{"module": "m", "seconds": 60}], "m (running 1 min)"),
        ("float-s", [{"module": "m", "seconds": "61.9"}], "m (running 1 min)"),
        ("3599-s", [{"module": "m", "seconds": 3599}], "m (running 59 min)"),
        ("3600-s", [{"module": "m", "seconds": 3600}],
         "m (running 1 h 00 min)"),
        ("7860-s", [{"module": "m", "seconds": 7860}],
         "m (running 2 h 11 min)"),
        ("seconds-none", [{"module": "m", "seconds": None}], "m"),
        ("seconds-text", [{"module": "m", "seconds": "soon"}], "m (running)"),
        ("seconds-object", [{"module": "m", "seconds": object()}],
         "m (running)"),
        ("seconds-nan", [{"module": "m", "seconds": float("nan")}],
         "m (running)"),
    ]
    for case_id, running, expected in cases:
        assert rs.describe_running(running) == expected, case_id


def test_flush_writes_runs_a_waiting_job_unless_a_write_is_in_progress(
        monkeypatch, subtests):
    """A flush runs what is left on the caller's thread, waits for a live writer, never races a busy one."""
    dead = types.SimpleNamespace(is_alive=lambda: False)
    alive = types.SimpleNamespace(is_alive=lambda: True)
    cases = [
        # id, pending, busy, thread, timeout, returned, ran, pending after
        ("dead-writer-left-a-job", "job", False, dead, 0.5, True, ["job"],
         None),
        ("no-writer-thread", "job", False, None, 0.5, True, ["job"], None),
        ("busy-writer-is-not-raced", "job", True, alive, 0.0, False, [],
         "job"),
        ("live-writer-deadline-passes", "job", False, alive, 0.05, True,
         ["job"], None),
        ("nothing-waiting", None, False, None, 0.0, True, [], None),
        ("negative-timeout-is-zero", "job", True, alive, -3.0, False, [],
         "job"),
    ]
    for case_id, pending, busy, thread, timeout, returned, ran_expected, \
            left in cases:
        with subtests.test(msg=case_id):
            state = {"pending": pending, "busy": busy, "thread": thread}
            monkeypatch.setattr(rs, "_WRITES", threading.Condition())
            monkeypatch.setattr(rs, "_WRITE_STATE", state)
            ran = []
            monkeypatch.setattr(rs, "_run_one_write", ran.append)
            assert rs._flush_writes(timeout=timeout) is returned
            assert ran == ran_expected
            assert state["pending"] == left
            assert state["busy"] is busy


def test_take_drafts_returns_only_well_formed_module_settings(
        monkeypatch, caplog, subtests):
    """Unreadable, corrupt or malformed drafts give ``{}``; a broken module entry is dropped alone."""
    cases = [
        # id, store, expected drafts, cleared, warning fragment
        ("store-cannot-be-read",
         _DraftStore(read_error=OSError("preference store cannot be read")),
         {}, [], "could not read session/drafts: preference store cannot be "
         "read"),
        ("nothing-stored", _DraftStore(raw=""), {}, [], None),
        ("truncated-json", _DraftStore(raw='{"modules": {"measure":'), {}, [],
         None),
        ("not-json", _DraftStore(raw="{not json"), {}, [], None),
        ("json-list", _DraftStore(raw=json.dumps(["not", "a", "mapping"])),
         {}, [], None),
        ("modules-not-a-mapping", _DraftStore(raw=json.dumps(
            {"modules": "odd"})), {}, [], None),
        ("broken-entry-dropped", _DraftStore(raw=json.dumps(
            {"modules": {"measure": ["bad"], "mask": {"src": "/data"}}})),
         {"mask": {"src": "/data"}}, [rs._DRAFTS_KEY], None),
        ("keys-become-strings", _DraftStore(raw=json.dumps(
            {"modules": {"7": {"a": 1}}})), {"7": {"a": 1}},
         [rs._DRAFTS_KEY], None),
    ]
    for case_id, store, expected, cleared, warning in cases:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(rs, "_store", lambda store=store: store)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                assert rs._take_drafts() == expected
            assert store.removed == cleared
            if warning is None:
                assert caplog.records == []
            else:
                assert [r.getMessage() for r in caplog.records] == [warning]


def test_state_path_lives_under_spacr_home_and_bad_records_read_as_none(
        spacr_home, caplog, subtests):
    """The record sits in ``SPACR_HOME``; corrupt, non-object and directory records are reported, not raised."""
    path = rs.state_path()
    assert path == spacr_home / rs.FILE_NAME
    assert rs.FILE_NAME == "restart_state.json"

    readers = [
        # id, content written (None = a directory), peek warning, discard
        ("missing", "absent", None, False, None),
        ("corrupt-json", "{not json at all", "could not read the restart "
         "state", True, None),
        ("json-list", "[1, 2, 3]", None, True, None),
        ("a-directory", None, "could not read the restart state", False,
         "could not remove the restart state"),
    ]
    for case_id, content, peek_warning, removed, discard_warning in readers:
        with subtests.test(msg=case_id):
            if path.is_dir():
                path.rmdir()
            path.parent.mkdir(parents=True, exist_ok=True)
            if content is None:
                path.mkdir()
            elif content != "absent":
                path.write_text(content, encoding="utf-8")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                assert rs.peek() is None
            messages = [r.getMessage() for r in caplog.records]
            if peek_warning is None:
                assert messages == []
            else:
                assert len(messages) == 1
                assert messages[0].startswith(peek_warning + ": ")
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                assert rs.discard() is removed
            messages = [r.getMessage() for r in caplog.records]
            if discard_warning is None:
                assert messages == []
            else:
                assert len(messages) == 1
                assert messages[0].startswith(discard_warning + ": ")
            assert not path.is_file()


def test_save_writes_a_verified_record_or_cancels_the_restart(
        spacr_home, monkeypatch, caplog, subtests):
    """A saved record round-trips; a wrong read-back or an unwritable home returns ``None``."""
    with subtests.test(msg="round-trip"):
        path = rs.save(
            module="measure",
            settings={"src": "/data", "channels": (1, 2)},
            running=[{"module": "measure", "seconds": 5}, "not a mapping"],
            run_folders=["/data/run1", ""],
            saved="2026-11-14T00:00:00+00:00",
        )
        assert path == spacr_home / rs.FILE_NAME
        assert rs.peek() == {
            "version": rs.SCHEMA_VERSION,
            "saved": "2026-11-14T00:00:00+00:00",
            "module": "measure",
            "settings": {"src": "/data", "channels": [1, 2]},
            "running": [{"module": "measure", "seconds": 5}],
            "run_folders": ["/data/run1"],
        }
        assert not path.with_suffix(".json.tmp").exists()
        assert rs.discard() is True

    with subtests.test(msg="reads-back-as-another-module"):
        class _WrongReadBack:
            dumps = staticmethod(json.dumps)

            @staticmethod
            def loads(*args, **kwargs):
                return {"module": "some other module"}

        monkeypatch.setattr(rs, "json", _WrongReadBack)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert rs.save(module="measure") is None
        assert [r.getMessage() for r in caplog.records] == [
            "could not save the restart state: the state read back as a "
            "different module"]
        monkeypatch.setattr(rs, "json", json)

    with subtests.test(msg="unwritable-home"):
        def _refuse(*args, **kwargs):
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(type(spacr_home), "mkdir", _refuse)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert rs.save(module="measure") is None
        assert [r.getMessage() for r in caplog.records] == [
            "could not save the restart state: read-only filesystem"]


def test_jsonable_makes_any_settings_value_storable():
    """Scalars pass, containers recurse, sets sort, anything else and deep nesting become strings."""
    nested = {"leaf": "bottom"}
    for _ in range(20):
        nested = {"down": nested}
    # The root is depth 0; the mapping reached at depth 13 is stringified.
    expected = nested
    for _ in range(13):
        expected = expected["down"]
    expected = str(expected)
    for _ in range(13):
        expected = {"down": expected}

    cases = [
        ("none", None, None),
        ("bool", True, True),
        ("int", 3, 3),
        ("float", 2.5, 2.5),
        ("str", "x", "x"),
        ("tuple", (1, (2, 3)), [1, [2, 3]]),
        ("set-sorted", {"c", "a", "b"}, ["a", "b", "c"]),
        ("frozenset-sorted", frozenset({2, 1}), [1, 2]),
        ("keys-stringified", {1: {2: "v"}}, {"1": {"2": "v"}}),
        ("path", Path("/data/x"), str(Path("/data/x"))),
        ("depth-cap", nested, expected),
    ]
    for case_id, value, encoded in cases:
        assert rs._jsonable(value) == encoded, case_id
