"""Restart-record edges: bad input, a failed write-back, and a stale stamp.

Every test points ``SPACR_HOME`` at its own tmp_path, so the record under
test is a real file on disk and ``save``/``peek``/``take``/``discard``
run unmocked against it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture()
def spacr_home(tmp_path, monkeypatch):
    """Redirect the restart record into an empty per-test directory."""
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_the_record_lands_under_spacr_home(spacr_home):
    """``state_path`` honours SPACR_HOME rather than the real ~/.spacr."""
    from spacr import restart_state

    path = restart_state.state_path()

    assert path.parent == spacr_home
    assert path.name == restart_state.FILE_NAME


def test_a_non_mapping_entry_is_skipped_not_crashed_on():
    """``describe_running`` walks past anything that is not a mapping."""
    from spacr.restart_state import describe_running

    assert describe_running([None, "measure", 7, {"module": "mask"}]) == "mask"


def test_a_nameless_entry_contributes_nothing():
    """An entry with neither module nor name has nothing to report."""
    from spacr.restart_state import describe_running

    assert describe_running([{"seconds": 90}, {"name": "   "},
                             {"module": "", "name": ""}]) == ""


def test_the_name_key_is_the_fallback_for_module():
    """``name`` is used when ``module`` is absent."""
    from spacr.restart_state import describe_running

    assert describe_running([{"name": "classify"}]) == "classify"


@pytest.mark.parametrize("seconds,expected", [
    (0, "running 0 s"),
    (59, "running 59 s"),
    (60, "running 1 min"),
    (3599, "running 59 min"),
    (3600, "running 1 h 00 min"),
    (7860, "running 2 h 11 min"),
])
def test_elapsed_reads_in_the_largest_unit_that_fits(seconds, expected):
    """Seconds below a minute stay seconds; the rest promote."""
    from spacr.restart_state import describe_running

    assert describe_running([{"module": "m", "seconds": seconds}]) == (
        f"m ({expected})")


@pytest.mark.parametrize("seconds", [None, "soon", object(), float("nan")])
def test_an_unreadable_elapsed_still_names_the_run(seconds):
    """A seconds value that will not become an int degrades to "running"."""
    from spacr.restart_state import describe_running

    text = describe_running([{"module": "m", "seconds": seconds}])

    assert text in ("m", "m (running)")
    assert text.startswith("m")


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


def test_a_saved_record_round_trips(spacr_home):
    """``save`` writes a record ``peek`` reads back with the same fields."""
    from spacr import restart_state

    path = restart_state.save(
        module="measure",
        settings={"src": "/data", "channels": (1, 2)},
        running=[{"module": "measure", "seconds": 5}, "not a mapping"],
        run_folders=["/data/run1", ""],
    )

    assert path is not None and path.is_file()
    document = restart_state.peek()
    assert document["module"] == "measure"
    assert document["version"] == restart_state.SCHEMA_VERSION
    assert document["settings"] == {"src": "/data", "channels": [1, 2]}
    assert document["running"] == [{"module": "measure", "seconds": 5}]
    assert document["run_folders"] == ["/data/run1"]


def test_a_record_that_reads_back_as_another_module_is_not_saved(
        spacr_home, monkeypatch, caplog):
    """A verify failure returns None so the caller cancels the restart."""
    import logging

    from spacr import restart_state

    class _WrongReadBack:
        dumps = staticmethod(json.dumps)

        @staticmethod
        def loads(*args, **kwargs):
            return {"module": "some other module"}

    monkeypatch.setattr(restart_state, "json", _WrongReadBack)

    with caplog.at_level(logging.WARNING, logger="spacr.restart_state"):
        result = restart_state.save(module="measure")

    assert result is None
    assert "different module" in caplog.text


def test_an_unwritable_home_cancels_the_restart(spacr_home, monkeypatch):
    """``save`` logs and returns None when the directory cannot be made."""
    from spacr import restart_state

    def _refuse(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr(type(spacr_home), "mkdir", _refuse)

    assert restart_state.save(module="measure") is None


def test_deeply_nested_settings_stop_recursing_and_stringify(spacr_home):
    """``_jsonable`` gives up below a depth cap instead of recursing forever."""
    from spacr import restart_state

    nested = {"leaf": "bottom"}
    for _ in range(20):
        nested = {"down": nested}

    encoded = restart_state._jsonable(nested)

    probe = encoded
    depth = 0
    while isinstance(probe, dict) and "down" in probe:
        probe = probe["down"]
        depth += 1
    assert depth <= 13
    assert isinstance(probe, str)
    assert "leaf" in probe


def test_a_set_of_settings_is_sorted_so_the_record_is_stable(spacr_home):
    """Sets have no order, so they are written sorted by their repr."""
    from spacr import restart_state

    assert restart_state._jsonable({"c", "a", "b"}) == ["a", "b", "c"]
    assert restart_state._jsonable(frozenset({2, 1})) == [1, 2]


def test_an_unencodable_value_becomes_its_string(spacr_home):
    """Anything JSON cannot carry is stored as ``str(value)``."""
    from spacr import restart_state

    from pathlib import Path

    assert restart_state._jsonable(Path("/data/x")) == "/data/x"


def test_peek_returns_none_when_nothing_was_saved(spacr_home):
    """A missing record is not an error."""
    from spacr import restart_state

    assert restart_state.peek() is None


def test_a_corrupt_record_is_reported_and_ignored(spacr_home, caplog):
    """Unparsable JSON logs a warning and reads as no record at all."""
    import logging

    from spacr import restart_state

    path = restart_state.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="spacr.restart_state"):
        assert restart_state.peek() is None

    assert "could not read the restart state" in caplog.text


def test_a_record_that_is_not_an_object_is_ignored(spacr_home):
    """A JSON list parses fine but is not a restart record."""
    from spacr import restart_state

    path = restart_state.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

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


def test_a_record_path_that_is_a_directory_is_reported_not_raised(
        spacr_home, caplog):
    """``discard`` swallows every removal error after logging it."""
    import logging

    from spacr import restart_state

    path = restart_state.state_path()
    path.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING, logger="spacr.restart_state"):
        assert restart_state.discard() is False

    assert "could not remove the restart state" in caplog.text


def test_the_restart_command_names_this_interpreter_and_entry_point():
    """The relaunch never depends on a ``spacr`` executable found on PATH."""
    import sys

    from spacr.restart_state import command

    assert command() == [sys.executable, "-m", "spacr.qt"]
