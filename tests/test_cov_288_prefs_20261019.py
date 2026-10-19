"""The preference writer's edges: a failed write, a superseded value, and a
flush that finds no live writer.

Each test swaps the module's pending table and state for private ones so
nothing leaks into the real preference writer.
"""

from __future__ import annotations

import threading
import types

import pytest

from spacr.qt import prefs


@pytest.fixture
def private(monkeypatch):
    stores = threading.Condition()
    pending = {}
    state = {"busy": False, "thread": None}
    monkeypatch.setattr(prefs, "_STORES", stores)
    monkeypatch.setattr(prefs, "_PENDING_STORES", pending)
    monkeypatch.setattr(prefs, "_STORE_STATE", state)
    return pending, state


def test_a_write_that_fails_is_logged_and_does_not_stop_the_batch(
        monkeypatch, caplog):
    seen = []

    class Boom:
        def __init__(self, name, kind):
            self.name = name

        def setValue(self, key, value):
            raise OSError("disk full")

        def sync(self):
            seen.append("sync")

    monkeypatch.setattr(prefs, "QSettings", Boom)
    monkeypatch.setattr(prefs.QSettings, "Format", lambda kind: kind,
                        raising=False)
    with caplog.at_level("WARNING"):
        prefs._write_stores({("a.ini", 0, "k"): 1})
    assert "a preference write failed" in caplog.text
    assert seen == []


def test_the_writer_keeps_a_value_that_was_replaced_while_it_wrote(
        monkeypatch, private):
    pending, state = private
    first, second = object(), object()
    key = ("x.ini", 0, "k")
    pending[key] = first
    batches = []

    def write(batch):
        batches.append(dict(batch))
        if len(batches) == 1:
            pending[key] = second

    monkeypatch.setattr(prefs, "_write_stores", write)

    class Stop(Exception):
        pass

    def wait():
        raise Stop

    monkeypatch.setattr(prefs._STORES, "wait", wait, raising=False)
    with pytest.raises(Stop):
        prefs._run_stores()
    assert [b[key] for b in batches] == [first, second], (
        "the newer value must be written next, not dropped")
    assert pending == {}
    assert state["busy"] is False


def test_a_flush_with_no_live_writer_stores_on_the_callers_thread(
        monkeypatch, private):
    pending, state = private
    written = []
    monkeypatch.setattr(prefs, "_write_stores", written.append)
    pending[("y.ini", 0, "k")] = 5
    state["thread"] = types.SimpleNamespace(is_alive=lambda: False)
    assert prefs._flush_stores(timeout=0.5) is True
    assert written == [{("y.ini", 0, "k"): 5}]
    assert pending == {}


def test_a_flush_while_a_write_is_busy_leaves_the_queue_to_the_writer(
        monkeypatch, private):
    pending, state = private
    written = []
    monkeypatch.setattr(prefs, "_write_stores", written.append)
    pending[("z.ini", 0, "k")] = 1
    state["busy"] = True
    state["thread"] = types.SimpleNamespace(is_alive=lambda: True)
    assert prefs._flush_stores(timeout=0.0) is False
    assert written == []
    assert len(pending) == 1


def test_a_batch_is_stored_in_the_file_it_was_set_in(tmp_path):
    from PySide6.QtCore import QSettings

    path = str(tmp_path / "prefs.ini")
    kind = int(QSettings.Format.IniFormat.value)
    prefs._write_stores({(path, kind, "a/b"): "1", (path, kind, "c"): "2"})
    stored = QSettings(path, QSettings.Format.IniFormat)
    assert stored.value("a/b") == "1" and stored.value("c") == "2"


def test_a_flush_gives_a_live_writer_its_time_and_then_reports_the_wait(
        monkeypatch, private):
    pending, state = private
    monkeypatch.setattr(prefs, "_write_stores", lambda batch: None)
    pending[("w.ini", 0, "k")] = 1
    state["thread"] = types.SimpleNamespace(is_alive=lambda: True)
    assert prefs._flush_stores(timeout=0.05) is True
    assert pending == {}
