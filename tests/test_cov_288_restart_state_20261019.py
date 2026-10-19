"""The restart-state writer's flush: no live writer, a busy writer, a patient wait."""

from __future__ import annotations

import threading
import types

import pytest

from spacr import restart_state as rs


@pytest.fixture
def private(monkeypatch):
    state = {"pending": None, "busy": False, "thread": None}
    monkeypatch.setattr(rs, "_WRITES", threading.Condition())
    monkeypatch.setattr(rs, "_WRITE_STATE", state)
    ran = []
    monkeypatch.setattr(rs, "_run_one_write", ran.append)
    return state, ran


def test_a_job_left_waiting_by_a_dead_writer_runs_on_the_callers_thread(private):
    state, ran = private
    state["pending"] = "job"
    state["thread"] = types.SimpleNamespace(is_alive=lambda: False)
    assert rs._flush_writes(timeout=0.5) is True
    assert ran == ["job"]
    assert state["pending"] is None


def test_a_flush_does_not_run_a_job_while_a_write_is_in_progress(private):
    state, ran = private
    state["pending"] = "job"
    state["busy"] = True
    state["thread"] = types.SimpleNamespace(is_alive=lambda: True)
    assert rs._flush_writes(timeout=0.0) is False
    assert ran == [] and state["pending"] == "job"


def test_a_flush_waits_for_a_live_writer_then_gives_up_at_the_deadline(private):
    state, ran = private
    state["pending"] = "job"
    state["thread"] = types.SimpleNamespace(is_alive=lambda: True)
    assert rs._flush_writes(timeout=0.05) is True
    assert ran == ["job"]


def test_a_flush_with_nothing_waiting_is_a_no_op(private):
    state, ran = private
    assert rs._flush_writes(timeout=0.0) is True
    assert ran == []
