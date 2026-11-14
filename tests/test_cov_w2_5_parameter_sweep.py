"""A sweep is judged by what its rows say, not by how many trials ran.

The value of a sweep row is that it carries the design it fitted, where the
named controls landed, and why a trial failed — so a trial that dies still
produces a row. These tests drive the orchestration with injected runners so
no regression is actually fitted, and read the rows and the CSV the sweep
writes as it goes.
"""
from __future__ import annotations

import builtins
import io

import pytest

from spacr import parameter_sweep as ps


def test_a_mode_that_is_neither_grid_nor_random_is_refused():
    """A typo must not silently enumerate in some third order."""
    with pytest.raises(ValueError, match="'grid' or 'random'"):
        ps.build_trials(ps.SweepSpace(), mode="exhaustive")


def test_without_systemd_run_there_is_no_containment(monkeypatch):
    """Containment is claimed only when the tool is actually there."""
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    ps.containment_available.cache_clear()
    try:
        assert ps.containment_available() is False
    finally:
        ps.containment_available.cache_clear()


def test_a_meminfo_without_memavailable_reports_no_limit(monkeypatch):
    """An unreadable memory figure must not stop a sweep starting."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return io.StringIO("MemTotal:  1000 kB\nSwapFree:  0 kB\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    assert ps.free_memory_gb() == float("inf")


def test_a_row_reproduces_its_own_regression_settings():
    """A sweep row carries every setting, so a trial can be re-run from it."""
    row = {"trial_id": 3, "status": "ok", "seconds": 12.0,
           "regression_type": "ridge", "alpha": "1",
           "positive_present": True, "positive_rank": 1,
           "folder": "/tmp/trial_3"}

    settings = ps.settings_for_trial({"src": "/data"}, row)

    assert settings["src"] == "/tmp/trial_3", "the trial's own folder wins"
    assert settings["regression_type"] == "ridge"
    assert settings["alpha"] == 1, "a stored literal is read back as one"
    assert "positive_present" not in settings
    assert "status" not in settings


def test_the_containment_note_states_the_actual_limits(monkeypatch):
    """The user is told which caps are on, and what happens when one bites."""
    monkeypatch.setattr(ps, "containment_available", lambda: True)

    note = ps.containment_note()

    assert ps.TRIAL_MEMORY_MAX in note
    assert ps.TRIAL_CPU_QUOTA in note
    assert "only that trial is stopped" in note


def test_without_containment_the_note_says_what_is_left(monkeypatch):
    """An uncapped sweep is a decision the user should make knowingly."""
    monkeypatch.setattr(ps, "containment_available", lambda: False)

    note = ps.containment_note()

    assert "unavailable" in note
    assert "cannot prevent a single trial from exhausting system memory" in note
    assert "Reduce the worker count" in note


def test_a_missing_value_in_a_row_is_not_a_setting():
    """A NaN in the CSV means "this trial had none", not "set it to NaN"."""
    row = {"trial_id": 1, "status": "ok", "seconds": 1.0,
           "regression_type": "ols", "alpha": float("nan")}

    settings = ps.settings_for_trial({"src": "/data"}, row)

    assert "alpha" not in settings
    assert settings["regression_type"] == "ols"
