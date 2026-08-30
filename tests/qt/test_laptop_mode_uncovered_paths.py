"""The measurement fallbacks, the unreadable override, and what ``apply`` writes.

Laptop mode is allowed to be wrong about the machine; it is not allowed to
raise on a platform whose ``os`` module is missing a call, and it is not
allowed to claim a preference changed when it did not.
"""

from __future__ import annotations

import os

import pytest

from spacr.qt import laptop_mode as LM


def test_usable_cores_falls_back_to_cpu_count_without_scheduler_affinity(monkeypatch):
    """``os.sched_getaffinity`` is Linux-only; elsewhere the machine is still counted."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert LM.usable_cores() == 6


def test_usable_cores_never_reports_zero_cores(monkeypatch):
    """``os.cpu_count`` returns None on an exotic platform; one core is the floor."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert LM.usable_cores() == 1


def test_usable_cores_survives_an_affinity_call_that_fails(monkeypatch):
    def _refuse(_pid):
        raise OSError("no such process")

    monkeypatch.setattr(os, "sched_getaffinity", _refuse)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert LM.usable_cores() == 3


def test_unreadable_memory_is_none_rather_than_an_exception(monkeypatch):
    """A platform without ``SC_PHYS_PAGES`` reports unknown, not a traceback."""
    def _refuse(_name):
        raise ValueError("unrecognised configuration name")

    monkeypatch.setattr(os, "sysconf", _refuse)
    assert LM.total_memory_gib() is None


def test_a_reading_with_no_memory_still_decides_and_omits_the_memory_phrase(monkeypatch):
    """`wanted` must cope with the None `total_memory_gib` can return."""
    def _refuse(_name):
        raise ValueError("unrecognised configuration name")

    monkeypatch.setattr(os, "sysconf", _refuse)
    monkeypatch.delenv(LM.OVERRIDE_VARIABLE, raising=False)
    reading = LM.measure()
    assert reading["memory_gib"] is None
    assert reading["little_memory"] is False
    _on, why = LM.wanted(reading)
    assert "GiB of memory" not in why


@pytest.mark.parametrize("raw", ["", "maybe", "2", "  "])
def test_an_unrecognised_override_leaves_the_decision_to_the_measurement(monkeypatch, raw):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, raw)
    assert LM.override() is None


def test_the_override_is_read_case_insensitively_and_untrimmed(monkeypatch):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "  ON  ")
    assert LM.override() is True
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, " Off ")
    assert LM.override() is False


def test_apply_with_no_argument_measures_the_machine(monkeypatch):
    """`apply(None)` must reach the same answer `wanted()` would."""
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "0")
    result = LM.apply()
    assert result["on"] is False
    assert LM.OVERRIDE_VARIABLE in result["why"]
    assert result["changed"] == []


def _ambient_seam(monkeypatch, preferences, enabled, written):
    """Stand in for the preferences module's ambient on/off pair.

    ``raising=True`` is deliberate and load-bearing. ``monkeypatch.setattr``
    with ``raising=False`` CREATES a name the module does not have, so a call
    to a misspelled getter would be manufactured by the fixture and the test
    would pass while the shipped code reached nothing -- which is exactly how
    a silent no-op in ``apply`` stayed hidden. Patching only names that
    already exist means a rename in preferences fails here first.
    """
    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: enabled)
    monkeypatch.setattr(preferences, "set_ambient_enabled", written.append)


def test_turning_the_mode_on_turns_the_ambient_backdrop_off(monkeypatch):
    from spacr.qt import preferences

    written = []
    _ambient_seam(monkeypatch, preferences, True, written)

    result = LM.apply(True)

    assert result["on"] is True
    assert written == [False], "the ambient backdrop was not actually turned off"
    assert result["changed"] == ["ambient_enabled=False"]
    assert "the caller" in result["why"]


def test_an_already_dark_backdrop_is_not_reported_as_changed(monkeypatch):
    """`changed` names what was written, so it must stay empty when nothing was."""
    from spacr.qt import preferences

    written = []
    _ambient_seam(monkeypatch, preferences, False, written)

    result = LM.apply(True)

    assert result["on"] is True
    assert written == []
    assert result["changed"] == []


def test_turning_the_mode_off_writes_no_preference_at_all(monkeypatch):
    from spacr.qt import preferences

    written = []
    _ambient_seam(monkeypatch, preferences, True, written)

    result = LM.apply(False)

    assert result["on"] is False
    assert written == []
    assert result["changed"] == []


def test_a_preference_store_that_will_not_answer_does_not_stop_the_mode(monkeypatch):
    """Laptop mode is decorative; a settings backend that fails must not sink it."""
    from spacr.qt import preferences

    def _broken():
        raise RuntimeError("settings backend is gone")

    for name in ("ambient_enabled", "get_ambient_enabled"):
        monkeypatch.setattr(preferences, name, _broken, raising=False)

    result = LM.apply(True)

    assert result["on"] is True
    assert result["changed"] == []


def test_describe_lists_every_drawing_cost_when_the_mode_is_on(monkeypatch):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "1")
    block = LM.describe()
    assert "a run computes exactly the same answer either way." in block
    for what, cost in LM.what_it_turns_down():
        assert f"  - {what}: {cost}" in block


def test_describe_is_a_single_line_when_the_mode_is_off(monkeypatch):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "0")
    block = LM.describe()
    assert block.count("\n") == 0
    assert "turned down" not in block
