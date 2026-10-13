"""Laptop mode: chosen automatically, overridable, and it says what it did.

Instruction 268's rule, in the maintainer's words: "just turning things of
is one strategy, but my best case scenario is being able to keep as many
features as possible just optimizing them so they also run on worse
hardware." So this is the FALLBACK, and it turns down what is DRAWN rather
than what a module does.
"""

from __future__ import annotations

import pytest

from spacr.qt import laptop_mode as LM


def _reading(cores=32, memory=64.0, override=None):
    return {"cores": cores, "memory_gib": memory,
            "few_cores": cores < LM.SMALL_CORE_COUNT,
            "little_memory": memory is not None and memory < LM.SMALL_MEMORY_GIB,
            "override": override}


def test_a_big_machine_is_left_alone():
    on, why = LM.wanted(_reading(cores=32, memory=64.0))
    assert on is False
    assert "32 usable core" in why


def test_few_cores_turns_it_on():
    on, why = LM.wanted(_reading(cores=2, memory=64.0))
    assert on is True
    assert "fewer than" in why


def test_little_memory_turns_it_on():
    on, why = LM.wanted(_reading(cores=32, memory=4.0))
    assert on is True
    assert "less than" in why


def test_both_reasons_are_given_when_both_hold():
    _on, why = LM.wanted(_reading(cores=2, memory=4.0))
    assert "cores" in why and "memory" in why


def test_the_override_wins_in_both_directions():
    on, why = LM.wanted(_reading(cores=32, memory=64.0, override=True))
    assert on is True and LM.OVERRIDE_VARIABLE in why
    on, why = LM.wanted(_reading(cores=1, memory=1.0, override=False))
    assert on is False and LM.OVERRIDE_VARIABLE in why


def test_the_reason_is_given_even_when_the_answer_is_no():
    """"spaCR decided not to" is a thing a user reports, and somebody has to
    be able to check it."""
    _on, why = LM.wanted(_reading())
    assert why.strip()
    assert LM.OVERRIDE_VARIABLE in why, "no way out is offered"


def test_it_says_what_it_turns_down_and_what_that_costs():
    for what, cost in LM.what_it_turns_down():
        assert what.strip() and cost.strip()


def test_nothing_it_touches_changes_what_a_run_computes():
    """The promise that makes an automatic decision acceptable."""
    changed = " ".join(w for w, _ in LM.what_it_turns_down()).lower()
    for computational in ("threshold", "permutation", "model", "channel",
                          "segmentation", "measurement"):
        assert computational not in changed, (
            f"laptop mode claims to change {computational!r}, which a run reads")


def test_describe_names_the_cost_when_it_is_on(monkeypatch):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "1")
    text = LM.describe()
    assert "turned down" in text
    assert "same answer either way" in text


def test_describe_is_short_when_it_is_off(monkeypatch):
    monkeypatch.setenv(LM.OVERRIDE_VARIABLE, "0")
    assert "turned down" not in LM.describe()


def test_applying_it_reports_what_it_actually_wrote():
    result = LM.apply(on=False)
    assert result["on"] is False
    assert result["changed"] == [], "it changed something while switched off"


def test_usable_cores_is_never_zero():
    assert LM.usable_cores() >= 1
