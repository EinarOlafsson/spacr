"""Observing a setting change records nothing when there is no analysis lock."""

from __future__ import annotations

from spacr import run_journal


def test_no_lock_on_this_pipeline_means_nothing_is_observed(monkeypatch):
    asked = []

    def find(app_key, src):
        asked.append((app_key, src))
        return None

    monkeypatch.setattr(run_journal, "_find_lock", find)
    assert run_journal._observe_settings_changes(
        {"src": "/plates/a"}, "mask", keys=["channels"]) is None
    assert asked == [("mask", "/plates/a")]


def test_settings_that_are_missing_are_treated_as_no_source(monkeypatch):
    asked = []
    monkeypatch.setattr(run_journal, "_find_lock",
                        lambda app_key, src: asked.append(src))
    assert run_journal._observe_settings_changes(None, "mask") is None
    assert asked == [None]
