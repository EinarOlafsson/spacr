"""``spacr.qt.memory_budget.what_to_drop``: idle first, then oldest under the ceiling.

One table pins the exact keys selected, in order, for each combination of
idle timeout and size ceiling, and which preference getters a call reaches:
each limit is looked up only when it was not named, and an unreadable
preference store falls back to the shipped defaults.
"""
from __future__ import annotations

import builtins

import pytest

from spacr.qt import memory_budget as mb

pytestmark = pytest.mark.qt

NOW = 1_000_000.0


def _entry(key, megabytes, minutes_idle):
    return (key, megabytes, NOW - minutes_idle * 60.0)


def test_what_to_drop_selects_idle_keys_then_least_recently_used_over_the_ceiling(
        monkeypatch, subtests):
    """Exact keys in selection order, and the preference getters each call reaches."""
    from spacr.qt import preferences

    prefs = {"idle": 999.0, "ceiling": 15}
    seen = []

    def get_idle_minutes():
        seen.append("idle")
        return prefs["idle"]

    def get_cache_ceiling_mb():
        seen.append("ceiling")
        return prefs["ceiling"]

    monkeypatch.setattr(preferences, "get_idle_minutes", get_idle_minutes)
    monkeypatch.setattr(preferences, "get_cache_ceiling_mb",
                        get_cache_ceiling_mb)

    two_fresh = [_entry("older", 10.0, 5), _entry("newer", 10.0, 1)]
    stale_fresh = [_entry("stale", 10.0, 30), _entry("fresh", 10.0, 1)]
    three = [_entry("newest", 100.0, 1), _entry("oldest", 40.0, 9),
             _entry("middle", 70.0, 5)]
    by_age = [("huge_and_fresh", 900.0, NOW - 10),
              ("small_and_old", 10.0, NOW - 60),
              ("medium_and_older", 100.0, NOW - 120)]
    cases = [
        # case, preferences (idle, ceiling), entries, kwargs, expected,
        # getters reached
        ("timeout named, ceiling from preferences", (999.0, 15), two_fresh,
         dict(idle_minutes=30), ["older"], ["ceiling"]),
        ("neither named", (999.0, 15), two_fresh, {}, ["older"],
         ["idle", "ceiling"]),
        ("ceiling named, timeout from preferences", (2.0, 1), stale_fresh,
         dict(ceiling_mb=4096), ["stale"], ["idle"]),
        ("neither named, the tiny ceiling evicts the fresh one too",
         (2.0, 1), stale_fresh, {}, ["stale", "fresh"], ["idle", "ceiling"]),
        ("a ceiling under every entry empties the cache", (0, 0), three,
         dict(idle_minutes=60, ceiling_mb=5), ["oldest", "middle", "newest"],
         []),
        ("a ceiling one entry fits under stops early", (0, 0), three,
         dict(idle_minutes=60, ceiling_mb=100), ["oldest", "middle"], []),
        ("size decides whether, age decides which", (0, 0), by_age,
         dict(idle_minutes=60, ceiling_mb=500),
         ["medium_and_older", "small_and_old", "huge_and_fresh"], []),
        ("the fresh large entry survives once the rest fit", (0, 0), by_age,
         dict(idle_minutes=60, ceiling_mb=900),
         ["medium_and_older", "small_and_old"], []),
        ("an idle entry goes under a roomy ceiling", (0, 0),
         [("stale", 1.0, NOW - 3600), ("live", 1.0, NOW - 5)],
         dict(idle_minutes=1, ceiling_mb=99999), ["stale"], []),
        ("idle entries keep input order and leave room", (0, 0),
         [_entry("b", 50.0, 90), _entry("a", 50.0, 120),
          _entry("c", 50.0, 1)], dict(idle_minutes=60, ceiling_mb=50),
         ["b", "a"], []),
        ("exactly at the ceiling is not over", (0, 0),
         [_entry("x", 30.0, 1), _entry("y", 20.0, 2)],
         dict(idle_minutes=60, ceiling_mb=50), [], []),
        ("keys come back as strings", (0, 0), [(7, 1.0, NOW - 7200)],
         dict(idle_minutes=1, ceiling_mb=100), ["7"], []),
        ("nothing cached", (0, 0), [], dict(idle_minutes=1, ceiling_mb=1),
         [], []),
    ]
    for case, (idle, ceiling), entries, kwargs, expected, reached in cases:
        with subtests.test(msg=case):
            prefs.update(idle=idle, ceiling=ceiling)
            seen.clear()
            assert mb.what_to_drop(entries, NOW, **kwargs) == expected
            assert seen == reached

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.endswith("preferences") or name == "preferences":
            raise ImportError("preferences unavailable")
        return real_import(name, *args, **kwargs)

    default_idle = NOW - mb.DEFAULT_IDLE_MINUTES * 60.0
    unreadable = [
        ("just past the default timeout", [("stale", 1.0, default_idle - 1)],
         {}, ["stale"]),
        ("just inside the default timeout", [("fresh", 1.0, default_idle + 1)],
         {}, []),
        ("over the default ceiling", [
            ("a", mb.DEFAULT_CACHE_CEILING_MB, NOW - 2),
            ("b", 1.0, NOW - 1)], {}, ["a"]),
        ("a named timeout survives the fallback",
         [("stale", 1.0, NOW - 120)], dict(idle_minutes=1), ["stale"]),
        ("a named ceiling survives the fallback",
         [("x", 10.0, NOW - 2), ("y", 10.0, NOW - 1)], dict(ceiling_mb=10),
         ["x"]),
    ]
    for case, entries, kwargs, expected in unreadable:
        with subtests.test(msg=f"unreadable preferences: {case}"), \
                monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", refuse)
            seen.clear()
            assert mb.what_to_drop(entries, NOW, **kwargs) == expected
            assert seen == []


def test_the_stored_budget_is_clamped_to_the_published_limits(
        qapp, tmp_path, monkeypatch, subtests):
    """A hand-edited idle timeout or ceiling reads back inside [MIN, MAX], exactly at the edges."""
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    store = QSettings(str(tmp_path / "qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    cases = [
        # (id, stored idle, idle read, stored ceiling, ceiling read)
        ("above both maxima", "100000", 480.0, "999999999", 131072),
        ("exactly the maxima", "480", 480.0, "131072", 131072),
        ("below both minima", "-5", 0.0, "1", 128),
        ("inside the limits", "7.5", 7.5, "4096", 4096),
    ]
    for case_id, idle, idle_read, ceiling, ceiling_read in cases:
        with subtests.test(msg=case_id):
            store.setValue(preferences._KEY_IDLE_MINUTES, idle)
            store.setValue(preferences._KEY_CACHE_CEILING, ceiling)
            assert preferences.get_idle_minutes() == idle_read
            assert preferences.get_cache_ceiling_mb() == ceiling_read
    assert (mb.MIN_IDLE_MINUTES, mb.MAX_IDLE_MINUTES,
            mb.MIN_CACHE_CEILING_MB, mb.MAX_CACHE_CEILING_MB) == (
        0.0, 480.0, 128, 131072)
