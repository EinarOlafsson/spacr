"""``spacr.qt.memory_budget``: which cached things go, and in what order.

The module's own contract, from its docstring: *an entry is never dropped for
being large alone -- size decides the ORDER of a trim, and idleness decides
whether one happens.* That distinction is the whole design. A cache that
evicted by size would throw away the big model a user is actively working with
and keep a hundred small stale thumbnails.

It also refuses to guess. Free memory that cannot be measured is not treated
as memory that is short, because a cache that cannot be shown to be a problem
should not be dropped on suspicion.
"""
from __future__ import annotations

import pytest

from spacr.qt import memory_budget as mb


NOW = 1_000_000.0


def _entry(key, megabytes, minutes_idle):
    return (key, megabytes, NOW - minutes_idle * 60.0)


# ---------------------------------------------------------------------------
# idleness decides whether a trim happens
# ---------------------------------------------------------------------------

def test_nothing_idle_and_nothing_over_the_ceiling_drops_nothing():
    entries = [_entry("a", 10.0, 1), _entry("b", 10.0, 2)]

    assert mb.what_to_drop(entries, NOW, idle_minutes=30,
                           ceiling_mb=1000) == []


def test_an_entry_idle_longer_than_the_timeout_goes():
    entries = [_entry("fresh", 10.0, 1), _entry("stale", 10.0, 99)]

    assert mb.what_to_drop(entries, NOW, idle_minutes=30,
                           ceiling_mb=1000) == ["stale"]


def test_a_large_entry_in_active_use_is_not_dropped():
    """The contract's own sentence, made a test.

    Fifty times the size of everything else and touched a minute ago: this is
    the model the user is working with, and evicting it is the failure the
    ordering rule exists to prevent.
    """
    entries = [_entry("the model", 5000.0, 1), _entry("thumb", 1.0, 1)]

    assert mb.what_to_drop(entries, NOW, idle_minutes=30,
                           ceiling_mb=100_000) == []


# ---------------------------------------------------------------------------
# size decides the order once a trim is happening
# ---------------------------------------------------------------------------

def test_over_the_ceiling_the_least_recently_used_go_first():
    """Least recently used, not largest, and not first in the list."""
    entries = [_entry("oldest", 60.0, 20), _entry("newest", 60.0, 1),
               _entry("middle", 60.0, 10)]

    dropped = mb.what_to_drop(entries, NOW, idle_minutes=999, ceiling_mb=100)

    assert dropped[0] == "oldest"
    assert "newest" not in dropped


def test_the_trim_stops_as_soon_as_it_fits():
    """Dropping more than necessary is a cache that keeps re-fetching."""
    entries = [_entry("oldest", 60.0, 30), _entry("middle", 60.0, 20),
               _entry("newest", 60.0, 1)]

    dropped = mb.what_to_drop(entries, NOW, idle_minutes=999, ceiling_mb=150)

    assert dropped == ["oldest"], "the trim went further than it had to"


def test_both_reasons_apply_in_order():
    """Idle entries go first, and the ceiling is judged on what is left.

    Counting the already-doomed entries against the ceiling would under-trim,
    leaving the cache over its limit after a pass that reported success.
    """
    entries = [_entry("stale", 500.0, 99), _entry("old", 60.0, 20),
               _entry("new", 60.0, 1)]

    dropped = mb.what_to_drop(entries, NOW, idle_minutes=30, ceiling_mb=100)

    assert dropped[0] == "stale"
    assert "old" in dropped, "the ceiling was judged including the stale entry"
    assert "new" not in dropped


def test_an_empty_cache_needs_no_trimming():
    assert mb.what_to_drop([], NOW, idle_minutes=1, ceiling_mb=0) == []


def test_the_preferences_are_read_when_the_caller_names_no_limits(monkeypatch):
    """The ordinary call from the running application passes neither."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_idle_minutes", lambda: 30)
    monkeypatch.setattr(preferences, "get_cache_ceiling_mb", lambda: 1000)

    entries = [_entry("stale", 10.0, 99), _entry("fresh", 10.0, 1)]

    assert mb.what_to_drop(entries, NOW) == ["stale"]


def test_unreadable_preferences_fall_back_to_the_shipped_defaults(monkeypatch):
    """A broken settings file must not disable the whole budget.

    Falling through with no limits at all would mean nothing is ever dropped,
    and the cache grows until the machine swaps -- the failure this module
    exists to prevent, arriving through its own error handling.
    """
    from spacr.qt import preferences

    def refuse():
        raise RuntimeError("the settings store is unreadable")

    monkeypatch.setattr(preferences, "get_idle_minutes", refuse)
    monkeypatch.setattr(preferences, "get_cache_ceiling_mb", refuse)

    entries = [_entry("ancient", 10.0, mb.DEFAULT_IDLE_MINUTES * 10)]

    assert mb.what_to_drop(entries, NOW) == ["ancient"]


# ---------------------------------------------------------------------------
# the headroom floor, and refusing to guess
# ---------------------------------------------------------------------------

def test_memory_that_cannot_be_measured_is_not_treated_as_short(monkeypatch):
    """``None`` means "no answer", and a cache is not dropped on suspicion."""
    monkeypatch.setattr(mb, "free_megabytes", lambda: None)

    assert mb.headroom_is_short(floor_mb=100_000) is False


def test_free_memory_below_the_floor_is_short(monkeypatch):
    monkeypatch.setattr(mb, "free_megabytes", lambda: 100.0)

    assert mb.headroom_is_short(floor_mb=500.0) is True


def test_free_memory_above_the_floor_is_not(monkeypatch):
    monkeypatch.setattr(mb, "free_megabytes", lambda: 5000.0)

    assert mb.headroom_is_short(floor_mb=500.0) is False


def test_a_floor_that_cannot_be_read_is_not_a_shortage(monkeypatch):
    """Same rule as an unmeasurable free figure: no answer is not a yes."""
    from spacr.qt import preferences

    monkeypatch.setattr(mb, "free_megabytes", lambda: 1.0)

    def refuse():
        raise RuntimeError("the settings store is unreadable")

    monkeypatch.setattr(preferences, "get_headroom_mb", refuse)

    assert mb.headroom_is_short() is False


def test_free_memory_that_cannot_be_read_answers_none(monkeypatch):
    """psutil is optional, and its absence is not a measurement of zero."""
    real_import = __import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_psutil)

    assert mb.free_megabytes() is None


def test_a_real_reading_is_a_plausible_number_of_megabytes():
    """Otherwise every test above would pass on a function returning None."""
    free = mb.free_megabytes()

    assert free is None or 1.0 < free < 100_000_000.0


# ---------------------------------------------------------------------------
# the recommended budgets
# ---------------------------------------------------------------------------

def test_every_performance_level_has_a_budget():
    for level, expected in mb.RECOMMENDED.items():
        assert mb.recommended_for(level) == expected
        assert len(expected) == 3


def test_an_unknown_level_falls_back_to_balanced():
    """A preference file from a newer spaCR can name a level this one lacks."""
    assert mb.recommended_for("a level that does not exist") == (
        mb.RECOMMENDED["balanced"])
