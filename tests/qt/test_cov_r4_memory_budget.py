"""``spacr.qt.memory_budget``: the half-named call, and the trim that empties.

``tests/qt/test_the_memory_budget_drops_what_nothing_is_using.py`` drives the
two ends of ``what_to_drop``'s preference lookup -- the caller who names both
limits, and the caller who names neither. This file pins the middle, which is
what the running application actually does when one limit is a deliberate
choice and the other is the user's setting: **each limit is looked up
independently**, so naming one must not silently pull the other out of
preferences (or, worse, override the one that was named).

It also pins the trim that runs out of entries. Every other test of the size
ceiling stops early, because something in the cache always fits underneath it;
here nothing does, and the loop has to fall off its end and answer with the
whole cache rather than spin or keep the last entry back.
"""
from __future__ import annotations

from spacr.qt import memory_budget as mb


NOW = 1_000_000.0


def _entry(key, megabytes, minutes_idle):
    return (key, megabytes, NOW - minutes_idle * 60.0)


def _spy_preferences(monkeypatch, *, idle, ceiling):
    """Record which preference getters a call actually reaches."""
    from spacr.qt import preferences

    seen = []

    def get_idle_minutes():
        seen.append("idle")
        return idle

    def get_cache_ceiling_mb():
        seen.append("ceiling")
        return ceiling

    monkeypatch.setattr(preferences, "get_idle_minutes", get_idle_minutes)
    monkeypatch.setattr(preferences, "get_cache_ceiling_mb",
                        get_cache_ceiling_mb)
    return seen


# ---------------------------------------------------------------------------
# one limit named, one left to the preferences
# ---------------------------------------------------------------------------

def test_naming_the_timeout_still_reads_the_ceiling_from_preferences(
        monkeypatch):
    """A caller's timeout is kept, and the ceiling still comes from settings.

    The absence assertion (``"idle" not in seen``) is only worth making
    because the second call in this same test makes it present: omit the
    timeout and the getter is reached.
    """
    seen = _spy_preferences(monkeypatch, idle=999.0, ceiling=15)

    # Both entries are fresh under the caller's 30-minute timeout, so nothing
    # is idle-dropped; 20 MB against the preferences ceiling of 15 MB is over,
    # so the least recently used goes for size.
    entries = [_entry("older", 10.0, 5), _entry("newer", 10.0, 1)]

    assert mb.what_to_drop(entries, NOW, idle_minutes=30) == ["older"]
    assert seen == ["ceiling"], "the named timeout must not be looked up"

    # ... and the same entries with nothing named do reach the idle getter,
    # whose 999-minute timeout keeps both from going stale.
    seen.clear()
    assert mb.what_to_drop(entries, NOW) == ["older"]
    assert seen == ["idle", "ceiling"]


def test_naming_the_ceiling_still_reads_the_timeout_from_preferences(
        monkeypatch):
    """The mirror image: the caller's ceiling is kept, the timeout is read."""
    seen = _spy_preferences(monkeypatch, idle=2.0, ceiling=1)

    # The preferences timeout of 2 minutes makes "stale" idle; "fresh" is not,
    # and the caller's roomy 4096 MB ceiling -- not the 1 MB in preferences --
    # leaves it in place.
    entries = [_entry("stale", 10.0, 30), _entry("fresh", 10.0, 1)]

    assert mb.what_to_drop(entries, NOW, ceiling_mb=4096) == ["stale"]
    assert seen == ["idle"], "the named ceiling must not be looked up"

    # ... and omitting the ceiling does reach its getter, whose 1 MB ceiling
    # then evicts the entry the caller's 4096 MB had kept.
    seen.clear()
    assert mb.what_to_drop(entries, NOW) == ["stale", "fresh"]
    assert seen == ["idle", "ceiling"]


# ---------------------------------------------------------------------------
# the trim that runs out of things to give up
# ---------------------------------------------------------------------------

def test_a_ceiling_under_every_entry_gives_up_the_whole_cache():
    """Nothing fits, so the trim reaches the end of the cache and stops there.

    Every other ceiling test breaks out early because some entry is small
    enough to keep. With a 5 MB ceiling nothing is, and the loop must run off
    its end and answer with all three keys rather than hold the last one back
    on the strength of a bound it still exceeds.
    """
    entries = [
        _entry("newest", 100.0, 1),
        _entry("oldest", 40.0, 9),
        _entry("middle", 70.0, 5),
    ]

    # Fresh under the timeout, so idleness drops nothing: this is a pure trim.
    dropped = mb.what_to_drop(entries, NOW, idle_minutes=60, ceiling_mb=5)

    assert dropped == ["oldest", "middle", "newest"]

    # And the same cache under a ceiling one of them fits below stops early,
    # which is what makes the answer above a property of the ceiling and not
    # of the function always emptying itself.
    assert mb.what_to_drop(entries, NOW, idle_minutes=60,
                           ceiling_mb=100) == ["oldest", "middle"]
