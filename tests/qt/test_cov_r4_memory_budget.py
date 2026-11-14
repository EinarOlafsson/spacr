"""``spacr.qt.memory_budget``: the recommendation table and the headroom reading.

``what_to_drop`` is covered by ``tests/qt/test_memory_budget_what_to_drop.py``.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# 352: the arms nothing exercised -- the recommendation table, the two
# preference fallbacks, and the trim's second reason.
# ---------------------------------------------------------------------------

def test_every_performance_level_has_a_recommendation():
    """A level with no row would silently get "balanced" in the panel.

    `recommended_for` falls back to balanced for an unknown name, which is
    the right behaviour for a corrupted setting and the wrong one for a
    level spaCR actually ships -- there it would show the user numbers for
    a posture they did not choose, with nothing saying so.
    """
    from spacr.qt import memory_budget as mb
    from spacr.qt.preferences import PERFORMANCE_LEVELS

    for level in PERFORMANCE_LEVELS:
        assert level in mb.RECOMMENDED, level
    for level, (idle, cache, headroom) in mb.RECOMMENDED.items():
        assert idle > 0 and cache > 0 and headroom > 0, level

    # The levels are ordered, and so are their budgets: a "performance"
    # posture that cached less than a laptop would be the table saying the
    # opposite of its own names.
    ordered = ["laptop", "extra_performance", "performance", "balanced",
               "workstation"]
    caches = [mb.RECOMMENDED[name][1] for name in ordered]
    assert caches == sorted(caches), caches


def test_an_unknown_level_is_given_the_balanced_budget():
    """A corrupted preference must not leave the panel with no numbers."""
    from spacr.qt import memory_budget as mb

    assert mb.recommended_for("not a level") == mb.RECOMMENDED["balanced"]
    assert mb.recommended_for("") == mb.RECOMMENDED["balanced"]


def test_memory_that_cannot_be_measured_is_not_treated_as_short(monkeypatch):
    """"A cache that cannot be shown to be a problem is not dropped on
    suspicion" -- so an unreadable machine must answer False, not True."""
    from spacr.qt import memory_budget as mb

    monkeypatch.setattr(mb, "free_megabytes", lambda: None)
    assert mb.headroom_is_short() is False
    assert mb.headroom_is_short(floor_mb=999999) is False


def test_the_floor_comes_from_preferences_and_survives_their_absence(
        monkeypatch):
    """The floor is a preference, and an unreadable one is not a crisis."""
    from spacr.qt import memory_budget as mb

    monkeypatch.setattr(mb, "free_megabytes", lambda: 100.0)
    assert mb.headroom_is_short(floor_mb=200.0) is True
    assert mb.headroom_is_short(floor_mb=50.0) is False

    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.endswith("preferences") or name == "preferences":
            raise ImportError("preferences unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    # Unreadable floor: refuse to call it short rather than guess one.
    assert mb.headroom_is_short() is False


def test_free_megabytes_is_none_when_psutil_will_not_answer(monkeypatch):
    """Not every machine has psutil, and none of them should crash for it."""
    from spacr.qt import memory_budget as mb
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert mb.free_megabytes() is None
