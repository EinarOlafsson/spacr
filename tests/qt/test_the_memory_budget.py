"""The user sets the memory budget, and nothing claims to unload a library."""
from __future__ import annotations

import sys
import time

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

from spacr.qt import memory_budget as mb
from spacr.qt import preferences as P


def test_a_session_that_opens_no_deep_learning_module_never_imports_torch(
        qapp):
    """"asserted by sys.modules and not by a timing"."""
    import spacr.qt.app as app_module

    win = app_module.MainWindow()
    win.show()
    qapp.processEvents()
    try:
        assert "torch" not in sys.modules
        for key in ("mask", "measure", "regression"):
            win._on_nav_selected(key)
            qapp.processEvents()
        assert "torch" not in sys.modules, (
            "opening ordinary modules pulled in 477 MB of torch")
    finally:
        win.close()


def test_every_level_has_a_recommendation_and_the_hardware_it_suits():
    """"system configuration recomendations for each level"."""
    for level in P.PERFORMANCE_LEVELS:
        idle, cache, headroom = mb.recommended_for(level)
        assert idle >= 0 and cache > 0 and headroom > 0
        assert level in mb.HARDWARE_NOTES
        assert len(mb.HARDWARE_NOTES[level]) > 10


def test_the_recommendations_are_monotonic_with_the_level():
    """Laptop keeps least, Workstation most, and the order is the scale."""
    caches = [mb.RECOMMENDED[l][1] for l in P.PERFORMANCE_LEVELS]
    idles = [mb.RECOMMENDED[l][0] for l in P.PERFORMANCE_LEVELS]
    assert caches == sorted(caches), caches
    assert idles == sorted(idles), idles


def test_idle_entries_are_dropped():
    now = time.time()
    entries = [
        ("fresh", 10.0, now - 60),          # 1 minute idle
        ("stale", 10.0, now - 3600),        # an hour idle
    ]
    doomed = mb.what_to_drop(entries, now, idle_minutes=15.0,
                             ceiling_mb=10_000)
    assert doomed == ["stale"]


def test_over_the_ceiling_the_least_recently_used_goes_first():
    """A cache under pressure gives up what it is least likely to want."""
    now = time.time()
    entries = [
        ("newest", 100.0, now - 10),
        ("middle", 100.0, now - 20),
        ("oldest", 100.0, now - 30),
    ]
    doomed = mb.what_to_drop(entries, now, idle_minutes=600.0,
                             ceiling_mb=150)
    assert doomed[0] == "oldest"
    assert "newest" not in doomed


def test_size_alone_never_drops_an_entry():
    """Idleness decides WHETHER a trim happens; size decides the order."""
    now = time.time()
    entries = [("huge", 5000.0, now)]
    assert mb.what_to_drop(entries, now, idle_minutes=600.0,
                           ceiling_mb=10_000) == []


def test_headroom_that_cannot_be_measured_drops_nothing():
    """A cache that cannot be shown to be a problem is not dropped on
    suspicion."""
    real = mb.free_megabytes
    mb.free_megabytes = lambda: None
    try:
        assert mb.headroom_is_short(1_000_000) is False
    finally:
        mb.free_megabytes = real


def test_the_three_settings_round_trip(monkeypatch):
    store = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return store.get(key, default)

        def setValue(self, key, value):
            store[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)

    P.set_headroom_mb(4096)
    P.set_idle_minutes(7.5)
    P.set_cache_ceiling_mb(8192)
    assert P.get_headroom_mb() == 4096
    assert P.get_idle_minutes() == 7.5
    assert P.get_cache_ceiling_mb() == 8192

    # Nonsense in the store reads as the default rather than raising.
    store[P._KEY_HEADROOM] = "not a number"
    assert P.get_headroom_mb() == mb.DEFAULT_HEADROOM_MB


def test_the_controls_are_on_the_performance_tab(qtbot):
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    assert dlg.findChild(QSpinBox, "HeadroomMb") is not None
    assert dlg.findChild(QDoubleSpinBox, "CacheIdleMinutes") is not None
    assert dlg.findChild(QSpinBox, "CacheCeilingMb") is not None


def test_no_setting_claims_to_unload_a_library(qtbot):
    """"If a setting is named for that, it is lying."

    The help is read out of the hint bar's register rather than off the
    control: Preferences moves every tooltip there so a control answers in
    the strip instead of in a window over it.
    """
    from spacr.qt.widgets.hint_bar import HintBar

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    bar = dlg.findChildren(HintBar)[0]
    # Keyed by the row's LABEL, which is the hover target -- the help was
    # moved there deliberately, so the register is searched by content.
    said = list(getattr(bar, "_hints", {}).values())
    assert said, "the strip registered nothing at all"

    # It has to say so outright, because the request asked for exactly the
    # thing that cannot be done.
    unloadable = [text for text in said if "cannot be unloaded" in str(text)]
    assert unloadable, "no setting says a library cannot be unloaded"

    # And every level's recommendation is quoted somewhere, which is what
    # makes a number a decision.
    floor = [text for text in said if "must stay free" in str(text)]
    assert floor, "the headroom floor is not explained"
    for level in P.PERFORMANCE_LEVELS:
        assert P.PERFORMANCE_LABELS[level] in floor[0], (
            f"{level} has no recommendation in the headroom help")
