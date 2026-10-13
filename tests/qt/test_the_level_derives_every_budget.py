"""286: Laptop keeps the least, Workstation the most, monotonic between.

TESTED ON DERIVED POLICY, NOT ON A MEASURED WORKLOAD. What is asserted is
the numbers each level hands the code that decides what to keep -- live
figures, the idle timeout and the cache ceiling the budget sweep enforces,
whether an animated backdrop is drawn -- not resident memory of a running
application, which depends on what the user opened and is not a property of
the setting.

The budget sweep used to enforce the same idle timeout and ceiling at every
level (15 min, 2048 MB) unless somebody typed a number: the per-level values
existed only as suggestions in a tooltip, so "all cache decisions derive
from the level" was true of the tooltip and not of the sweep. An untouched
budget now follows the level; a number the user sets is kept at every level.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import laptop_mode as LM
from spacr.qt import memory_budget as mb
from spacr.qt import preferences as P

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _process_backdrop_state_is_reset():
    """Moving the level applies laptop constraints to this PROCESS; clear
    that state before and after so nothing leaks into the next test."""
    saved = os.environ.get("SPACR_NO_BACKDROP")

    def _clear():
        LM._suppressed_here = False
        os.environ.pop("SPACR_NO_BACKDROP", None)

    _clear()
    try:
        yield
    finally:
        _clear()
        if saved is not None:
            os.environ["SPACR_NO_BACKDROP"] = saved


@pytest.fixture
def store(monkeypatch, tmp_path):
    real = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(P, "_settings", lambda: real)
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return real


def test_an_untouched_budget_follows_the_level(store):
    for level in P.PERFORMANCE_LEVELS:
        P.set_performance_level(level)
        idle, cache, headroom = mb.recommended_for(level)
        assert P.get_idle_minutes() == idle, level
        assert P.get_cache_ceiling_mb() == cache, level
        assert P.get_headroom_mb() == headroom, level


def test_a_number_the_user_set_is_kept_at_every_level(store):
    P.set_cache_ceiling_mb(3000)
    P.set_idle_minutes(7.5)
    for level in P.PERFORMANCE_LEVELS:
        P.set_performance_level(level)
        assert P.get_cache_ceiling_mb() == 3000, level
        assert P.get_idle_minutes() == 7.5, level


def _derived_policy(level):
    """What the level hands the code that decides what to keep."""
    P.set_performance_level(level)
    return {
        "live figures kept": P.live_figure_allowance(),
        "minutes an unused cache entry is kept": P.get_idle_minutes(),
        "cache ceiling (MB)": P.get_cache_ceiling_mb(),
        "retention scale": P.retention_scale(),
    }


def test_retained_state_is_monotonic_from_laptop_to_workstation(store):
    rows = [_derived_policy(level) for level in P.PERFORMANCE_LEVELS]
    for name in rows[0]:
        values = [row[name] for row in rows]
        assert values == sorted(values), (
            f"{name} is not monotonic along {P.PERFORMANCE_LEVELS}: {values}")
        assert values[0] == min(values) and values[-1] == max(values)
        assert values[0] < values[-1], (
            f"{name}: Laptop does not keep less than Workstation ({values})")


def test_laptop_draws_no_backdrop_and_workstation_keeps_the_users(store):
    """The one continuous background cost a level controls."""
    P.set_performance_level("balanced")
    P.set_ambient_animation("blobs")
    drawn = {}
    for level in P.PERFORMANCE_LEVELS:
        P.set_performance_level(level)
        drawn[level] = P.get_ambient_enabled()
    assert drawn["laptop"] is False
    assert drawn["workstation"] is True
    ordered = [int(drawn[level]) for level in P.PERFORMANCE_LEVELS]
    assert ordered == sorted(ordered), drawn


def test_the_dialog_moves_an_untouched_budget_with_the_level(
        store, qtbot, qt_theme_applied):
    """Save used to write all three numbers whatever they were, which froze
    the budget at the first Save. A number still at the old level's value
    moves with the level; a number the user typed stays."""
    from PySide6.QtWidgets import (QComboBox, QDialogButtonBox,
                                   QDoubleSpinBox, QSpinBox)

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    combo = dlg.findChild(QComboBox, "PerformanceLevel")
    cache = dlg.findChild(QSpinBox, "CacheCeilingMb")
    idle = dlg.findChild(QDoubleSpinBox, "CacheIdleMinutes")
    headroom = dlg.findChild(QSpinBox, "HeadroomMb")
    assert cache.value() == mb.recommended_for("balanced")[1]

    headroom.setValue(3072)
    combo.setCurrentIndex(combo.findData("laptop"))
    assert idle.value() == mb.recommended_for("laptop")[0]
    assert cache.value() == mb.recommended_for("laptop")[1]
    assert headroom.value() == 3072, "a number the user typed was moved"

    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert P.get_performance_level() == "laptop"
    assert P.get_cache_ceiling_mb() == mb.recommended_for("laptop")[1]
    assert P.get_headroom_mb() == 3072

    P.set_performance_level("workstation")
    assert P.get_cache_ceiling_mb() == mb.recommended_for("workstation")[1], (
        "Save froze the cache ceiling at the level it was saved under")
    assert P.get_headroom_mb() == 3072
