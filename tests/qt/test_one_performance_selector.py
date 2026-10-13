"""One performance level, five choices, and no second control to disagree."""
from __future__ import annotations

import itertools

import pytest
from PySide6.QtWidgets import QComboBox

from spacr.qt import preferences as P


@pytest.fixture
def store(monkeypatch):
    """A preference store this test owns."""
    values = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return values.get(key, default)

        def setValue(self, key, value):
            values[key] = value

        def remove(self, key):
            values.pop(key, None)

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return values


def test_exactly_five_levels_in_resource_order():
    assert P.PERFORMANCE_LEVELS == (
        "laptop", "extra_performance", "performance", "balanced",
        "workstation")


def test_retention_is_monotonic_along_that_order():
    """The order is meant to BE a resource scale, not a list of words."""
    scales = [P.PERFORMANCE_RETENTION[l] for l in P.PERFORMANCE_LEVELS]
    assert scales == sorted(scales), scales
    assert scales[0] < scales[-1]
    assert P.PERFORMANCE_RETENTION["laptop"] == min(scales)
    assert P.PERFORMANCE_RETENTION["workstation"] == max(scales)


def test_every_level_keeps_at_least_one_live_figure(store):
    """A level that kept none would make right-click restyling useless."""
    for level in P.PERFORMANCE_LEVELS:
        assert P.live_figure_allowance(level) >= 1


def test_the_allowance_is_monotonic_too(store):
    counts = [P.live_figure_allowance(l) for l in P.PERFORMANCE_LEVELS]
    assert counts == sorted(counts), counts


def test_every_level_explains_its_hardware():
    """"Each level's tooltip states the intended hardware profile"."""
    for level in P.PERFORMANCE_LEVELS:
        note = P.PERFORMANCE_NOTES[level]
        assert len(note) > 80, f"{level} is not explained"
        assert P.PERFORMANCE_LABELS[level]


@pytest.mark.parametrize(
    "legacy_mode,legacy_laptop",
    list(itertools.product(
        ("extra_performance", "performance", "balanced", "", "nonsense"),
        ("on", "off", "automatic", ""))))
def test_migration_from_every_legacy_pair(store, legacy_mode, legacy_laptop):
    """"migration tests cover every legacy mode crossed with on/off/automatic"."""
    if legacy_mode:
        store[P._KEY_SPACR_MODE] = legacy_mode
    if legacy_laptop:
        store[P._KEY_LAPTOP_MODE] = legacy_laptop

    level = P.get_performance_level()
    assert level in P.PERFORMANCE_LEVELS

    if legacy_laptop == "on":
        assert level == "laptop", "an explicit Laptop user lost their choice"
    elif legacy_mode in P.SPACR_MODES:
        assert level == legacy_mode, "a saved mode was silently changed"
    else:
        assert level == P.DEFAULT_PERFORMANCE_LEVEL

    # Idempotent: a second read does not migrate again or drift.
    assert P.get_performance_level() == level
    assert P.get_performance_level() == level


def test_migration_is_durable(store):
    """"Perform the migration once" -- the level is written, not recomputed."""
    store[P._KEY_LAPTOP_MODE] = "on"
    assert P.get_performance_level() == "laptop"
    assert store.get(P._KEY_PERFORMANCE_LEVEL) == "laptop"

    # Even if the legacy values now say something else, the stored level wins.
    store[P._KEY_LAPTOP_MODE] = "off"
    store[P._KEY_SPACR_MODE] = "balanced"
    assert P.get_performance_level() == "laptop"


def test_the_old_settings_derive_rather_than_override(store):
    """No hidden Laptop override may change a choice made in the selector."""
    for level in P.PERFORMANCE_LEVELS:
        P.set_performance_level(level)
        assert P.get_performance_level() == level
        assert P.get_spacr_mode() == P.spacr_mode_for_level(level)
        assert P.get_laptop_mode() == ("on" if level == "laptop" else "off")


def test_an_unknown_level_is_refused(store):
    with pytest.raises(ValueError):
        P.set_performance_level("turbo")


def test_the_dialog_has_one_control_and_not_two(qtbot):
    """"no separate Laptop mode control"."""
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    names = {c.objectName() for c in dlg.findChildren(QComboBox)}
    assert "PerformanceLevel" in names
    assert "LaptopMode" not in names
    assert "SpacrMode" not in names

    combo = next(c for c in dlg.findChildren(QComboBox)
                 if c.objectName() == "PerformanceLevel")
    assert [combo.itemData(i) for i in range(combo.count())] == \
        list(P.PERFORMANCE_LEVELS)
