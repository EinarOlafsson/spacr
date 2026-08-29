"""The recent-path helper and application preferences share one Qt store."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

from spacr.qt import preferences, prefs


def _ini_factory(root: Path):
    """Map an organization/application pair to an isolated real INI file."""
    def make(organization: str, application: str) -> QSettings:
        parent = root / organization
        parent.mkdir(parents=True, exist_ok=True)
        return QSettings(
            str(parent / f"{application}.ini"), QSettings.IniFormat)

    return make


def test_recent_paths_and_preferences_use_the_same_store(tmp_path, monkeypatch):
    make = _ini_factory(tmp_path)
    monkeypatch.setattr(prefs, "QSettings", make)
    monkeypatch.setattr(preferences, "QSettings", make)
    monkeypatch.setattr(prefs, "_MIGRATED_FILES", set())

    prefs.set_last_source("annotate", "/data/annotate")
    preferences.set_theme("dark")

    recent_store = prefs._s()
    preference_store = preferences._settings()
    assert recent_store.fileName() == preference_store.fileName()
    assert recent_store.value("recent/annotate/last") == "/data/annotate"
    assert preference_store.value("prefs/theme") == "dark"


def test_both_legacy_spellings_migrate_without_being_changed(
        tmp_path, monkeypatch):
    make = _ini_factory(tmp_path)
    monkeypatch.setattr(prefs, "QSettings", make)
    monkeypatch.setattr(prefs, "_MIGRATED_FILES", set())

    legacy = make("Olafsson Lab", "spaCR")
    case_drift = make("Olafsson Lab", "SpaCR")
    current = make("spacr", "qt")
    legacy.setValue("recent/annotate/last", "/legacy/annotate")
    legacy.setValue("recent/existing/last", "/legacy/must-not-win")
    case_drift.setValue("recent/mask/last", "/legacy/mask")
    current.setValue("recent/existing/last", "/canonical/value")
    for store in (legacy, case_drift, current):
        store.sync()

    legacy_before = {key: legacy.value(key) for key in legacy.allKeys()}
    case_before = {key: case_drift.value(key) for key in case_drift.allKeys()}

    assert prefs.get_last_source("annotate") == "/legacy/annotate"
    assert prefs.get_last_source("mask") == "/legacy/mask"
    assert prefs.get_last_source("existing") == "/canonical/value"
    assert {key: legacy.value(key) for key in legacy.allKeys()} == legacy_before
    assert {
        key: case_drift.value(key) for key in case_drift.allKeys()
    } == case_before
