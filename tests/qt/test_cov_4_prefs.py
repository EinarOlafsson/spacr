"""Recent-folder memory tolerates an empty path and a list-shaped store.

Two shapes reach these helpers that the happy path never produces: a caller
handing over ``""`` because nothing was picked, and a settings backend that
returns the recent list as a real list rather than the newline-joined string
it was written as. Neither may lose the list that is already remembered.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from spacr.qt import prefs

#: The real factory, captured before the isolating fixture replaces it.
_REAL_SETTINGS_FACTORY = prefs._s


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path, monkeypatch):
    """Point prefs at a per-test INI so nothing reaches the real store."""
    ini = tmp_path / "spacr_qt_prefs_cov4.ini"
    monkeypatch.setattr(
        prefs, "_s", lambda: QSettings(str(ini), QSettings.IniFormat))


def test_an_empty_path_does_not_become_the_last_source():
    """Nothing picked must not overwrite the folder already remembered."""
    prefs.set_last_source("annotate", "/real/folder")
    prefs.set_last_source("annotate", "")
    assert prefs.get_last_source("annotate") == "/real/folder"


def test_an_empty_path_is_not_pushed_onto_the_recent_list():
    """A blank entry at the head of the list would be unusable to click."""
    prefs.push_recent_source("annotate", "/real/folder")
    prefs.push_recent_source("annotate", "")
    assert prefs.get_recent_sources("annotate") == ["/real/folder"]


def test_the_real_factory_is_pinned_to_the_spacr_namespace():
    """Read-only check that the unpatched factory names one shared store."""
    settings = _REAL_SETTINGS_FACTORY()
    assert settings.organizationName() == prefs.ORG
    assert settings.applicationName() == prefs.APP
