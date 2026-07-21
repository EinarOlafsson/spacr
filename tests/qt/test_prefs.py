"""QSettings-backed prefs helpers."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from spacr.qt import prefs


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path, monkeypatch):
    """Redirect prefs._s() to a per-test INI file so tests don't stomp
    each other or the developer's real settings store."""
    ini = tmp_path / "spacr_qt_prefs.ini"

    def _factory():
        return QSettings(str(ini), QSettings.IniFormat)

    monkeypatch.setattr(prefs, "_s", _factory)
    yield


def test_last_source_round_trip():
    assert prefs.get_last_source("annotate") == ""
    prefs.set_last_source("annotate", "/some/path")
    assert prefs.get_last_source("annotate") == "/some/path"


def test_push_recent_source_dedups_and_orders():
    prefs.push_recent_source("annotate", "/a")
    prefs.push_recent_source("annotate", "/b")
    prefs.push_recent_source("annotate", "/a")     # promote
    items = prefs.get_recent_sources("annotate")
    assert items[:2] == ["/a", "/b"]


def test_recent_capped_at_limit():
    for i in range(15):
        prefs.push_recent_source("annotate", f"/dir{i}")
    items = prefs.get_recent_sources("annotate", limit=8)
    assert len(items) == 8
    # Newest first
    assert items[0] == "/dir14"


def test_different_app_keys_isolated():
    prefs.push_recent_source("annotate", "/annotate/x")
    prefs.push_recent_source("make_masks", "/masks/y")
    assert prefs.get_last_source("annotate") == "/annotate/x"
    assert prefs.get_last_source("make_masks") == "/masks/y"
