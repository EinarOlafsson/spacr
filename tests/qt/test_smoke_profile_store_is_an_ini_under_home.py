"""A smoke or isolated profile keeps its preferences in an INI file under HOME.

macOS, 2026-10-20: a "fresh profile" frozen smoke with HOME moved still
read and wrote the user's real com.spacr.qt (GUI scale, onboarding and
installation-review flags), because the native store goes through cfprefsd,
which ignores HOME; the Windows registry ignores it too. With
``SPACR_DISTRIBUTION_SMOKE=1`` or ``SPACR_ISOLATED_PROFILE=1`` every store
is an INI file under ``$HOME/.spacr/settings`` on every platform; an
ordinary run keeps the native store.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences, prefs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A fresh HOME, no portable mode, no smoke or isolation flag yet."""
    folder = tmp_path / "profile"
    folder.mkdir()
    monkeypatch.setenv("HOME", str(folder))
    monkeypatch.setenv("USERPROFILE", str(folder))
    monkeypatch.setenv("SPACR_PORTABLE", "0")
    for name in ("SPACR_HOME", "SPACR_DISTRIBUTION_SMOKE", "SPACR_ISOLATED_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(preferences, "_STORE", threading.local())
    return folder


@pytest.mark.parametrize("flag", ["SPACR_DISTRIBUTION_SMOKE", "SPACR_ISOLATED_PROFILE"])
def test_the_smoke_store_is_an_ini_under_the_given_home(monkeypatch, home, flag):
    monkeypatch.setenv(flag, "1")
    args = prefs._store_args("spacr", "qt")
    assert args[:2] == (QSettings.IniFormat, QSettings.UserScope)
    settings_folder = Path(home).resolve() / ".spacr" / "settings"
    store = preferences._real_store()
    path = Path(store.fileName()).resolve()
    assert path.suffix == ".ini" and path.is_relative_to(settings_folder)
    store.setValue("tests/isolated_profile", 1)
    store.sync()
    assert path.is_file()


def test_an_ordinary_run_keeps_the_native_store(home):
    assert prefs._store_args("spacr", "qt") == ("spacr", "qt")
