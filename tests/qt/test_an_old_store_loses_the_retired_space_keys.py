"""An older preference store stops carrying the retired Space theme's keys.

Instruction 364 retired `prefs/space_variant` and `prefs/space_seed` on
2026-09-09 (61c896555): their accessors went, because the theme they served
cannot be selected. The two key names stayed in `spacr.qt.preferences` with a
note saying they were kept "so a stored value can still be recognised and
cleared" -- and nothing cleared them. Measured on 2026-09-19: no line in the
package read or removed either key, so every store written before the
retirement went on carrying two values that look live and are not.

Each test writes the store an older spaCR left behind and then reads it the
way a launch does. The last one goes through `apply_preferences_to_app`, the
call `spacr.qt.app.launch` makes at startup, because that is where a user's
old store is first opened.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences as P  # noqa: E402

pytestmark = pytest.mark.qt

RETIRED = ("prefs/space_variant", "prefs/space_seed")
OLD_VALUES = {"prefs/space_variant": "nebula", "prefs/space_seed": 1234}


@pytest.fixture(autouse=True)
def _a_fresh_process(monkeypatch):
    """Forget which stores this process has already cleared.

    The clearing runs once per store file per process. Each test here has
    its own sandboxed store, but starting from an empty record keeps one
    test's store from standing in for another's.
    """
    monkeypatch.setattr(P, "_SPACE_KEYS_CLEARED", set(), raising=False)
    monkeypatch.setattr(P, "_SAFE_MODE", False)


def _write_old_store(values):
    """Leave ``values`` in the store exactly as an older spaCR would have."""
    store = QSettings("spacr", "qt")
    for key, value in values.items():
        store.setValue(key, value)
    store.sync()
    return store.fileName()


def _stored():
    """A fresh read of the store, as the next process would see it."""
    store = QSettings("spacr", "qt")
    store.sync()
    return {key: store.value(key) for key in store.allKeys()}


def test_the_key_names_are_the_ones_the_module_retired():
    """The test names what the module names, so neither drifts alone."""
    assert (P._KEY_SPACE_VARIANT, P._KEY_SPACE_SEED) == RETIRED


@pytest.mark.parametrize("key", RETIRED)
def test_an_old_store_holding_the_key_loses_it_on_the_first_theme_read(key):
    _write_old_store({key: OLD_VALUES[key], "prefs/theme": "light"})
    assert key in _stored(), "the old store was not written"

    assert P.get_theme() == "light"

    after = _stored()
    assert key not in after, f"{key} is still in the store: {after}"
    assert after.get("prefs/theme") == "light", (
        "the clearing took a live preference with it")


def test_a_store_that_still_names_the_space_theme_falls_back_and_is_cleared():
    """The oldest store of all: Space chosen, a sky picked, a seed fixed.

    `get_theme` already mapped the unknown `space` to the default. The two
    Space values go as well, and the theme value itself stays, since
    rewriting a user's stored choice is not this change's business.
    """
    _write_old_store({"prefs/theme": "space", **OLD_VALUES,
                      "prefs/cell_variant": "microtubules"})

    assert P.get_theme() == P.DEFAULT_THEME

    after = _stored()
    assert not set(RETIRED) & set(after), after
    assert after.get("prefs/theme") == "space"
    assert after.get("prefs/cell_variant") == "microtubules", (
        "cell_variant is a LIVE key of the Cell theme and must survive")


def test_a_clean_store_is_not_rewritten():
    """A store with nothing to remove is only read, never written."""
    path = _write_old_store({"prefs/theme": "dark"})
    before = open(path, "rb").read()
    mtime = os.stat(path).st_mtime_ns

    assert P.get_theme() == "dark"

    assert open(path, "rb").read() == before
    assert os.stat(path).st_mtime_ns == mtime


def test_safe_mode_leaves_the_store_alone(monkeypatch):
    """Safe mode reads nothing it was given, so it removes nothing either."""
    _write_old_store(OLD_VALUES)
    monkeypatch.setattr(P, "_SAFE_MODE", True)

    P.get_theme()

    assert set(RETIRED) <= set(_stored())


def test_the_launch_path_clears_an_old_store(qapp):
    """Where a user meets it: the preferences a launch applies.

    `spacr.qt.app.launch` calls `apply_preferences_to_app`, which resolves
    the theme before anything else paints.
    """
    _write_old_store({**OLD_VALUES, "prefs/theme": "dark"})

    P.apply_preferences_to_app(qapp)

    assert not set(RETIRED) & set(_stored())
