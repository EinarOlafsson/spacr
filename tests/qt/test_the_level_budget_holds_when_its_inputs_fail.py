"""286's level-derived budget, where its inputs fail or have nothing to say.

``test_the_level_derives_every_budget.py`` and
``test_the_level_migration_removes_what_it_replaced.py`` hold what the level
decides when everything works. This file holds the promises the same helpers
make for when something does not:

* ``_level_budget`` never raises -- a background budget sweep with no user in
  front of it reads it, so an unreadable store costs a default;
* ``_budget_follows_level`` moves nothing when the level did not change or
  either side has no recommendation to compare with;
* ``_level_is_durable`` answers False on any doubt, so the migration keeps
  the obsolete answers that are still the only record of the user's choice;
* ``_backdrop_follows_the_level`` failing does not stop the level being saved.
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
    """Setting the level applies laptop constraints to this PROCESS."""
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


# ---------------------------------------------------------------------------
# _level_budget
# ---------------------------------------------------------------------------

class _Unreadable:
    """A preference file on a share that went away."""

    def value(self, *_args, **_kwargs):
        raise OSError("the preference file cannot be read")


def test_the_budget_is_the_levels_own_recommendation(store):
    P.set_performance_level("laptop")
    assert P._level_budget() == mb.recommended_for("laptop")
    assert P._level_budget() != (mb.DEFAULT_IDLE_MINUTES,
                                 mb.DEFAULT_CACHE_CEILING_MB,
                                 mb.DEFAULT_HEADROOM_MB)


def test_an_unreadable_store_costs_the_sweep_a_default_not_an_exception(
        monkeypatch):
    monkeypatch.setattr(P, "_settings", lambda: _Unreadable())
    monkeypatch.setattr(P, "_SAFE_MODE", False)

    assert P._level_budget() == (mb.DEFAULT_IDLE_MINUTES,
                                 mb.DEFAULT_CACHE_CEILING_MB,
                                 mb.DEFAULT_HEADROOM_MB)


# ---------------------------------------------------------------------------
# _budget_follows_level
# ---------------------------------------------------------------------------

class _Combo:
    def __init__(self, level):
        self.level = level

    def currentData(self):
        return self.level


class _Spin:
    def __init__(self, value):
        self._value = value
        self.moved_to = []

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value
        self.moved_to.append(value)


def _spins_at(level):
    return tuple(_Spin(value) for value in mb.RECOMMENDED[level])


def test_a_new_level_moves_the_untouched_numbers_and_keeps_a_typed_one():
    idle, cache, headroom = _spins_at("balanced")
    headroom.setValue(3072)
    headroom.moved_to.clear()
    last = ["balanced"]

    P._budget_follows_level(_Combo("laptop"), last, (idle, cache, headroom))

    assert (idle.value(), cache.value()) == mb.RECOMMENDED["laptop"][:2]
    assert headroom.moved_to == [], "a number the user typed was moved"
    assert last == ["laptop"]


def test_choosing_the_same_level_again_moves_nothing():
    spins = _spins_at("balanced")
    last = ["balanced"]

    P._budget_follows_level(_Combo("balanced"), last, spins)

    assert [spin.moved_to for spin in spins] == [[], [], []]
    assert last == ["balanced"]


def test_a_level_with_no_recommendation_moves_nothing_either_way():
    """There is nothing to compare an unknown level's numbers with, so
    nothing can be said to be untouched."""
    spins = _spins_at("balanced")
    last = ["balanced"]

    P._budget_follows_level(_Combo("not-a-level"), last, spins)
    assert [spin.moved_to for spin in spins] == [[], [], []]
    assert last == ["not-a-level"], "the next change compared with a stale level"

    P._budget_follows_level(_Combo("laptop"), last, spins)
    assert [spin.moved_to for spin in spins] == [[], [], []]
    assert last == ["laptop"]


# ---------------------------------------------------------------------------
# _level_is_durable
# ---------------------------------------------------------------------------

class _Store:
    """A dict-backed store whose status or read-back can fail."""

    def __init__(self, values=(), *, status_raises=False,
                 read_back_raises=False):
        self.values = dict(values)
        self.status_raises = status_raises
        self.read_back_raises = read_back_raises
        self.level_reads = 0
        self.removed = []

    def value(self, key, default=None, type=None):
        if key == P._KEY_PERFORMANCE_LEVEL:
            self.level_reads += 1
            # The first read is the migration's question; any later one is
            # the durability check reading the written level back.
            if self.read_back_raises and self.level_reads > 1:
                raise RuntimeError("the store went away after the write")
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.removed.append(key)
        self.values.pop(key, None)

    def sync(self):
        pass

    def status(self):
        if self.status_raises:
            raise RuntimeError("the store cannot report its status")
        return QSettings.Status.NoError


def test_a_level_that_reads_back_with_no_error_is_durable():
    store = _Store({P._KEY_PERFORMANCE_LEVEL: "laptop"})
    assert P._level_is_durable(store, "laptop") is True


@pytest.mark.parametrize("failure", [
    {"status_raises": True},
    {"read_back_raises": True},
])
def test_a_store_that_cannot_answer_is_not_durable(failure):
    store = _Store({P._KEY_PERFORMANCE_LEVEL: "laptop"}, **failure)
    store.level_reads = 1          # past the migration's own read
    assert P._level_is_durable(store, "laptop") is False


@pytest.mark.parametrize("failure", [
    {"status_raises": True},
    {"read_back_raises": True},
])
def test_the_migration_keeps_the_old_answers_when_durability_is_in_doubt(
        monkeypatch, failure):
    legacy = {P._KEY_LAPTOP_MODE: "on", P._KEY_SPACR_MODE: "performance"}
    store = _Store(legacy, **failure)
    monkeypatch.setattr(P, "_settings", lambda: store)
    monkeypatch.setattr(P, "_SAFE_MODE", False)

    assert P.get_performance_level() == "laptop"
    assert store.removed == []
    assert store.values[P._KEY_LAPTOP_MODE] == "on"


# ---------------------------------------------------------------------------
# _backdrop_follows_the_level
# ---------------------------------------------------------------------------

def test_this_runs_backdrop_follows_each_level_that_is_set(store, monkeypatch):
    asked = []
    monkeypatch.setattr(LM, "apply", lambda on=None: asked.append(on) or {})

    P.set_performance_level("laptop")
    P.set_performance_level("workstation")

    assert asked == [True, False]


def test_a_backdrop_that_cannot_follow_does_not_stop_the_level_being_saved(
        store, monkeypatch):
    def broken(on=None):
        raise RuntimeError("the backdrop could not be reached")

    monkeypatch.setattr(LM, "apply", broken)

    P.set_performance_level("laptop")

    assert P.get_performance_level() == "laptop"
