"""286: the performance level is the only thing that decides laptop constraints.

"No hidden Laptop override may continue changing a choice made in the
five-level selector." Three paths used to be exactly that:

* `launch` called ``laptop_mode.apply()`` with no argument, which MEASURES
  the machine, so a two-core box chosen as Workstation still lost its
  backdrop at every start;
* ``set_laptop_mode`` wrote the obsolete key and applied the measurement
  too, while ``get_laptop_mode`` answered from the level -- a setter and a
  getter that disagreed;
* leaving Extra Performance (which Laptop maps onto) put the stashed
  animation back through ``set_ambient_animation``, which turns the backdrop
  ON -- so a user who had switched it off got it back by moving the level
  down to Laptop and up again.

`laptop_mode._suppressed_here` is PROCESS state, not a preference: it is
what lets ``apply(False)`` lift only a suppression THIS process made, never
crash recovery's, and never the user's stored answer. Every test here
resets it and ``SPACR_NO_BACKDROP`` before and after, so no test can leak a
suppression into the next one.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import laptop_mode as LM
from spacr.qt import preferences as P

pytestmark = pytest.mark.qt

_NO_BACKDROP = "SPACR_NO_BACKDROP"


@pytest.fixture(autouse=True)
def _process_backdrop_state_is_reset():
    """Clear the process-local suppression before AND after each test."""
    saved = os.environ.get(_NO_BACKDROP)

    def _clear():
        LM._suppressed_here = False
        os.environ.pop(_NO_BACKDROP, None)

    _clear()
    try:
        yield
    finally:
        _clear()
        if saved is not None:
            os.environ[_NO_BACKDROP] = saved


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A real INI preference store this test owns."""
    real = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(P, "_settings", lambda: real)
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return real


def _stored_ambient_enabled(store) -> bool:
    """The USER'S answer, read past every process-local suppression."""
    return P._as_bool(store.value(P._KEY_AMBIENT_ENABLED, True), True)


@pytest.mark.parametrize("down,up", [
    ("laptop", "balanced"),
    ("laptop", "performance"),
    ("extra_performance", "workstation"),
])
def test_a_backdrop_the_user_switched_off_stays_off_through_the_levels(
        store, down, up):
    """THE DEFECT. Down to a minimising level and back up restored the
    animation through a setter that also switches the backdrop on."""
    P.set_performance_level("balanced")
    P.set_ambient_animation("blobs")
    P.set_ambient_enabled(False)

    P.set_performance_level(down)
    P.set_performance_level(up)

    assert _stored_ambient_enabled(store) is False, (
        f"moving the level to {down} and back to {up} switched on a backdrop "
        "the user had switched off")
    assert P.get_ambient_enabled() is False
    assert P.get_ambient_animation() == "blobs", "the animation choice was lost"


def test_a_suppression_this_process_made_is_not_stashed_as_the_users_choice(
        store):
    """A launch at Laptop suppresses the backdrop for this run only. The
    stash must record the stored answer (on), not what the suppression makes
    `get_ambient_enabled` say (off) -- and leaving Laptop lifts the
    suppression this process made."""
    P.set_performance_level("balanced")
    P.set_ambient_animation("blobs")
    LM.apply(True)
    assert LM._suppressed_here and P.get_ambient_enabled() is False

    P.set_performance_level("laptop")
    P.set_performance_level("balanced")

    assert _stored_ambient_enabled(store) is True, (
        "a process-local suppression was saved as the user's preference")
    assert os.environ.get(_NO_BACKDROP) is None, (
        "the level left Laptop but this run is still suppressed")
    assert P.get_ambient_enabled() is True


def test_a_suppression_this_process_did_not_make_is_never_lifted(store):
    """Crash recovery sets the same variable after two failed starts. No
    level change may hand the backdrop back to a driver that crashed on it."""
    P.set_ambient_animation("blobs")
    os.environ[_NO_BACKDROP] = "1"

    P.set_performance_level("laptop")
    P.set_performance_level("workstation")

    assert os.environ.get(_NO_BACKDROP) == "1"
    assert P.get_ambient_enabled() is False


def test_the_legacy_setter_writes_the_level_and_measures_nothing(
        store, monkeypatch):
    """`set_laptop_mode` is the old API. It must say the same thing as the
    getter, never write the key the migration removed, and never let a
    hardware measurement decide."""
    monkeypatch.setattr(LM, "wanted",
                        lambda reading=None: (True, "a two-core machine"))
    P.set_ambient_animation("blobs")
    P.set_performance_level("workstation")

    P.set_laptop_mode("automatic")
    assert P.get_performance_level() == "workstation"
    assert os.environ.get(_NO_BACKDROP) is None, (
        "'automatic' let the machine measurement override Workstation")

    P.set_laptop_mode("on")
    assert P.get_performance_level() == "laptop"
    assert P.get_laptop_mode() == "on"

    P.set_laptop_mode("off")
    assert P.get_laptop_mode() == "off"
    assert P.get_performance_level() == P.DEFAULT_PERFORMANCE_LEVEL
    assert not store.contains(P._KEY_LAPTOP_MODE), (
        "the obsolete laptop-mode key was written again")


@pytest.mark.parametrize("level,expected", [
    ("workstation", False),
    ("balanced", False),
    ("laptop", True),
])
def test_launch_follows_the_level_and_not_the_machine(
        launched, monkeypatch, level, expected):
    """The launch path. A reading that says "small machine" must not turn
    laptop constraints on for a level that is not Laptop."""
    from spacr.qt import app as app_mod

    seen = []
    monkeypatch.setattr(LM, "wanted",
                        lambda reading=None: (True, "a two-core machine"))
    monkeypatch.setattr(
        LM, "apply",
        lambda on=None: seen.append(on) or {"on": bool(on), "why": "",
                                            "changed": []})
    monkeypatch.setattr(P, "get_performance_level", lambda: level)

    assert app_mod.launch([]) == 0
    assert seen == [expected], (
        f"launch at {level} asked laptop_mode for {seen}, not the level's "
        f"answer {expected}")


# The one stand-in QApplication that survives `launch`; borrowed, as
# test_cov_r8_launch_laptop_mode.py does.
from tests.qt.test_cov_qt_app import launched  # noqa: E402,F401
