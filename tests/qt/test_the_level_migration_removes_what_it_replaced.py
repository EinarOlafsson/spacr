"""286: migrate once, and remove the obsolete settings only when the level is
durable.

"Perform the migration once and remove the obsolete preference only after
the new value is durable." The migration wrote the level but never removed
``prefs/laptop_mode`` or ``prefs/spacr_mode``, and ``set_spacr_mode`` kept
writing the second one -- two stored answers to the one question the
selector asks. And in safe mode, where every read returns a default but
writes reach the real store, the first read "migrated" those defaults and
replaced the user's real level with Balanced.

Every legacy spaCR mode is crossed with Laptop ``on``, ``off``,
``automatic`` and unset, against a real INI file, and "repeated startup" is
two fresh interpreter processes rather than two calls in this one.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as P

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]

LEGACY_MODES = ("extra_performance", "performance", "balanced", "", "nonsense")
LEGACY_LAPTOP = ("on", "off", "automatic", "")
PAIRS = list(itertools.product(LEGACY_MODES, LEGACY_LAPTOP))


def _expected(mode: str, laptop: str) -> str:
    """The contract's mapping, written out rather than imported."""
    if laptop == "on":
        return "laptop"
    if mode in ("extra_performance", "performance", "balanced"):
        return mode
    return P.DEFAULT_PERFORMANCE_LEVEL


def _write_legacy(path: Path, mode: str, laptop: str) -> None:
    """A preference file as a pre-286 build left it."""
    old = QSettings(str(path), QSettings.IniFormat)
    if mode:
        old.setValue(P._KEY_SPACR_MODE, mode)
    if laptop:
        old.setValue(P._KEY_LAPTOP_MODE, laptop)
    old.sync()
    del old


def _reopen(path: Path) -> QSettings:
    return QSettings(str(path), QSettings.IniFormat)


@pytest.fixture
def ini(tmp_path, monkeypatch):
    """A fresh QSettings object per read, on one file -- as the app does."""
    path = tmp_path / "prefs.ini"
    monkeypatch.setattr(P, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return path


@pytest.mark.parametrize("mode,laptop", PAIRS)
def test_every_legacy_pair_migrates_and_the_obsolete_keys_go(ini, mode,
                                                             laptop):
    _write_legacy(ini, mode, laptop)

    assert P.get_performance_level() == _expected(mode, laptop)

    on_disk = _reopen(ini)
    assert on_disk.value(P._KEY_PERFORMANCE_LEVEL) == _expected(mode, laptop)
    assert not on_disk.contains(P._KEY_LAPTOP_MODE), (
        "the obsolete laptop-mode key survived a durable migration")
    assert not on_disk.contains(P._KEY_SPACR_MODE), (
        "the obsolete spaCR-mode key survived a durable migration")

    # Idempotent in-process.
    assert P.get_performance_level() == _expected(mode, laptop)


_STARTUP = r"""
import json, sys
from PySide6.QtCore import QSettings
from spacr.qt import preferences as P
levels = {}
for path in sys.argv[1:]:
    P._settings = (lambda p=path: QSettings(p, QSettings.IniFormat))
    levels[path] = P.get_performance_level()
print(json.dumps({"module": P.__file__, "levels": levels}))
"""


def _fresh_process_startup(paths):
    """One launch of spaCR's preference layer in a NEW interpreter."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    done = subprocess.run(
        [sys.executable, "-c", _STARTUP, *map(str, paths)],
        capture_output=True, text=True, env=env, timeout=120, check=True)
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert Path(report["module"]).resolve().is_relative_to(ROOT), (
        f"the child imported {report['module']}, not this tree")
    return report["levels"]


def test_repeated_startup_in_fresh_processes_migrates_once(tmp_path):
    """Two launches, each a new process, over every legacy pair.

    Between them an older spaCR writes its keys back -- a downgrade and an
    upgrade on one machine. The second launch must keep the level the first
    one migrated to: the migration happened once and the stored level wins.
    """
    paths = {}
    for index, (mode, laptop) in enumerate(PAIRS):
        path = tmp_path / f"pair{index}.ini"
        _write_legacy(path, mode, laptop)
        paths[str(path)] = _expected(mode, laptop)

    first = _fresh_process_startup(paths)
    assert first == paths
    for path in paths:
        on_disk = _reopen(Path(path))
        assert not on_disk.contains(P._KEY_LAPTOP_MODE), path
        assert not on_disk.contains(P._KEY_SPACR_MODE), path

    for path in paths:
        _write_legacy(Path(path), "balanced", "on")

    second = _fresh_process_startup(paths)
    assert second == paths, "a second startup migrated again"


class _Store:
    """A dict-backed store whose durability can be made to fail."""

    def __init__(self, values, *, sync_raises=False, status=None,
                 drops_writes=False):
        self.values = dict(values)
        self.sync_raises = sync_raises
        self._status = status
        self.drops_writes = drops_writes
        self.removed = []

    def value(self, key, default=None, type=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        if not self.drops_writes:
            self.values[key] = value

    def remove(self, key):
        self.removed.append(key)
        self.values.pop(key, None)

    def sync(self):
        if self.sync_raises:
            raise OSError("disk full")

    def status(self):
        return (QSettings.Status.NoError if self._status is None
                else self._status)


@pytest.mark.parametrize("failure", [
    {"sync_raises": True},
    {"status": QSettings.Status.AccessError},
    {"drops_writes": True},
])
def test_the_obsolete_keys_stay_until_the_level_is_durable(monkeypatch,
                                                           failure):
    """A level that did not reach the disk must not cost the only record of
    what the user chose."""
    legacy = {P._KEY_LAPTOP_MODE: "on", P._KEY_SPACR_MODE: "performance"}
    store = _Store(legacy, **failure)
    monkeypatch.setattr(P, "_settings", lambda: store)
    monkeypatch.setattr(P, "_SAFE_MODE", False)

    assert P.get_performance_level() == "laptop"
    assert store.removed == []
    assert store.values[P._KEY_LAPTOP_MODE] == "on"
    assert store.values[P._KEY_SPACR_MODE] == "performance"


def test_safe_mode_never_writes_a_migrated_default_over_the_real_level(
        tmp_path, monkeypatch):
    """Safe mode reads defaults and writes for real. A migration run on
    those defaults replaced a real Workstation with Balanced."""
    path = tmp_path / "real.ini"
    real = QSettings(str(path), QSettings.IniFormat)
    real.setValue(P._KEY_PERFORMANCE_LEVEL, "workstation")
    real.sync()
    del real

    monkeypatch.setattr(P, "_SAFE_MODE", True)
    monkeypatch.setattr(
        P, "_settings",
        lambda: P._DefaultsForReadingRealForWriting(
            QSettings(str(path), QSettings.IniFormat)))

    assert P.get_performance_level() == P.DEFAULT_PERFORMANCE_LEVEL
    assert _reopen(path).value(P._KEY_PERFORMANCE_LEVEL) == "workstation", (
        "safe mode overwrote the user's real performance level")


@pytest.mark.parametrize("write", [
    lambda: P.set_spacr_mode("performance"),
    lambda: P.set_performance_level("workstation"),
    lambda: P.set_performance_level("laptop"),
])
def test_one_stored_value_not_two(ini, write):
    """"There is one stored performance level." The posture is derived."""
    write()
    on_disk = _reopen(ini)
    assert on_disk.contains(P._KEY_PERFORMANCE_LEVEL)
    assert not on_disk.contains(P._KEY_SPACR_MODE), (
        "the spaCR-mode key is still written beside the level")
    assert not on_disk.contains(P._KEY_LAPTOP_MODE)
