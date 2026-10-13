"""A memory-budget sweep one test installs must not tick inside the next.

`resource_cleanup.register()` -- run by every `launch` and every
`register_self_registering_modules()` -- parents a repeating QTimer to the
session-wide QApplication and connects a hook to the run registry. Nothing
took either back, so the sweep kept ticking for the rest of the session and
fired wherever a later test spun the event loop.

pytest-qt spins it once more after EVERY test's fixtures are set up. So a
tick landed after a test had pointed `preferences._settings` at its own
store and before its body ran. The budget reads its per-level defaults
through `get_performance_level` (286), so that tick migrated the test's
EMPTY store to Balanced and saved it, and
`test_the_level_migration_removes_what_it_replaced.py` then read a level its
body never wrote. Measured: 1 failure in 169 on landing, and 2 of 80 cases
under load, each preceded by three level reads during SETUP -- the sweep's
idle timeout, cache ceiling and headroom.

THE TWO TESTS RUN IN FILE ORDER, as the suite's serial runs do
(``-p no:randomly``). Shuffled, the second can only pass without proving
anything; it cannot fail spuriously.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as P
from spacr.qt import resource_cleanup as rc

pytestmark = pytest.mark.qt


def test_a_test_installs_the_sweep_and_the_run_hook(qapp, monkeypatch):
    """What `launch` does. The tick is made due by the next test's setup,
    which turns a once-every-five-seconds race into a certainty."""
    monkeypatch.setattr(rc, "BUDGET_SWEEP_INTERVAL_MS", 1)
    assert rc.install_budget_sweep() is True
    assert rc.install_run_hook() is True
    assert rc._BUDGET_TIMER is not None and rc._BUDGET_TIMER.isActive()


@pytest.fixture
def own_store(tmp_path, monkeypatch):
    """This test's preference store, in place before pytest-qt's
    post-setup event spin -- exactly where the migration tests' is."""
    path = tmp_path / "prefs.ini"
    monkeypatch.setattr(P, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return path


def test_the_next_test_starts_with_no_sweep_and_an_untouched_store(
        qapp, own_store):
    written = (own_store.read_text(encoding="utf-8")
               if own_store.exists() else "")
    assert "performance_level" not in written, (
        "a budget sweep left running by the previous test ticked during this "
        f"test's setup and wrote into its store:\n{written}")

    timer = rc._BUDGET_TIMER
    assert timer is None or not timer.isActive(), (
        "the previous test's budget sweep is still ticking")
    assert rc._INSTALLED is False, (
        "the previous test's run-registry hook is still connected")
