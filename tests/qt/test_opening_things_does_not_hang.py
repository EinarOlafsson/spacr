"""Opening a module or Preferences finishes, and finishes quickly.

Instruction 314. A maintainer reported Regression HANGING -- twice, so
not a one-off -- and Preferences taking "a very long time". Instruction
284's ratchet fails a module at ten seconds, and as 314 puts it, a hang
has no number at all: the existing guard is tuned for slow, not for
stuck.

These are the missing numbers. They are deliberately GENEROUS -- several
times the measured cost on the machine that runs them -- because the
failure being caught is seconds-to-forever, not a few hundred
milliseconds of drift. A tight budget here would fail on a loaded CI box
and teach everyone to ignore it.

WHAT IS ALREADY GUARDED, so a reader does not go looking again: the two
paths that could block the GUI thread on the heavy-import lock both
yield instead. `AppScreen._install_ambient` defers on a timer when the
lock is busy -- 83% of a measured 3148 ms block was the backdrop
constructor waiting for a lock the preloader held -- and
`fractal_travel` acquires with a timeout rather than blocking. The
preloader itself runs on a worker thread.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.app import MainWindow

#: Seconds. Measured here at 1.34 s for the slowest module (Regression)
#: and 0.42 s for a first Preferences build; these are the ceilings a
#: hang would blow through, not a target to tune against.
MODULE_BUDGET = 20.0
PREFERENCES_BUDGET = 10.0

#: The modules 284's ratchet named, plus Regression, which is the one
#: reported. Ordered cheapest first so a failure names the first thing
#: that broke rather than the last.
MODULES = ("measure", "mask", "classify_merged", "regression")


@pytest.fixture(scope="module")
def window(qapp_module_scope=None):
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    made = MainWindow()
    made.show()
    application.processEvents()
    yield made
    made.close()


@pytest.mark.parametrize("key", MODULES)
def test_a_module_opens_rather_than_hanging(window, key):
    """Each module opens, and the time it took is reported on failure.

    Driven through `_on_nav_selected`, the same entry point a tile
    click uses, so a screen that only opens from a test would not pass.
    """
    from PySide6.QtWidgets import QApplication

    started = time.perf_counter()
    window._on_nav_selected(key)
    QApplication.instance().processEvents()
    elapsed = time.perf_counter() - started

    assert key in window._screens, f"{key} did not open at all"
    assert elapsed < MODULE_BUDGET, (
        f"{key} took {elapsed:.1f} s to open, which is not slow, it is "
        "stuck")


def test_preferences_builds_rather_than_hanging(window):
    """The dialog is BUILT, not exec'd.

    `exec()` blocks until a human closes it, so a test that called it
    would hang forever headlessly and prove nothing about the cost the
    user reported. The complaint was about how long it took to APPEAR,
    which is construction.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt.preferences import PreferencesDialog

    started = time.perf_counter()
    dialog = PreferencesDialog(window)
    QApplication.instance().processEvents()
    elapsed = time.perf_counter() - started

    assert dialog is not None
    assert elapsed < PREFERENCES_BUDGET, (
        f"Preferences took {elapsed:.1f} s to build")
    dialog.close()


def test_the_backdrop_never_waits_on_the_heavy_lock(window):
    """THE MECHANISM BEHIND THE REPORTED FREEZE, pinned.

    The GPU backdrop takes the heavy-import lock to build its GL
    context, and the startup preloader holds that lock for a whole
    module import. Installing regardless is what froze the GUI thread.
    The screen must open undecorated and retry, rather than wait.

    Asserted on the source because the race cannot be reproduced
    reliably in a test: what must be true is that the code CHECKS before
    it installs, and that check is what a refactor would remove.
    """
    import inspect

    from spacr.qt.screens.app_screen import AppScreen

    body = inspect.getsource(AppScreen._install_ambient)
    assert "_heavy_lock_is_free" in body, (
        "the backdrop installs without checking the heavy lock; a module "
        "opened soon after launch will freeze behind the preloader")
    assert "singleShot" in body, (
        "the backdrop does not retry, so a busy lock loses it entirely")
