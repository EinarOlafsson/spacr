"""The QC field browser polls its button without touching the filesystem.

THE FREEZE, 2026-09-04. `QCFieldBrowser.__init__` starts a 400 ms `QTimer`
parented to the dialog, so `_sync_action` runs in the GUI thread's event
loop for as long as the browser is open. Its second statement was
`self._file_state()`, and that was:

    Path(merged_dir, f"{field}.npy").is_file()
    Path(...).is_symlink()
    qc_quarantine.is_quarantined(merged_dir, field)
      -> quarantine_dir_for -> _merged_dir -> Path.resolve()   # realpath walk

Three stats and a realpath walk, over a path the user chose. Measured on the
maintainer's machine that day: one `os.path.exists` under `/nas_mnt`, an
`autofs` mount whose share was asleep, had NOT RETURNED AFTER TWENTY
SECONDS -- the stat is what triggers the automount. Opening the field
browser on a NAS plate therefore stalled the event loop a couple of times a
second, for the life of the dialog. It leaves no traceback, because a
stalled event loop is not a crash; it was reported as "opening map barcodes
crashes spacr", hover flicker, and glimpses of other screens.

The fix routes both questions through `spacr/qt/path_probe.py`, which
answers from a cache and probes in the background, and primes that cache
from `load_qc_field` -- which already stats both copies, on a worker.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldTarget,
    QCFieldVerdict,
)

pytestmark = pytest.mark.qt

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration is a test
#: nobody runs.
SLOW_S = 8.0

#: Only paths under the browser's own plate are made slow. Patching every
#: stat in the process would stall pytest's own bookkeeping as readily as
#: the dialog's, and prove nothing about this file.
SLOW_MARK = "asleep_plate"


@pytest.fixture(autouse=True)
def _fresh_cache():
    path_probe.forget()
    yield
    path_probe.forget()


def _target(tmp_path, name=SLOW_MARK):
    plate = tmp_path / name
    (plate / "merged").mkdir(parents=True)
    return QCFieldTarget(
        field="plate1_A01_1", plate_root=str(plate),
        merged_dir=str(plate / "merged"),
        verdicts=(QCFieldVerdict("cell", "fail", ("empty_field",), "empty"),))


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Hand back a switch that makes the plate take :data:`SLOW_S` to answer.

    This is what an `autofs` mount does while its share spins up, and it is
    the state the maintainer's machine was in. `Path.is_file`, `is_symlink`
    and `resolve` all go through `os.stat`/`os.lstat`, so one patch covers
    the old code and the new probe worker alike -- the point of the test is
    WHICH THREAD waits, not who calls what.

    A switch rather than an autouse patch because a browser has to be BUILT
    before its timer can poll, and building one on a plate that takes eight
    seconds per stat is a minute of test nobody learns anything from.
    """
    real_stat, real_lstat = os.stat, os.lstat

    def slow(real):
        def wrapper(path, *args, **kwargs):
            if SLOW_MARK in str(path):
                time.sleep(SLOW_S)
            return real(path, *args, **kwargs)
        return wrapper

    def put_to_sleep():
        monkeypatch.setattr(os, "stat", slow(real_stat))
        monkeypatch.setattr(os, "lstat", slow(real_lstat))

    return put_to_sleep


def test_the_button_poll_returns_before_the_mount_does(
        qtbot, tmp_path, sleeping_mount):
    """The property the freeze violated: the 400 ms poll never waits.

    The browser is built while the plate still answers and only then put to
    sleep, because it is the POLL that ran forever -- construction is one
    event the user asked for, and the timer is thousands they did not.
    """
    target = _target(tmp_path)
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    # Forget what construction primed: an unseen path is the case that
    # actually blocked, since there was nothing cached to answer from.
    path_probe.forget()
    sleeping_mount()

    started = time.monotonic()
    browser._sync_action()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"_sync_action() took {elapsed:.1f}s -- the 400 ms timer is stat-ing "
        "the merged folder on the GUI thread again, which is the freeze")


def test_a_field_state_read_is_cheap_on_a_sleeping_plate(
        qtbot, tmp_path, sleeping_mount):
    """`_file_state` itself, with nothing cached and nothing primed."""
    target = _target(tmp_path)
    sleeping_mount()
    browser = QCFieldBrowser.__new__(QCFieldBrowser)
    browser._targets = (target,)
    browser._index = 0

    started = time.monotonic()
    active, quarantined = browser._file_state()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"_file_state() took {elapsed:.1f}s -- it is still stat-ing inline")
    assert active is True, (
        "the active copy must answer optimistically, the way file_list does: "
        "a field drawn as present and then corrected has learned something, "
        "one drawn as missing and then corrected cried wolf")
    assert quarantined is False, (
        "an unknown quarantine copy must answer False, or an unprobed field "
        "renders the 'both copies exist' dead end and disables the button")


def test_the_button_corrects_itself_once_the_probe_lands(qtbot, tmp_path):
    """Answering optimistically is only honest if the truth still arrives.

    With the poll timer stopped, the only thing that can repaint this button
    is `path_probe.probes.answered` -- so this is `_on_probe_answered`, and
    nothing else, being tested.
    """
    target = _target(tmp_path, name="gone_plate")
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)
    browser._run_timer.stop()

    path_probe.forget()
    browser._sync_action()
    assert browser._quarantine.isEnabled(), (
        "an unprobed field should be offered, not pre-emptively refused")

    qtbot.waitUntil(
        lambda: "out of date" in browser._action_status.text().lower(),
        timeout=10000)
    assert not browser._quarantine.isEnabled()


def test_the_loading_job_settles_both_states_before_it_paints(
        qtbot, tmp_path):
    """The load already stats both copies, on a worker; reuse those answers.

    Optimism is the right default but it is not good enough on its own: the
    button must say "already gone" on the first frame for a field that is,
    and "resolve duplicate copies" for one that is in both folders, and
    neither can be read off an unprobed cache. So the loading job records
    both answers before its callback runs.
    """
    target = _target(tmp_path, name="primed_plate")
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    active = os.path.join(target.merged_dir, "plate1_A01_1.npy")
    quarantined = os.path.join(
        target.plate_root, "merged_quarantined", "plate1_A01_1.npy")

    assert path_probe.known(active) is False, (
        "the loader learned this file was missing and threw the answer away")
    assert path_probe.known(quarantined) is False
    assert "already gone" in browser._load_status.text().lower()
    assert "out of date" in browser._action_status.text().lower()
    assert not browser._quarantine.isEnabled()


def test_a_copy_that_vanishes_elsewhere_is_still_noticed(qtbot, tmp_path):
    """The cache must not be allowed to go permanently stale.

    `_file_state` used to stat on every tick, so a copy deleted from outside
    the application -- a crashed run tidying up, the user removing one of two
    duplicates -- corrected the button within 400 ms. The probe cache has no
    expiry of its own, so the browser retires its own two keys periodically.
    """
    target = _target(tmp_path, name="vanishing_plate")
    active = os.path.join(target.merged_dir, "plate1_A01_1.npy")
    open(active, "wb").close()

    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)
    assert path_probe.known(active) is True

    os.unlink(active)
    qtbot.waitUntil(lambda: path_probe.known(active) is False, timeout=10000)
    qtbot.waitUntil(
        lambda: "out of date" in browser._action_status.text().lower(),
        timeout=10000)


def test_a_re_check_waits_for_the_answer_it_already_asked_for(qtbot, tmp_path):
    """Never re-arm a probe that has not landed.

    A probe against a mount that has stopped responding parks a thread for
    `path_probe.PROBE_TIMEOUT_S`. Re-arming on a timer regardless would
    queue a fresh one every two seconds against a share that is not going to
    answer any of them, which is a thread leak dressed up as a refresh.
    """
    target = _target(tmp_path, name="outstanding_plate")
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    active, quarantined = browser._field_paths()
    path_probe.forget(active)
    assert path_probe.known(active) is None

    browser._recheck_files()

    assert path_probe.known(quarantined) is False, (
        "the settled key was retired while the other one was outstanding")


def test_a_dismissed_dialog_cannot_be_called_into_by_a_late_probe(
        qtbot, tmp_path):
    """`path_probe.probes` is process-wide and outlives every dialog.

    And Escape does not go through `closeEvent`: `QDialog.done` deletes a
    `WA_DeleteOnClose` dialog outright. So the connection cannot depend on a
    teardown hook -- it is a bound method, which Qt retires with the object
    it belongs to. A signal still wired to a destroyed QDialog is a hard
    crash, not an exception.
    """
    from PySide6.QtCore import Qt
    from shiboken6 import isValid

    target = _target(tmp_path, name="closed_plate")
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)
    browser.show()

    qtbot.keyClick(browser, Qt.Key_Escape)
    qtbot.waitUntil(lambda: not isValid(browser), timeout=5000)

    # The emission a background probe would have made a moment later.
    path_probe.probes.answered.emit(
        os.path.join(target.merged_dir, "plate1_A01_1.npy"), True)


def test_the_slot_survives_a_widget_that_died_mid_emission(qtbot, tmp_path):
    """The other half of the same hazard: an emission already in flight.

    Qt can be part-way through delivering when the C++ half goes, and
    reaching a destroyed child through a live wrapper raises RuntimeError.
    Swallowed here, because a repaint that arrives too late is nothing and a
    traceback out of a signal is a lost session.
    """
    target = _target(tmp_path, name="racing_plate")
    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    def gone(*_args):
        raise RuntimeError("Internal C++ object already deleted.")

    browser._sync_action = gone
    active, _quarantined = browser._field_paths()

    browser._on_probe_answered(active, False)
