"""The benchmark's preferences state, and the two backstops behind it.

``test_cov_r4_startup_benchmark`` pinned the controller's refusals around
Home and the sidebar. What that left is the state between them -- the real
Preferences dialog, which is measured as a budgeted interval of its own --
and the two things that stop a run nothing else can stop:

* the WALL-CLOCK backstop. Every other deadline in this controller is a
  ``QTimer``, and a GUI thread that cannot return to the event loop cannot
  deliver one. The wall timer runs on a plain Python thread and leaves by
  ``os._exit`` for the same reason, after printing where every thread was.
* the WATCHDOG SEAL. A result window may only be closed at an observed
  watchdog beat, so a beat that has not moved past the paint means "not
  yet, ask again" rather than a boundary invented from the wall clock.
  Both the readiness interval and the preferences interval seal this way,
  and both must survive being asked before the beat arrives.

The preferences half is four separate refusals -- a dialog that will not
construct, a settle that arrives in the wrong phase, a settle that arrives
before the beat, and a dialog whose C++ half has already gone -- and after
every one of them the sweep has to go on to the module keys, because
Preferences is one row of the artifact and not the run.

``_ready`` and ``_settle_*`` are called directly here for the reason the
round 4 file gives: they are the callbacks ``subscribe_readiness`` and
``QTimer.singleShot`` install, and the production publisher needs a really
painted screen tree.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog, QWidget                # noqa: E402

from spacr.qt import startup_benchmark as sb                  # noqa: E402
from spacr.qt import timing                                   # noqa: E402

pytestmark = pytest.mark.qt

SETTLE_WAIT_MS = 3000


def _home_entry(**overrides):
    """A readiness record for Home, as ``timing`` publishes one."""
    entry = {
        "detail": "__home__",
        "name": "interactive Home",
        "started_at": 0.0,
        "at": 0.0,
        "duration_s": 0.25,
        "budget_s": timing.HOME_BUDGET_S,
        "within_budget": True,
    }
    entry.update(overrides)
    return entry


@pytest.fixture
def window(qtbot):
    """A window with the one attribute the controller reads: ``_sidebar``."""
    holder = QWidget()
    qtbot.addWidget(holder)
    holder._sidebar = SimpleNamespace(_items=[])
    return holder


@pytest.fixture
def make_controller(qapp, tmp_path):
    """Build controllers and retire each one before the next test runs."""
    made = []

    def build(window, keys=(), timeout_s=5.0, name="benchmark.json", **kwargs):
        controller = sb.BenchmarkController(
            qapp, window, keys, str(tmp_path / name), timeout_s=timeout_s,
            **kwargs)
        made.append(controller)
        return controller

    yield build

    for controller in made:
        controller._finished = True
        controller.timeout.stop()
        timing.unsubscribe_readiness(controller._ready)


def _artifact(controller):
    return json.loads(open(controller.output, encoding="utf-8").read())


class _RecordingDialog(QDialog):
    """The Preferences dialog reduced to what the controller uses."""

    made = []

    def __init__(self, parent=None):
        super().__init__(parent)
        type(self).made.append(self)


# ---------------------------------------------------------------------------
# the wall-clock backstop
# ---------------------------------------------------------------------------

def test_a_gui_thread_that_cannot_run_a_qtimer_is_killed_where_it_stands(
        make_controller, window, monkeypatch, tmp_path):
    """The backstop that is not a QTimer, because a QTimer is what is stuck.

    Arming it is pinned in ``test_real_startup_readiness``; what it DOES was
    not. It is deliberately blunt -- ``os._exit``, no Qt shutdown, because a
    clean shutdown needs the wedged event loop -- so the diagnostic has to be
    written before it goes: the reason with the deadline it exceeded, and
    every thread's stack, which is the only evidence of where the GUI thread
    stopped. Status 124 is what the parent reads as the ratchet failure.
    """
    armed = []

    class _CapturingWallTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            armed.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setenv(sb.HARD_TIMEOUT_ENV, "1")
    monkeypatch.setattr(sb.threading, "Timer", _CapturingWallTimer)
    controller = make_controller(window, timeout_s=2.0)
    assert len(armed) == 1, "the wall-clock backstop was never armed"

    report = tmp_path / "stderr.txt"
    exits = []
    with open(report, "w", encoding="utf-8") as handle:
        # faulthandler writes to a file DESCRIPTOR, so this has to be a real
        # file rather than a capture buffer.
        monkeypatch.setattr(sb.sys, "stderr", handle)
        monkeypatch.setattr(sb.os, "_exit", exits.append)
        armed[0].callback()          # the wall clock running out

    assert exits == [124], "a wedged benchmark was left running"
    written = report.read_text(encoding="utf-8")
    assert "spaCR benchmark hard timeout" in written
    assert "within 2.0 seconds" in written, (
        "the report does not say which deadline was exceeded")
    assert "Current thread" in written and "_hard_timed_out" in written, (
        "no thread stacks were dumped, so there is no evidence of where the "
        "GUI thread stopped")


# ---------------------------------------------------------------------------
# sealing a readiness interval at a watchdog beat
# ---------------------------------------------------------------------------

def test_a_paint_is_not_sealed_until_the_watchdog_has_moved_past_it(
        make_controller, window, qtbot, monkeypatch):
    """A beat at or before the paint spans backwards across it.

    Sealing there would let a later raw interval begin before the closed
    window ended, and the stall inventory of two results would overlap. The
    controller asks again instead, and the record it is holding stays held
    -- it is not dropped and it is not measured against a wall-clock guess.
    """
    beat = {"at": 1.0}
    monkeypatch.setattr(timing, "last_gui_beat_at", lambda: beat["at"])
    controller = make_controller(window)

    controller._ready(_home_entry(at=5.0))
    qtbot.wait(sb.SETTLE_MS * 4)

    assert controller._pending is not None, (
        "the paint was thrown away rather than retried")
    assert controller.results == [], (
        "the interval was sealed at a beat that precedes the paint")

    # The watchdog moves past the paint: the same retry now seals, at the
    # beat rather than at the clock.
    beat["at"] = 9.0
    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    assert controller._pending is None
    assert controller.results[0]["stall_window_ended_at"] == 9.0
    assert controller.results[0]["at"] == 5.0, (
        "the retry re-stamped the paint it was waiting for")


# ---------------------------------------------------------------------------
# opening the Preferences dialog
# ---------------------------------------------------------------------------

def test_with_no_factory_the_real_preferences_dialog_is_what_is_measured(
        make_controller, window, monkeypatch):
    """The factory is a test seam; production has none and builds the dialog.

    The benchmark exists to measure what the user gets, so the default path
    imports ``PreferencesDialog`` and parents it on the live window. It is
    substituted here (the real one owns the preference store and every
    settings page), but the substitution is on the module the controller
    imports from, so the import and the parenting are the production ones.
    """
    from spacr.qt import preferences as preferences_module

    _RecordingDialog.made = []
    monkeypatch.setattr(preferences_module, "PreferencesDialog",
                        _RecordingDialog, raising=False)
    controller = make_controller(window, measure_preferences=True)

    controller._open_preferences()

    assert len(_RecordingDialog.made) == 1, (
        "the dialog the user opens was never built")
    dialog = _RecordingDialog.made[0]
    assert dialog.parent() is window, "the dialog was not parented on the app"
    assert controller._preferences_dialog is dialog
    assert dialog.isVisible(), "the dialog was never shown, so nothing painted"

    # A factory given: it is used INSTEAD, which is what makes the import
    # above the no-factory path rather than the only path.
    made_by_factory = []

    def _factory():
        made_by_factory.append(QDialog(window))
        return made_by_factory[-1]

    second = make_controller(window, measure_preferences=True,
                             preferences_factory=_factory, name="factory.json")
    second._open_preferences()
    assert len(_RecordingDialog.made) == 1
    assert second._preferences_dialog is made_by_factory[0]


def test_a_preferences_dialog_that_will_not_open_is_one_failed_row(
        make_controller, window, qtbot):
    """A dialog that raises is a recorded failure, not a dead sweep.

    Preferences is one budgeted row of the artifact. When it cannot be
    built the row says so, the phase moves on to the module keys, and the
    run still reaches its ordinary end -- otherwise a broken settings page
    would hide every module measurement behind it.
    """
    def _explodes():
        raise RuntimeError("the settings page would not build")

    controller = make_controller(window, measure_preferences=True,
                                 preferences_factory=_explodes)
    controller.phase = "preferences"

    controller._open_preferences()

    failure = controller.results[-1]
    assert failure["detail"] == "__preferences__"
    assert failure["name"] == "interactive preferences"
    assert failure["error"] == (
        "RuntimeError: the settings page would not build")
    assert failure["within_budget"] is False
    assert failure["budget_s"] == sb.PREFERENCES_BUDGET_S
    assert controller._preferences_dialog is None, (
        "a dialog that never opened was left to be closed")
    assert controller.phase == "module", (
        "the sweep stayed in a phase whose dialog cannot be built")

    qtbot.waitUntil(lambda: controller._finished, timeout=SETTLE_WAIT_MS)
    benchmark = _artifact(controller)["benchmark"]
    assert benchmark["exit_reason"] == "registry sweep complete"
    assert any("the settings page would not build" in line
               for line in benchmark["violations"])


# ---------------------------------------------------------------------------
# settling the Preferences interval
# ---------------------------------------------------------------------------

def test_a_preferences_settle_outside_its_own_phase_measures_nothing(
        make_controller, window):
    """``_settle_preferences`` is a queued single-shot: by the time it runs
    the deadline may have fired, or the run may have ended. Either one has
    already recorded this interval, and a second row for it would report a
    dialog that was measured twice.
    """
    late = make_controller(window, measure_preferences=True)
    late.phase = "module"
    late._settle_preferences()
    assert late.results == [], "a settle after the phase moved on was measured"

    ended = make_controller(window, measure_preferences=True, name="ended.json")
    ended.phase = "preferences"
    ended._finished = True
    ended._settle_preferences()
    assert ended.results == [], "a settle after the run ended was measured"

    # In its own phase, in a live run, the same call is the measurement.
    live = make_controller(window, measure_preferences=True, name="live.json")
    live.phase = "preferences"
    live._settle_preferences()
    assert [row["detail"] for row in live.results] == ["__preferences__"]
    assert live.phase == "module", "the sweep did not move on after measuring"


def test_the_preferences_readiness_stamp_survives_waiting_for_the_beat(
        make_controller, window, qtbot, monkeypatch):
    """The interval is stamped once and sealed later, possibly much later.

    ``_settle_preferences`` retries itself until the watchdog has moved past
    the paint. Re-stamping readiness on each retry would charge the dialog
    for the time spent waiting for a beat, so the first stamp is kept and
    only the window's END moves.
    """
    beat = {"at": 1.0}
    monkeypatch.setattr(timing, "last_gui_beat_at", lambda: beat["at"])
    controller = make_controller(window, measure_preferences=True)
    controller.phase = "preferences"
    controller._preferences_started_elapsed = 2.0
    controller._preferences_ready_at = 4.0

    controller._settle_preferences()

    assert controller.results == [], (
        "the interval was sealed at a beat that precedes the paint")
    assert controller._preferences_ready_at == 4.0, (
        "waiting for the beat moved the readiness stamp")

    beat["at"] = 12.0
    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    measured = controller.results[-1]
    assert measured["at"] == 4.0
    assert measured["stall_window_ended_at"] == 12.0
    assert measured["duration_s"] == pytest.approx(2.0), (
        "the dialog was charged for the wait for a watchdog beat")


def test_a_dialog_whose_c_half_has_gone_is_still_let_go_of(make_controller,
                                                           window):
    """``_close_preferences_dialog`` runs from ``_finish`` as well.

    By then Qt may have destroyed the dialog under the Python wrapper, and
    ``close()`` on the wrapper raises. Holding on to it would keep the run
    from finishing over a window that no longer exists.
    """
    class _AlreadyDeleted:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1
            raise RuntimeError("Internal C++ object already deleted.")

        def deleteLater(self):                            # noqa: N802 (Qt)
            raise AssertionError("a dialog that would not close was deleted")

    controller = make_controller(window, measure_preferences=True)
    gone = _AlreadyDeleted()
    controller._preferences_dialog = gone
    controller._preferences_ready_at = 3.0

    controller._close_preferences_dialog()

    assert gone.closes == 1, "the close was never attempted"
    assert controller._preferences_dialog is None, (
        "the controller kept a dialog it cannot close")
    assert controller._preferences_ready_at is None

    # A dialog that closes: really closed, so the raise above is what the
    # handler caught and not a close that never happened.
    live = QDialog(window)
    live.show()
    assert live.isVisible()
    controller._preferences_dialog = live
    controller._close_preferences_dialog()
    assert not live.isVisible()
    assert controller._preferences_dialog is None


# ---------------------------------------------------------------------------
# the deadline, in the preferences phase
# ---------------------------------------------------------------------------

def test_a_preferences_dialog_that_never_paints_expires_as_its_own_row(
        make_controller, window, qtbot):
    """The deadline has to name the state it expired in.

    ``__preferences__`` is the artifact's key for this interval; recording a
    registry key instead would attribute a hung settings dialog to whichever
    module screen came next -- and, since ``current_key`` is still ``None``
    here, would write the string ``"None"`` into the artifact.
    """
    controller = make_controller(window, measure_preferences=True,
                                 timeout_s=5.0)
    controller.phase = "preferences"

    controller._timed_out()
    assert controller._timeout_pending is True

    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    expired = controller.results[-1]
    assert expired["detail"] == "__preferences__"
    assert expired["name"] == "interactive preferences"
    assert expired["error"].startswith("no painted usable state within")

    # The same deadline in the module phase names the key it was pressing,
    # which is what makes the branch above the preferences one.
    other = make_controller(window, keys=("mask",), name="module.json")
    other.phase = "module"
    other.current_key = "mask"
    other._timed_out()
    qtbot.waitUntil(lambda: bool(other.results), timeout=SETTLE_WAIT_MS)
    assert other.results[-1]["detail"] == "mask"
