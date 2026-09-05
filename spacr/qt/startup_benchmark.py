"""Controlled real-entry-point startup and module-readiness benchmark.

This module is imported only when ``SPACR_BENCHMARK_JSON`` names an output
file.  The ordinary application therefore pays no import or QObject cost for
it.  The benchmark still runs the ordinary ``spacr.qt.run`` path: once Home
has really painted, it presses each live sidebar button and waits for the
production timing probe to observe a painted, enabled control.  After the
last registry key it writes one JSON artifact and exits Qt deliberately.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Callable, Iterable, Optional

from PySide6.QtCore import QObject, QTimer

from . import timing

OUTPUT_ENV = "SPACR_BENCHMARK_JSON"
RUN_LABEL_ENV = "SPACR_BENCHMARK_RUN"
TIMEOUT_ENV = "SPACR_BENCHMARK_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 30.0
SETTLE_MS = 32
PREFERENCES_BUDGET_S = 3.0
PREFERENCES_HANG_TIMEOUT_S = 10.0
HARD_TIMEOUT_ENV = "SPACR_BENCHMARK_HARD_TIMEOUT"
HARD_TIMEOUT_GRACE_S = 1.0


def _timeout_seconds() -> float:
    """How long the startup benchmark waits before giving up.

    :returns: the timeout in seconds.
    """
    try:
        return max(1.0, min(300.0, float(os.environ.get(
            TIMEOUT_ENV, DEFAULT_TIMEOUT_S))))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S


class BenchmarkController(QObject):
    """Advance through Home and an exact snapshot of the live app registry.

    :param app: the running :class:`QApplication`, also this object's parent.
    :param window: the main window to drive.
    :param keys: the module keys to visit, in order.
    :param output: where to write the measurements.
    :param timeout_s: how long one module may take before the run is
        abandoned. ``None`` uses the default.
    :param live_keys: called for the registry as it stands NOW, so a run can
        check the snapshot in ``keys`` still matches the app it is driving.
    :param measure_preferences: whether to open Preferences and time it.
        ``None`` decides from the environment.
    :param preferences_factory: builds the Preferences dialog, so a test can
        supply one without the real dialog.
    """

    def __init__(self, app, window, keys: Iterable[str], output: str, *,
                 timeout_s: Optional[float] = None,
                 live_keys: Optional[Callable[[], Iterable[str]]] = None,
                 measure_preferences: Optional[bool] = None,
                 preferences_factory: Optional[Callable[[], object]] = None,
                 ) -> None:
        """Drive the benchmark: open each module in turn and time it.

        :param app: the running QApplication.
        :param window: the main window to drive.
        :param keys: the modules to open, in order.
        :param output: where to write the artifact.
        :param timeout_s: how long one module may take before the run fails.
        :param live_keys: modules whose readiness is signalled rather than polled.
        :param measure_preferences: whether to time the Preferences dialog too.
        :param preferences_factory: how to build that dialog.
        """
        super().__init__(app)
        self.app = app
        self.window = window
        self.keys = tuple(str(key) for key in keys)
        self._live_keys = live_keys
        self.output = str(output)
        self.timeout_s = _timeout_seconds() if timeout_s is None else float(
            timeout_s)
        self.results: list[dict] = []
        self.phase = "home"
        self.current_key: Optional[str] = None
        self.index = 0
        self._pending: Optional[dict] = None
        self._timeout_pending = False
        self._finished = False
        self._written = False
        self._measure_preferences = (
            bool(os.environ.get(OUTPUT_ENV, "").strip())
            if measure_preferences is None else bool(measure_preferences)
        )
        self._preferences_factory = preferences_factory
        self._preferences_dialog = None
        self._preferences_started_elapsed = 0.0
        self._preferences_ready_at: Optional[float] = None
        self._hard_timeout = None
        self._armed_timeout_s = self.timeout_s
        self._attempt_started = time.perf_counter()
        self._attempt_started_elapsed = timing.elapsed()

        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self._timed_out)
        timing.subscribe_readiness(self._ready)
        app.aboutToQuit.connect(self._application_quit)
        self._arm_timeout()

    def _arm_timeout(self, timeout_s: Optional[float] = None) -> None:
        """Start the clock that fails the run if a module hangs.

        A HANG MUST NOT BE A HANG. Without this a module that never signals
        ready leaves the benchmark waiting for ever, which in CI is an hour of
        a runner rather than a failure anyone can read.

        :param timeout_s: how long to allow.
        """
        self._disarm_timeout()
        self._timeout_pending = False
        self._attempt_started = time.perf_counter()
        self._attempt_started_elapsed = timing.elapsed()
        self._armed_timeout_s = (
            self.timeout_s if timeout_s is None else float(timeout_s)
        )
        self.timeout.start(max(1, int(self._armed_timeout_s * 1000.0)))
        if os.environ.get(HARD_TIMEOUT_ENV, "").strip() == "1":
            wall = threading.Timer(
                self._armed_timeout_s + HARD_TIMEOUT_GRACE_S,
                self._hard_timed_out,
            )
            wall.daemon = True
            self._hard_timeout = wall
            wall.start()

    def _disarm_timeout(self) -> None:
        """Stop both the event-loop deadline and its wall-clock backstop."""
        self.timeout.stop()
        wall = self._hard_timeout
        self._hard_timeout = None
        if wall is not None:
            wall.cancel()

    def _hard_timed_out(self) -> None:
        """Terminate a benchmark worker whose GUI thread cannot run QTimer.

        This path is enabled only in the dedicated benchmark subprocess. It
        deliberately uses ``os._exit``: a normal Qt shutdown needs the very
        event loop that is wedged. The parent preserves the last checkpoint
        and records exit status 124 as a ratchet failure.
        """
        try:
            import faulthandler

            print(
                "spaCR benchmark hard timeout: GUI thread did not return "
                f"within {self._armed_timeout_s:.1f} seconds",
                file=sys.stderr,
                flush=True,
            )
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        finally:
            os._exit(124)

    def _ready(self, entry: dict) -> None:
        """Record that one module finished opening.

        :param entry: the module's timing entry.
        """
        if (self._finished or self._pending is not None
                or self._timeout_pending):
            return
        expected = "__home__" if self.phase == "home" else self.current_key
        if entry.get("detail") != expected:
            return
        expected_name = (
            "interactive Home" if self.phase == "home"
            else "interactive module"
        )
        if entry.get("name") != expected_name:
            return
        self._disarm_timeout()
        self._pending = dict(entry)
        # Let the 16 ms watchdog report a timer delayed by the click handler
        # before this interval is sealed.  The readiness timestamp remains
        # the first settled paint; only the stall inventory waits two frames.
        QTimer.singleShot(SETTLE_MS, self._settle_ready)

    @staticmethod
    def _sealed_stall_window_end(
            observed_at: float, retry: Callable[[], None]) -> Optional[float]:
        """Return the first observed watchdog beat after readiness.

        A wall-clock snapshot between watchdog beats is not a stable boundary:
        the next beat begins at the preceding beat and therefore spans backward
        across that snapshot.  Sealing at a beat instead makes every later raw
        interval begin exactly at or after the closed result window.  Unit
        environments without the production watchdog retain the current clock
        fallback because no later watchdog trace can appear there.
        """
        latest = timing.last_gui_beat_at()
        if latest is None:
            return timing.elapsed()
        if latest <= float(observed_at):
            QTimer.singleShot(SETTLE_MS, retry)
            return None
        return float(latest)

    def _settle_ready(self) -> None:
        """Let the event loop drain before the timing is taken.

        A MODULE IS NOT READY WHEN ITS CONSTRUCTOR RETURNS: deferred work is
        still queued, and timing it there measures the constructor rather than
        the wait a user actually sees.
        """
        if self._pending is None or self._finished:
            return
        entry = self._pending
        end = self._sealed_stall_window_end(
            float(entry.get("at", 0.0)), self._settle_ready)
        if end is None:
            return
        self._pending = None
        state = timing.snapshot()
        start = float(entry.get("started_at", 0.0))
        interval_stalls = timing.stalls_between(start, end, state["stalls"])
        # Preserve the exact window used for the derived stall fields.  The
        # readiness timestamp precedes the two-frame settling interval above,
        # so ``started_at``/``at`` alone cannot reproduce this calculation.
        # The parent benchmark driver independently recomputes every value
        # below from the raw watchdog trace and these two boundaries.
        entry["stall_window_started_at"] = start
        entry["stall_window_ended_at"] = end
        entry["worst_event_loop_stall_ms"] = max(
            (float(row["overlap_ms"]) for row in interval_stalls), default=0.0)
        entry["worst_overlapping_frame_interval_ms"] = max(
            (float(row["late_ms"]) for row in interval_stalls), default=0.0)
        entry["event_loop_stall_budget_met"] = (
            entry["worst_event_loop_stall_ms"] < timing.STALL_BUDGET_MS
        )
        entry["stall_samples"] = len(interval_stalls)
        self.results.append(entry)
        self._checkpoint()
        print(
            f"benchmark ready: {entry['detail']} "
            f"{entry['duration_s']:.3f}s, worst gap "
            f"{entry['worst_event_loop_stall_ms']:.0f}ms",
            flush=True,
        )
        if self.phase == "home":
            self.phase = (
                "preferences" if self._measure_preferences else "module"
            )
        else:
            self.index += 1
            self.current_key = None
        self._advance_after_watchdog()

    def _advance_after_watchdog(self) -> None:
        """Do not let checkpoint/prewarm work contaminate the next click."""
        checkpoint_ended = timing.elapsed()

        def _after_beat() -> None:
            """Continue once the GUI thread has answered again."""
            latest = timing.last_gui_beat_at()
            if latest is not None and latest <= checkpoint_ended:
                QTimer.singleShot(SETTLE_MS, _after_beat)
                return
            self._advance()

        # Unit environments may not install the production watchdog; ``None``
        # deliberately falls through after the same two-frame settle.
        QTimer.singleShot(SETTLE_MS, _after_beat)

    def _advance(self) -> None:
        """Move on to the next module."""
        if self._finished:
            return
        if self.phase == "preferences":
            self._open_preferences()
            return
        if self.index >= len(self.keys):
            QTimer.singleShot(SETTLE_MS, self._finish)
            return
        key = self.keys[self.index]
        self.current_key = key
        self._arm_timeout()
        buttons = [
            button for button in getattr(self.window._sidebar, "_items", ())
            if str(button.property("navKey") or "") == key
        ]
        # NINE KEYS HAVE TWO ROWS, and neither is a mistake. A module that is
        # folded onto a host's masthead AND keeps its registry row -- profiler,
        # train_compare, convert, external_masks, layer_viewer, lineage,
        # tabulate, plate_view, investigate_hit -- is drawn once from the
        # registry and once as an indented child of its host. Both are hidden
        # (the registry row because the key is in `TILELESS_APPS`, the child
        # until its host is expanded), so a user never sees a duplicate.
        #
        # BUT THIS DRIVER DEMANDED EXACTLY ONE AND ERRORED ON ALL NINE, which
        # is how instruction 284's ratchet stopped measuring nine modules
        # without anyone noticing -- 314's own suspicion, that "the ratchet is
        # not running on the path the user actually takes", made concrete.
        #
        # The REGISTRY row is the one to press: it is the module's own
        # identity, and the fold child is a second door onto the same screen
        # that resolves through `open_module` rather than plain navigation.
        if len(buttons) > 1:
            registry_rows = [b for b in buttons
                             if not b.property("isFoldChild")]
            if len(registry_rows) == 1:
                buttons = registry_rows
        if len(buttons) != 1:
            self._record_error(
                key, f"expected one live sidebar button, found {len(buttons)}")
            return
        button = buttons[0]
        if not button.isEnabled():
            self._record_error(key, "the live sidebar button is disabled")
            return
        # QAbstractButton.click() is the same signal path as a user release:
        # Sidebar.nav_selected -> MainWindow._on_nav_selected.  Calling the
        # screen factory directly is the constructor proxy this benchmark
        # exists to replace.
        try:
            button.click()
        except BaseException as error:                      # noqa: BLE001
            self._record_error(
                key, f"{type(error).__name__}: {error}", already_stopped=True)

    def _open_preferences(self) -> None:
        """Construct and paint the real Preferences dialog under a budget."""
        self._preferences_started_elapsed = timing.elapsed()
        self._preferences_ready_at = None
        self._arm_timeout(min(self.timeout_s, PREFERENCES_HANG_TIMEOUT_S))
        try:
            if self._preferences_factory is None:
                from .preferences import PreferencesDialog

                dialog = PreferencesDialog(self.window)
            else:
                dialog = self._preferences_factory()
            self._preferences_dialog = dialog
            dialog.show()
        except BaseException as error:                       # noqa: BLE001
            self._record_error(
                "__preferences__",
                f"{type(error).__name__}: {error}",
            )
            return
        QTimer.singleShot(SETTLE_MS * 2, self._settle_preferences)

    def _settle_preferences(self) -> None:
        """Let the Preferences dialog finish laying out before it is timed."""
        if self._finished or self.phase != "preferences":
            return
        if self._preferences_ready_at is None:
            self._preferences_ready_at = timing.elapsed()
        ready_at = self._preferences_ready_at
        ended = self._sealed_stall_window_end(
            ready_at, self._settle_preferences)
        if ended is None:
            return
        self._disarm_timeout()
        state = timing.snapshot()
        stalls = timing.stalls_between(
            self._preferences_started_elapsed, ended, state["stalls"])
        duration = max(0.0, ready_at - self._preferences_started_elapsed)
        worst = max(
            (float(row["overlap_ms"]) for row in stalls), default=0.0)
        raw_worst = max(
            (float(row["late_ms"]) for row in stalls), default=0.0)
        self.results.append({
            "name": "interactive preferences",
            "detail": "__preferences__",
            "at": ready_at,
            "started_at": self._preferences_started_elapsed,
            "event_loop_started_at": state.get("event_loop_started_at"),
            "duration_s": duration,
            "budget_s": PREFERENCES_BUDGET_S,
            "within_budget": duration <= PREFERENCES_BUDGET_S,
            "stall_window_started_at": self._preferences_started_elapsed,
            "stall_window_ended_at": ended,
            "worst_event_loop_stall_ms": worst,
            "worst_overlapping_frame_interval_ms": raw_worst,
            "event_loop_stall_budget_met": worst < timing.STALL_BUDGET_MS,
            "stall_samples": len(stalls),
        })
        self._close_preferences_dialog()
        self.phase = "module"
        self._checkpoint()
        self._advance_after_watchdog()

    def _close_preferences_dialog(self) -> None:
        """Close the timed Preferences dialog."""
        dialog = self._preferences_dialog
        self._preferences_dialog = None
        self._preferences_ready_at = None
        if dialog is None:
            return
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass

    def _timed_out(self) -> None:
        """Fail the run, naming the module that did not become ready."""
        if self._finished or self._timeout_pending:
            return
        if self.phase == "home":
            detail = "__home__"
        elif self.phase == "preferences":
            detail = "__preferences__"
        else:
            detail = str(self.current_key)
        self._timeout_pending = True
        timing.cancel_interactive(detail=str(detail))
        # Let an overdue watchdog beat run before sealing the failed interval.
        # Readiness is rejected while this is pending, so the deadline stays
        # decisive even when a paint was queued behind the same long block.
        QTimer.singleShot(
            SETTLE_MS,
            lambda: self._record_error(
                detail,
                f"no painted usable state within {self._armed_timeout_s:.1f} seconds",
                already_stopped=True,
            ),
        )

    def _record_error(self, detail: str, message: str, *,
                      already_stopped: bool = False) -> None:
        # ``already_stopped`` means the Qt single-shot has fired; its wall
        # timer is independent and must still be cancelled before this method
        # checkpoints or advances.
        """Record a failure without losing the timings already taken.

        :param detail: what was being done.
        :param message: what went wrong.
        :param already_stopped: whether the run had already been halted.
        """
        del already_stopped
        self._disarm_timeout()
        self._timeout_pending = False
        timing.cancel_interactive(detail=str(detail))
        duration = (
            timing.elapsed() if self.phase == "home"
            else max(0.0, time.perf_counter() - self._attempt_started)
        )
        attempt_started_elapsed = (
            0.0 if self.phase == "home" else self._attempt_started_elapsed
        )
        state = timing.snapshot()
        interval_stalls = timing.stalls_between(
            attempt_started_elapsed, float(state["elapsed_s"]), state["stalls"])
        worst_stall = max(
            (float(row["overlap_ms"]) for row in interval_stalls), default=0.0)
        raw_worst = max(
            (float(row["late_ms"]) for row in interval_stalls), default=0.0)
        self.results.append({
            "name": (
                "interactive Home" if self.phase == "home"
                else "interactive preferences" if self.phase == "preferences"
                else "interactive module"
            ),
            "detail": str(detail),
            "duration_s": duration,
            "budget_s": (
                timing.HOME_BUDGET_S if self.phase == "home"
                else PREFERENCES_BUDGET_S if self.phase == "preferences"
                else timing.MODULE_BUDGET_S
            ),
            "within_budget": False,
            "stall_window_started_at": attempt_started_elapsed,
            "stall_window_ended_at": float(state["elapsed_s"]),
            "worst_event_loop_stall_ms": worst_stall,
            "worst_overlapping_frame_interval_ms": raw_worst,
            "event_loop_stall_budget_met": worst_stall < timing.STALL_BUDGET_MS,
            "stall_samples": len(interval_stalls),
            "error": str(message),
        })
        self._checkpoint()
        print(f"benchmark failed: {detail}: {message}", flush=True)
        if self.phase == "home":
            # Without a usable Home, no click path exists to benchmark.  Do
            # not disguise that by calling private factories instead.
            self._finish("Home never became interactive")
            return
        if self.phase == "preferences":
            self._close_preferences_dialog()
            self.phase = "module"
            self._advance_after_watchdog()
            return
        self.index += 1
        # The screen can finish painting after its overdue timeout has been
        # delivered.  Clear the key before the next settled advance so
        # that late readiness cannot terminate this attempt a second time and
        # skip the following registry row.
        self.current_key = None
        self._advance_after_watchdog()

    def _current_registry_keys(self) -> tuple[str, ...]:
        """The modules the registry holds right now.

        :returns: the keys.
        """
        if self._live_keys is None:
            return self.keys
        return tuple(str(key) for key in self._live_keys())

    def _violations(self, current_keys: Iterable[str]) -> list[str]:
        """Modules that appeared or vanished during the run.

        A REGISTRY THAT MOVES INVALIDATES THE COMPARISON: the benchmark times
        a fixed list, and a module registered halfway through means the
        numbers describe two different applications.

        :param current_keys: the keys as they are now.
        :returns: the discrepancies.
        """
        violations: list[str] = []
        final_keys = tuple(str(key) for key in current_keys)
        if final_keys != self.keys:
            violations.append(
                "the live registry changed during the benchmark sweep")
        measured = [
            str(row.get("detail")) for row in self.results
            if row.get("detail") not in {"__home__", "__preferences__"}
        ]
        if measured != list(self.keys):
            violations.append(
                "measured app sequence does not equal the live registry exactly")
        by_detail = {str(row.get("detail")): row for row in self.results}
        expected = (
            ("__home__", "__preferences__", *self.keys)
            if self._measure_preferences else ("__home__", *self.keys)
        )
        for detail in expected:
            row = by_detail.get(detail)
            if row is None:
                violations.append(f"{detail}: missing readiness record")
                continue
            if row.get("error"):
                violations.append(f"{detail}: {row['error']}")
            if row.get("within_budget") is not True:
                violations.append(
                    f"{detail}: no readiness record meeting the "
                    f"{row.get('budget_s')} s budget")
            if row.get("event_loop_stall_budget_met") is not True:
                violations.append(
                    f"{detail}: event-loop stall reached the 500 ms ceiling")
        return violations

    def _artifact(self, exit_reason: str) -> dict:
        """Assemble the run's results into the artifact structure.

        :param exit_reason: why the run ended.
        :returns: the artifact.
        """
        artifact = timing.snapshot()
        final_keys = self._current_registry_keys()
        measured = [
            str(row.get("detail")) for row in self.results
            if row.get("detail") not in {"__home__", "__preferences__"}
        ]
        artifact["benchmark"] = {
            "run": os.environ.get(RUN_LABEL_ENV, "benchmark"),
            "exit_reason": str(exit_reason),
            "registry_keys": list(self.keys),
            "registry_count": len(self.keys),
            "final_registry_keys": list(final_keys),
            "registry_stable": final_keys == self.keys,
            "measured_keys": measured,
            "measured_count": len(measured),
            "registry_matches_measurements": measured == list(self.keys),
            "preferences_measured": self._measure_preferences,
            "preferences_budget_s": PREFERENCES_BUDGET_S,
            "results": list(self.results),
            "violations": self._violations(final_keys),
        }
        return artifact

    def _persist(self, exit_reason: str) -> str:
        """Atomically replace the artifact; return an error message or ``""``."""
        temporary = f"{self.output}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.output)),
                        exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self._artifact(exit_reason), handle,
                          indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.output)
        except Exception as error:                           # noqa: BLE001
            try:
                os.unlink(temporary)
            except OSError:
                pass
            return str(error)
        return ""

    def _write(self, exit_reason: str) -> None:
        """Write the artifact to disk.

        :param exit_reason: why the run ended.
        """
        if self._written:
            return
        error = self._persist(exit_reason)
        if error:
            print(f"could not write spaCR benchmark artifact: {error}")
            return
        self._written = True

    def _checkpoint(self) -> None:
        """Preserve completed states even if a later screen kills the process."""
        self._persist("registry sweep in progress")

    def _finish(self, reason: str = "registry sweep complete") -> None:
        """Write the artifact and stop the application.

        :param reason: why the run ended.
        """
        if self._finished:
            return
        self._finished = True
        self._disarm_timeout()
        self._close_preferences_dialog()
        timing.unsubscribe_readiness(self._ready)
        self._write(reason)
        self.app.quit()

    def _application_quit(self) -> None:
        """Quit, whether the run succeeded or failed."""
        if not self._written:
            self._write("application quit before registry sweep completed")


def maybe_start(app, window) -> Optional[BenchmarkController]:
    """Install the controller named by the environment, or return ``None``."""
    output = os.environ.get(OUTPUT_ENV, "").strip()
    if not output:
        return None
    from .app import APPS

    def _live_keys() -> tuple[str, ...]:
        """Every registered app key, read when the benchmark runs."""
        return tuple(key for key, _name, _description, _section in APPS)

    keys = _live_keys()
    if len(keys) != len(set(keys)):
        raise ValueError("the live application registry contains duplicate keys")
    return BenchmarkController(
        app, window, keys, output, live_keys=_live_keys)
