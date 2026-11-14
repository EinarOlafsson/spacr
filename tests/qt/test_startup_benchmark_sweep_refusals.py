"""The start-up benchmark's sweep when a screen, the stream or the clock disagrees.

``BenchmarkController`` subscribes to :mod:`spacr.qt.timing`'s readiness
stream, seals each interval at a watchdog beat, presses each registry key's one
live door, and records a refusal (never a skip) when a key cannot be measured.
``_ready`` is called directly with a readiness dict of ``timing``'s own shape:
it is the callback ``subscribe_readiness`` installs, and the production
publisher needs a really painted screen tree.  The happy path lives in
``test_the_startup_benchmark_writes_what_it_measured``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog, QPushButton, QWidget  # noqa: E402

from spacr.qt import startup_benchmark as sb  # noqa: E402
from spacr.qt import timing  # noqa: E402

pytestmark = pytest.mark.qt

WAIT_MS = 3000


def _entry(**overrides):
    """A readiness record for Home, as ``timing`` publishes one."""
    entry = {"detail": "__home__", "name": "interactive Home",
             "started_at": 0.0, "at": 0.0, "duration_s": 0.25,
             "budget_s": timing.HOME_BUDGET_S, "within_budget": True}
    entry.update(overrides)
    return entry


def _nav_button(key, *, enabled=True, cls=QPushButton, fold_child=False):
    button = cls()
    button.setProperty("navKey", key)
    button.setProperty("isFoldChild", fold_child)
    button.setEnabled(enabled)
    return button


class _ExplodingButton(QPushButton):
    """A live button whose press raises, as a broken screen factory does."""

    def click(self):  # noqa: D102
        raise RuntimeError("the screen would not build")


@pytest.fixture
def window(qtbot):
    """A window with the one attribute the controller reads: ``_sidebar``."""
    holder = QWidget()
    qtbot.addWidget(holder)
    holder._sidebar = SimpleNamespace(_items=[])
    return holder


@pytest.fixture
def make_controller(qapp, tmp_path):
    """Build controllers and retire each one however the test ends."""
    made = []

    def build(window, keys=("mask",), timeout_s=5.0, name="benchmark.json",
              **kwargs):
        kwargs.setdefault("measure_preferences", False)
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
    with open(controller.output, encoding="utf-8") as handle:
        return json.load(handle)["benchmark"]


def test_ready_seals_only_the_expected_record_once(make_controller, window,
                                                   qtbot, monkeypatch,
                                                   subtests):
    """Another probe's record, a second record and a finished run are all ignored."""
    with subtests.test(msg="another probe's record is not Home's"):
        controller = make_controller(window, keys=())
        controller._ready(_entry(name="interactive module"))
        controller._ready(_entry(detail="mask"))
        assert controller._pending is None and controller.results == []
        controller._ready(_entry())
        qtbot.waitUntil(lambda: bool(controller.results), timeout=WAIT_MS)
        assert [(row["name"], row["detail"]) for row in controller.results] \
            == [("interactive Home", "__home__")]
        assert controller.phase == "module"
        assert "door" not in controller.results[0]

    with subtests.test(msg="Home goes on to Preferences when asked"):
        dialogs = []

        def factory():
            dialog = QDialog(window)
            dialogs.append(dialog)
            return dialog

        controller = make_controller(window, keys=(), name="prefs.json",
                                     measure_preferences=True,
                                     preferences_factory=factory)
        controller._ready(_entry())
        qtbot.waitUntil(lambda: bool(controller.results), timeout=WAIT_MS)
        assert controller.phase == "preferences"
        qtbot.waitUntil(lambda: controller._finished, timeout=WAIT_MS)
        assert [(row["detail"], row["name"], row["budget_s"])
                for row in controller.results] == [
            ("__home__", "interactive Home", timing.HOME_BUDGET_S),
            ("__preferences__", "interactive preferences",
             sb.PREFERENCES_BUDGET_S)]
        assert len(dialogs) == 1 and controller._preferences_dialog is None
        assert _artifact(controller)["preferences_measured"] is True

    with subtests.test(msg="a second record while one settles is dropped"):
        controller = make_controller(window, keys=(), name="second.json")
        controller._ready(_entry(duration_s=0.25))
        held = controller._pending
        controller._ready(_entry(duration_s=99.0))
        assert controller._pending is held
        qtbot.waitUntil(lambda: bool(controller.results), timeout=WAIT_MS)
        assert [row["duration_s"] for row in controller.results] == [0.25]

    with subtests.test(msg="a run that finished first keeps it out"):
        ended = make_controller(window, name="ended.json")
        ended._ready(_entry())
        assert ended._pending is not None
        ended._finish("stopped while a paint was settling")
        qtbot.wait(sb.SETTLE_MS * 4)
        assert ended.results == []
        assert _artifact(ended)["exit_reason"] == (
            "stopped while a paint was settling")
        ended._ready(_entry())
        assert ended.results == []

    with subtests.test(msg="sealed only once the watchdog passes the paint"):
        beat = {"at": 1.0}
        monkeypatch.setattr(timing, "last_gui_beat_at", lambda: beat["at"])
        controller = make_controller(window, keys=(), name="beat.json")
        controller._ready(_entry(at=5.0))
        qtbot.wait(sb.SETTLE_MS * 4)
        assert controller._pending is not None and controller.results == []
        beat["at"] = 5.0
        qtbot.wait(sb.SETTLE_MS * 3)
        assert controller.results == []
        beat["at"] = 9.0
        qtbot.waitUntil(lambda: bool(controller.results), timeout=WAIT_MS)
        row = controller.results[0]
        assert controller._pending is None
        assert (row["at"], row["stall_window_started_at"],
                row["stall_window_ended_at"]) == (5.0, 0.0, 9.0)
        assert (row["worst_event_loop_stall_ms"], row["stall_samples"],
                row["event_loop_stall_budget_met"]) == (0.0, 0, True)


def test_each_key_is_opened_through_its_one_door_or_refused(
        make_controller, window, qtbot, subtests):
    """Two rows, no row, a disabled or raising row each give an exact result row."""
    # case, sidebar items, palette?, error (None = opened), door
    cases = [
        ("two live buttons", lambda: [_nav_button("mask"),
                                      _nav_button("mask")], False,
         "expected one live sidebar button, found 2", "sidebar"),
        ("disabled button", lambda: [_nav_button("mask", enabled=False)],
         False, "the live sidebar button is disabled", "sidebar"),
        ("button that raises",
         lambda: [_nav_button("mask", cls=_ExplodingButton)], False,
         "RuntimeError: the screen would not build", "sidebar"),
        ("no row and no other door", lambda: [], False,
         "AttributeError", "command palette"),
        ("no row: the command palette", lambda: [], True, None,
         "command palette"),
        ("one row", lambda: [_nav_button("other"), _nav_button("mask")],
         False, None, "sidebar"),
        ("a registry row wins over its fold child",
         lambda: [_nav_button("mask", fold_child=True),
                  _nav_button("mask")], False, None, "sidebar"),
    ]
    for index, (case, items, palette, error, door) in enumerate(cases):
        with subtests.test(msg=case):
            window._sidebar._items = items()
            pressed, navigated = [], []
            for button in window._sidebar._items:
                button.clicked.connect(
                    lambda _c=False, b=button: pressed.append(b))
            if palette:
                window._on_nav_selected = navigated.append
            elif hasattr(window, "_on_nav_selected"):
                del window._on_nav_selected
            controller = make_controller(window, keys=("mask",),
                                         name=f"door{index}.json")
            controller._ready(_entry())
            if error is None:
                qtbot.waitUntil(lambda: bool(pressed or navigated),
                                timeout=WAIT_MS)
                assert controller._door == door
                assert controller.current_key == "mask"
                assert navigated == (["mask"] if palette else [])
                if not palette:
                    rows = [b for b in window._sidebar._items
                            if b.property("navKey") == "mask"
                            and not b.property("isFoldChild")]
                    assert pressed == rows
                continue
            qtbot.waitUntil(lambda: controller._finished, timeout=WAIT_MS)
            assert pressed == []
            failure = controller.results[-1]
            assert (failure["detail"], failure["name"], failure["door"],
                    failure["within_budget"], failure["budget_s"]) == (
                "mask", "interactive module", door, False,
                timing.MODULE_BUDGET_S)
            if error == "AttributeError":
                assert failure["error"].startswith("AttributeError: ")
                assert "_on_nav_selected" in failure["error"]
            else:
                assert failure["error"] == error
            artifact = _artifact(controller)
            assert artifact["exit_reason"] == "registry sweep complete"
            assert artifact["measured_keys"] == ["mask"]
            assert f"mask: {failure['error']}" in artifact["violations"]


def test_a_deadline_names_the_state_it_expired_in(make_controller, window,
                                                  qtbot, subtests):
    """Home, Preferences and a module each expire as their own row, once."""
    with subtests.test(msg="Home never paints: the sweep ends"):
        window._sidebar._items = [_nav_button("mask")]
        controller = make_controller(window, keys=("mask",), timeout_s=0.001,
                                     name="home.json")
        qtbot.waitUntil(lambda: controller._finished, timeout=WAIT_MS)
        row = controller.results[-1]
        assert (row["detail"], row["name"], row["error"],
                row["budget_s"]) == (
            "__home__", "interactive Home",
            "no painted usable state within 0.0 seconds",
            timing.HOME_BUDGET_S)
        assert "door" not in row
        artifact = _artifact(controller)
        assert artifact["exit_reason"] == "Home never became interactive"
        assert artifact["measured_keys"] == []
        controller._timed_out()
        qtbot.wait(sb.SETTLE_MS * 3)
        assert len(controller.results) == 1

    # phase, current key, expected detail, name, budget
    phases = [
        ("module", "mask", "mask", "interactive module",
         timing.MODULE_BUDGET_S),
        ("preferences", None, "__preferences__", "interactive preferences",
         sb.PREFERENCES_BUDGET_S),
    ]
    for phase, key, detail, name, budget in phases:
        with subtests.test(msg=f"expiry in the {phase} phase"):
            controller = make_controller(
                window, keys=("mask",), timeout_s=5.0, name=f"{phase}.json",
                measure_preferences=phase == "preferences")
            controller.phase = phase
            controller.current_key = key
            controller._timed_out()
            assert controller._timeout_pending is True
            # A late paint cannot reopen the expiring state.
            controller._ready(_entry(detail="mask",
                                     name="interactive module"))
            assert controller._pending is None
            controller._timed_out()
            qtbot.waitUntil(lambda: bool(controller.results), timeout=WAIT_MS)
            qtbot.wait(sb.SETTLE_MS * 2)
            rows = [r for r in controller.results if r["detail"] == detail]
            assert len(rows) == 1
            assert (rows[0]["name"], rows[0]["error"], rows[0]["budget_s"],
                    rows[0]["within_budget"]) == (
                name, "no painted usable state within 5.0 seconds", budget,
                False)
            assert controller._timeout_pending is False
