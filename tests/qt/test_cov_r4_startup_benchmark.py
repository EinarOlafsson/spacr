"""What the start-up benchmark does when a screen will not behave.

The happy path -- Home paints, every sidebar button is pressed, one JSON
artifact is written -- is held by ``test_startup_benchmark_driver`` and by
``test_the_startup_benchmark_writes_what_it_measured``. This file holds the
other half: the controller's behaviour when the registry, the sidebar, the
deadline or the readiness stream disagree with it.

Each of these was a way for the benchmark to report a number that was not
measured -- a screen skipped because its button was missing, a screen counted
twice because a late paint arrived after its deadline, an artifact rewritten
by a second finish with a different reason. The controller records a refusal
instead, and these pin the refusal.

``_ready`` is exercised by calling it with a readiness dict of
:mod:`spacr.qt.timing`'s own shape: it is the callback ``subscribe_readiness``
installs, and the production publisher needs a really painted screen tree,
which is what the driver test launches an application for.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QPushButton, QWidget            # noqa: E402

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
        "duration_s": 0.25,
        "budget_s": timing.HOME_BUDGET_S,
        "within_budget": True,
    }
    entry.update(overrides)
    return entry


def _nav_button(key, *, enabled=True, cls=QPushButton):
    button = cls()
    button.setProperty("navKey", key)
    button.setEnabled(enabled)
    return button


class ExplodingButton(QPushButton):
    """A live button whose press raises, as a broken screen factory does."""

    def click(self):                                          # noqa: D102
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
    """Build controllers and retire each one before the next test runs."""
    made = []

    def build(window, keys=("mask",), timeout_s=5.0, name="benchmark.json"):
        controller = sb.BenchmarkController(
            qapp, window, keys, str(tmp_path / name), timeout_s=timeout_s)
        made.append(controller)
        return controller

    yield build

    for controller in made:
        controller._finished = True
        controller.timeout.stop()
        timing.unsubscribe_readiness(controller._ready)


def _artifact(controller):
    return json.loads(open(controller.output, encoding="utf-8").read())


# ---------------------------------------------------------------------------
# the readiness stream
# ---------------------------------------------------------------------------

def test_a_readiness_record_for_another_probe_is_not_home(make_controller,
                                                          window, qtbot):
    """The detail matches but the name does not, so it is a different probe.

    ``timing`` publishes every readiness record to every subscriber. Home's
    interval may only be sealed by Home's own record; accepting one that
    merely carried the right detail would time the wrong screen.
    """
    controller = make_controller(window)

    controller._ready(_home_entry(name="interactive module"))
    assert controller._pending is None
    assert controller.results == []

    controller._ready(_home_entry())

    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    assert controller.results[0]["name"] == "interactive Home"
    assert controller.phase == "module"


def test_a_second_readiness_while_one_settles_is_dropped(make_controller,
                                                         window, qtbot):
    """Two records for one interval would measure one screen twice."""
    controller = make_controller(window)

    controller._ready(_home_entry(duration_s=0.25))
    assert controller._pending is not None
    controller._ready(_home_entry(duration_s=99.0))

    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    assert [row["duration_s"] for row in controller.results] == [0.25]


def test_a_run_that_finished_first_keeps_the_settling_record_out(
        make_controller, window, qtbot):
    """A record still settling when the sweep ends is not appended late.

    Both halves are driven here: the same record lands when the run is live,
    and is dropped when the run has already written its artifact.
    """
    live = make_controller(window)
    live._ready(_home_entry())
    qtbot.waitUntil(lambda: bool(live.results), timeout=SETTLE_WAIT_MS)
    assert len(live.results) == 1

    ended = make_controller(window, name="ended.json")
    ended._ready(_home_entry())
    assert ended._pending is not None
    ended._finish("stopped while a paint was settling")

    qtbot.wait(sb.SETTLE_MS * 4)
    assert ended.results == []
    assert _artifact(ended)["benchmark"]["exit_reason"] == (
        "stopped while a paint was settling")


# ---------------------------------------------------------------------------
# the sidebar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nav_keys,found", [(("measure",), 0),
                                            (("mask", "mask"), 2)])
def test_a_key_without_exactly_one_live_button_is_recorded(make_controller,
                                                           window, qtbot,
                                                           nav_keys, found):
    """A registry key with no button, or two, is a refusal -- not a skip.

    The benchmark exists to press the buttons a user presses. If a key were
    quietly skipped the artifact would report a complete sweep of a registry
    it never touched.
    """
    window._sidebar._items = [_nav_button(key) for key in nav_keys]
    controller = make_controller(window, keys=("mask",))

    controller._ready(_home_entry())

    qtbot.waitUntil(lambda: controller._finished, timeout=SETTLE_WAIT_MS)
    failure = controller.results[-1]
    assert failure["detail"] == "mask"
    assert failure["error"] == (
        f"expected one live sidebar button, found {found}")
    assert failure["within_budget"] is False
    violations = _artifact(controller)["benchmark"]["violations"]
    assert any("expected one live sidebar button" in line
               for line in violations)


def test_a_disabled_button_is_reported_rather_than_pressed(make_controller,
                                                           window, qtbot):
    """Clicking a disabled button measures nothing and reports success."""
    pressed = []
    disabled = _nav_button("mask", enabled=False)
    disabled.clicked.connect(lambda: pressed.append("mask"))
    window._sidebar._items = [disabled]
    controller = make_controller(window, keys=("mask",))

    controller._ready(_home_entry())
    qtbot.waitUntil(lambda: controller._finished, timeout=SETTLE_WAIT_MS)

    assert controller.results[-1]["error"] == (
        "the live sidebar button is disabled")
    assert pressed == []

    # The same button, enabled, is really clicked: the controller presses the
    # live control rather than calling the screen factory behind it.
    live = _nav_button("mask")
    live.clicked.connect(lambda: pressed.append("mask"))
    window._sidebar._items = [live]
    second = make_controller(window, keys=("mask",), name="enabled.json")
    second._ready(_home_entry())
    qtbot.waitUntil(lambda: pressed == ["mask"], timeout=SETTLE_WAIT_MS)


def test_a_button_that_raises_names_the_exception(make_controller, window,
                                                  qtbot):
    """A screen that dies on construction is one failed key, not a dead run."""
    window._sidebar._items = [_nav_button("mask", cls=ExplodingButton)]
    controller = make_controller(window, keys=("mask",))

    controller._ready(_home_entry())

    qtbot.waitUntil(lambda: controller._finished, timeout=SETTLE_WAIT_MS)
    assert controller.results[-1]["error"] == (
        "RuntimeError: the screen would not build")
    assert _artifact(controller)["benchmark"]["exit_reason"] == (
        "registry sweep complete")


# ---------------------------------------------------------------------------
# the deadline
# ---------------------------------------------------------------------------

def test_a_home_that_never_paints_ends_the_sweep(make_controller, window,
                                                 qtbot):
    """Without a usable Home there is no click path left to benchmark.

    The deadline seals the interval, the artifact says Home never became
    interactive, and the deadline cannot fire twice into the same run.
    """
    window._sidebar._items = [_nav_button("mask")]
    controller = make_controller(window, keys=("mask",), timeout_s=0.001)

    qtbot.waitUntil(lambda: controller._finished, timeout=SETTLE_WAIT_MS)

    assert controller.results[-1]["detail"] == "__home__"
    assert controller.results[-1]["error"].startswith(
        "no painted usable state within")
    artifact = _artifact(controller)["benchmark"]
    assert artifact["exit_reason"] == "Home never became interactive"
    assert artifact["measured_keys"] == []

    # A second expiry into a finished run adds nothing.
    controller._timed_out()
    qtbot.wait(sb.SETTLE_MS * 3)
    assert len(controller.results) == 1


def test_a_late_paint_cannot_reopen_an_expired_screen(make_controller, window,
                                                      qtbot):
    """Readiness is refused while the sealing of a timeout is in flight."""
    window._sidebar._items = [_nav_button("mask")]
    controller = make_controller(window, keys=("mask",), timeout_s=5.0)
    controller.phase = "module"
    controller.current_key = "mask"

    controller._timed_out()
    assert controller._timeout_pending is True
    controller._ready(_home_entry(detail="mask", name="interactive module"))
    assert controller._pending is None

    qtbot.waitUntil(lambda: bool(controller.results), timeout=SETTLE_WAIT_MS)
    assert controller.results[0]["error"].startswith("no painted usable state")


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------

def test_a_second_finish_does_not_restate_the_reason(make_controller, window,
                                                     tmp_path):
    """``_finish`` latches, so a later reason cannot overwrite the first.

    ``_written`` is cleared first on purpose: without the latch in ``_finish``
    the write guard alone would let the second call replace the artifact, and
    the test would not be able to tell the two guards apart.
    """
    controller = make_controller(window)
    controller._finish("registry sweep complete")

    controller._written = False
    controller._finish("something else entirely")

    assert _artifact(controller)["benchmark"]["exit_reason"] == (
        "registry sweep complete")


def test_quitting_twice_leaves_the_first_artifact(make_controller, window):
    """The quit handler writes once; a second quit is not a second reason."""
    controller = make_controller(window)

    controller._application_quit()
    first = _artifact(controller)["benchmark"]["exit_reason"]
    assert first == "application quit before registry sweep completed"

    controller.results.append({"detail": "mask", "error": "later"})
    controller._application_quit()

    artifact = _artifact(controller)["benchmark"]
    assert artifact["exit_reason"] == first
    assert artifact["results"] == []


def test_an_artifact_path_that_cannot_be_made_reports_and_leaves_nothing(
        make_controller, window, tmp_path):
    """The folder is a file, so neither the artifact nor its temporary exist."""
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    controller = make_controller(window)
    controller.output = str(blocker / "benchmark.json")

    error = controller._persist("registry sweep complete")

    assert "not-a-folder" in error
    assert blocker.read_text(encoding="utf-8") == "x"
    assert [p.name for p in tmp_path.iterdir()] == ["not-a-folder"]


# ---------------------------------------------------------------------------
# installing the controller
# ---------------------------------------------------------------------------

def test_no_output_path_installs_no_controller(qapp, window, tmp_path,
                                               monkeypatch):
    """The ordinary application must pay nothing for the benchmark."""
    monkeypatch.setenv(sb.OUTPUT_ENV, "   ")

    assert sb.maybe_start(qapp, window) is None

    monkeypatch.setenv(sb.OUTPUT_ENV, str(tmp_path / "live.json"))
    controller = sb.maybe_start(qapp, window)
    try:
        from spacr.qt.app import APPS

        assert controller is not None
        assert controller.keys == tuple(key for key, *_rest in APPS)
    finally:
        controller._finished = True
        controller.timeout.stop()
        timing.unsubscribe_readiness(controller._ready)


def test_a_registry_with_a_duplicate_key_is_refused(qapp, window, tmp_path,
                                                    monkeypatch):
    """Two rows for one key would measure one screen and report two."""
    monkeypatch.setenv(sb.OUTPUT_ENV, str(tmp_path / "dup.json"))
    monkeypatch.setattr(
        "spacr.qt.app.APPS",
        [("mask", "Mask", "make masks", "Segment"),
         ("mask", "Mask again", "the same key", "Segment")])

    with pytest.raises(ValueError, match="duplicate keys"):
        sb.maybe_start(qapp, window)

    assert not os.path.exists(str(tmp_path / "dup.json"))
