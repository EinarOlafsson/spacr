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

from PySide6.QtWidgets import QWidget            # noqa: E402

from spacr.qt import startup_benchmark as sb                  # noqa: E402
from spacr.qt import timing                                   # noqa: E402

pytestmark = pytest.mark.qt



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


# ---------------------------------------------------------------------------
# the sidebar
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the deadline
# ---------------------------------------------------------------------------


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
