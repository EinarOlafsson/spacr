"""``BenchmarkController``'s deadline, and the artifact it must not lose.

The benchmark exists because every earlier measurement of spaCR's start-up was
taken by a script that imitated the application. This controller clicks the
real sidebar buttons in the real window, so the thing it produces is the only
evidence there is -- and a benchmark that loses its artifact when a screen
kills the process has measured nothing.

Two behaviours are covered here that the driver-level tests cannot reach,
because they are about the controller's own bookkeeping rather than about a
launch: how the deadline is read from the environment, and what happens to the
file when writing it fails.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt import startup_benchmark as sb                        # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# the deadline
# ---------------------------------------------------------------------------

def test_the_default_deadline_is_used_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv(sb.TIMEOUT_ENV, raising=False)

    assert sb._timeout_seconds() == pytest.approx(sb.DEFAULT_TIMEOUT_S)


@pytest.mark.parametrize("raw,expected", [("45", 45.0), ("1.5", 1.5)])
def test_a_deadline_in_the_environment_is_honoured(monkeypatch, raw, expected):
    monkeypatch.setenv(sb.TIMEOUT_ENV, raw)

    assert sb._timeout_seconds() == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [("0", 1.0), ("-30", 1.0),
                                          ("100000", 300.0)])
def test_an_absurd_deadline_is_clamped_rather_than_obeyed(monkeypatch, raw,
                                                          expected):
    """A zero or negative deadline fires before the window exists.

    The benchmark would then report every screen as timed out, which looks
    exactly like a catastrophic regression and is a typo in an environment
    variable. The ceiling is the other half: an unattended run that waits five
    minutes per screen is a hung CI job.
    """
    monkeypatch.setenv(sb.TIMEOUT_ENV, raw)

    assert sb._timeout_seconds() == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "soon", "10s", "None"])
def test_a_deadline_that_is_not_a_number_falls_back_to_the_default(monkeypatch,
                                                                   raw):
    """An unparseable value must not take the run down before it starts.

    This is a diagnostic tool; refusing to launch because its optional
    tuning knob was mistyped would cost the measurement the user asked for.
    """
    monkeypatch.setenv(sb.TIMEOUT_ENV, raw)

    assert sb._timeout_seconds() == pytest.approx(sb.DEFAULT_TIMEOUT_S)


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------

@pytest.fixture
def controller(qapp, qtbot, tmp_path, monkeypatch):
    """A controller wired to a real QApplication and a bare window."""
    from PySide6.QtWidgets import QWidget

    window = QWidget()
    qtbot.addWidget(window)
    made = sb.BenchmarkController(qapp, window, ("mask", "measure"),
                                  str(tmp_path / "benchmark.json"),
                                  timeout_s=5.0)
    yield made
    made.timeout.stop()
    from spacr.qt import timing
    timing.unsubscribe_readiness(made._ready)


def test_the_artifact_is_replaced_atomically_not_written_in_place(controller,
                                                                  tmp_path):
    """A partial JSON file is worse than none: the reader parses it.

    The write goes to a temporary beside the target and is then renamed, so a
    process killed mid-write leaves the previous complete artifact rather than
    half of the new one.
    """
    error = controller._persist("registry sweep complete")

    assert error == ""
    written = json.loads((tmp_path / "benchmark.json").read_text())
    assert written["benchmark"]["exit_reason"] == "registry sweep complete"
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.name != "benchmark.json"]
    assert leftovers == [], f"a temporary was left behind: {leftovers}"


def test_a_failed_write_reports_why_and_leaves_no_temporary(controller,
                                                            tmp_path,
                                                            monkeypatch):
    """The cleanup is the point, not the message.

    A benchmark run writes a checkpoint after every screen. If a failed write
    left its temporary behind, an unattended sweep over thirty screens would
    litter the output folder with thirty partial files -- and the next reader
    has no way to tell them from the artifact.
    """
    def refuse(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(json, "dump", refuse)

    error = controller._persist("registry sweep complete")

    assert "no space left" in error
    assert not (tmp_path / "benchmark.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_does_not_mark_the_artifact_written(controller,
                                                           monkeypatch, capsys):
    """``_written`` is what stops a second attempt, so a failure must not set it.

    Marking it written on failure means the quit handler skips its last-chance
    write, and a run that hit one transient error produces no artifact at all.
    """
    monkeypatch.setattr(controller, "_persist", lambda reason: "disk gone")

    controller._write("registry sweep complete")

    assert controller._written is False
    assert "could not write" in capsys.readouterr().out


def test_a_successful_write_is_not_repeated(controller, tmp_path):
    """The quit handler and the normal finish both call it."""
    controller._write("registry sweep complete")
    assert controller._written is True
    first = (tmp_path / "benchmark.json").stat().st_mtime_ns

    controller._write("a different reason entirely")

    assert (tmp_path / "benchmark.json").stat().st_mtime_ns == first
    written = json.loads((tmp_path / "benchmark.json").read_text())
    assert written["benchmark"]["exit_reason"] == "registry sweep complete"


def test_quitting_early_still_leaves_an_artifact_saying_so(controller,
                                                           tmp_path):
    """A run the user closed is a result, not a missing file.

    Without this the artifact simply would not exist, and a reader cannot tell
    "the benchmark never ran" from "somebody closed the window at screen 4".
    """
    controller._application_quit()

    written = json.loads((tmp_path / "benchmark.json").read_text())
    assert written["benchmark"]["exit_reason"] == (
        "application quit before registry sweep completed")
