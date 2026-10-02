"""Calibrate the added item 380 instruments without claiming native signoff."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def harness():
    """Load the same executable harness used for diagnostic receipts."""
    path = Path(__file__).resolve().parents[2] / "tools" / "perf_paint.py"
    spec = importlib.util.spec_from_file_location("spacr_missing_perf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observer_sees_an_injected_gui_stall(qtbot, qapp, harness):
    """A known input stall must appear in independent event-loop sampling."""
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    widget.show()
    qapp.processEvents()

    def block():
        """Inject a known GUI-thread defect into the measured action."""
        time.sleep(.06)

    row = harness._observe_activity(qapp, widget, .18, block)
    assert row["input_dispatch_and_events_ms"] >= 55
    assert row["max_event_loop_gap_ms"] >= 55
    assert "physical presented/dropped frames unknown" in row["frame_scope"]


def test_a_failed_input_witness_fails_the_instrument(qtbot, qapp, harness):
    """A failed action must not be serialized as fast successful input."""
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)

    def failed():
        """Represent an input whose expected application state never appeared."""
        raise RuntimeError("no state change")

    with pytest.raises(RuntimeError, match="no state change"):
        harness._observe_activity(qapp, widget, .12, failed)


def test_pointer_actions_have_real_state_witnesses(qtbot, qapp, harness):
    """Exercise the actual module splitter and canonical Home-grid hover."""
    from spacr.qt import preferences

    with harness._settings_elsewhere():
        preferences._set_tooltip_delay(2.0)
        rows = harness.measure_pointer_interaction("mask")
        assert preferences._get_tooltip_delay() == 2.0
    drags = [r for r in rows if r["action"] == "drag splitter"]
    hovers = [r for r in rows if r["action"] == "hover Home grid"]
    assert len(drags) == len(hovers) == 3
    assert all(r["pane_sizes_before"] != r["pane_sizes_after"] for r in drags)
    assert len({r["module"] for r in hovers}) == 2
    assert all(r["configured_hint_delay_ms"] == 2000 for r in hovers)
    assert all(r["hint_witness_elapsed_ms"] >= 2000 for r in hovers)
    assert all(r["input_dispatch_and_events_ms"] is not None for r in rows)


def test_hidden_splitter_is_rejected(qtbot, qapp, harness):
    """Unexercised hidden controls cannot satisfy pointer acceptance."""
    from PySide6.QtWidgets import QSplitter, QWidget

    splitter = QSplitter()
    qtbot.addWidget(splitter)
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    with pytest.raises(RuntimeError, match="not visible"):
        harness._drag_splitter(splitter, qapp)


def test_environment_uses_actual_qt_backend_and_display(qapp, harness):
    """Environment metadata must describe the running Qt platform plugin."""
    row = harness._environment()
    assert row["qt_platform"] == qapp.platformName()
    assert row["displays"]
    assert row["displays"][0]["logical_width"] > 0
    assert "no physical display or lower-end signoff" in row["acceptance_scope"]


def test_black_screensaver_fallback_cannot_pass(qapp, monkeypatch, harness):
    """The real screensaver's missing-renderer fallback is not an FPS result."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt import screensaver

    class BlackFallback(QWidget):
        """Represent the documented no-renderer fallback without a GPU job."""

        def __init__(self):
            """Keep the real fallback state visible to the harness."""
            super().__init__()
            self._backdrop = None

    monkeypatch.setattr(screensaver, "Screensaver", BlackFallback)
    with harness._settings_elsewhere():
        with pytest.raises(RuntimeError, match="no measurable CPU backdrop"):
            harness.measure_screensaver(.1, ["dark"])
