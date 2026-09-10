"""The performance harness is calibrated against a defect already fixed.

Instruction 380's first instruction: "WRITE THE HARNESS FIRST, and prove
it catches a defect that is already known before trusting it on anything
else -- 350 did exactly this against the Annotate instance and it is why
that sweep is believable. The one-backdrop regression is the ideal
calibration case: a harness that cannot see 950 paints/s for an invisible
layer is not measuring paint."

So this file does not test spaCR. It tests the INSTRUMENT: two backdrop
layers must read as about twice the paints of one, an interaction that
blocks must read as dropped frames, and the widget count must count the
widgets that exist rather than the ones that happen to be children of the
application object.

WHY NOT `QWidget.grab()`, which is the obvious way to check a backdrop:
380's own WATCH list says it does not capture the GL-backed
`AmbientWidget`, so a grab-based check reports a black window and cannot
tell a working backdrop from a broken one. It was found the hard way,
after an A/B across two commits came out byte-identical. The widget's own
paint counter is the instrument that counts the thing that costs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "perf_paint.py"


@pytest.fixture(scope="module")
def harness():
    if not TOOL.is_file():
        pytest.skip("the performance harness is not in this checkout")
    spec = importlib.util.spec_from_file_location("spacr_perf_paint", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _painted_over(qtbot, layers: int, ms: int = 700) -> int:
    """Frames painted by ``layers`` backdrops in ``ms`` milliseconds."""
    from PySide6.QtCore import QEventLoop, QTimer

    from spacr.qt.widgets import ambient

    widgets = []
    for _ in range(layers):
        widget = ambient.AmbientWidget()
        widget.resize(640, 480)
        qtbot.addWidget(widget)
        widget.show()
        widgets.append(widget)
    qtbot.waitExposed(widgets[0]) if widgets else None

    before = ambient.total_frames_painted()
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    painted = ambient.total_frames_painted() - before
    for widget in widgets:
        widget.hide()
    return painted


def test_one_layer_paints_at_all(qtbot):
    """A harness that reads zero cannot tell a stopped backdrop from a fast
    one, and every ratio below would be meaningless."""
    assert _painted_over(qtbot, 1) > 0, (
        "the backdrop painted nothing at all, so this instrument is not "
        "measuring paint on this machine")


def test_a_second_layer_is_seen_as_a_second_layer(qtbot):
    """The one-backdrop regression, reproduced deliberately.

    Two full-size animated fields were being shaded and blitted per frame
    for a picture nobody could see. The instrument has to notice, and the
    only assertion that is safe on a loaded machine is a RATIO: absolute
    frame counts move with the load, and the ratio does not.
    """
    one = _painted_over(qtbot, 1)
    two = _painted_over(qtbot, 2)
    assert one > 0
    assert two >= 1.5 * one, (
        f"two layers painted {two} frames against one layer's {one}. An "
        "instrument that cannot see the duplicated layer would have "
        "reported the regression this harness was calibrated on as no "
        "change at all")


def test_the_widget_count_counts_top_level_windows(harness, qtbot):
    """`findChildren` on the application object misses every window.

    The first version of the theme measurement reported "1049 widgets" as
    "1", because a top-level window is not a CHILD of the application --
    and a per-widget cost divided by one widget reads as a flattering
    result rather than as a broken denominator.
    """
    from PySide6.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    window = QWidget()
    qtbot.addWidget(window)
    window.show()

    assert len(app.allWidgets()) > len(app.findChildren(QWidget)), (
        "this machine's Qt reports the same count both ways, so the "
        "distinction this test exists for cannot be shown here")


def test_an_interaction_that_blocks_reads_as_dropped_frames(harness):
    """The latency rule, asserted on the arithmetic rather than on a run.

    "Measure INPUT LATENCY and DROPPED FRAMES, not wall-clock -- a 200 ms
    hitch on a keystroke is felt, and a 200 ms total spread over ten
    frames is not." One frame at 60 Hz is the unit.
    """
    assert 16.0 < harness.FRAME_MS < 17.0
    assert int(200.0 // harness.FRAME_MS) == 11
    assert int(10.0 // harness.FRAME_MS) == 0


def test_the_record_says_what_else_the_machine_was_doing(harness):
    """A baseline taken under load is a fact about the load."""
    environment = harness._environment()
    assert environment["platform"] and environment["python"]
    assert "qt_platform" in environment
    # Load average is POSIX-only; where it exists it must be recorded,
    # because every number in the file is smaller when it is high.
    import os

    if hasattr(os, "getloadavg"):
        assert "load_1m" in environment
