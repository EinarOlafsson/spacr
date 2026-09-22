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


# ---------------------------------------------------------------------------
# The magnifier row, added for item 407
# ---------------------------------------------------------------------------
#
# Item 407's WHAT IS LEFT said the harness "was not run with the toggle on",
# and it could not be: there was no row for it. The magnifier is the one
# control on Make Masks that puts a segmentation model behind a moving mouse,
# which is the thing 380 measures. The same rule applies to this row as to
# every other one here -- test the INSTRUMENT, not the application.


def test_the_magnifier_row_measures_both_scopes_and_both_box_sizes(harness):
    """A row per scope, on a real screen with a real field open.

    Small on purpose: what is asserted is that the instrument reports, not
    how fast this machine is. The numbers themselves belong in a run of the
    tool, beside the load average that moves them.

    THE LARGEST BOX IS A ROW OF ITS OWN because every per-move path in the
    box is counted over the box's pixels and item 417 let the Size box go
    as high as the field is wide. A harness that only ever measured 128 px
    would report a control that is fine at one setting and say nothing
    about the same control at 2,048.
    """
    rows = harness.measure_magnifier(field_px=192, moves=4)

    assert [row.get("scope") for row in rows] == ["region", "image",
                                                  "region", "image"], (
        "the harness did not measure both scopes: " + repr(rows))
    assert rows[-2]["box_px"] == rows[-1]["box_px"] == 192 \
        > rows[0]["box_px"], (
        "the largest box the field allows was not measured in both "
        "scopes: " + repr(rows))
    for row in rows:
        assert row["measurement"] == "magnifier"
        assert row["mode"], "the row does not say which model ran"
        assert row["moves"] == 4
        assert row["slowest_move_ms"] >= row["median_move_ms"] > 0
        assert row["first_move_ms"] > 0
        assert "model_still_running" in row


def test_the_magnifier_row_measures_the_box_over_a_mask_that_is_there(
        harness):
    """The field is opened with a mask, so the Overlap rule has work to do.

    What the box draws is what a click would ADD, which is the rule
    against the mask already painted. Over an empty mask that path does
    not run at all, so a harness whose field has no mask file reports a
    magnifier nobody has measured and a row of numbers that look fine.
    """
    rows = harness.measure_magnifier(field_px=192, moves=4)

    ghosted = [row for row in rows if row.get("ghosted")]
    assert ghosted, (
        "no row saw the Overlap rule take anything away, so the box's own "
        "per-move work went unmeasured: " + repr(rows))


def test_the_magnifier_row_leaves_the_users_own_settings_alone(harness):
    """A measurement must not edit the machine it measures.

    Driving the real screen opens a folder, and opening a folder is
    remembered: `_open_folder` ends in `prefs.push_recent_source`, which
    writes `$HOME/.config/spacr/qt.conf`. Run by hand as the tool
    documents itself, that put a TemporaryDirectory at the head of the
    maintainer's recent-folder list and left `last` pointing at a path
    that had already been deleted.
    """
    from spacr.qt import prefs

    mine = "/a/folder/the/user/chose"
    prefs.push_recent_source("make_masks", mine)
    harness.measure_magnifier(field_px=192, moves=2)

    assert prefs.get_last_source("make_masks") == mine, (
        "the harness overwrote the last folder the user opened")
    assert prefs.get_recent_sources("make_masks")[0] == mine, (
        "the harness pushed its own temporary folder onto the recent list")


def test_the_magnifier_row_says_what_else_the_machine_was_doing(harness):
    """A latency is the measurement load moves most, so the load is on the
    ROW and not only in the record's environment."""
    import os

    rows = harness.measure_magnifier(field_px=192, moves=2)
    if hasattr(os, "getloadavg"):
        assert all("load_1m" in row for row in rows)


def test_a_slow_move_reads_as_dropped_frames_in_the_magnifier_row(
        harness, monkeypatch):
    """Calibration: the dropped-frame count follows the measurement.

    A row that reported 0 whatever it measured would pass every reading of
    this harness and mean nothing. Shrinking one frame to a tenth of a
    millisecond must make every move above it read as dropped.
    """
    monkeypatch.setattr(harness, "FRAME_MS", 0.1)
    rows = harness.measure_magnifier(field_px=192, moves=3)

    assert all(row["frames_dropped_worst_move"] > 0 for row in rows), (
        "no move read as a dropped frame at a tenth of a millisecond: "
        + repr(rows))
