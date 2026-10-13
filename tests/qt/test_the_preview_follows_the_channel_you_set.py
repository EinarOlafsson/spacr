"""The displayed plane follows the channel the user types, not only the object.

Item 442. ``_follow_object_channel`` already moved the view onto the primary
object's channel, and its own docstring gives the reason: tuning diameter,
flow and background against the wrong plane with nothing on screen saying so.
It was wired to ONE trigger -- changing WHICH object is primary -- while the
four channel spinners were wired only to a repaint. So with cell already
primary, typing 2 into cell channel repainted the plane already on screen.

The plate is written to disk and Yokogawa-named, as the other live-preview
tests do, because the panel reads real files to know what channels exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from spacr.qt.widgets.live_preview import LivePreviewPanel  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


@pytest.fixture
def plate(tmp_path: Path) -> Path:
    """Two fields, three channels, one plane each."""
    root = tmp_path / "plate"
    root.mkdir()
    for field in range(1, 3):
        for chan in range(1, 4):
            tifffile.imwrite(
                root / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif",
                np.full((8, 8), field * 10 + chan, dtype=np.uint16))
    return root


@pytest.fixture
def panel(qtbot, plate: Path):
    widget = LivePreviewPanel()
    qtbot.addWidget(widget)
    widget.load_image(sorted(plate.iterdir())[0])
    return widget


def _assert_shown(panel, qtbot, channel):
    """The selected channel file and its pixels agree with the table."""
    qtbot.waitUntil(lambda: not panel._image_loaders)
    assert panel._column_channels[panel._table_col] == channel
    assert panel._image_path.name == (
        f"plate1_A01_T0001F001L01A01Z01C{channel + 1:02d}.tif")
    np.testing.assert_array_equal(panel._image, 11 + channel)


def _choose_object(panel, caption: str) -> None:
    box = panel._object_box
    for index in range(box.count()):
        if box.itemText(index).strip().lower() == caption:
            box.setCurrentIndex(index)
            return
    pytest.skip(f"this build offers no {caption!r} object")


def test_typing_a_channel_moves_the_view_onto_it(panel, qtbot):
    """The report: cell is primary, Ch 1 is up, cell channel becomes 2."""
    _choose_object(panel, "cell")
    panel._cell_channel.setValue(1)
    _assert_shown(panel, qtbot, 1)

    panel._cell_channel.setValue(2)

    _assert_shown(panel, qtbot, 2)


def test_each_compartment_is_followed(panel, qtbot):
    """Not only cell: nucleus and pathogen are set the same way."""
    for caption, spinner in (("nucleus", panel._nucleus_channel),
                             ("pathogen", panel._pathogen_channel)):
        _choose_object(panel, caption)
        spinner.setValue(1)
        _assert_shown(panel, qtbot, 1)
        spinner.setValue(2)
        _assert_shown(panel, qtbot, 2)


def test_a_channel_the_set_does_not_have_leaves_the_view_alone(panel, qtbot):
    """Three channels are written as C01-C03 and shown as Ch 0, 1 and 2.

    Typing 7 into cell channel asks for a plane that is not in this set. The
    view stays where it is rather than blanking or picking something near it:
    a spinner can pass through any number on its way to the one the user
    means, and the settings still say 7 for the run to complain about.
    """
    _choose_object(panel, "cell")
    panel._cell_channel.setValue(1)
    _assert_shown(panel, qtbot, 1)

    panel._cell_channel.setValue(7)

    _assert_shown(panel, qtbot, 1)


def test_two_objects_leave_the_view_alone(panel, qtbot):
    """With cell + nucleus neither channel is the answer, so nothing moves.

    The rule `_follow_object_channel` already had, kept: a view that flickered
    between two compartments while both are being segmented is worse than one
    that stays where the user put it.
    """
    _choose_object(panel, "cell")
    panel._cell_channel.setValue(1)
    _assert_shown(panel, qtbot, 1)
    _choose_object(panel, "cell + nucleus")
    before = panel._image_path

    panel._cell_channel.setValue(3)

    assert panel._image_path == before


def test_the_canvas_is_repainted_once_per_keystroke(panel, qtbot, monkeypatch):
    """The follow repaints when it moves, so the caller must not repaint again.

    Two repaints per keystroke means the full-size image is redrawn twice
    while a number is being typed into a spinner.
    """
    _choose_object(panel, "cell")
    panel._cell_channel.setValue(1)
    _assert_shown(panel, qtbot, 1)

    calls = []
    monkeypatch.setattr(type(panel), "_refresh_canvases",
                        lambda self: calls.append(1))

    panel._cell_channel.setValue(2)
    _assert_shown(panel, qtbot, 2)
    assert len(calls) == 1, calls

    calls.clear()
    panel._cell_channel.setValue(2)
    assert calls == []

    calls.clear()
    panel._cell_channel.setValue(9)
    assert len(calls) == 1, calls
