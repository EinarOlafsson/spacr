"""The brush and the track panel when the gesture, the disk or the model says no.

A curation tool edits data that is not recoverable from anywhere else, so
every refusal here has to be visible. A brush that paints on a gesture nobody
made destroys a mask; a ledger that could not be written but reported success
makes a curated dataset indistinguishable from a raw one; a save dialog the
user cancelled must not be read as a save.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.curation import MaskCuration  # noqa: E402
from spacr.layers import LayerError, LayerStack, Spacing  # noqa: E402
from spacr.qt import curation_tool as ct  # noqa: E402
from spacr.qt import layer_viewer as lv  # noqa: E402

pytestmark = pytest.mark.qt


def _stack(size=64, step=1.0, units="px"):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, step, units=units)
    stack.add_image(np.zeros((size, size), np.uint16), name="image",
                    spacing=spacing)
    stack.add_labels(np.zeros((size, size), np.int64), name="mask",
                     spacing=spacing)
    return stack


def _canvas(qtbot, stack=None):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    return canvas


def _tracks(spec):
    rows = []
    for track_id, frames in spec.items():
        for frame in frames:
            rows.append({"frame": frame, "track_id": track_id,
                         "original_label": 100 + frame,
                         "x": float(frame), "y": float(track_id)})
    return pd.DataFrame(rows, columns=["frame", "track_id", "original_label",
                                       "x", "y"])


class _Event:
    """A mouse or key event, only as much of one as the tool reads."""

    def __init__(self, *, button=None, buttons=None, key=None, text=""):
        self._button = button
        self._buttons = buttons
        self._key = key
        self._text = text

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons

    def key(self):
        return self._key

    def text(self):
        return self._text


# ---------------------------------------------------------------------------
# what the brush refuses to paint
# ---------------------------------------------------------------------------

def test_a_brush_refuses_to_be_built_on_anything_but_a_curation_session():
    """The session is what records the edit, so a stand-in is not enough.

    A brush attached to a bare labels layer would paint perfectly well and
    record nothing, which is exactly the state ``is_curated`` exists to be
    able to rule out.
    """
    with pytest.raises(LayerError, match="a brush paints into a MaskCuration"):
        ct.BrushTool(object())


def test_a_middle_click_does_not_open_a_stroke(qtbot):
    """Only left and right paint; the middle button belongs to the canvas.

    Middle-drag pans in the viewer. Opening a stroke on it would paint a line
    across the mask every time the user moved the view.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    tool = ct.BrushTool(MaskCuration(stack["mask"], artifact="mask"))

    handled = tool.press(canvas, {"y": 10.0, "x": 10.0},
                         _Event(button=Qt.MiddleButton))

    assert handled is False
    assert not stack["mask"].data.any()


def test_the_brush_stops_painting_when_no_button_is_down(qtbot):
    """A move with the button up is the cursor travelling, not a stroke.

    Without this the brush paints wherever the pointer went after the button
    came up, which destroys a mask in the time it takes to reach for undo.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    session = MaskCuration(stack["mask"], artifact="mask")
    session.label = 5
    tool = ct.BrushTool(session)
    tool.press(canvas, {"y": 10.0, "x": 10.0},
               _Event(button=Qt.LeftButton))
    painted = int((stack["mask"].data == 5).sum())
    assert painted

    handled = tool.move(canvas, {"y": 30.0, "x": 30.0},
                        _Event(buttons=Qt.NoButton))

    assert handled is False
    assert int((stack["mask"].data == 5).sum()) == painted


def test_releasing_without_a_stroke_open_is_not_an_edit(qtbot):
    """A release the tool never saw a press for closes nothing.

    ``end_stroke`` on a stroke that was never opened would record an empty
    ledger entry and give the undo button something to take back.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    session = MaskCuration(stack["mask"], artifact="mask")
    tool = ct.BrushTool(session)

    handled = tool.release(canvas, {"y": 10.0, "x": 10.0},
                           _Event(button=Qt.LeftButton))

    assert handled is False
    assert list(session.log.edits) == []


def test_a_key_the_brush_does_not_use_is_passed_back_to_the_canvas(qtbot):
    """The brush claims three keys; everything else belongs to the viewer.

    Swallowing the rest would break the canvas's own shortcuts for as long as
    the brush was switched on.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    tool = ct.BrushTool(MaskCuration(stack["mask"], artifact="mask"))
    before = tool.session.radius

    handled = tool.key(canvas, _Event(key=Qt.Key_G, text="g"))

    assert handled is False
    assert tool.session.radius == before


def test_a_dab_the_layer_refuses_is_logged_and_leaves_the_mask_alone(
        qtbot, monkeypatch, caplog):
    """A paint the layer cannot do must not take the drag down with it.

    The dab runs inside a mouse-move handler. An exception there escapes into
    Qt's event loop mid-stroke, leaving the stroke open and the tool armed.
    """
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    session = MaskCuration(stack["mask"], artifact="mask")
    tool = ct.BrushTool(session)

    def refuse(world):
        raise LayerError("the brush is outside the layer")

    monkeypatch.setattr(session, "paint", refuse)

    with caplog.at_level("ERROR", logger="spacr.qt.curation_tool"):
        handled = tool.press(canvas, {"y": 10.0, "x": 10.0},
                             _Event(button=Qt.LeftButton))

    assert handled is True
    assert not stack["mask"].data.any()
    assert any("Could not paint" in record.message
               for record in caplog.records)


# ---------------------------------------------------------------------------
# the brush button
# ---------------------------------------------------------------------------

def test_the_brush_button_attaches_and_detaches_the_tool(qtbot):
    """The toggle is the only way a user gets the mouse back.

    Leaving the tool attached after the button came up means every click on
    the image paints, with a button that says it is off.
    """
    canvas = _canvas(qtbot)
    panel = ct.BrushPanel(canvas)
    qtbot.addWidget(panel)

    assert panel.tool is None

    panel.paint_button.setChecked(True)
    assert isinstance(panel.tool, ct.BrushTool)
    assert canvas.tool is panel.tool

    panel.paint_button.setChecked(False)
    assert panel.tool is None
    assert canvas.tool is None


def test_a_ledger_that_cannot_be_written_says_so_in_the_badge(qtbot,
                                                              monkeypatch):
    """A failed write must not be reported as a written ledger.

    The badge is what tells the user this mask has a correction record. A
    silent failure there produces a curated mask with no evidence that it was
    curated, which is the one thing the ledger exists to prevent.
    """
    canvas = _canvas(qtbot)
    panel = ct.BrushPanel(canvas, artifact="mask.tif")
    qtbot.addWidget(panel)
    logged: list = []
    panel.logged.connect(logged.append)

    def refuse(path=None):
        raise OSError("Read-only file system")

    monkeypatch.setattr(panel.session, "save_log", refuse)

    assert panel.save_log() is None
    assert "Could not write the ledger" in panel.badge.text()
    assert "Read-only file system" in panel.badge.text()
    assert logged == []


# ---------------------------------------------------------------------------
# the track panel
# ---------------------------------------------------------------------------

def test_opening_a_table_through_the_seam_replaces_the_session(qtbot):
    """``set_tracks`` is how a screen hands the panel a table it already has.

    Going via a file would mean writing the frame out and reading it back,
    and the ledger would then be beside a temporary file rather than beside
    the tracks the run produced.
    """
    panel = ct.TrackCurationPanel()
    qtbot.addWidget(panel)
    assert panel.session is None

    panel.set_tracks(_tracks({1: [0, 1, 2], 2: [3, 4]}),
                     artifact="/runs/plate1/tracks.csv")

    assert panel.session is not None
    assert panel.session.artifact == "/runs/plate1/tracks.csv"
    assert panel.track_list.count() == 2


def test_loading_a_file_that_is_not_a_table_says_which_file(qtbot, tmp_path):
    """The status line is the only place the failure can appear.

    A run has one tracks file per plate, so "could not read" without the path
    does not say which plate to go and look at.
    """
    broken = tmp_path / "tracks.csv"
    broken.write_bytes(b"\x00\x01\x02not a csv")
    panel = ct.TrackCurationPanel()
    qtbot.addWidget(panel)

    assert panel.load(str(tmp_path / "missing.csv")) is None
    assert "Could not read" in panel.status.text()
    assert "missing.csv" in panel.status.text()
    assert panel.session is None


def test_an_operation_with_too_few_tracks_selected_says_how_many_are_needed(
        qtbot):
    """A button that declines silently is indistinguishable from a broken one.

    Join needs two tracks and split needs one, and the selection is in a list
    the user may not be looking at.
    """
    panel = ct.TrackCurationPanel(tracks=_tracks({1: [0, 1], 2: [2, 3]}))
    qtbot.addWidget(panel)
    panel.track_list.clearSelection()

    assert panel.join_selected() is False
    assert "Select 2 track(s) first" in panel.status.text()
    assert "0 selected" in panel.status.text()

    panel.track_list.item(0).setSelected(True)
    assert panel.join_selected() is False
    assert "1 selected" in panel.status.text()
    # One IS enough for a split, so the guard is not simply always refusing.
    panel.frame_spin.setValue(1)
    assert panel.split_selected() is True


def test_cancelling_the_save_dialog_writes_nothing(qtbot, monkeypatch):
    """A cancelled dialog returns an empty path, which is not a destination.

    Writing to it would either raise or, worse, create a file called nothing
    in the working directory and report it as the curated table.
    """
    panel = ct.TrackCurationPanel(tracks=_tracks({1: [0, 1], 2: [2, 3]}))
    qtbot.addWidget(panel)
    saved: list = []
    panel.saved.connect(saved.append)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    assert panel.save() is None
    assert saved == []


def test_a_table_that_cannot_be_written_says_so_and_emits_nothing(qtbot,
                                                                  monkeypatch):
    """The saved signal is what marks the tracks curated downstream."""
    panel = ct.TrackCurationPanel(tracks=_tracks({1: [0, 1], 2: [2, 3]}),
                                  artifact="/runs/plate1/tracks.csv")
    qtbot.addWidget(panel)
    saved: list = []
    panel.saved.connect(saved.append)

    def refuse(target):
        raise OSError("No space left on device")

    monkeypatch.setattr(panel.session, "save", refuse)

    assert panel.save() is None
    assert "Could not write the tracks" in panel.status.text()
    assert "No space left on device" in panel.status.text()
    assert saved == []


def test_saving_with_nothing_open_writes_nothing(qtbot):
    """No session means no table, and no dialog either."""
    panel = ct.TrackCurationPanel()
    qtbot.addWidget(panel)

    assert panel.save() is None
