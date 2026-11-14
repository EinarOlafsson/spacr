"""The brush and the track panel when the gesture, the disk or the user says no.

A brush must paint only on a left or right drag with the button held, and a
refused dab must be logged rather than raised; a track save writes only to a
real destination and emits ``saved`` only for a table that was written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from spacr.curation import MaskCuration
from spacr.layers import LayerError, LayerStack, Spacing
from spacr.qt import curation_tool as ct
from spacr.qt import layer_viewer as lv

pytestmark = pytest.mark.qt


def _stack(size=64):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, 1.0, units="px")
    stack.add_image(np.zeros((size, size), np.uint16), name="image",
                    spacing=spacing)
    stack.add_labels(np.zeros((size, size), np.int64), name="mask",
                     spacing=spacing)
    return stack


def _canvas(qtbot, stack):
    canvas = lv.LayerCanvas(stack)
    qtbot.addWidget(canvas)
    canvas.resize(200, 200)
    canvas._ensure_canvas()
    return canvas


def _tracks():
    rows = [{"frame": frame, "track_id": track_id,
             "original_label": 100 + frame,
             "x": float(frame), "y": float(track_id)}
            for track_id, frames in {1: [0, 1], 2: [2, 3]}.items()
            for frame in frames]
    return pd.DataFrame(rows, columns=["frame", "track_id", "original_label",
                                       "x", "y"])


class _Event:
    """Only as much of a mouse event as the brush reads."""

    def __init__(self, *, button=None, buttons=None):
        self._button = button
        self._buttons = buttons

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons


def test_brush_tool_paints_only_on_a_held_left_or_right_drag(
        subtests, qtbot, monkeypatch, caplog):
    """Middle clicks and button-up moves are declined; refused dabs are logged."""
    for case_id, button, label, expected_label in [
            ("left paints the active label", Qt.LeftButton, 5, 5),
            ("right erases to zero", Qt.RightButton, 5, 0)]:
        with subtests.test(msg=case_id):
            stack = _stack()
            canvas = _canvas(qtbot, stack)
            stack["mask"].data[:] = 9
            session = MaskCuration(stack["mask"], artifact="mask")
            session.label = label
            tool = ct.BrushTool(session)
            assert tool.press(canvas, {"y": 10.0, "x": 10.0},
                              _Event(button=button)) is True
            assert stack["mask"].data[10, 10] == expected_label
            painted = int((stack["mask"].data == expected_label).sum())
            assert tool.move(canvas, {"y": 30.0, "x": 30.0},
                             _Event(buttons=button)) is True
            assert stack["mask"].data[30, 30] == expected_label
            assert tool.move(canvas, {"y": 50.0, "x": 50.0},
                             _Event(buttons=Qt.NoButton)) is False
            assert stack["mask"].data[50, 50] == 9
            assert int((stack["mask"].data == expected_label).sum()) \
                == 2 * painted
            assert tool.release(canvas, {"y": 50.0, "x": 50.0},
                                _Event()) is True
            assert tool.release(canvas, {"y": 50.0, "x": 50.0},
                                _Event()) is False
            assert tool.move(canvas, {"y": 40.0, "x": 40.0},
                             _Event(buttons=button)) is False
            assert stack["mask"].data[40, 40] == 9

    with subtests.test(msg="a middle click opens no stroke"):
        stack = _stack()
        canvas = _canvas(qtbot, stack)
        tool = ct.BrushTool(MaskCuration(stack["mask"], artifact="mask"))
        assert tool.press(canvas, {"y": 10.0, "x": 10.0},
                          _Event(button=Qt.MiddleButton)) is False
        assert tool.move(canvas, {"y": 10.0, "x": 10.0},
                         _Event(buttons=Qt.MiddleButton)) is False
        assert tool.release(canvas, {"y": 10.0, "x": 10.0}, _Event()) is False
        assert not stack["mask"].data.any()

    with subtests.test(msg="a dab the layer refuses is logged, not raised"):
        stack = _stack()
        canvas = _canvas(qtbot, stack)
        session = MaskCuration(stack["mask"], artifact="mask")
        tool = ct.BrushTool(session)
        calls = []

        def refuse(world):
            calls.append(world)
            raise LayerError("the brush is outside the layer")

        monkeypatch.setattr(session, "paint", refuse)
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.curation_tool"):
            assert tool.press(canvas, {"y": 10.0, "x": 10.0},
                              _Event(button=Qt.LeftButton)) is True
            assert tool._dab({"y": 11.0, "x": 11.0}) == 0
        assert calls == [{"y": 10.0, "x": 10.0}, {"y": 11.0, "x": 11.0}]
        assert not stack["mask"].data.any()
        assert [r.getMessage() for r in caplog.records
                if r.name == "spacr.qt.curation_tool"] == ["Could not paint"] * 2


def test_track_curation_panel_save_writes_only_to_a_real_destination(
        subtests, qtbot, tmp_path, monkeypatch):
    """No session, a cancelled dialog or a failed write all return None and emit nothing."""
    def _panel(**kwargs):
        panel = ct.TrackCurationPanel(**kwargs)
        qtbot.addWidget(panel)
        saved = []
        panel.saved.connect(saved.append)
        return panel, saved

    def _dialog(answer, asked):
        def dialog(parent, caption, suggested, filter_):
            asked.append((caption, suggested.rsplit("/", 1)[-1], filter_))
            return answer, "CSV"
        return staticmethod(dialog)

    with subtests.test(msg="nothing open: no dialog, no table"):
        asked = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            _dialog(str(tmp_path / "x.csv"), asked))
        panel, saved = _panel()
        assert panel.save() is None
        assert asked == [] and saved == []

    with subtests.test(msg="a cancelled dialog writes nothing"):
        asked = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName", _dialog("", asked))
        monkeypatch.chdir(tmp_path)
        panel, saved = _panel(tracks=_tracks())
        assert panel.save() is None
        assert asked == [("Save the curated tracks", "tracks_curated.csv",
                          "CSV (*.csv)")]
        assert saved == []
        assert list(tmp_path.glob("*.csv")) == []

    with subtests.test(msg="a chosen file is written and announced"):
        target = tmp_path / "chosen.csv"
        asked = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            _dialog(str(target), asked))
        panel, saved = _panel(tracks=_tracks())
        assert panel.save() == str(target)
        assert saved == [str(target)]
        assert len(asked) == 1
        assert sorted(set(pd.read_csv(target)["track_id"])) == [1, 2]

    with subtests.test(msg="an explicit path skips the dialog"):
        target = tmp_path / "explicit.csv"
        asked = []
        monkeypatch.setattr(QFileDialog, "getSaveFileName", _dialog("", asked))
        panel, saved = _panel(tracks=_tracks())
        assert panel.save(str(target)) == str(target)
        assert asked == [] and saved == [str(target)]
        assert target.is_file()

    with subtests.test(msg="a table that cannot be written says so"):
        panel, saved = _panel(tracks=_tracks(),
                              artifact="/runs/plate1/tracks.csv")
        targets = []

        def refuse(target):
            targets.append(target)
            raise OSError("No space left on device")

        monkeypatch.setattr(panel.session, "save", refuse)
        assert panel.save() is None
        assert targets == ["/runs/plate1/tracks.csv"]
        assert panel.status.text() == (
            "Could not write the tracks: No space left on device")
        assert saved == []
