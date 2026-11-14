"""The layer viewer's dispatch: tools on the canvas, the viewer's selection and status, the layer panel.

A canvas tool owns the events it consumes and the canvas keeps the rest; the
viewer's status line and selection follow only live layers; the layer panel
writes back only edits that make sense (a listed blending mode, a non-empty
contrast range) and Auto resets the limits from the layer's own intensities.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from spacr.layers import FieldKey, LayerStack
from spacr.qt import layer_viewer as lv

pytestmark = pytest.mark.qt


def _stack(size=40, block=slice(20, 30), label=17, field=True):
    stack = LayerStack()
    image = np.full((size, size), 1000, dtype=np.uint16)
    image[block, block] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((size, size), dtype=np.int32)
    mask[block, block] = label
    key = (FieldKey(values=dict(zip(FieldKey.columns(),
                                    ("plate1", "A", "1", "1"))))
           if field else None)
    stack.add_labels(mask, name="mask", field=key)
    return stack


def _sized(qtbot, widget, width=320, height=320):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    return widget


def _move_event(x=40.0, y=40.0):
    return QMouseEvent(QMouseEvent.MouseMove, QPointF(x, y), QPointF(x, y),
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)


class _Recorder(lv.CanvasTool):
    """A tool that consumes what it is told to and counts its detaches."""

    def __init__(self, consume_keys=False, consume_moves=False):
        self.consume_keys = consume_keys
        self.consume_moves = consume_moves
        self.keys = []
        self.moves = []
        self.detached = 0

    def key(self, view, event):
        self.keys.append(event.key())
        return self.consume_keys

    def move(self, view, world, event):
        self.moves.append(world)
        return self.consume_moves

    def detach(self):
        self.detached += 1


def test_layer_canvas_routes_events_to_its_tool_first(subtests, qtbot):
    """Re-attaching keeps the tool; consumed keys and moves stop at the tool."""
    with subtests.test(msg="attaching the same tool again does not detach it"):
        canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
        tool = _Recorder()
        assert canvas.set_tool(tool) is None
        assert canvas.focusPolicy() == Qt.StrongFocus
        assert canvas.set_tool(tool) is tool
        assert tool.detached == 0 and canvas.tool is tool
        other = _Recorder()
        assert canvas.set_tool(other) is tool
        assert tool.detached == 1 and canvas.tool is other
        assert canvas.set_tool(None) is other
        assert other.detached == 1 and canvas.tool is None
        assert canvas.focusPolicy() == Qt.NoFocus

    for case_id, consume, accepted in [
            ("a consumed key is accepted", True, True),
            ("an unconsumed key is passed on", False, False)]:
        with subtests.test(msg=case_id):
            canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
            tool = _Recorder(consume_keys=consume)
            canvas.set_tool(tool)
            event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
            event.setAccepted(False)
            canvas.keyPressEvent(event)
            assert tool.keys == [Qt.Key_Escape]
            assert event.isAccepted() is accepted

    with subtests.test(msg="a consumed move does not also pan"):
        canvas = _sized(qtbot, lv.LayerCanvas(_stack()))
        before = canvas._ensure_canvas()
        tool = _Recorder(consume_moves=True)
        canvas.set_tool(tool)
        hovered = []
        canvas.hovered.connect(hovered.append)
        canvas.mouseMoveEvent(_move_event(40.0, 40.0))
        assert len(tool.moves) == 1
        assert set(tool.moves[0]) == {"y", "x"}
        assert canvas.canvas is before


def _flat_stack():
    stack = LayerStack()
    image = np.full((40, 40), 1000, dtype=np.uint16)
    image[10:20, 10:20] = 4000
    stack.add_image(image, name="image", contrast_limits=(0.0, 4095.0))
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[10:20, 10:20] = 3
    stack.add_labels(mask, name="mask")
    return stack


def test_layer_viewer_selection_status_and_dialogs_follow_live_layers(
        subtests, qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Stale rows, hovers, custom colours and cancelled dialogs leave the state true."""
    with subtests.test(msg="a stale row keeps the real selection"):
        stack = _flat_stack()
        viewer = _sized(qtbot, lv.LayerViewer(stack), 420, 420)
        rows = viewer.layer_list
        assert [rows.item(i).text() for i in range(rows.count())] == [
            "mask", "image"]
        rows.setCurrentRow(0)
        assert stack.selected.name == "mask"
        described = viewer.status.text()
        assert described.startswith("mask (labels)")
        stale = rows.item(1)
        stale.setData(Qt.UserRole, "a layer that was removed")
        assert rows._layer_for(stale) is None
        rows.setCurrentRow(1)
        assert stack.selected.name == "mask"
        assert viewer.status.text() == described
        assert viewer.remove_button.isEnabled() is True

    with subtests.test(msg="hover writes coordinates only with no selection"):
        stack = _flat_stack()
        viewer = _sized(qtbot, lv.LayerViewer(stack), 420, 420)
        stack.select(None)
        assert viewer.status.text() == "No layer selected"
        viewer.canvas.hovered.emit({"y": 12.5, "x": 4.0})
        assert viewer.status.text() == "y 12.5 · x 4"
        stack.select(stack["image"])
        described = viewer.status.text()
        assert described.startswith("image (image)")
        viewer.canvas.hovered.emit({"y": 31.25, "x": 9.0})
        assert viewer.status.text() == described

    with subtests.test(msg="a custom colour is shown without recolouring it"):
        stack = LayerStack()
        stack.add_image(np.full((20, 20), 500, dtype=np.uint16),
                        name="image", colormaps="#ff00aa")
        viewer = _sized(qtbot, lv.LayerViewer(stack), 420, 420)
        combo = viewer.colormap_combo
        assert combo.currentText() == "#ff00aa"
        assert combo.findText("#ff00aa") >= 0
        assert stack["image"].colormap.name == "#ff00aa"
        stack["image"].colormap = "magenta"
        stack.select(None)
        stack.select(stack["image"])
        assert combo.currentText() == "magenta"
        assert combo.findText("#ff00aa") == -1
        assert stack["image"].colormap.name == "magenta"

    with subtests.test(msg="a cancelled mask dialog adds no layer"):
        from PIL import Image

        mask_png = tmp_path / "mask.png"
        labels = np.zeros((16, 16), dtype=np.uint8)
        labels[4:9, 4:9] = 7
        Image.fromarray(labels).save(mask_png)
        viewer = _sized(qtbot, lv.LayerViewer(), 420, 420)
        chosen = [str(mask_png)]
        monkeypatch.setattr(lv.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (chosen[0], "")))
        viewer.add_mask_button.click()
        assert viewer.stack.names == ("mask",)
        assert int(viewer.stack["mask"].data.max()) == 7
        assert int((viewer.stack["mask"].data == 7).sum()) == 25
        status = viewer.status.text()
        chosen[0] = ""
        viewer.add_mask_button.click()
        assert viewer.stack.names == ("mask",)
        assert viewer.status.text() == status
        assert "Could not load" not in viewer.status.text()


def test_layer_panel_writes_back_only_sensible_edits(subtests, qtbot):
    """Unlisted blending and empty contrast ranges are ignored; Auto refits."""
    def _panel():
        stack = _stack(field=False)
        panel = lv._LayerPanel(stack)
        qtbot.addWidget(panel)
        stack.select("image")
        panel.sync()
        return stack, panel

    with subtests.test(msg="an unlisted blending mode leaves the combo alone"):
        stack, panel = _panel()
        assert [panel.blending_combo.itemData(i)
                for i in range(panel.blending_combo.count())][:3] == [
            "translucent", "additive", "opaque"]
        assert panel.blending_combo.currentIndex() == 0
        panel.blending_combo.blockSignals(True)
        panel.blending_combo.setCurrentIndex(2)
        panel.blending_combo.blockSignals(False)
        stack.selected._blending = "not-a-listed-mode"
        panel.sync()
        assert panel.blending_combo.currentIndex() == 2
        assert stack.selected.blending == "not-a-listed-mode"
        stack.selected._blending = "additive"
        panel.sync()
        assert panel.blending_combo.currentIndex() == 1

    with subtests.test(msg="an empty or inverted contrast range is left alone"):
        stack, panel = _panel()
        assert stack.selected.contrast_limits(0) == (0.0, 4095.0)
        panel.high_spin.setValue(panel.low_spin.value())
        assert stack.selected.contrast_limits(0) == (0.0, 4095.0)
        panel.low_spin.setValue(10.0)
        assert stack.selected.contrast_limits(0) == (0.0, 4095.0)
        panel.high_spin.setValue(3000.0)
        assert stack.selected.contrast_limits(0) == (10.0, 3000.0)

    with subtests.test(msg="Auto resets the limits from the layer's intensities"):
        stack, panel = _panel()
        layer = stack.selected
        assert (panel.low_spin.value(), panel.high_spin.value()) == (
            0.0, 4095.0)
        panel._on_auto()
        assert layer.contrast_limits(0) == (1000.0, 4000.0)
        assert (panel.low_spin.value(), panel.high_spin.value()) == (
            1000.0, 4000.0)

    with subtests.test(msg="Auto and limits do nothing on a labels layer"):
        stack, panel = _panel()
        stack.select("mask")
        panel.sync()
        panel._on_auto()
        panel._on_limits()
        assert stack["image"].contrast_limits(0) == (0.0, 4095.0)
        assert panel.low_spin.isEnabled() is False
