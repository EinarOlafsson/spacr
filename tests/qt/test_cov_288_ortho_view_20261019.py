"""Ortho view: a selection naming a label the array does not hold, and an event that changes no pixel."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.layers import FieldKey, LayerEvent, LayerStack, Spacing
from spacr.qt import ortho_view as ov
from spacr.qt.linked_selection import Selection

pytestmark = pytest.mark.qt

CONFOCAL = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
KEY = FieldKey(values=dict(zip(FieldKey.columns(), ("plate1", "A", "1", "1"))))


def _view(qtbot):
    stack = LayerStack()
    stack.add_image(np.zeros((10, 64, 64), np.uint16), name="volume",
                    spacing=CONFOCAL, contrast_limits=(0.0, 4000.0))
    mask = np.zeros((10, 64, 64), np.int32)
    mask[5, 30:34, 30:34] = 17
    stack.add_labels(mask, name="mask", spacing=CONFOCAL, field=KEY)
    view = ov.OrthoView(stack)
    qtbot.addWidget(view)
    view.resize(520, 520)
    return view, stack


def test_a_selected_object_with_no_pixels_leaves_the_crosshair_where_it_was(qtbot, monkeypatch):
    view, stack = _view(qtbot)
    view.move_to(z=0.0)
    monkeypatch.setattr(stack["mask"], "labels", lambda: np.array([99]))
    view.on_linked_selection_changed(Selection(keys=["plate1_A_1_1_99"], source="umap"))
    assert stack["mask"].selected_label == 99
    assert view.slice_index("z") == 0


def test_a_layer_event_that_changes_no_pixel_composites_nothing(qtbot, monkeypatch):
    view, stack = _view(qtbot)
    redraws = []
    for panel in view.panels.values():
        monkeypatch.setattr(panel, "_redraw", lambda panel=panel: redraws.append(panel))
    view._on_layers_changed(LayerEvent(kind="selected", layer=None, index=-1))
    assert redraws == []
    stack["mask"].visible = False
    assert len(redraws) == 3
