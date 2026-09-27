"""Item 288: deleting an object whole, while a stroke is open or history is full.

``MaskCuration.delete_object`` removes every element of one label as one
undoable stroke. Two situations it has to handle that a plain delete does
not reach:

* a brush stroke is still open (the user is mid-drag when they delete). The
  stroke is closed first, so it stays its own undo step and the delete is
  another -- rather than the delete being folded into the stroke, or the
  stroke's dabs being lost;
* the undo history is already full. The delete is still recorded and the
  oldest stroke is the one dropped, so the history stays bounded.
"""
from __future__ import annotations

import numpy as np

from spacr.curation import MaskCuration
from spacr.layers import LabelsLayer, Spacing


def _layer():
    data = np.zeros((1, 21, 21), dtype=np.int64)
    data[0, 2:5, 2:5] = 7
    return LabelsLayer(data, name="mask", spacing=Spacing.from_map(
        {"z": 1.0, "y": 1.0, "x": 1.0}, units="um"))


def test_a_delete_mid_stroke_closes_the_stroke_first():
    layer = _layer()
    session = MaskCuration(layer)
    session.begin_stroke()
    session.paint({"z": 0.0, "y": 15.0, "x": 15.0}, label=3, radius=1.0)

    edit = session.delete_object(7)

    assert edit is not None and edit.kind == "delete"
    assert not (layer.data == 7).any()
    assert (layer.data == 3).any()
    assert len(session) == 2, "the stroke and the delete are two undo steps"
    assert [e.kind for e in session.log.edits] == ["paint", "delete"]

    session.undo()
    assert (layer.data == 7).sum() == 9, "undo brings the object back whole"
    assert (layer.data == 3).any(), "and leaves the stroke alone"
    session.undo()
    assert not (layer.data == 3).any()


def test_a_delete_with_the_history_full_drops_the_oldest_stroke():
    layer = _layer()
    session = MaskCuration(layer, history=1)
    session.paint({"z": 0.0, "y": 15.0, "x": 15.0}, label=3, radius=1.0)
    assert len(session) == 1

    session.delete_object(7)

    assert len(session) == 1
    assert [e.kind for e in session.log.edits] == ["paint", "delete"]
    session.undo()
    assert (layer.data == 7).sum() == 9
    assert session.undo() is None, "the paint fell off the bounded history"
    assert (layer.data == 3).any()
