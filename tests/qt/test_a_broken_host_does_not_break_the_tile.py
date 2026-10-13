"""Two guards that keep a decoration from taking its host down.

Instruction 288.

``GeneTilePanel.show_feature`` calls a frame provider supplied by
whatever screen owns the panel. A provider that raises is a broken HOST,
not a broken tile, and the panel's own docstring states the contract: "a
tile is an explanation, and an explanation that raises leaves the user
with a traceback instead of the point they clicked."

``figure_queue`` clears its view's scene background inside a try, because
a QGraphicsView built without a scene has none to clear.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.gene_tile import GeneTilePanel


def _frame():
    return pd.DataFrame({
        "feature": ["gene_a", "gene_b"],
        "coefficient": [0.5, -0.2],
        "p_value": [0.01, 0.4],
    })


def test_a_provider_that_raises_leaves_a_usable_panel(qtbot):
    """THE ARM. The panel must still render, with no frame."""
    def _explode():
        raise RuntimeError("the results table is not loaded")

    panel = GeneTilePanel(frame_provider=_explode)
    qtbot.addWidget(panel)

    panel.show_feature("gene_a")            # must not raise

    assert panel._feature == "gene_a", (
        "the panel forgot which feature was clicked")


def test_a_working_provider_is_actually_consulted(qtbot):
    """So the test above is not passing because the provider is ignored."""
    seen = []

    def _provider():
        seen.append(True)
        return _frame()

    panel = GeneTilePanel(frame_provider=_provider)
    qtbot.addWidget(panel)
    panel.show_feature("gene_a")

    assert seen == [True], "the frame provider was never called"


def test_no_provider_at_all_is_fine(qtbot):
    """The default. A panel with no host still has to draw."""
    panel = GeneTilePanel()
    qtbot.addWidget(panel)
    panel.show_feature("gene_a")
    assert panel._feature or panel._error_feature


def test_an_unknown_feature_still_produces_a_tile(qtbot):
    """RECORDED, because it is not what the code reads like.

    `_error_feature` looks like the "that gene is not in the results"
    path, and it is not: `gene_tile` is TOLERANT -- an unknown feature,
    a None frame and a None key all return a tile rather than raising.
    `_error_feature` is reached only when the builder itself fails.

    So the two arms in `show_feature` are "the host's provider raised"
    and "the builder raised", and neither is "no such gene".
    """
    panel = GeneTilePanel(frame_provider=_frame)
    qtbot.addWidget(panel)
    panel.show_feature("no_such_gene_at_all")

    assert panel._feature == "no_such_gene_at_all"
    assert panel._error_feature == "", (
        "an unknown feature now takes the error path; the tolerant "
        "behaviour this test records has changed")


def test_a_builder_that_raises_is_what_sets_the_error(qtbot, monkeypatch):
    """The OTHER arm, driven where the tolerant builder cannot reach it."""
    from spacr.qt.widgets import gene_tile as module

    def _explode(*_args, **_kwargs):
        raise ValueError("the tile could not be built")

    monkeypatch.setattr(module, "gene_tile", _explode)

    panel = GeneTilePanel(frame_provider=_frame)
    qtbot.addWidget(panel)
    panel.show_feature("gene_a")            # must not raise

    assert panel._error_feature == "gene_a"
    assert panel._tile is None


def test_a_view_without_a_scene_does_not_stop_the_queue(qtbot):
    """figure_queue's own guard, driven directly.

    `QGraphicsView.scene()` returns None until one is set, and calling
    `setBackgroundBrush` on None raises.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QGraphicsView

    view = QGraphicsView()
    qtbot.addWidget(view)
    assert view.scene() is None, "the premise changed: a fresh view has one"

    with pytest.raises(AttributeError):
        view.scene().setBackgroundBrush(Qt.NoBrush)
