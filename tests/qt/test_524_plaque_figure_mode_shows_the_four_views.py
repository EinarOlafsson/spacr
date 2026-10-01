"""Item 524: Figure mode offers the four views, Plaque mode keeps them.

    "i cannot see the cell probability, flows, and masks tabs in the figure
     mode for the plaque assay, please add these, and also for plaque mode
     if they are not there already" -- the maintainer, 2026-09-25

In Figure mode the Masks, Flows and Cell probability views show what the
segmented wells gave, each well at its place on the figure; before a well
is segmented, or for an array the segmenter did not give, the shared
widget's sentence says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402
from spacr.qt.widgets import segmentation_views as sv  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import test_plaque_assay_has_plaque_and_figure_modes as modes  # noqa: E402

WELL_A = (100, 180, 100, 180)
WELL_B = (100, 180, 220, 300)


def _labels(shape):
    labels = np.zeros(shape[:2], dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:45, 30:45] = 2
    return labels


def _flows(shape):
    flow = np.zeros(shape[:2] + (3,), dtype=np.uint8)
    flow[..., 1] = 180
    prob = np.full(shape[:2], -4.0, dtype=np.float32)
    prob[10:20, 10:20] = 5.0
    return {"flow_rgb": flow, "cellprob": prob}


def _segment_with_flows(crop):
    return _labels(crop.shape), _flows(crop.shape)


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def figure(panel, tmp_path):
    return modes._figure_panel(panel, tmp_path)


def _show(panel, view):
    panel.views().set_view(view)
    if panel.views().is_showing_message():
        return panel.views().message_text()
    return panel._view.array()


def _expected(segmented_wells):
    labels = np.zeros((400, 400), dtype=np.int32)
    flow = np.zeros((400, 400, 3), dtype=np.uint8)
    prob = np.full((400, 400), -30.0, dtype=np.float32)
    for y0, y1, x0, x1 in segmented_wells:
        crop = _labels((y1 - y0, x1 - x0))
        found = crop > 0
        labels[y0:y1, x0:x1][found] = crop[found] + labels.max()
        flows = _flows((y1 - y0, x1 - x0))
        flow[y0:y1, x0:x1] = flows["flow_rgb"]
        prob[y0:y1, x0:x1] = flows["cellprob"]
    return labels, flow, prob


def test_plaque_mode_offers_the_four_views(panel):
    panel.set_mode("plaque")
    selector = panel._view_selector
    assert selector.isVisibleTo(panel)
    assert [selector.itemData(i) for i in range(selector.count())] == [
        sv.OVERLAY, sv.MASKS, sv.FLOWS, sv.CELLPROB]


def test_figure_mode_offers_the_four_views(figure):
    selector = figure._view_selector
    assert figure.mode() == ppv.FIGURE_MODE
    assert selector.isVisibleTo(figure)
    assert selector.count() == 4


def test_before_a_well_is_segmented_the_views_say_so(figure):
    for view in (sv.MASKS, sv.FLOWS, sv.CELLPROB):
        assert _show(figure, view) == sv.WAITING
    overlay = _show(figure, sv.OVERLAY)
    assert isinstance(overlay, np.ndarray) and overlay.shape[:2] == (400, 400)


def test_each_view_shows_the_segmented_wells_in_place(figure):
    figure.select_well(0)
    assert figure.preview_selected_well(segment=_segment_with_flows)
    labels, flow, prob = _expected([WELL_A])
    assert np.array_equal(_show(figure, sv.MASKS), sv.render_labels(labels))
    assert np.array_equal(_show(figure, sv.FLOWS), flow)
    assert np.array_equal(_show(figure, sv.CELLPROB),
                          sv.render_cellprob(prob))
    assert figure.find_plaques_in_all_wells(segment=_segment_with_flows)
    labels, flow, prob = _expected([WELL_A, WELL_B])
    assert labels.max() == 4
    assert np.array_equal(_show(figure, sv.MASKS), sv.render_labels(labels))
    assert np.array_equal(_show(figure, sv.FLOWS), flow)
    assert np.array_equal(_show(figure, sv.CELLPROB),
                          sv.render_cellprob(prob))
    overlay = _show(figure, sv.OVERLAY)
    assert overlay.shape[:2] == (400, 400)


def test_a_segmenter_without_flows_is_said_on_those_views(figure):
    assert figure.find_plaques_in_all_wells(segment=modes._segment_crop)
    assert isinstance(_show(figure, sv.MASKS), np.ndarray)
    assert _show(figure, sv.FLOWS) == sv.NO_FLOWS
    assert _show(figure, sv.CELLPROB) == sv.NO_CELLPROB


def test_segment_well_hands_back_the_flows_of_its_crop():
    region = type("R", (), {"x0": 20, "y0": 10, "x1": 80, "y1": 50})()
    result = ppv.segment_well(np.zeros((100, 100, 3), np.uint8), region, {},
                              segment=_segment_with_flows)
    assert result["flow_rgb"].shape == (40, 60, 3)
    assert result["cellprob"].shape == (40, 60)
    plain = ppv.segment_well(np.zeros((100, 100, 3), np.uint8), region, {},
                             segment=modes._segment_crop)
    assert plain["flow_rgb"] is None and plain["cellprob"] is None
