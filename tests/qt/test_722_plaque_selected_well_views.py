"""The selected plaque well follows the figure's view without stale pixels."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from spacr.qt.widgets import plaque_preview as ppv
from spacr.qt.widgets import segmentation_views as sv


BOXES = ((100, 100, 180, 180), (220, 100, 300, 180))


def _segment(crop):
    second = int(crop[0, 0, 0]) > 80
    labels = np.zeros(crop.shape[:2], dtype=np.int32)
    start = 35 if second else 10
    labels[start:start + 12, start:start + 12] = 1
    flows = np.zeros(crop.shape, dtype=np.uint8)
    flows[..., 0 if second else 1] = 160 if second else 210
    probability = np.full(crop.shape[:2], -5.0, dtype=np.float32)
    probability[labels > 0] = 4.0 if second else 1.0
    return labels, {"flow_rgb": flows, "cellprob": probability}


@pytest.fixture
def figure(qtbot, monkeypatch, tmp_path):
    monkeypatch.setitem(ppv._SESSION, "style", ppv.OverlayStyle())
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    image = np.full((400, 400, 3), 9, dtype=np.uint8)
    image[100:180, 100:180] = (41, 19, 70)
    image[100:180, 220:300] = (111, 93, 27)
    Image.fromarray(image).save(tmp_path / "figure.png")
    panel = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings({"plaque_mode": "figure", "confirm_annotations": False})
    panel.load_source_async(str(tmp_path))

    def detect(image, weights, **kwargs):
        return [SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1, confidence=0.9)
                for x0, y0, x1, y1 in BOXES]

    assert panel.run_preview(detect=detect, read_text=lambda path: [])
    return panel


def _choose(figure, view):
    selector = figure._view_selector
    selector.setCurrentIndex(selector.findData(view))
    assert selector.currentData() == figure.views().view() == view


def _expected(figure, index, view):
    region = figure._figure["regions"][index]
    crop = figure._figure["image"][region.y0:region.y1, region.x0:region.x1]
    labels, outputs = _segment(crop)
    if view == sv.OVERLAY:
        picture = crop.copy()
        ppv.render_overlay(picture, labels, figure._overlay_style)
        return picture
    if view == sv.MASKS:
        return sv.render_labels(labels)
    if view == sv.FLOWS:
        return outputs["flow_rgb"]
    return sv.render_cellprob(outputs["cellprob"])


@pytest.mark.parametrize("view", (sv.OVERLAY, sv.MASKS, sv.FLOWS, sv.CELLPROB))
def test_the_dropdown_and_well_selection_show_that_wells_actual_view(figure, view):
    assert figure.find_plaques_in_all_wells(segment=_segment)
    figure.select_well(0)
    figure._well_view.zoom(1.75)
    zoom, pan = figure._well_view._scale, figure._well_view._pan
    _choose(figure, view)
    assert figure._well_views.view() == view
    assert np.array_equal(figure._well_view.array(), _expected(figure, 0, view))
    assert figure._well_view._scale == zoom and figure._well_view._pan == pan
    first = figure._well_view.array().copy()
    assert figure._view.array().shape[:2] == (400, 400)
    figure.select_well(1)
    assert figure.selected_well() == 1
    assert figure.views().view() == figure._well_views.view() == view
    assert np.array_equal(figure._well_view.array(), _expected(figure, 1, view))
    assert not np.array_equal(figure._well_view.array(), first)
    assert figure._view.array().shape[:2] == (400, 400)


def test_unavailable_well_outputs_hide_the_previous_wells_picture(figure):
    figure.select_well(0)
    assert figure.preview_selected_well(segment=_segment)
    _choose(figure, sv.FLOWS)
    assert np.array_equal(figure._well_view.array(), _expected(figure, 0, sv.FLOWS))
    figure.select_well(1)
    for view in (sv.MASKS, sv.FLOWS, sv.CELLPROB):
        _choose(figure, view)
        assert figure._well_views.view() == view
        assert figure._well_views.message_text() == sv.WAITING
        assert figure._well_views.picture() is None
        assert not figure._well_view.isVisibleTo(figure._well_views)
    assert figure.preview_selected_well(segment=lambda crop: _segment(crop)[0])
    for view, message in ((sv.FLOWS, sv.NO_FLOWS), (sv.CELLPROB, sv.NO_CELLPROB)):
        _choose(figure, view)
        assert figure._well_views.message_text() == message
        assert figure._well_views.picture() is None
        assert not figure._well_view.isVisibleTo(figure._well_views)
    _choose(figure, sv.MASKS)
    assert np.array_equal(figure._well_view.array(), _expected(figure, 1, sv.MASKS))
    figure.select_well(0)
    assert figure._well_views.view() == sv.MASKS
    assert np.array_equal(figure._well_view.array(), _expected(figure, 0, sv.MASKS))
