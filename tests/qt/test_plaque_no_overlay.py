"""Plaque's None view shows source pixels without changing segmentation."""
import numpy as np

from spacr.qt.widgets import plaque_preview as ppv
from spacr.qt.widgets import segmentation_views as sv
from tests.qt.test_722_plaque_selected_well_views import figure, _segment


def test_none_view_keeps_original_figure_and_selected_well_pixels(figure):
    assert figure.find_plaques_in_all_wells(segment=_segment)
    labels = {key: value['labels'].copy() for key, value in figure._wells.items()}
    selector = figure._view_selector
    assert selector.findData('None') >= 0
    selector.setCurrentIndex(selector.findData('None'))
    assert figure.views().view() == 'None'
    assert np.array_equal(figure._view.array(), figure._figure['image'])
    for index in (0, 1):
        figure.select_well(index)
        region = figure._figure['regions'][index]
        crop = figure._figure['image'][region.y0:region.y1, region.x0:region.x1]
        assert figure._well_views.view() == 'None'
        assert np.array_equal(figure._well_view.array(), crop)
        assert np.array_equal(figure._wells[index]['labels'], labels[index])
    selector.setCurrentIndex(selector.findData(sv.OVERLAY))
    assert not np.array_equal(figure._view.array(), figure._figure['image'])


def test_none_view_in_plaque_mode_removes_only_the_display_overlay(qtbot, monkeypatch, tmp_path):
    from PIL import Image
    image = np.full((40, 50, 3), (19, 54, 93), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / 'plaque.png')
    monkeypatch.setattr(ppv, 'missing_papers_packages', lambda *a, **k: [])
    panel = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.load_source_async(str(tmp_path))
    labels = np.zeros((40, 50), dtype=np.int32)
    labels[5:20, 7:24] = 1
    assert panel.run_preview(segment=lambda path: labels)
    assert not np.array_equal(panel._view.array(), image)
    panel._view_selector.setCurrentIndex(panel._view_selector.findData('None'))
    assert np.array_equal(panel._view.array(), image)
    assert np.array_equal(panel._plaque_result['labels'], labels)
    assert panel._plaque_result['count'] == 1
